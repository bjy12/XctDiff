"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
import pdb
BERT_MODEL_DIM=768

def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock_Condition(nn.Module):
    """支持多尺度条件的ResNet块"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, multi_cond_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.has_cond = exists(multi_cond_dim)
        input_dim = dim + (multi_cond_dim or 0)
        
        self.block1 = Block(input_dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(input_dim, dim_out, 1) if dim != dim_out or self.has_cond else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        # 如果存在条件，在通道维度上拼接
        if exists(cond):
            x = torch.cat([x, cond], dim=1)

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        #pdb.set_trace()
        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias
        # pos_bias 8 32 32  # sim 1,1024 8 32 32 
        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        #pdb.set_trace()
        return self.to_out(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)

        q = q * self.scale
        sim = einsum('b h c i, b h c j -> b h i j', q, k)
        
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', x=x.shape[2], y=x.shape[3], z=x.shape[4])
        return self.to_out(out)
    
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        multi_level_cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8,
        has_self_cond=False,
    ):
        super().__init__()
        self.channels = channels  
        # video 
        # temporal attention and its relative positional encoding
        # rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        # def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
        #     dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        # # realistically will not be able to generate that many frames of video... yet
        # self.time_rel_pos_bias = RelativePositionBias(
        #     heads=attn_heads, max_distance=32)

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        channels = channels * 2 if has_self_cond else channels
        # liner init  
        self.init_conv = nn.Conv3d(channels, init_dim, 1)

        # self.init_temporal_attn = Residual(
        #     PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        #pdb.set_trace()
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )


        # text conditioning
        # self.has_cond = exists(cond_dim) or use_bert_text_cond
        #cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        #self.null_cond_emb = nn.Parameter(
        #    torch.randn(1, cond_dim)) if self.has_cond else None

        #cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        #block_klass = partial(ResnetBlock, groups=resnet_groups)
        #block_klass = partial(ResnetBlock_Condition , groups = resnet_groups)
        #block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        #block_klass = partial(ResnetBlock_Condition, groups=resnet_groups)
        #block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # 下采样路径
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            # 只在后两层使用attention (当ind >= 2时)
            use_attention = ind >= 3
            
            self.downs.append(nn.ModuleList([
                ResnetBlock_Condition(
                    dim_in, 
                    dim_out,
                    time_emb_dim=time_dim,
                    groups=resnet_groups,
                    multi_cond_dim=multi_level_cond_dim if ind < 3 else None
                ),
                ResnetBlock(
                    dim_out, 
                    dim_out,
                    time_emb_dim=time_dim,
                    groups=resnet_groups
                ),
                SpatialAttention(dim_out) if use_attention else nn.Identity(),
                nn.Conv3d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        # 中间层保留attention
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SpatialAttention(mid_dim)  # 保留中间层的attention
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 上采样路径
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            
            # 只在前两层使用attention (对应下采样的深层)
            use_attention = ind < 3
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                SpatialAttention(dim_in) if use_attention else nn.Identity(),
                nn.ConvTranspose3d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        # 输出层
        self.final_conv = nn.Sequential(
            ResnetBlock(dim * 2, dim),
            nn.Conv3d(dim, default(out_dim, channels), 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        self_cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):
        # assert not (self.has_cond and not exists(cond)
        #             ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        #pdb.set_trace()
        # focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
        #     (batch,), prob_focus_present, device=device))

        # time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        #x = torch.cat([x , cond[0]] , dim=1)
        #pdb.set_trace()
        #x = torch.cat([x, self_cond], dim=1) if self_cond is not None else x
        x = self.init_conv(x)
        r = x.clone()

        # x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        # if self.has_cond:
        #     batch, device = x.shape[0], x.device
        #     mask = prob_mask_like((batch,), null_cond_prob, device=device)
        #     cond = torch.where(rearrange(mask, 'b -> b 1'),
        #                        self.null_cond_emb, cond)
        #     t = torch.cat((t, cond), dim=-1)

        h = []
        #pdb.set_trace()
        # 下采样
        for level, (block1, block2, attn, downsample) in enumerate(self.downs):
            # 只在第一个block使用condition
            if level < 3 and exists(cond):
                #pdb.set_trace()
                current_cond = cond[f'level_{level}']
                x = block1(x, t, cond=current_cond)
            else:
                x = block1(x, t)
                
            x = block2(x, t)  # 第二个block不使用condition
            #pdb.set_trace()
            x = attn(x)
            h.append(x)
            x = downsample(x)
        #pdb.set_trace()

        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        #pdb.set_trace()
        return self.final_conv(x)

if __name__ == '__main__':
    data = torch.rand((1,8,32,32,32), dtype=torch.float32)
    data1 = torch.rand((1, 8, 32, 32, 32), dtype=torch.float32)
    model = Unet3D(dim=32, channels=8, out_dim=8, has_self_cond=True)
    t = torch.randint(0, 1000, (1,)).long()
    result = model(data, t, self_cond=data1)
    print(result.shape)
