import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_3tuple
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
import pdb
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 4, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 4, 1, 2, 3)
        x = self.drop(x)
        return x

class Cluster(nn.Module):
    def __init__(self, dim, out_dim,
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param proposal_d: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param fold_d: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv3d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((proposal_w, proposal_h, proposal_d))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.fold_d = fold_d
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h, d]
        #pdb.set_trace()
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h d -> (b e) c w h d", e=self.heads)
        value = rearrange(value, "b (e c) w h d -> (b e) c w h d", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0, d0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0 and d0 % self.fold_d == 0, \
                f"Ensure the feature map size ({w0}*{h0}*{w0}) can be divided by fold " \
                f"{self.fold_w}*{self.fold_h}*{self.fold_d}"
            x = rearrange(x, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                          f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1],ks[2]]
            value = rearrange(value, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                              f2=self.fold_h, f3=self.fold_d)
        b, c, w, h, d = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H,C_D], we set M = C_W*C_H and N = w*h*d
        value_centers = rearrange(self.centers_proposal(value), 'b c w h d -> b (w h d) c')  # [b,C_W,C_H,c]
        b, c, ww, hh, dd = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h d -> b (w h d) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        # a small bug: mask.sum should be sim.sum according to Eq. (1),
        # mask can be considered as a hard version of sim in our implementation.
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h d) c -> b c w h d", w=ww, h=hh)  # center shape
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h d) c -> b c w h d", w=w, h=h)  # cluster shape

        if self.fold_w > 1 and self.fold_h > 1 and self.fold_d > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2 f3) c w h d -> b c (f1 w) (f2 h) (f3 d)", f1=self.fold_w,
                            f2=self.fold_h, f3=self.fold_d)
        out = rearrange(out, "(b e) c w h d -> b (e c) w h d", e=self.heads)
        out = self.proj(out)
        return out


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim,
                                   proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                                   fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                                   heads=heads, head_dim=head_dim, return_center=return_center)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # The following technique is useful to train deep ContextClusters.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            #pdb.set_trace()
            x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))
            #pdb.set_trace()
            x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        else:
            x = x + self.token_mixer(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x
    
def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim



def create_pos(x):
    _ , c , w , h , d = x.shape
        # 生成每个维度的坐标
    range_w = torch.arange(w, device=x.device, dtype=torch.float32)
    range_h = torch.arange(h, device=x.device, dtype=torch.float32)
    range_d = torch.arange(d, device=x.device, dtype=torch.float32)

    range_w = range_w / (w - 1.0)
    range_h = range_h / (h - 1.0)
    range_d = range_d / (d - 1.0)

    grid_w, grid_h, grid_d = torch.meshgrid(range_w, range_h, range_d, indexing='ij')

    position_encoding = torch.stack([grid_w, grid_h, grid_d], dim=-1)

    position_encoding = (position_encoding - 0.5) * 2. 

    # 调整维度顺序并添加batch维度
    position_encoding = position_encoding.permute(3, 0, 1, 2).unsqueeze(0)
    
    # 扩展到batch size
    position_encoding = position_encoding.expand(x.shape[0], -1, -1, -1, -1)

    return position_encoding


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Quant_Context_Cluster(nn.Module):
    def __init__(self, quant_size , quant_channels, embed_dim = 32 , out_dim= 8 , **kwargs):
        super().__init__()

        embedding_channels = 3+quant_channels

        self.pos_quant_embedding = PointRecuder(patch_size = 1 , stride = 1 , padding=0 , 
                                                in_chans=embedding_channels  , embed_dim=embed_dim)
        
        self.cluster_blocks = ClusterBlock(dim=embed_dim , mlp_ratio=2  , norm_layer=GroupNorm ,  drop = 0.  , use_layer_scale=True,
                                           layer_scale_init_value=1e-5 , 
                                           proposal_w= 4, proposal_d = 4, proposal_h= 4, 
                                           fold_w=2 ,fold_d=2 ,fold_h = 2 ,
                                           heads = 8 , head_dim = embed_dim,return_center =False )
        
        self.out_conv = torch.nn.Conv3d(32 , out_dim , 1)
    def embedding(self , x , pos):
        #pdb.set_trace()
        x = torch.cat([x , pos] , dim=1)
        x = self.pos_quant_embedding(x)

        return x 
    def forward(self, x ):
        #pdb.set_trace()
        pos = create_pos(x)
        x = self.embedding(x , pos)
        x = self.cluster_blocks(x)
        x = self.out_conv(x)
        #pdb.set_trace()
        return x 
