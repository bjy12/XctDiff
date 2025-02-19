import sys 
import torch
import torch.nn as nn
import pdb
from ldm.modules.view_fusion.fusion_base_modules import MLP
from einops import rearrange
import math
import torch.nn.functional as F



def group_sizes(num_elements, num_groups):
    """Local helper to compute the group sizes, when distributing
    num_elements across num_groups while keeping group sizes as close
    as possible."""
    sizes = torch.full(
        (num_groups,), math.floor(num_elements / num_groups),
        dtype=torch.long)
    sizes += torch.arange(num_groups) < num_elements - sizes.sum()
    return sizes


def expand_group_feat(A, num_groups, num_channels):
    if num_groups == 1:
        A = A.view(-1, 1)
    elif num_groups < num_channels:
        # Expand compatibilities to features of the same group
        sizes = group_sizes(num_channels, num_groups).to(A.device)
        A = A.repeat_interleave(sizes, dim=1)
    return A



class ViewPoints_Aggregation(nn.Module):
    def __init__(self, in_view_channels:int , in_points_channels:int , view_embedd_channels:int , 
                 points_embedd_channels:int , nc_inner:int , num_groups:int , use_points_mlp:bool,
                 scaling:bool , eps:float , gating:bool, g_type:str,
                 deep_set_feat):
        super(ViewPoints_Aggregation, self).__init__()
        # parameters
        """
        in_view_channels  输入view feature 的通道数 
        in_points_channels 输入points feature 的通道数 
        view_embedd_channels 将view feature映射到高维空间的通道数
        poitns_embedd_channels 将points feature映射到高维空间的通道数
        nc_inner  将view feature映射到nc_inner个组
        """    
        self.view_embedd_channels = view_embedd_channels
        self.in_view_channels = in_view_channels
        self.in_points_channels = in_points_channels
        embed_dim = 20


        self.view_embedder = nn.Sequential(
            nn.Linear(in_view_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.view_encoder = DeepSetFeat(embed_dim ,nc_inner , **deep_set_feat)
        self.points_encoder = MLP([in_points_channels , points_embedd_channels , points_embedd_channels] , bias=False)
        
        #self.E_score = nn.Linear(nc_inner , num_groups , bias=True)  replace nn.liner
        self.E_score = MLP([nc_inner , num_groups] , bias=True)

        self.scaling = scaling
        self.eps = eps
        self.num_groups = num_groups
        self.points_embedd_c = points_embedd_channels


        self.use_gate = gating
        self.g_type = g_type
        self.G = Gating(num_groups ,bias = True) if gating else None
        self.use_points_mlp = use_points_mlp




    def points_encoder_process(self, points_feature):
        #pdb.set_trace()
        b , v, n , c  = points_feature.shape

        points_feature = rearrange(points_feature, 'b v n c -> (b v) n  c')

        points_feature = self.points_encoder(points_feature)

        points_feature = rearrange(points_feature, '(b v) n c -> b v n c', v=v, n=n)

        return points_feature
    
    def compute_view_attention(self, view_features , scaling , eps):

        """
        softmax(xi) = exp(xi) / Σ_j exp(xj)
        """
        #pdb.set_trace()
        # view_feature  是
        b , v , n , c  = view_features.shape
        # 
        view_features = rearrange(view_features, 'b v n c -> (b v) n c') # 1 20000 32 
        #pdb.set_trace()
        view_score = self.E_score(view_features) #  1 20000 2  （2是group num 对于每个点把特征分解为2组）

        view_score = rearrange(view_score , '(b v) n g -> b n v g', v=v)

        scale_factor = 1.0 / math.sqrt(v)

        scale_scores = scale_factor * view_score

        attention = torch.softmax(scale_scores , dim=2)

        attention = attention.permute(0,2,1,3)  # b v n group
        
        #pdb.set_trace()

        return attention

    def apply_attention(self, points_feature, view_score):
        """
        应用视角注意力权重到点特征，实现多视角融合。
        Args:
            points_feature: (b, v, n, c)  # 输入特征(batch, 视角数, 点数, 通道数）
            view_score: (b, v, n, k)      # 注意力权重(k为特征组数)
        Returns:
            fused_features: (b, n, c)     # 融合后的特征
        """
        # 1. 检查输入维度
        #pdb.set_trace()
        b, v, n, c = points_feature.shape
        _, _, _, k = view_score.shape
        assert c % k == 0, "通道数c必须能被组数k整除"

        # 2. 将特征按组拆分
        features_per_group = c // k
        points_feature_grouped = points_feature.view(b, v, n, k, features_per_group)  # (b, v, n, k, c//k)

        # 3. 扩展注意力权重以匹配特征形状
        attention_scores = view_score.unsqueeze(-1)                  # (b, v, n, k, 1)
        attention_scores = attention_scores.expand(-1, -1, -1, -1, features_per_group)  # 虚拟扩展，无内存复制

        # 4. 加权并融合视角维度
        weighted_features = points_feature_grouped * attention_scores  # (b, v, n, k, c//k)
        fused_features = weighted_features.sum(dim=1)                 # (b, n, k, c//k)

        # 5. 合并组维度
        fused_features = fused_features.reshape(b, n, c)              # (b, n, c)

        return fused_features
    def apply_gate(self, points_pool , view_score , g_type):
        """
        points_pool : b n c 
        view_score : b v n group_num   对于每个点在每个视角下的每个特征组的权重  
        g_type : max  ( max for maximum over views )
        """

        #pdb.set_trace()
    
        b , v , n , num_goups = view_score.shape
        _ , _ , c = points_pool.shape
        features_per_group = c // num_goups
        if g_type == 'max':

            g_score = torch.max(view_score,dim=1)[0]

        gatting = self.G(g_score)
        #pdb.set_trace()

        gatting = rearrange(gatting , " (b n ) num_groups -> b n num_groups" , b = b , n = n)

        gatting = gatting.unsqueeze(-1)
        gatting = gatting.expand(-1,-1,-1,features_per_group)
        gatting = gatting.reshape(b,n,-1)

        gated_features = points_pool * gatting



        return gated_features
    def view_embedding(self , view_feature):
        b , v , p , c  = view_feature.shape
        view_feature = view_feature.reshape(-1,c)
        view_feature = self.view_embedder(view_feature)
        view_feature = view_feature.reshape(b,v,p,-1)
        return view_feature


   
    def forward(self , view_feature , points_feature):

        #pdb.set_trace()
        view_feature = self.view_embedding(view_feature)
        #pdb.set_trace()
        # encoder every points each view feature , 学习每个视角的特征并融合每个视角上每个点的池化特征
        view_features =  self.view_encoder(view_feature)  
        #pdb.set_trace()
        # encoder every points feature each view
        #pdb.set_trace()
        if self.use_points_mlp:
            points_feature = self.points_encoder_process(points_feature)
        # computre view attention 
        #pdb.set_trace()
        view_score = self.compute_view_attention(view_features , self.scaling , self.eps) 
        # apply attention 
        #pdb.set_trace()
        fusion_points = self.apply_attention(points_feature , view_score)

        if self.use_gate:
            fusion_points  = self.apply_gate(fusion_points , view_score , self.g_type)

        return fusion_points 



    







class DeepSetFeat(nn.Module):
    """Produce element-wise set features based on shared learned
    features.

    Inspired from:
        DeepSets: https://arxiv.org/abs/1703.06114
        PointNet: https://arxiv.org/abs/1612.00593
    """

    _POOLING_MODES = ['max', 'mean', 'min', 'sum']
    _FUSION_MODES = ['residual', 'concatenation', 'both']

    def __init__(
            self, d_in, d_out, pool='max', fusion='concatenation',
            use_num=False, **kwargs):
        super(DeepSetFeat, self).__init__()

        # Initialize the set-pooling mechanism to aggregate features of
        # elements-level features to set-level features
        self.pool = pool.split('_')
        assert all([p in self._POOLING_MODES for p in self.pool]), \
            f"Unsupported pool='{pool}'. Expected elements of: {self._POOLING_MODES}"
        #pdb.set_trace()    
        self.f_pool = lambda x: torch.cat([
            self._pool_single(x, p) for p in self.pool], dim=-1)

        # Initialize the fusion mechanism to merge set-level and
        # element-level features
        if fusion == 'residual':
            self.f_fusion = lambda a, b: a + b
        elif fusion == 'concatenation':
            self.f_fusion = lambda a, b: torch.cat((a, b), dim=-1)
        elif fusion == 'both':
            self.f_fusion = lambda a, b: torch.cat((a, a + b), dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown fusion='{fusion}'. Please choose among "
                f"supported modes: {self._FUSION_MODES}.")
        self.fusion = fusion

        # Initialize the MLPs
        self.d_in = d_in
        self.d_out = d_out
        self.use_num = use_num
        self.mlp_elt_1 = MLP(
            [d_in, d_out, d_out], bias=False)
        in_set_mlp = d_out * len(self.pool) + self.use_num
        self.mlp_set = MLP(
            [in_set_mlp, d_out, d_out], bias=False)
        in_last_mlp = d_out if fusion == 'residual' else d_out * 2
        self.mlp_elt_2 = MLP(
            [in_last_mlp, d_out, d_out], bias=False)
        
    def _pool_single(self, x, pool_type):
        """单个池化操作
        Args:
            x: tensor of shape [B , V, N , C] (N个点，V个视角，C维特征)
            pool_type: 池化类型
        Returns:
            pooled: tensor of shape [B,N,C]
        """
        if pool_type == 'max':
            return torch.max(x, dim=0)[0]
        elif pool_type == 'mean':
            return torch.mean(x, dim=0)
        elif pool_type == 'min':
            return torch.min(x, dim=0)[0]
        elif pool_type == 'sum':
            return torch.sum(x, dim=0)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")


    def forward(self, x):
        #pdb.set_trace()
        # x  每个点在每个视角下的视角特征 
        b , v , n , c  = x.shape   # b view_number , n_points , manual_feature
        #pdb.set_trace()
        # 1. 映射每个点在每个视角下的视角特征
        x = rearrange(x, 'b v n c -> (b v)  n c') #   all_v(b*v) n c 
        x = self.mlp_elt_1(x) # all_v n c   mlp 学习每个点在每个视角下的视角特征
        #pdb.set_trace() 
        x_set = self.f_pool(x) # 10000 32 每个点的视角特征的池化结果   pool view feature  [n , pool] 
        #pdb.set_trace()
        x_set = x_set.repeat(b*v , 1, 1 )
        #pdb.set_trace()
        x_set = self.mlp_set(x_set) # 10000 32  mlp 学习池化结果 learning pool feature  [b , n , c_pool]

        #pdb.set_trace()
        x_out = self.f_fusion(x, x_set) # 融合每个点的视角特征和池化结果  
        x_out = self.mlp_elt_2(x_out) #
        x_out = rearrange(x_out, '(b v) n  c  -> b v n c', v=v, n=n)
        #pdb.set_trace()

        return x_out

    def extra_repr(self) -> str:
        repr_attr = ['pool', 'fusion', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])
    

class Gating(nn.Module):
    """Rectified-tanh gating mechanism with learnable linear correction."""
    def __init__(self, num_groups, weight=True, bias=True, activation='tanh+'):
        super(Gating, self).__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_groups)) if weight \
            else None
        self.bias = nn.Parameter(torch.zeros(1, num_groups)) if bias else None
        if activation == 'tanh+':
            self.activation = lambda x: torch.tanh_(F.relu(x, inplace=True))
        elif activation == 'sigmoid':
            self.activation = lambda x: torch.sigmoid_(x)
        else:
            raise ValueError(f"Activation '{activation}' not supported for Gating")

    def forward(self, x):
        #pdb.set_trace()
        if self.weight is not None:
            x *= self.weight
        if self.bias is not None:
            x += self.bias
        return torch.tanh(
            F.relu(x, inplace=True)).view(-1, self.num_groups).squeeze(1)

    def extra_repr(self) -> str:
        return f'num_groups={self.num_groups}, ' \
            f'weight={self.weight is not None}, bias={self.bias is not None}'