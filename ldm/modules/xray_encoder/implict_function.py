import torch
import torch.nn as nn
from torch.nn.functional import silu
import pdb

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim  ## self.kwargs["input_dims"] + self.kwargs["input_dims"] * 2 * self.kwargs["num_freqs"]

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class DummyNeRF(nn.Module):
    def __init__(self, input_ch , output_ch):
        """ """
        super(DummyNeRF, self).__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        print(f"[DummyNeRF] Input ch : {self.input_ch}, Output ch : {self.output_ch}")

        ## for dimension reduction
        if self.input_ch != self.output_ch:
            self.linear = nn.Linear(self.input_ch, self.output_ch)

    def forward(self, x , l_features):
        res_features = l_features 
        # pdb.set_trace()

        if self.input_ch != self.output_ch:
            func_implict = self.linear(torch.cat([x , l_features] , dim=-1 ))
        output = torch.cat((func_implict , res_features) , dim=-1 )
        return output
    

class Implict_Fuc_Network(nn.Module):
    def __init__(self, pos_dim , local_f_dim , num_layer , hidden_dim , output_dim , skips=[2] ,last_activation='relu',use_silu=False, no_activation=False):
        super().__init__()

        self.in_dim = pos_dim + local_f_dim

        self.num_layers = num_layer 

        self.skips = skips

        self.layers = nn.ModuleList([])  # 添加这行
        # 第一层
        self.layers.append(nn.Linear(self.in_dim, hidden_dim))
        # 中间隐藏层
        for i in range(1, self.num_layers - 1):
            if i in skips:
                # 跳跃连接层的输入维度需要增加原始输入维度
                self.layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 最后一层输出
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # 激活函数配置
        self.activations = nn.ModuleList()
        if no_activation:
            self.activations = nn.ModuleList([nn.Identity() for _ in range(self.num_layers - 1)])
        elif use_silu:
            self.activations = nn.ModuleList([nn.SiLU() for _ in range(self.num_layers - 1)])
        else:
            self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(self.num_layers - 1)])
        
        # 最后一层激活函数
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "silu":
            self.activations.append(nn.SiLU())
        elif no_activation:
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError("Unknown last activation")
        
    def forward(self, pos_feature, local_feature , global_feats):
        #pdb.set_trace()
        input_features = torch.cat([ pos_feature , global_feats ] , dim=-1)
        res_feats = local_feature   #
        #res_feats = self.global_feat_processor(res_feats) 
        #pdb.set_trace()
        x = input_features
        # 前向传播
        for i in range(len(self.layers)):
            # 在跳跃连接层，将原始输入特征拼接到当前特征
            if i in self.skips:
                #pdb.set_trace()
                x = torch.cat([pos_feature, x], dim=-1)
            
            linear = self.layers[i]
            activation = self.activations[i]
            
            x = linear(x)
            x = activation(x)
        #pdb.set_trace()
        output = torch.cat([x , res_feats] , dim=2)
        #pdb.set_trace()


        
        return output



if __name__ == "__main__":
    pos_dim = 63      # 位置特征维度，比如xyz坐标
    local_f_dim = 64 # 局部特征维度
    hidden_dim = 128 # 隐藏层维度
    output_dim = 64   # 输出维度
    num_layer = 4    # 层数
    skips=[2] 
    implict_func_net = Implict_Fuc_Network(pos_dim=pos_dim , local_f_dim=local_f_dim , num_layer=num_layer ,
                                           hidden_dim=hidden_dim , output_dim=output_dim , skips=skips,
                                           last_activation='silu',use_silu=True, no_activation=True)
    implict_func_net = implict_func_net.to('cuda')
    pos_feature = torch.randn(1 , pos_dim).to('cuda')
    local_feature = torch.randn(1 , local_f_dim).to('cuda')
    output = implict_func_net(pos_feature , local_feature)
    print(output.shape)
