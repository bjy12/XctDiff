import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
#from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.xray_encoder.Res_Unet_multi import ResNetGLEncoder
from ldm.modules.xray_encoder.unet import UNet
from ldm.modules.xray_encoder.implict_function import get_embedder , Implict_Fuc_Network
from ldm.modules.embedding.modules import Encoder, Decoder
from ldm.modules.view_fusion.view_agg_points import ViewPoints_Aggregation
import pdb
def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]
class MLP(nn.Module):
    def __init__(self, mlp_list, use_bn=False):
        super().__init__()

        layers = []
        for i in range(len(mlp_list) - 1):
            layers += [nn.Conv2d(mlp_list[i], mlp_list[i + 1], kernel_size=1)]
            if use_bn:
                layers += [nn.BatchNorm2d(mlp_list[i + 1])]
            layers += [nn.LeakyReLU(inplace=True),]
        
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
class XrayCondition(pl.LightningModule):
    def __init__(self,
                 Xrayencoder_config,
                 implict_fuc_config,
                 aggregation_config,
                 latent_res,
                 xray_encoder,
                 ckpt_path=None,
                 image_key='xray',
                 points_key='proj_points',
                 ignore_keys=[],
                 monitor=None,
                 scheduler_config=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 combine= 'mlp',
                 use_view_feature = False, 
                 use_ema=False , 
                 level_dict = {}
                 ):
        super().__init__()
        self.image_key = image_key
        self.points_key = points_key
        self.latent_res = latent_res
        self.scale_level = level_dict
        if self.scale_level is not None:
            multi_scale_levels = True
        pdb.set_trace()
        if xray_encoder == 'res_unet':
             self.image_feature_extractor = ResNetGLEncoder(**Xrayencoder_config)
        elif xray_encoder == 'vanilla':
             self.image_feature_extractor = UNet(**Xrayencoder_config)
        self.combine = combine
        self.use_view_feature = use_view_feature
        self.view_mixer = MLP([2, 2 // 2, 1])
        self.pos_implic , _  = get_embedder(multires=10 , i=0)
        self.multi_level_res = {'level_0':32, 'level_1':16, 'level_2':8}
        if self.use_view_feature:
            self.view_aggregator = ViewPoints_Aggregation(**aggregation_config)

        self.implict_fn = Implict_Fuc_Network(**implict_fuc_config)

 
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap,
        #                                 sane_index_shape=sane_index_shape)
        # self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scheduler_config = scheduler_config

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    def points_wise(self ,proj_feats, projs_points , view_feature):
        #pdb.set_trace()
        multi_level_condition = {}
        #pdb.set_trace()
        
        for idx , level in enumerate(self.scale_level):
            #pdb.set_trace
            p_points = projs_points[level]
            proj_f = [proj_feats[idx]]
            #pdb.set_trace()
            p_feats = self.forward_points(proj_f , p_points)
            if view_feature is not None:
                v_feat = view_feature[level]
                b , _ , _ ,_= v_feat.shape
                agg_points_feature = self.view_aggregator(v_feat , p_feats)
                res = self.multi_level_res[level]
                multi_level_condition[f'{level}'] = agg_points_feature.reshape(b , -1 , res ,res , res)
            else:
                #!to do for exp 
                multi_level_condition[f'{level}'] = p_feats
        #pdb.set_trace()
        return multi_level_condition
                
    def encode(self, x):
        #pdb.set_trace()
        projs = x['proj']
        proj_points = x['proj_points']
        coords = x['coord']
        if self.use_view_feature:
            view_features = x['c_view_feature']
        else:
            view_features = None
        #pdb.set_trace()
        b , m , c , w , h = projs.shape
        projs = projs.reshape(b * m, c, w, h) # B', C, W, H

        #pdb.set_trace()
        #* first version not use global feature 
        proj_feats , _  = self.image_feature_extractor(projs)
        #pdb.set_trace()
        #proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        #pdb.set_trace()
        for i in range(len(proj_feats)):
            #pdb.set_trace()
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_).contiguous() # B, M, C, W, H
        #pdb.set_trace()
        multi_level_condition =  self.points_wise(proj_feats , proj_points , view_features )
        condition_feats = multi_level_condition
        #pdb.set_trace()
        # points_feats =  self.forward_points(proj_feats, proj_points)    
        # #pdb.set_trace()
        # if self.use_view_feature:
        #     #pdb.set_trace()
        #     view_feature = x['c_view_feature']
        #     #pdb.set_trace()
        #     agg_points_feature = self.view_aggregator(view_feature , points_feats)
        #     condition_feats = agg_points_feature.permute(0,2,1)
        #     #pdb.set_trace()
        # #condition_feats = points_feats
        # #points_feats = points_feats.reshape(b, -1, self.latent_res, self.latent_res , self.latent_res) # B, C , latent_res , latent_res , latent_res 
        # #pdb.set_trace()
        # #pos_embed = self.pos_implic(coords)
        # #pdb.set_trace()
        # #condition_feats = self.implict_fn(pos_embed , points_feats , global_feats)
        # #pdb.set_trace()
        # condition_feats = condition_feats.reshape(b, -1, self.latent_res, self.latent_res , self.latent_res) # B, C , latent_res , latent_res , latent_res 
        #pdb.set_trace()
        return condition_feats
    def forward_points(self, proj_feats , projs_points):
        n_view = proj_feats[0].shape[1]
        # 1. query view-specific features
        p_list = []
        #pdb.set_trace()
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                #pdb.set_trace()
                feat = proj_f[:, i, ...] # B, C, W, H
                p = projs_points[:, i, ...] # B, N, 2
                #pdb.set_trace()
                p_feats = index_2d(feat, p) # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.stack(p_list, dim=-1) # B, C, N, M
        #pdb.set_trace()
         # 2. cross-view fusion
        if self.combine == 'max':
            p_feats = F.max_pool2d(p_feats, (1, n_view))
            p_feats = p_feats.squeeze(-1) # B, C, N
            condition_feats = p_feats  # 保存用于condition的特征
        elif self.combine == 'mlp':
            #pdb.set_trace()
            p_feats = p_feats.permute(0, 3, 1, 2)
            condition_feats = self.view_mixer(p_feats)  # 保存MLP输出的特征
            p_feats = condition_feats.squeeze(1)
        elif self.combine == 'mean':
            p_feats = torch.mean(p_feats, dim=-1) # B, C, N
            condition_feats = p_feats    
        elif self.combine == 'view_agg':
            p_feats = p_feats.permute(0,3,2,1)

        return p_feats
    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, projs_points, return_pred_indices=False):
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, x_key, y_key):
        pdb.set_trace()
        x, y = batch[x_key], batch[y_key] , 
        return x, y

    def training_step(self, batch, batch_idx):
        proj, proj_points = self.get_input(batch, self.image_key, self.points_key)

        return 

    def validation_step(self, batch, batch_idx):
        pdb.set_trace()
        x, y = self.get_input(batch, self.image_key, self.points_key )
        xrec, qloss, ind = self(x, return_pred_indices=True)
        q_loss = qloss.mean()

        # L2 loss
        rec_loss = F.mse_loss(y.contiguous(), xrec.contiguous()).mean()

        total_loss = rec_loss + q_loss
        self.log(f"val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.encoder.parameters()) + \
                 list(self.decoder.parameters()) + \
                 list(self.quantize.parameters()) + \
                 list(self.quant_conv.parameters()) + \
                 list(self.post_quant_conv.parameters())

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt], scheduler

        return [opt]

class XrayEmbeddingInterface(XrayCondition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        return dec