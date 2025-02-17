import numpy as np
import torch
import pytorch_lightning as pl
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.ct_encoder.image_points_cluster import Quant_Context_Cluster
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pdb
class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        
        self.quant_cluster = Quant_Context_Cluster(quant_size=ddconfig['ch'] ,quant_channels=embed_dim,  )
        
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.val_metrics = {'psnr': [], 'ssim': []}

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

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

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        #pdb.set_trace()
        h = self.encoder(x)
        #pdb.set_trace()
        h = self.quant_conv(h)
        h = self.quant_cluster(h)

        #pdb.set_trace()
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

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

    def forward(self, input, return_pred_indices=False):
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix)

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix)
        x_norm = self.normalize_volume(x)
        xrec_norm = self.normalize_volume(xrec)
        
        # pdb.set_trace()
        # create random idx
        if batch_idx == 0:
            x_slices = self.get_center_slices(x_norm)
            xrec_slices = self.get_center_slices(xrec_norm)
            
            # 记录三个方向的对比图和指标
            for direction, (slice_gt, slice_rec) in zip(
                ['axial', 'sagittal', 'coronal'],
                zip(x_slices, xrec_slices)
            ):
                self.log_slice_comparison(slice_gt, slice_rec, direction, suffix)
    
        total_psnr, total_ssim = self.calculate_metrics(x_norm, xrec_norm)
        
        self.val_metrics['psnr'].append(total_psnr)
        self.val_metrics['ssim'].append(total_ssim)

        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        del log_dict_ae[f"val{suffix}/rec_loss"]

        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    def on_validation_epoch_end(self):

        avg_psnr = np.mean(self.val_metrics['psnr'])
        avg_ssim = np.mean(self.val_metrics['ssim'])
        self.log('val/avg_psnr', avg_psnr, sync_dist=True)
        self.log('val/avg_ssim', avg_ssim, sync_dist=True)
        self.logger.experiment.add_scalar('Metrics/avg_psnr', avg_psnr, self.global_step)
        self.logger.experiment.add_scalar('Metrics/avg_ssim', avg_ssim, self.global_step)
  
        # 清空累积器
        self.val_metrics = {'psnr': [], 'ssim': []}
        self.val_metrics_ema = {'psnr': [], 'ssim': []}

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def normalize_volume(self, volume):
        """将体数据标准化到[0,1]范围"""
        return (volume + 1.0) / 2.0
    
    def get_center_slices(self, volume):
        """获取三个方向的中心切片
        Args:
            volume (torch.Tensor): Shape [B, C, D, H, W]
        Returns:
            tuple: (axial, sagittal, coronal) 切片
        """
        B, C, D, H, W = volume.shape
        center_d = D // 2
        center_h = H // 2
        center_w = W // 2
        
        axial = volume[:, :, center_d, :, :]     # [B, C, H, W]
        sagittal = volume[:, :, :, :, center_w]  # [B, C, D, H]
        coronal = volume[:, :, :, center_h, :]   # [B, C, D, W]
        
        return axial, sagittal, coronal
    def tensor_to_numpy(self, tensor):
        """将tensor转换为numpy数组，并确保值域在[0,1]
        Args:
            tensor (torch.Tensor): 输入tensor [B,C,H,W] or [B,C,D,H,W]
        Returns:
            np.ndarray: numpy数组
        """
        # 确保tensor在CPU上
        tensor = tensor.detach().cpu()
        
        # 如果是batch，只取第一个样本
        if tensor.dim() == 5:  # 3D volume [B,C,D,H,W]
            tensor = tensor[0]  # [C,D,H,W]
        elif tensor.dim() == 4:  # 2D image [B,C,H,W]
            tensor = tensor[0]  # [C,H,W]
            
        # 转换为numpy并调整通道顺序
        array = tensor.numpy()
        
        # 确保值域在[0,1]之间
        array = np.clip(array, 0, 1)
        
        return array
    def log_metrics_to_tensorboard(self, psnr, ssim, direction, suffix=""):
        """记录PSNR和SSIM到Tensorboard
        Args:
            psnr (float): PSNR值
            ssim (float): SSIM值
            direction (str): 切片方向
            suffix (str): 日志后缀
        """
        # 记录到tensorboard的scalar部分
        self.logger.experiment.add_scalar(
            f'Metrics{suffix}/PSNR_{direction}',
            psnr,
            self.global_step
        )
        self.logger.experiment.add_scalar(
            f'Metrics{suffix}/SSIM_{direction}',
            ssim,
            self.global_step
        )
        
        # 同时也通过log方法记录
        self.log(f'val{suffix}/psnr_{direction}', psnr,
                on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'val{suffix}/ssim_{direction}', ssim,
                on_step=False, on_epoch=True, sync_dist=True)
    def calculate_metrics(self, gt, pred):
        """计算PSNR和SSIM
        Args:
            gt (torch.Tensor): Ground truth tensor
            pred (torch.Tensor): Predicted tensor
        Returns:
            tuple: (psnr, ssim) 指标值
        """
        # 转换为numpy数组
        gt_np = self.tensor_to_numpy(gt)
        pred_np = self.tensor_to_numpy(pred)
        
        # 计算PSNR
        try:
            psnr = peak_signal_noise_ratio(
                gt_np, 
                pred_np, 
                data_range=1.0
            )
        except Exception as e:
            print(f"PSNR calculation failed: {e}")
            psnr = 0.0
        
        # 计算SSIM
        try:
            # 对于3D数据，我们计算每个切片的SSIM然后取平均
            if gt_np.ndim == 4:  # [C,D,H,W]
                ssim_vals = []
                for d in range(gt_np.shape[1]):  # 遍历深度方向
                    ssim = structural_similarity(
                        gt_np[:, d],
                        pred_np[:, d],
                        channel_axis=0,
                        data_range=1.0
                    )
                    ssim_vals.append(ssim)
                ssim = np.mean(ssim_vals)
            else:  # 2D数据 [C,H,W]
                ssim = structural_similarity(
                    gt_np,
                    pred_np,
                    channel_axis=0,
                    data_range=1.0
                )
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            ssim = 0.0
            
        return psnr, ssim   

    def log_slice_comparison(self, slice_gt, slice_rec, direction, suffix=""):
        """记录单个方向的切片对比和指标"""
        # 创建对比图
        comparison = torch.cat([slice_gt, slice_rec], dim=-1)
        
        # 记录图像
        self.logger.experiment.add_images(
            f'val{suffix}/reconstruction_{direction}',
            comparison,
            self.global_step,
            dataformats='NCHW'
        )
        
        # 计算并记录PSNR和SSIM
        psnr, ssim = self.calculate_metrics(slice_gt, slice_rec)
        
        # 记录指标
        self.log_metrics_to_tensorboard(psnr, ssim, direction, suffix)
        
    
        


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        h = self.quant_cluster(h)
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
