import random
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR
from ldm.models.implict_autoencoder.model import LIIF
from ldm.modules.diffusionmodules.model import Encoder_Implict , Decoder_Implict
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.models.implict_autoencoder.util import to_pixel_samples ,to_pixel_samples_3d
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import pdb
def disabled_train(self, mode=True):
    return self


class IND(nn.Module):
    def __init__(self, ddconfig, liifconfig):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv3d(ddconfig["z_channels"], ddconfig["z_channels"], 1),
            Decoder_Implict(**ddconfig)
        )
        self.inr = LIIF(in_dim=ddconfig['ch']*ddconfig['ch_mult'][0], out_dim=1, **liifconfig)

    def forward(self, z, coord=None, cell=None, output_size=None, return_img=True, bsize=0):
        #pdb.set_trace()
        h = self.decoder(z)
        #pdb.set_trace()
        return self.inr(h, coord=coord, cell=cell,return_img=return_img ,output_size=output_size)


class FirstStageModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 liifconfig,
                 lossconfig,
                 trainconfig=None,
                 valconfig=None,
                 scheduler_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 nll_weight=0.0,
                 use_saconv=False,
                 freeze_encoder=False,
                 q_sample=True
                 ):
        super().__init__()
        self.trainconfig = trainconfig
        self.valconfig = valconfig
        #pdb.set_trace()
        self.q_sample = q_sample
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        #pdb.set_trace()
        self.encoder = nn.Sequential(
            Encoder_Implict(**ddconfig),
            nn.Conv3d(2*ddconfig["z_channels"], 2*ddconfig["z_channels"], 1) if ddconfig["double_z"] \
            else nn.Conv3d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        )

        self.decoder = IND(ddconfig=ddconfig, liifconfig=liifconfig)

        self.loss = instantiate_from_config(lossconfig)
        self.use_posterior = ddconfig["double_z"]

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze_encoder:
            self.encoder = self.encoder.eval()
            self.encoder.train = disabled_train
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.val_metrics = {'psnr': [], 'ssim': []}


    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif 'model' in ckpt:
            sd = torch.load(path, map_location="cpu")["model"]['sd']
        else:
            raise NotImplementedError

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # For div2k begin
        # for k in keys:
        #     if k.startswith('encoder'):
        #         sd['encoder.0' + k[7:]] = sd[k]
        #         del sd[k]
        #     elif k.startswith('quant_conv'):
        #         sd['encoder.1' + k[10:]] = sd[k]
        #         del sd[k]
        # end
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        #pdb.set_trace()
        if self.use_posterior:
            return DiagonalGaussianDistribution(h)
        else:
            return h

    def decode(self, z, coord=None, cell=None, output_size=None, return_img=True, bsize=0):
        #pdb.set_trace()
        return self.decoder(z, coord=coord, cell=cell, output_size=output_size, return_img=return_img, bsize=bsize)

    def forward(self, input, coord=None, cell=None, output_size=None, 
                return_img=True, bsize=0, sample_posterior=True):
        #pdb.set_trace()
        posterior = self.encode(input)
        if not self.use_posterior:
            z = posterior
        elif sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, coord, cell, output_size, return_img, bsize)
        return dec, posterior

    def get_input(self, batch, config):
        #pdb.set_trace()
        image = batch['image']
        #coord = batch['coord']

        image = image.to(memory_format=torch.contiguous_format).float()
        #coord = coord.to(memory_format=torch.contiguous_format).float()
        gt = image
        fconfig = {'bsize': config.get('bsize', 0)}
        #pdb.set_trace()
        fconfig = {'return_img': config.get('return_img',True)}
        #pdb.set_trace()
        if 'sample_q' in config:

            sample_q = config['sample_q']
            #pdb.set_trace()
            coord, cell, gt = to_pixel_samples_3d(gt)
            #pdb.set_trace()
            sample_lst = np.random.choice(
                gt.shape[1], sample_q, replace=False)
            #pdb.set_trace()
            coord = coord[:,sample_lst,:]
            cell = cell[:,sample_lst,:]
            gt = gt[:,sample_lst,:]
            #pdb.set_trace()
            fconfig.update(
                coord=coord,
                cell=cell
            )
        else:
            fconfig.update(
                output_size=gt.shape[-1],
            )
        #pdb.set_trace()
        return image , gt , fconfig

    def training_step(self, batch, batch_idx):
        image, gt , fconfig = self.get_input(batch, self.trainconfig)
        #pdb.set_trace()
        #fconfig.update(return_img=False)
        #pdb.set_trace()
        print(" training_step")
        reconstructions, posterior = self(image,**fconfig )
        #pdb.set_trace()
        rec_loss, log_dict = self.loss(gt, reconstructions, split="train")
        self.log("rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return rec_loss

    def validation_step(self, batch, batch_idx ,suffix=""):
        image, gt , fconfig = self.get_input(batch, self.valconfig)
        #pdb.set_trace()
        print("validation_steps")
        reconstructions, posterior = self(image,  **fconfig  )
        rec_loss, log_dict = self.loss(gt, reconstructions, split="val")
        
        x_norm = self.normalize_volume(image)
        xrec_norm = self.normalize_volume(reconstructions)
        #pdb.set_trace()
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

        self.val_metrics['psnr'].append(total_psnr)
        self.val_metrics['ssim'].append(total_ssim)

        self.log("val/rec_loss", log_dict["val/rec_loss"])
        self.log_dict(log_dict)
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
        lr = self.learning_rate
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
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

        return opt

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inp, gt, fconfig = self.get_input(batch, self.valconfig) # TODO
        inp = inp.to(self.device)
        if not only_inputs:
            xrec, posterior = self(inp, **fconfig)
            log["reconstructions"] = xrec
            log["gt"] = gt
        log["inputs"] = inp
        return log
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
        