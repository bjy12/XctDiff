import argparse
import pytorch_lightning as pl
import csv


from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#from data.lidc_idri import get_loader
from data.pelivic_ae import get_pelvic_loader
from ldm.util import instantiate_from_config
from ldm.modules.loggers.logger import ImageLogger
from metrics import calculate_mae , calculate_psnr , calculate_ssim
import torch
import nibabel as nib
import os
from tqdm import tqdm 
import numpy as np
import SimpleITK as sitk

from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

import pdb
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="the config path",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="the checkpoint file path",
    )
    parser.add_argument(
        "--out_dirs",
        type=str,
        help="the output directory",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="the generation step (default 50)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="generation eta (default 0.0)",
    )
    parser.add_argument(
        "--vis_option",
        type=bool,
        default=True,
        help="is option to turn visualization"
    )
    return parser.parse_args()

def sitk_save(path, image, spacing=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)

def save_case_metrics(metrics, case_name, output_dir):
    """Save metrics for a specific case to a CSV file"""
    import csv
    import os
    
    csv_path = os.path.join(output_dir, 'case_metrics.csv')
    is_new_file = not os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            # Write header if this is a new file
            writer.writerow(['case_name', 'psnr', 'ssim', 'mae'])
        
        # Write metrics for this case
        writer.writerow([
            case_name,
            metrics['psnr'][0],
            metrics['ssim'][0],
            metrics['mae'][0]
        ])

def evaluate_batch(generated_samples, target_samples):
    """Calculate metrics for a batch of samples"""
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }
    
    for gen, target in zip(generated_samples, target_samples):
        if gen.ndim == 4:  # (C, H, W, D)
            gen = gen[0]   # Now (H, W, D)
        if target.ndim == 4:
            target = target[0]
        
        # Calculate metrics
        #pdb.set_trace()
        psnr_val = calculate_psnr(gen, target)
        ssim_val = calculate_ssim(gen, target)
        mae_val = calculate_mae(gen, target)
        
        metrics['psnr'].append(psnr_val)
        metrics['ssim'].append(ssim_val)
        metrics['mae'].append(mae_val)
    
    return metrics
def save_metrics(metrics_dict, output_path):
    """Save metrics to a text file"""
    with open(output_path, 'w') as f:
        for metric_name, value in metrics_dict.items():
            f.write(f"{metric_name}: {value}\n")
def calculate_statistics(values):
    """Calculate mean, std, min, max for a list of values"""
    if not values:  # 如果列表为空
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0
        }
    
    values = np.array(values)
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }
if __name__ == '__main__':

    hparams = get_parser()

    cfg = OmegaConf.load(hparams.cfg_path)
    loader = get_pelvic_loader(cfg.data , train_mode='ldm')
    val_loader = loader[1]
    # loading model from cfg and load checkpoint
    #cfg.model.params.ckpt_path = hparams.ckpt_path
    out_dir = hparams.out_dirs
    os.makedirs(out_dir , exist_ok=True)

    vis_option = hparams.vis_option
    model = instantiate_from_config(cfg.model)
    model = model.cuda()
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }
    # DDIM sampler
    ddim_sampler = DDIMSampler(model)
    ddim_steps = hparams.ddim_steps
    eta = hparams.eta
    #pdb.set_trace()
    shape = (cfg.model.params.channels, cfg.model.params.image_size, cfg.model.params.image_size , cfg.model.params.image_size)
    B = 1 
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing validation set")):  
                  # print(xray.shape)
            #pdb.set_trace()
            name = batch['filename']
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            z ,c , x, xrec, xc = model.get_input(batch, k = cfg.model.params.first_stage_key,
                                                return_first_stage_outputs=True,
                                                force_c_encode=True,
                                                return_original_cond=True,
                                                bs=1)    
            #pdb.set_trace()
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size=B, shape=shape, conditioning=c,
                                                         verbose=False, eta=eta)
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clip(x_samples, -1, 1)
            x_samples = x_samples.cpu().numpy()
            x = x.cpu().numpy()


            #pdb.set_trace()
            batch_metrics = evaluate_batch(x_samples , x)
            # 保存每个case的指标
            save_case_metrics(batch_metrics, name[0], out_dir)
            # test metrics 
            # Store metrics
            for metric_name, values in batch_metrics.items():
                all_metrics[metric_name].extend(values)
            if vis_option == True:
                for i in range(B):
                    output_path = os.path.join(
                        out_dir,
                        f"sample_{name}_{i}.nii.gz"
                    )
                    image =  (x_samples[i] + 1) / 2.0
                    #pdb.set_trace()
                    sitk_save(path=output_path, spacing=np.array([2.5 ,2.5,2.5]) ,image=image[0] , uint8=True)
    
   # 计算并保存最终的统计结果
    final_metrics = {}
    for metric_name, values in all_metrics.items():
        stats = calculate_statistics(values)
        for stat_name, value in stats.items():
            final_metrics[f'{metric_name}_{stat_name}'] = value

    # 保存指标
    metrics_path = os.path.join(out_dir, 'metrics.txt')
    save_metrics(final_metrics, metrics_path)
    
    print("\nValidation Results:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # 打印详细的统计信息
    print("\nDetailed Statistics:")
    for metric_name in ['psnr', 'ssim', 'mae']:
        print(f"\n{metric_name.upper()} Statistics:")
        print(f"Mean: {final_metrics[f'{metric_name}_mean']:.4f}")
        print(f"Std:  {final_metrics[f'{metric_name}_std']:.4f}")
        print(f"Min:  {final_metrics[f'{metric_name}_min']:.4f}")
        print(f"Max:  {final_metrics[f'{metric_name}_max']:.4f}")