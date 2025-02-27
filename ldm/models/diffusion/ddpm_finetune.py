import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


import random
from ldm.util import default, instantiate_from_config
from ldm.modules.diffusionmodules.util import extract_into_tensor
from ldm.models.diffusion.ddpm import LatentDiffusion

import pdb


class FinetuneingDiffusion(LatentDiffusion):
    def __init__(self, *args , **kwargs):
        super().__init__(*args, **kwargs)
    
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.train()
    
    def shared_step(self, batch):
        #pdb.set_trace()
        x , c , y = self.get_input(batch , self.first_stage_key , return_x=True)

        loss = self(x,c,y)

        return loss
    
    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        
        loss , loss_dict = self.shared_step(batch)
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def forward(self, x, c, y , *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        #pdb.set_trace()
        return self.p_losses(x,c, t, y,*args, **kwargs)
    
    def get_fine_loss(self, pred, target, m, s):
        if self.loss_type == 'l1':
            loss = (m / s) * (target - pred).abs().mean(dim=[1, 2, 3])
        elif self.loss_type == 'l2':
            loss = ((m / s) ** 2) * torch.nn.functional.mse_loss(target, pred, reduction='none').mean(dim=[1, 2, 3])
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss    
    def p_losses(self, x_start, cond, t, y,noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        #pdb.set_trace()
        x_noisy , mean , std = self.q_sample(x_start=x_start, t=t, noise=noise ,return_mean_std=True)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'     

        if self.parameterization == "x0":
            target = x_start
            x_0_pred = model_output
        elif self.parameterization == "eps":
            target = noise
            x_0_pred = (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_noisy.shape) * x_noisy -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_noisy.shape) * model_output
            )
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        
        y_pred = self.decode_first_stage(x_0_pred, output_size=int(y.shape[-1]), requires_grad=True)

        loss_l1_recon = self.get_fine_loss(x_0_pred, x_start, mean, std).mean()

        loss_dict.update({f'{prefix}/loss_l1_recon': loss_l1_recon})
        loss_l1_image = self.get_fine_loss(y_pred, y, mean, std).mean()


        loss_dict.update({f'{prefix}/loss_l1_image':loss_l1_image})

        loss = 1.0 * loss_l1_recon + 1.0 * loss_l1_image

        loss_dict.update({f'{prefix}/loss': loss})


        return loss, loss_dict
    

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt       
