import os
from datetime import datetime
import argparse
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

#from data.lidc_idri import get_loader
#from data.pelivic_ae import get_pelvic_loader
from data.pelivic_ae_multi_scale import get_pelvic_loader
from ldm.util import instantiate_from_config
from ldm.modules.loggers.logger import ImageLogger
import shutil
import pdb

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="the config path",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200000,
        help="number of training iterations"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="training gpu number"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="accumulate grad batches"
    )
    parser.add_argument(
        "--every_n_train_steps",
        type=int,
        default=10000,
        help="frequency for saving checkpoint"
    )
    parser.add_argument(
        '--train_mode',
        type=str,
        default='autoencoder',
        help='train mode: autoencoder or ldm'
    )
    # 添加log目录参数
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="directory to save logs and checkpoints"
    )
    # 添加实验名称参数
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="experiment name"
    )

    return parser.parse_args()
def setup_logging_and_checkpoints(cfg_path, log_dir, exp_name=None):
    # 如果没有指定实验名称，使用时间戳
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验目录
    experiment_dir = os.path.join(log_dir, exp_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard_logs')
    
    # 创建所需的目录
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 复制配置文件
    cfg_name = os.path.basename(cfg_path)
    shutil.copy2(cfg_path, os.path.join(experiment_dir, cfg_name))
    
    return experiment_dir

if __name__ == '__main__':

    hparams = get_parser()
    #pdb.set_trace()
    cfg = OmegaConf.load(hparams.cfg_path)
    # 设置实验目录
    experiment_dir = setup_logging_and_checkpoints(
        hparams.cfg_path, 
        hparams.log_dir,
        hparams.exp_name
    )    
    
    pl.seed_everything(hparams.seed)


    # loading datasets
    loader = get_pelvic_loader(cfg.data , hparams.train_mode)

    # loading model from cfg
    model = instantiate_from_config(cfg.model)

    # configure learning rate
    model.learning_rate = cfg.model.base_learning_rate

    callbacks = []
    # val/loss_ema

    # 其他checkpoints
    epoch_ckpt = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, 'checkpoints'),
        filename='last_epoch',
        every_n_epochs=10,
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(epoch_ckpt)
    callbacks.append(ModelCheckpoint(monitor=cfg.model.params.monitor,
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))

    callbacks.append(ModelCheckpoint(every_n_train_steps=hparams.every_n_train_steps, save_top_k=-1,
                                     filename='{epoch}-{step}-{train/rec_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    accelerator = 'gpu' if hparams.gpus == 1 else 'ddp'

    # 设置logger
    logger = pl.loggers.TensorBoardLogger(
        save_dir = experiment_dir,
        name = None,  # 设置为None避免创建子目录
        version = '',  # 设置为空字符串避免创建版本子目录
        default_hp_metric = False
    )
    trainer = pl.Trainer(
        # precision=hparams.precision,
        gpus=hparams.gpus,
        callbacks=callbacks,
        max_steps=hparams.max_steps,
        accelerator=accelerator,
        accumulate_grad_batches=1,
        logger = logger,
        default_root_dir=experiment_dir
    )

    trainer.fit(model, loader[0], loader[1])

