import glob
import os

import numpy as np
import torch

from src.options import read_yaml
from src.utils.logging import get_scalars_from_tb_log

# PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_lowrank/*/events*'
PATH_LOG = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_full/*/events*'
logs_files = glob.glob(PATH_LOG)
train_loss_best = np.Inf
for i,log_file in enumerate(logs_files):
    chkpt_file = glob.glob(os.path.dirname(log_file)+ '/checkpoints/*ckpt')
    chkpt = torch.load(chkpt_file[0],map_location=torch.device('cpu') )
    if chkpt['hyper_parameters']["opts"]['residual_flag'] is False:
        continue
    train_loss = get_scalars_from_tb_log(log_file, 'train_loss').min()
    train_loss_best = np.minimum(train_loss,train_loss_best)
    if train_loss_best == train_loss:
        file_best = log_file
        hparams = chkpt['hyper_parameters']["opts"]

version = os.path.basename(os.path.dirname(file_best))

print("best version: ", version, ", train-loss: ", train_loss_best)
