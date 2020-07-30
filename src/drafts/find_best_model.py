import glob
import os

import numpy as np

from src.utils.logging import get_scalars_from_tb_log

PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_lowrank/*/events*'
logs_files = glob.glob(PATH)
train_loss_best = np.Inf
for file in logs_files:
    train_loss = get_scalars_from_tb_log(file,'train_loss').min()
    train_loss_best = np.minimum(train_loss,train_loss_best)
    if train_loss_best == train_loss:
        file_best = file

version = os.path.basename(os.path.dirname(file_best))

print("best version: ", version, ", train-loss: ", train_loss_best)
