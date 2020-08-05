import glob
import os

import numpy as np
import torch

from src.options import read_yaml
from src.utils.logging import get_scalars_from_tb_log

k_best_models = 5
PATH_EXP = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_imagemethod_5to6_full'

PATH = os.path.join(PATH_EXP,'*','*','*ckpt')
# PATH = os.path.join(PATH_EXP,'*','events*')
files = glob.glob(PATH)
best_loss = [np.Inf] * k_best_models
best_hparams = [None] * k_best_models
best_version = best_hparams.copy()

for i,file in enumerate(files):
    ckpt = torch.load(file,map_location=torch.device('cpu'))
    loss = ckpt['checkpoint_callback_best_model_score'].numpy()
    version = os.path.basename(os.path.dirname(os.path.dirname(file)))
    if any(loss < best_loss):
        i = np.argmax(loss < best_loss)
        best_loss.insert(i,loss)
        best_hparams.insert(i,ckpt['hyper_parameters'])
        best_version.insert(i,version)
        del best_loss[-1]
        del best_hparams[-1]
        del best_version[-1]

print("best version: ", best_version, "\n loss: ", best_loss)
