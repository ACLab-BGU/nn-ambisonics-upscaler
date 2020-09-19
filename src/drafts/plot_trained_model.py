import glob
import os

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src import options
from src.models.fc_model import BaseModelLT, default_opts, FC
from src.utils import get_experiments_dir

from src.utils.complex_tensors_old import complextorch2numpy
import src.utils.visualizations as vis

import matplotlib.pyplot as plt

# --- load model ---
version = 58

CKPT_PATH = os.path.join(get_experiments_dir(),'cloud_experiments','fc_imagemethod_5to6_full')
CKPT_PATH = os.path.join(CKPT_PATH,'version_' + str(version),'*','*ckpt')
CKPT_PATH = glob.glob(CKPT_PATH)[0]

CKPT_PATH = "/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/fc/image_method/fc_imagemethod_rank30/version_7/checkpoints/epoch=27.ckpt"

# model = BaseModelLT.load_from_checkpoint(CKPT_PATH,data_path=default_opts['data_path'])
model = FC.load_from_checkpoint(CKPT_PATH)
model.setup('fit')
model.setup('test')
model.eval()
print(model.hparams)

# --- load data &  estimate performance over all dataset ---
# loader_train = DataLoader(model.dataset_train, batch_size=model.hparams.batch_size, num_workers=model.hparams.num_workers)
# loader_test = DataLoader(model.dataset_test, batch_size=model.hparams.batch_size, num_workers=model.hparams.num_workers)
# trainer = Trainer(logger=None)
# trainer.test(model,test_dataloaders=loader_test)

# --- loop - draw a single sample and plot ---
loader = DataLoader(model.dataset_train, batch_size=1, num_workers=0)
# loader = DataLoader(model.dataset_test, batch_size=1, num_workers=0)
for x,y in loader:

    y_hat = model(x)

    x = complextorch2numpy(x[0])
    y = complextorch2numpy(y[0])
    y_hat = complextorch2numpy(y_hat[0])

    vis.compare_covs(x, y_hat, y)
    vis.compare_power_maps(x, y_hat, y)

    plt.close('all')
    pass