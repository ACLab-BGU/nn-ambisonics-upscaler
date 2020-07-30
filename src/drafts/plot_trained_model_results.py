import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src import options
from src.models.fc_model import BaseModelLT

# --- load model ---
# CKPT_PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_lowrank/version_462/checkpoints/epoch=99.ckpt'
from src.utils.complex_tensors import complextorch2numpy

CKPT_PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_lowrank/version_467/checkpoints/epoch=99.ckpt'

checkpoint = torch.load(CKPT_PATH,map_location=torch.device('cpu'))
opts = checkpoint['hyper_parameters']['opts'].copy()
del opts['data_path'],opts['logs_path']

base_opts = options.get_default_opts(opts)
full_opts = options.prepare_opts(base_opts,opts)
full_opts = options.validate_opts(full_opts)

model = BaseModelLT(opts)
model.setup('test')
model.load_state_dict(checkpoint['state_dict'])
trainer = Trainer()
model.eval()

# --- load data ---
loader = DataLoader(model.dataset_train, batch_size=model.hparams.batch_size, num_workers=model.hparams.num_workers)

# --- estimate performance over all dataset ---
# trainer.test(model,test_dataloaders=loader)

# --- loop - draw a single sample and plot ---
loader = DataLoader(model.dataset_train, batch_size=1, num_workers=0)
for x,y in loader:
    y_hat = model(x)

    x = complextorch2numpy(x[0])
    y = complextorch2numpy(y[0])
    y_hat = complextorch2numpy(y_hat[0])