import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src import options
from src.models.fc_model import BaseModelLT

from src.utils.complex_tensors import complextorch2numpy
import src.utils.visualizations as vis

# --- load model ---

# CKPT_PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_lowrank/version_467/checkpoints/epoch=99.ckpt'
# CKPT_PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_full/version_222/checkpoints/epoch=99.ckpt'
CKPT_PATH = '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments/cloud_experiments/fc_hp_tune/fc_imagemethod_full/version_202/checkpoints/epoch=99.ckpt'

checkpoint = torch.load(CKPT_PATH,map_location=torch.device('cpu'))
opts = checkpoint['hyper_parameters']['opts'].copy()
del opts['data_path'],opts['logs_path']

base_opts = options.get_default_opts(opts)
full_opts = options.prepare_opts(base_opts,opts)
full_opts = options.validate_opts(full_opts)

model = BaseModelLT(opts)
model.setup('fit')
model.load_state_dict(checkpoint['state_dict'])
trainer = Trainer(logger=None)
model.eval()

# --- load data &  estimate performance over all dataset ---
# loader = DataLoader(model.dataset_train, batch_size=model.hparams.batch_size, num_workers=model.hparams.num_workers)
# trainer.test(model,test_dataloaders=loader)

# --- loop - draw a single sample and plot ---
loader = DataLoader(model.dataset_train, batch_size=1, num_workers=0)
for x,y in loader:

    y_hat = model(x)

    x = complextorch2numpy(x[0])
    y = complextorch2numpy(y[0])
    y_hat = complextorch2numpy(y_hat[0])

    vis.compare_covs(x, y_hat, y)
    vis.compare_power_maps(x, y_hat, y)

    pass