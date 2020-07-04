from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from src.models.fc_model import BaseModelLT
from src.options import prepare_opts

default_opts = {
    # ---folders---
    "data_path": '/Users/ranweisman/Google Drive/My Drive/My Documents/MATLAB/Research/nn-ambisonics-upscaler/raw/free-field',
    "logs_path": '/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/experiments',
    # ---network structure---
    "input_size": 512, # TODO: fix hardcoding
    "output_size": 4802, # TODO: fix hardcoding
    "output_shape": [2,49,49], # TODO: fix hardcoding
    "hidden_layers": 3,
    "hidden_sizes": [1600,2700,3800],
    "loss": nn.MSELoss(),
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": None,
    "batch_size": 10,
    "num_workers": 6,
    "train_val_split": [0.8,0.2],
    # ---optimization---
    "lr": 3e-4,
    "max_epochs": 2,
    "gpus": 0
}


"""
Additional Trainer args:
fast_dev_run = false (for debugging)
overfit_batches=0.01 (overfit over small datasets)
limit_train_batches=0.1, limit_val_batches=0.01, limit_test_batches=0.01 (use less data, to make epochs shorter)
progress_bar_refresh_rate (How often to refresh progress bar (in steps). Value 0 disables progress bar)
auto_lr_find (activate an algorithm to auto-find the learning rate)

check_val_every_n_epoch (Check val every n train epochs)
val_check_interval (How often within one training epoch to check the validation set)
log_save_interval (Writes logs to disk this often)
row_log_interval (How often to add logging rows (does not write to disk)
resume_from_checkpoint (to resume training from a specific checkpoint pass in the path here. This can be a URL.)
"""

opts = dict(prepare_opts(default_opts))
model = BaseModelLT(opts)
logger = TensorBoardLogger(opts['logs_path'],name='my_model_name')
trainer = Trainer(weights_summary='full', max_epochs=opts['max_epochs'], gpus=opts['gpus'],
                  default_root_dir=opts['logs_path'], logger=logger)
trainer.fit(model)
trainer.test()