import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.fc_model import BaseModelLT
from src.options import prepare_opts

default_config_path = os.path.abspath("../options/fc_default_config.yaml")
config = {
    # ---folders---
    "data_path":  os.path.abspath('../../data/raw/free-field'),
    "logs_path":  os.path.abspath('../../experiments'),
    "model_name": 'fc_freefield',
    # ---network structure---
    "rank": None, # None -> output is full matrix, Int -> output is low rank matrix transformed into full matrix
    "hidden_layers": 2,
    "hidden_sizes": [1900,2500],
    "loss": 'mse', # 'mse'
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": None,
    "batch_size": 30,
    "num_workers": 6,
    "train_val_split": [0.9,0.1],
    "preload_data": True,
    # ---optimization---
    "lr": 3e-4,
    "max_epochs": 1000,
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

if __name__ == '__main__':
    # opts = dict(prepare_opts(default_config_path,config))
    opts = dict(prepare_opts(config))
    model = BaseModelLT(opts)
    logger = TensorBoardLogger(opts['logs_path'],name=opts['model_name'])
    trainer = Trainer(weights_summary='full', max_epochs=opts['max_epochs'], gpus=opts['gpus'],
                      default_root_dir=opts['logs_path'], logger=logger)
    trainer.fit(model)
    trainer.test()