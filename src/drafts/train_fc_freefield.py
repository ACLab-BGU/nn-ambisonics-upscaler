import os

from src.train import train
from src.utils import get_experiments_dir, get_data_dir

opts = {
    # ---folders---
    "data_path":  os.path.join(get_data_dir(),'SCM','free-field'),
    "logs_path":  os.path.join(get_experiments_dir(),'fc'),
    "experiment_name": 'fc_freefield_full',
    # ---network structure---
    "model_name": 'fc',
    "input_sh_order": 3,
    "rank": None, # None -> output is full matrix, Int -> output is low rank matrix transformed into full matrix
    "hidden_layers": 1,
    "hidden_sizes": [3500],
    "residual_flag": False,
    "residual_only": False,
    # ---data---
    "batch_size": 25,
    "num_workers": 15,
    # ---optimization---
    "lr": 1e-3,
    "lr_sched_thresh": 0.01,
    "lr_sched_patience": 5,
    "max_epochs": 1000,
    "gpus": -1
}

train(opts)
