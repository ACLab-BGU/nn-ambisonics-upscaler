import os

from src.train import train
from src.utils import get_experiments_dir

hidden_layers = 1
train({"model_name":"cnn",
       "data_path": "/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/data/whitenoise_10_reflections",
       "logs_path": os.path.join(get_experiments_dir(),'cnn'),
       "experiment_name": "cnn_10reflections_overfit_260820",
       "max_epochs": 1000,
       "num_workers": 15,
       "lr": 0.001,
       "lr_sched_patience": 5,
       "lr_sched_thresh": 0.1,
       "bias": True,
       "hidden_layers": hidden_layers,
       "kernel_widths": [1]*(hidden_layers+1),
       "strides": [1]*(hidden_layers+1),
       "hidden_channels": [1] * hidden_layers,
       "sh_order_sig": 0,
       "sh_order_scm": 0,
       "residual_flag": False,
       "force_residual": True,
       })
