from src.train import train

hidden_layers = 1
train({"model_name":"cnn",
       "data_path": "/Users/ranweisman/PycharmProjects/nn-ambisonics-upscaler/data/whitenoise_0_reflections",
       "experiment_name": "deleteme",
       "max_epochs": 1000,
       "num_workers": 15,
       "lr": 0.001,
       "lr_sched_patience": 5,
       "lr_sched_thresh": 0.1,
       "hidden_layers": hidden_layers,
       "kernel_widths": [1]*(hidden_layers+1),
       "strides": [1]*(hidden_layers+1),
       "hidden_channels": [500],
       "sh_order_sig": 3,
       "sh_order_scm": 4,
       "residual_flag": False,
       "force_residual": True,
       })
