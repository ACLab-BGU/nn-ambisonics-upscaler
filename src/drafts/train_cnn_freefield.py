from src.train import train

hidden_layers = 0
train({"model_name":"cnn",
       "data_path": r"/Users/tomshlomo/PycharmProjects/nn-ambisonics-upscaler2/data/whitenoise_10_reflections",
       "experiment_name": "cnn_0_refs_deep_3_to_4_hiddens_1_kernel_1_channels_500_10_refs",
       "max_epochs": 1000,
       "num_workers": 15,
       "lr": 0.001,
       "lr_sched_patience": 3,
       "lr_sched_thresh": 0.1,
       "hidden_layers": hidden_layers,
       "kernel_widths": [1]*(hidden_layers+1),
       "strides": [1]*(hidden_layers+1),
       "hidden_channels": [],
       "sh_order_sig": 3,
       "sh_order_scm": 3,
       "gpus": 0,
       })