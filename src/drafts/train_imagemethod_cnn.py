from src.train import train

train({"model_name":"cnn",
       "data_path": r"/Users/tomshlomo/PycharmProjects/nn-ambisonics-upscaler2/data/whitenoise_0_reflections",
       "experiment_name": "cnn_10_refs_deep",
       "max_epochs": 1000,
       "num_workers": 8,
       "lr": 0.001,
       "lr_sched_patience": 10,
       "lr_sched_thresh": 0.01,
       "hidden_layers": 7,
       "kernel_widths": [5,5,4,4,3,3,2,2],
       "strides": [1]*8,
       "hidden_channels": [50, 100, 200, 400, 800, 400, 200],
       "sh_order_sig": 3,
       "sh_order_scm": 6,
       "gpus": 0,
       })