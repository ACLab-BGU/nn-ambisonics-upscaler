# %%
import os

import torch

from src.train import train

# %%
from src.utils import get_data_dir

hidden_layers = 3
model = train({"model_name": "cnn",
               "data_path": os.path.join(get_data_dir(), "whitenoise_10_reflections"),
               "experiment_name": "3_to_4_conv2d_10_refs",
               "max_epochs": 15,
               "num_workers": 8,
               "lr": 0.001,
               "lr_sched_patience": 3,
               "lr_sched_thresh": 0.1,
               "hidden_layers": hidden_layers,
               "kernel_widths": [(10, 1)] * (hidden_layers + 1),
               "strides": [(1, 1)] * (hidden_layers + 1),
               "hidden_channels": [200]*hidden_layers,
               "sh_order_sig": 3,
               "sh_order_scm": 4,
               "gpus": -1,
               "residual_flag": True,
               "bias": True,
               "complex_conv": True,
               "fast_dev_run": False,
               "bandwidth": 2000,
               })
#
# # %%
# print(model.conv_layers[0].weight[0, :, 0])
# print(model.conv_layers[0].weight[1, :, 0])
# print(model.conv_layers[0].bias)
#
# U = model.conv_layers[0].weight[:, :, 0].detach().numpy()
# print(U)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(model.freq_weights.detach().numpy())
# plt.show()
pass
# print(U @ U.T)
#
# #%%
# x, y = next(iter(model.train_dataloader()))
#
# #%%
# # x should be of shape (N, 2, Q_in, T)
# N = 1
# Q_in = 1
# T = 100
# x = torch.ones((N, 2, Q_in, T))
# y = model(x)
