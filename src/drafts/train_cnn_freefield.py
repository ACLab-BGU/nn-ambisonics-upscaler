# %%
import torch

from src.train import train

# %%
hidden_layers = 2
model = train({"model_name": "cnn",
               "data_path": r"../../data/whitenoise_10_reflections",
               "experiment_name": "3_to_4_hiddens_2_conv2_kernel_20,1_channels_100_with_res_10_refs",
               "max_epochs": 1000,
               "num_workers": 8,
               "lr": 0.01,
               "lr_sched_patience": 3,
               "lr_sched_thresh": 0.1,
               "hidden_layers": hidden_layers,
               "kernel_widths": [(5, 1)] * (hidden_layers + 1),
               "strides": [(2, 1)] * (hidden_layers + 1),
               "hidden_channels": [100, 100],
               "sh_order_sig": 3,
               "sh_order_scm": 4,
               "gpus": -1,
               "residual_flag": True,
               "bias": True,
               "complex_conv": False,
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
