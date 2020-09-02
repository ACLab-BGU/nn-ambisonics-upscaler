# %%
import torch

from src.train import train

# %%
hidden_layers = 0
model = train({"model_name": "cnn",
               "data_path": r"/Users/tomshlomo/PycharmProjects/nn-ambisonics-upscaler2/data/whitenoise_inf_reflections",
               "experiment_name": "1_to_1_hiddens_0_kernel_1_no_res_10_refs",
               "max_epochs": 6,
               "num_workers": 12,
               "lr": 0.1,
               "lr_sched_patience": 3,
               "lr_sched_thresh": 0.1,
               "hidden_layers": hidden_layers,
               "kernel_widths": [(1, 1)] * (hidden_layers + 1),
               "strides": [1] * (hidden_layers + 1),
               "hidden_channels": [],
               "sh_order_sig": 0,
               "sh_order_scm": 0,
               "gpus": 0,
               "residual_flag": False,
               "bias": False,
               "complex_conv": True,
               "fast_dev_run": False,
               "bandwidth": 500,
               })
#
# # %%
print(model.conv_layers[0].weight[0, :, 0])
print(model.conv_layers[0].weight[1, :, 0])
# print(model.conv_layers[0].bias)
#
U = model.conv_layers[0].weight[:, :, 0].detach().numpy()
print(U)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(model.freq_weights.detach().numpy())
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
