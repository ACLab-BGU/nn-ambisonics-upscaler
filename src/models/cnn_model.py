import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sig2scm_dataset import Dataset
from src.models.base_model import BaseModel
from src.utils import get_data_dir, get_experiments_dir
from src.utils.complex_tensors import calc_scm, ComplexConv1d

default_opts = {
    # ---folders---
    "data_path": os.path.join(get_data_dir(), 'whitenoise_0_reflections'),
    "logs_path": get_experiments_dir(),
    "experiment_name": 'cnn_free_field',
    # ---data options---
    "frequency": 1000.,
    # ---network structure---
    "model_name": 'cnn',
    "complex_conv": True,
    "bias": True,
    "hidden_layers": 2,
    "kernel_widths": [1, 2, 3],
    "strides": [1, 1, 1],
    "hidden_channels": [25 * 2, 36 * 2],  # *2 for real/imag
    "residual_flag": True,
    "force_residual": True,
    "loss": 'mse',  # 'mse'
    "sh_order_sig": float("inf"),
    "sh_order_scm": float("inf"),
    "time_len_sig": float("inf"),
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": None,
    "batch_size": 25,
    "num_workers": 0,
    "train_val_split": [0.9, 0.1],
    "preload_data": False,
    # ---optimization---
    "lr": 3e-4,
    "lr_sched_thresh": 0.01,
    "lr_sched_patience": 10,
    "max_epochs": 1000,
    "gpus": -1
}


class CNN(BaseModel):
    def __init__(self, opts):
        # REQUIRED
        super().__init__(opts)

        # get input and output sizes
        self._get_input_output_sizes()

        # define network parameters
        if len(self.hparams.hidden_channels) == 0:
            channels = list(np.linspace(self.hparams.input_channels,
                                        self.hparams.output_channels,
                                        self.hparams.hidden_layers + 2,
                                        dtype=np.int))
        else:
            channels = [self.hparams.input_channels, *self.hparams.hidden_channels, self.hparams.output_channels]

        lst = []
        zipped = zip(channels, channels[1:], self.hparams.kernel_widths, self.hparams.strides)
        for channels_in, channels_out, kernel_size, stride in zipped:
            if self.hparams.complex_conv:
                conv_layer = ComplexConv1d
            else:
                conv_layer = nn.Conv1d
            lst.append(conv_layer(channels_in, channels_out, kernel_size, stride, bias=self.hparams.bias))

        self.conv_layers = nn.ModuleList(lst)
        self.alpha = nn.Parameter(torch.tensor(0., requires_grad=True))  # logit scaling of residual

    def forward(self, x):
        # REQUIRED (lightning)

        # x should be of shape (N, 2, Q_in, T)
        assert x.shape[1] == 2, "1st dim (real-imag) must be 2"

        N, _, Q_in, T = x.shape
        low_order_scm = None

        # merge real/imag channels to a single axis
        if self.hparams.residual_flag:
            low_order_scm = calc_scm(x, smoothing_dim=3, channels_dim=2, real_imag_dim=1, batch_dim=0)

        if not self.hparams.complex_conv:
            x = x.view((N, 2 * Q_in, T))

        # apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
            if layer != self.conv_layers[-1]:  # no activation at last layer
                x = F.relu(x)

        if not self.hparams.complex_conv:
            #  x should now be of shape (N, 2*Q_out, T).
            #  Reshape to (N, 2, Q_out, T) before applying time smoothing
            Q_out = x.shape[1] // 2
            T = x.shape[2]
            x = x.view((N, 2, Q_out, T))

        # calculate SCM using time smoothing
        x = calc_scm(x, smoothing_dim=3, channels_dim=2, real_imag_dim=1, batch_dim=0)

        # x is of shape (N, 2, Q_out, Q_out)
        if self.hparams.residual_flag:
            if self.hparams.force_residual:
                x_low_block = low_order_scm
            else:
                x_low_block = x[:, :, :Q_in, :Q_in].clone()
                beta = torch.sigmoid(self.alpha)
                x_low_block = (1 - beta) * x_low_block + beta * low_order_scm

            x[:, :, :Q_in, :Q_in] = x_low_block

        return x

    def _get_dataset_args(self):
        dataset_args = dict(frequency=self.hparams.frequency, sh_order_sig=self.hparams.sh_order_sig,
                            sh_order_scm=self.hparams.sh_order_scm, time_len_sig=self.hparams.time_len_sig)

        return dataset_args

    def _get_dataset_class(self):
        return Dataset

    def _get_loss(self):
        loss_dict = {'mse': lambda x, y: nn.MSELoss()(x, y) * np.prod(x.shape[1:])}
        return loss_dict[self.hparams.loss]

    def _get_input_output_sizes(self):
        # determine the number of channels of the input and output layers from data.

        dataset = Dataset(self.hparams.data_path, train=True, preload=False, **self.dataset_args)
        x, target = next(iter(dataset))
        #  input shape should be (2, Q_in, T), target shape should be (2, Q_out, Q_out)
        assert x.shape[0] == 2, "0 dim (real-imag) of input must be 2"
        assert target.shape[0] == 2, "0 dim (real-imag) of target must be 2"
        assert target.shape[1] == target.shape[2], "target must be a stack of square matrices"

        self.hparams.input_channels = x.shape[1] * 2
        self.hparams.output_channels = target.shape[1] * 2

    def training_epoch_end(self, outputs):
        results = super().training_epoch_end(outputs)

        # TODO: wrap tensorboard stuff more elegantly, in a different function
        for i, layer in enumerate(self.conv_layers):
            self.logger.experiment.add_histogram("layer " + str(i) + " - weights", layer.weight, self.current_epoch)
            if layer.bias is not None:
                self.logger.experiment.add_histogram("layer " + str(i) + " - bias", layer.bias, self.current_epoch)

        return results
