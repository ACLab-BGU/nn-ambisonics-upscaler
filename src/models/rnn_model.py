import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sig2scm_dataset import Dataset
from src.models.base_model import BaseModel
from src.utils import get_data_dir, get_experiments_dir

default_opts = {
    # ---folders---
    "data_path": os.path.join(get_data_dir(), 'whitenoise_10_reflections'),
    "logs_path": get_experiments_dir(),
    "experiment_name": 'rnn_QA',
    # ---data options---
    "center_frequency": 2500.,
    "bandwidth": 400,
    # ---network structure---
    "model_name": 'rnn',
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


class NNIWF(BaseModel):
    def __init__(self, opts):
        # REQUIRED
        super().__init__(opts)

        # output shape is (2, Qz, Qx)
        # z shape is (2, Qz)
        self.z_shape = self.output_shape[:2]
        self.x_shape = self.input_shape[:2]  # TODO: make sure shapes are correct, what about frequency?

        # define network parameters
        self.input2rnn_input = FlattenInput2RNNInput(self.input_shape)
        self.rnn = nn.LSTM(input_size=self.input2rnn_input.get_output_shape(),
                           hidden_size=self.hparams.rnn_hidden_size,
                           num_layers=self.hparams.rnn_num_layers,
                           bidirectional=self.hparams.rnn_bidirectional)
        self.rnn_output_to_dz = FCHidden2dz(input_shape=self.hparams.rnn_hidden_size, output_shape=self.z_shape,
                                            hidden_sizes=self.hparams.fc_hidden_sizes)

    def forward(self, x):
        # REQUIRED (lightning)
        # TODO: doc the shape of x

        rnn_input = self.input2rnn_input(x)
        h = self.rnn(rnn_input)
        dz = self.rnn_output_to_dz(h)
        Rzx, x_transformed = self.iwf(x, dz)

        return Rzx, x_transformed

    def iwf(self, x, dz, return_type="Rzx"):
        # x is of shape (T, N, 2, Qin, F)
        x_transformed = complex_solve(x, Rx)

        Rzx = torch.zeros(T, 2, Qz, Qx)
        z = torch.zeros(T, 2, Qz)
        for t in torch.range(T):
            Rzx_prev = Rzx[t - 1] if t else 0.
            z_prime = complex_matmul(Rzx_prev, x_transformed[t], dims=[(), ()])
            z[t] = z_prime + dz[t]

            Rzx[t] = (t * Rzx_prev + complex_outerporod(z[t], x[t])) / (t + 1)

        if return_type == "Rzx":
            return Rzx, x_transformed
        elif return_type == "z":
            return z, x_transformed
        else:
            raise NotImplementedError

    def _get_dataset_args(self):
        # REQUIRED
        dataset_args = dict(center_frequency=self.hparams.center_frequency,
                            bandwidth=self.hparams.bandwidth,
                            sh_order_sig=self.hparams.sh_order_sig,
                            sh_order_scm=self.hparams.sh_order_scm,
                            time_len_sig=self.hparams.time_len_sig,
                            use_cross_scm=True)

        return dataset_args

    def _get_dataset_class(self):
        # REQUIRED
        return Dataset

    def _get_loss(self):
        # REQUIRED
        loss_dict = {'mse': lambda x, y: nn.MSELoss()(x, y) * np.prod(x.shape[1:])}
        return loss_dict[self.hparams.loss]


class Input2RNNInput(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    @abstractmethod
    def get_output_shape(self):
        pass


class FlattenInput2RNNInput(Input2RNNInput):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def forward(self, x):
        # x is of shape (N, 2, Q_in, F, T)
        # features is of shape (T, N, 2*Q_in*F)

        features = x.permute((4, 0, 1, 2, 3))
        features = features.view((*features.shape[:2], -1))
        return features

    def get_output_shape(self):
        _, Q_in, F, _ = self.input_shape
        return 2 * F * Q_in


class Hidden2dz(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = input_shape


class FCHidden2dz(Hidden2dz):
    def __init__(self, input_shape, output_shape, hidden_sizes=()):
        super().__init__(input_shape, output_shape)

        sizes = [torch.prod(self.input_shape), *hidden_sizes, torch.prod(self.output_shape)]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(sizes, sizes[1:])])

    def forward(self, x):
        # x is of shape (T, N, *self.input_shape)
        T, N, *_ = x.shape
        assert (x.shape[2:] == self.input_shape)

        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        # x is of shape (T, N, prod(self.output_shape))
        # reshape to (T, N, *self.output_shape)
        x = x.view((T, N, *self.output_shape))
