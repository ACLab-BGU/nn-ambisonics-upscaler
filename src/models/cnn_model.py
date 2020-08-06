import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.data.sig2scm_dataset import Dataset
from src.utils import get_data_dir, get_experiments_dir
from src.utils.complex_tensors import calc_scm

default_opts = {
    # ---folders---
    "data_path": os.path.join(get_data_dir(), '03082020_free_field'),
    "logs_path": get_experiments_dir(),
    "experiment_name": 'cnn_free_field',
    # ---data options---
    "frequency": 1000.,
    # ---network structure---
    "model_name": 'cnn',
    "hidden_layers": 2,
    "kernel_widths": [1, 2, 3],
    "strides": [1, 1, 1],
    "hidden_channels": [25 * 2, 36 * 2],  # *2 for real/imag
    "residual_flag": True,  # TODO: implement residual paths
    "loss": 'mse',  # 'mse'
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": None,
    "batch_size": 25,
    "num_workers": 0,
    "train_val_split": [0.9, 0.1],
    "preload_data": False,
    # ---optimization---
    "lr": 3e-4,
    "max_epochs": 1000,
    "gpus": -1
}


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
class CNN(LightningModule):
    def __init__(self, opts):
        # REQUIRED
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opts)

        # get input and output sizes
        self._get_input_output_sizes()

        # define network parameters
        channels = [self.hparams.input_channels, *self.hparams.hidden_channels, self.hparams.output_channels]
        lst = []
        zipped = zip(channels, channels[1:], self.hparams.kernel_widths, self.hparams.strides)
        for channels_in, channels_out, kernel_size, stride in zipped:
            lst.append(nn.Conv1d(channels_in, channels_out, kernel_size, stride))

        self.conv_layers = nn.ModuleList(lst)
        loss_dict = {'mse': lambda x, y: nn.MSELoss()(x, y) * np.prod(x.shape[1:])}
        self.loss = loss_dict[self.hparams.loss]

    def _get_input_output_sizes(self):
        # determine the number of channels of the input and output layers from data.

        dataset = Dataset(self.hparams.data_path, self.hparams.frequency, train=True, preload=False)
        x, target = next(iter(dataset))
        #  input shape should be (2, Q_in, T), target shape should be (2, Q_out, Q_out)
        assert x.shape[0] == 2, "0 dim (real-imag) of input must be 2"
        assert target.shape[0] == 2, "0 dim (real-imag) of target must be 2"
        assert target.shape[1] == target.shape[2], "target must be a stack of square matrices"

        self.hparams.input_channels = x.shape[1] * 2
        self.hparams.output_channels = target.shape[1] * 2

    def forward(self, x):
        # REQUIRED
        assert x.shape[1] == 2, "1st dim (real-imag) must be 2"

        N, _, Q_in, T = x.shape
        x = x.view((N, 2 * Q_in, T))
        for layer in self.conv_layers:
            x = layer(x)
            if layer != self.conv_layers[-1]:  # no activation at last layer
                x = F.relu(x)

        #  x should now be of shape (N, 2*Q_out, T).
        #  Reshape to (N, Q_out, 2, T) before applying time smoothing
        Q_out = x.shape[1] // 2
        T = x.shape[2]
        x = x.view((N, 2, Q_out, T))

        # calculate SCM using time smoothing
        x = calc_scm(x, smoothing_dim=3, channels_dim=2, real_imag_dim=1, batch_dim=0)

        return x

    def configure_optimizers(self):
        # REQUIRED
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ----- Data Loaders -----

    def setup(self, stage):
        # OPTIONAL

        # train/val split
        if stage == 'fit':
            assert np.sum(self.hparams.train_val_split) == 1, 'invalid split arguments'
            dataset = Dataset(self.hparams.data_path, self.hparams.frequency, train=True,
                              preload=self.hparams.preload_data)
            train_size = round(self.hparams.train_val_split[0] * len(dataset))
            val_size = len(dataset) - train_size
            self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.dataset_test = Dataset(self.hparams.data_path, self.hparams.frequency, train=False,
                                        preload=self.hparams.preload_data)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        loader = DataLoader(self.dataset_train, batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers)
        return loader

    def val_dataloader(self):
        # OPTIONAL
        loader = DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    def test_dataloader(self):
        # OPTIONAL
        loader = DataLoader(self.dataset_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    # ----- Training Loop -----
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}

        return {'log': tensorboard_logs}

    # ----- Validation Loop -----
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # ----- Test Loop -----
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # OPTIONAL

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}

        return {'test_loss': avg_loss, 'log': tensorboard_logs}
