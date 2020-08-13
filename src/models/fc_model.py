import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.data.base_dataset import BasicDatasetLT
from src.models.base_model import BaseModel
from src.utils import get_data_dir, get_experiments_dir
from src.utils.complex_tensors import get_real_imag_parts, complex_outer_product, cat_real_imag_parts

default_opts = {
    # ---folders---
    "data_path":  os.path.join(get_data_dir(),'SCM','image-method'),
    "logs_path":  get_experiments_dir(),
    "experiment_name": 'fc_25rank_imagemethod',
    # ---network structure---
    "model_name": 'fc',
    "input_sh_order": 3,
    "rank": 25, # None -> output is full matrix, Int -> output is low rank matrix transformed into full matrix
    "hidden_layers": 3,
    "hidden_sizes": [1000,1500,2000],
    "residual_flag": True,
    "residual_only": False,
    "loss": 'mse', # 'mse'
    # ---data---
    # "dtype": torch.float32, # TODO: implement (does errors in saving hyperparameters)
    "transform": None,
    "batch_size": 25,
    "num_workers": 10,
    "train_val_split": [0.9,0.1],
    "preload_data": True,
    # ---optimization---
    "lr": 3e-4,
    "lr_sched_thresh": 0.01,
    "lr_sched_patience": 10,
    "max_epochs": 1000,
    "gpus": -1
}

# TODO: implement FC with new Dataset
class FC(BaseModel):
    def __init__(self, opts):
        # REQUIRED
        super().__init__(opts)

        # get input and output sizes
        self._get_input_output_sizes()

        # define network parameters
        if np.isscalar(self.hparams.hidden_sizes):
            self.hparams.hidden_sizes = [self.hparams.hidden_sizes] * self.hparams.hidden_layers
        sizes = [self.hparams.input_size, *self.hparams.hidden_sizes, self.hparams.last_layer_size]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(sizes, sizes[1:])])
        loss_dict = {'mse': lambda x, y: nn.MSELoss()(x, y) * np.prod(x.shape[1:])}
        self.loss = loss_dict[self.hparams.loss]

    def forward(self, x):
        # REQUIRED
        if self.hparams.residual_only:
            pad_size = self.hparams.output_shape[-1] - x.shape[-1]
            x = F.pad(x,(0,pad_size,0,pad_size))
            raise NotImplementedError()
            return x

        if self.hparams.residual_flag:
            x_orig = x.clone()

        x = torch.flatten(x, 1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        # split between full-matrix output OR low-rank output transformed to full matrix
        if self.hparams.rank is None:
            x = x.view((x.shape[0], *self.hparams.output_shape))
        else:
            x = x.view((x.shape[0],2,self.hparams.output_shape[-1],self.hparams.rank))
            x = cat_real_imag_parts(*complex_outer_product(get_real_imag_parts(x))) # TODO: wrap 3 functions together, for simpler syntax for outer-product

        if self.hparams.residual_flag:
            x[:, :, :x_orig.shape[-2], :x_orig.shape[-1]] += x_orig

        return x

    def _get_dataset_args(self):
        dataset_args = dict(input_sh_order=self.hparams.input_sh_order)

        return dataset_args

    def _get_dataset_class(self):
        return BasicDatasetLT

    def _get_loss(self):
        loss_dict = {'mse': lambda x, y: nn.MSELoss()(x, y) * np.prod(x.shape[1:])}
        return loss_dict[self.hparams.loss]

    def _get_input_output_sizes(self):
        # automatically get the input and output sizes
        # (values are casted into normal ints, to beautify the yaml format)
        dataset = BasicDatasetLT(self.hparams.data_path, train=True, preload=False, input_sh_order=self.hparams.input_sh_order)
        input,target = next(iter(dataset))
        self.hparams.input_size = np.prod(input.shape).tolist()
        self.hparams.output_shape = list(target.shape)
        if self.hparams.rank is None:
            self.hparams.last_layer_size = np.prod(self.hparams.output_shape).tolist()
        else:
            self.hparams.last_layer_size = np.prod((2,self.hparams.output_shape[-1],self.hparams.rank)).tolist()

# Delete this class (BaseModelLT)
class BaseModelLT(LightningModule):
    def __init__(self, opts):
        # REQUIRED
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opts)

        # get input and output sizes
        self._get_input_output_sizes()

        # define network parameters
        if np.isscalar(self.hparams.hidden_sizes):
            self.hparams.hidden_sizes = [self.hparams.hidden_sizes] * self.hparams.hidden_layers
        sizes = [self.hparams.input_size, *self.hparams.hidden_sizes, self.hparams.last_layer_size]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(sizes, sizes[1:])])
        loss_dict = {'mse': lambda x,y: nn.MSELoss()(x,y) * np.prod(x.shape[1:])}
        self.loss = loss_dict[self.hparams.loss]


    def _get_input_output_sizes(self):
        # automatically get the input and output sizes
        # (values are casted into normal ints, to beautify the yaml format)
        dataset = BasicDatasetLT(self.hparams.data_path, train=True, preload=False, input_sh_order=self.hparams.input_sh_order)
        input,target = next(iter(dataset))
        self.hparams.input_size = np.prod(input.shape).tolist()
        self.hparams.output_shape = list(target.shape)
        if self.hparams.rank is None:
            self.hparams.last_layer_size = np.prod(self.hparams.output_shape).tolist()
        else:
            self.hparams.last_layer_size = np.prod((2,self.hparams.output_shape[-1],self.hparams.rank)).tolist()


    def forward(self, x):
        # REQUIRED
        if self.hparams.residual_only:
            pad_size = self.hparams.output_shape[-1] - x.shape[-1]
            x = F.pad(x,(0,pad_size,0,pad_size))
            raise NotImplementedError()
            return x

        if self.hparams.residual_flag:
            x_orig = x.clone()

        x = torch.flatten(x, 1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        # split between full-matrix output OR low-rank output transformed to full matrix
        if self.hparams.rank is None:
            x = x.view((x.shape[0], *self.hparams.output_shape))
        else:
            x = x.view((x.shape[0],2,self.hparams.output_shape[-1],self.hparams.rank))
            x = cat_real_imag_parts(*complex_outer_product(get_real_imag_parts(x))) # TODO: wrap 3 functions together, for simpler syntax for outer-product

        if self.hparams.residual_flag:
            x[:, :, :x_orig.shape[-2], :x_orig.shape[-1]] += x_orig

        return x


    def configure_optimizers(self):
        # REQUIRED
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ----- Data Loaders -----

    def setup(self, stage):
        # OPTIONAL

        # train/val split
        if stage == 'fit':
            assert np.sum(self.hparams.train_val_split)==1, 'invalid split arguments'
            dataset = BasicDatasetLT(self.hparams.data_path, train=True, preload=self.hparams.preload_data, input_sh_order=self.hparams.input_sh_order)
            train_size = round(self.hparams.train_val_split[0] * len(dataset))
            val_size = len(dataset) - train_size
            self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.dataset_test = BasicDatasetLT(self.hparams.data_path, train=False, preload=self.hparams.preload_data, input_sh_order=self.hparams.input_sh_order)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        loader = DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
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

