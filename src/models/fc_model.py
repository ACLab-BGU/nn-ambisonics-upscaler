import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.data.base_dataset import BasicDatasetLT
from src.utils.complex_tensors import get_real_imag_parts, complex_outer_product, cat_real_imag_parts


class BaseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, hidden_sizes=200):
        super(BaseModel, self).__init__()
        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes] * hidden_layers
        sizes = [input_size, *hidden_sizes, output_size]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(sizes, sizes[1:])])

    def forward(self, x):

        x = torch.flatten(x, 1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        return x


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
class BaseModelLT(LightningModule):
    def __init__(self, opts):
        # REQUIRED
        super().__init__()

        # save options
        self.opts = opts

        # get input and output sizes
        self._get_input_output_sizes()

        # save hyperparameters for logging
        self.save_hyperparameters('opts')

        # define network parameters
        if np.isscalar(opts['hidden_sizes']):
            opts['hidden_sizes'] = [opts['hidden_sizes']] * opts['hidden_layers']
        sizes = [opts['input_size'], *opts['hidden_sizes'], opts['last_layer_size']]
        self.linears = nn.ModuleList([nn.Linear(in_size, out_size)
                                      for in_size, out_size in zip(sizes, sizes[1:])])
        loss_dict = {'mse': lambda x,y: nn.MSELoss()(x,y) * np.prod(x.shape[1:])}
        self.loss = loss_dict[opts['loss']]


    def _get_input_output_sizes(self):
        # automatically get the input and output sizes
        dataset = BasicDatasetLT(self.opts['data_path'], train=True, preload=False)
        input,target = next(iter(dataset))
        self.opts['input_size'] = np.prod(input.shape)
        self.opts['output_shape'] = target.shape
        if self.opts['rank'] is None:
            self.opts['last_layer_size'] = np.prod(self.opts['output_shape'])
        else:
            self.opts['last_layer_size'] = np.prod((2,self.opts['output_shape'][-1],self.opts['rank']))


    def forward(self, x):
        # REQUIRED
        x = torch.flatten(x, 1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)

        # split between full-matrix output OR low-rank output transformed to full matrix
        if self.opts['rank'] is None:
            x = x.view((x.shape[0], *self.opts['output_shape']))
        else:
            x = x.view((x.shape[0],2,self.opts['output_shape'][-1],self.opts['rank']))
            x = cat_real_imag_parts(*complex_outer_product(get_real_imag_parts(x))) # TODO: wrap 3 functions together, for simpler syntax for outer-product
        return x

    def configure_optimizers(self):
        # REQUIRED
        return optim.Adam(self.parameters(), lr=self.opts['lr'])

    # ----- Data Loaders -----

    def setup(self, stage):
        # OPTIONAL

        # train/val split
        assert np.sum(self.opts['train_val_split'])==1, 'invalid split arguments'
        dataset = BasicDatasetLT(self.opts['data_path'], train=True, preload=self.opts['preload_data'])
        train_size = round(self.opts['train_val_split'][0] * len(dataset))
        val_size = len(dataset) - train_size
        self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        # REQUIRED
        loader = DataLoader(self.dataset_train, batch_size=self.opts['batch_size'], num_workers=self.opts['num_workers'])
        return loader

    def val_dataloader(self):
        # OPTIONAL
        loader = DataLoader(self.dataset_val, batch_size=self.opts['batch_size'], num_workers=self.opts['num_workers'])
        return loader

    def test_dataloader(self):
        # OPTIONAL
        dataset = BasicDatasetLT(self.opts['data_path'], train=False)
        loader = DataLoader(dataset, batch_size=self.opts['batch_size'], num_workers=self.opts['num_workers'])
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

