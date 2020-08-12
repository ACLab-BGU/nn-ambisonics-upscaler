from abc import ABC, abstractmethod

import torch
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.data.base_dataset import BasicDatasetLT


class BaseModelNew(LightningModule,ABC):
    def __init__(self,opts):
        # REQUIRED
        print("before abstract init")
        super().__init__()
        print("after abstract init")

        # save hyperparameters
        self.save_hyperparameters(opts)

    @abstractmethod
    def forward(self, x):
        pass

    def configure_optimizers(self):
        # REQUIRED
        # TODO: let choose optimizer
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ----- Data Loaders -----

    def setup(self, stage):
        # OPTIONAL
        # TODO: change BasicDatasetLT
        # train/val split
        if stage == 'fit':
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


class SubModel(BaseModelNew):
    def __init__(self, opts):
        print("before sub init")
        super().__init__(opts)
        print("after sub init")
