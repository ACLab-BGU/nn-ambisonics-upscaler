from abc import ABC, abstractmethod

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader, random_split


class BaseModel(LightningModule, ABC):
    def __init__(self, opts):
        # REQUIRED (lightning)
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opts)

        # dataset stuff
        self._Dataset = self._get_dataset_class()
        self.dataset_args = self._get_dataset_args()

        # loss
        self.loss = self._get_loss()

    @abstractmethod
    def forward(self, x):
        # REQUIRED (lightning)
        pass

    @abstractmethod
    def _get_dataset_class(self):
        pass

    @abstractmethod
    def _get_dataset_args(self):
        pass

    @abstractmethod
    def _get_loss(self):
        pass

    def configure_optimizers(self):
        # REQUIRED (lightning)

        # TODO: control optimizer with hparams

        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=self.hparams.lr_sched_patience,
                                                         threshold=self.hparams.lr_sched_thresh, verbose=True)
        return [optimizer], [scheduler]

    # ----- Data Loaders -----
    def setup(self, stage):
        # OPTIONAL (lightning)

        # train/val split
        if stage == 'fit':
            assert np.sum(self.hparams.train_val_split) == 1, 'invalid split arguments'
            dataset = self._Dataset(self.hparams.data_path, train=True, preload=self.hparams.preload_data,
                                    **self.dataset_args)
            train_size = round(self.hparams.train_val_split[0] * len(dataset))
            val_size = len(dataset) - train_size
            self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])
        elif stage == 'test':
            self.dataset_test = self._Dataset(self.hparams.data_path, train=False, preload=self.hparams.preload_data,
                                              **self.dataset_args)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED (lightning)
        loader = DataLoader(self.dataset_train, batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers)
        return loader

    def val_dataloader(self):
        # OPTIONAL (lightning)
        loader = DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    def test_dataloader(self):
        # OPTIONAL (lightning)
        loader = DataLoader(self.dataset_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    # ----- Training Loop -----
    def training_step(self, batch, batch_idx):
        # REQUIRED (lightning)
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        # print(f"\nBATCH Train Loss: {loss}")

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL (lightning)
        avg_loss = torch.stack([x['batch_loss'] for x in outputs]).mean()
        tensorboard_logs = {'train_loss': avg_loss}

        print(f"\nTrain Loss: {avg_loss}")

        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    # ----- Validation Loop -----
    def validation_step(self, batch, batch_idx):
        # OPTIONAL (lightning)
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL (lightning)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        print(f"\nValidation Loss: {avg_loss}\n")

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # ----- Test Loop -----
    def test_step(self, batch, batch_idx):
        # OPTIONAL (lightning)
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # OPTIONAL (lightning)

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}

        return {'test_loss': avg_loss, 'log': tensorboard_logs}
