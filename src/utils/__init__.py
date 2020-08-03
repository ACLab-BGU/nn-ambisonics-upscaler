import os
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def get_data_dir():
    project_dir = str(Path(__file__).parent.parent.parent.absolute())
    path = os.path.join(project_dir, 'data')
    return path


def get_experiments_dir():
    project_dir = str(Path(__file__).parent.parent.parent.absolute())
    path = os.path.join(project_dir, 'experiments')
    return path


def prepare_logger(opts):
    ''' prepare logger using the given options '''
    logger = TensorBoardLogger(opts['logs_path'], name=opts['experiment_name'])
    return logger


def prepare_trainer(opts, logger=None):
    ''' prepare the trainer using the given options '''
    optional_arguments = ['max_epochs','gpus','fast_dev_run','overfit_batches','limit_train_batches',
                 'progress_bar_refresh_rate','auto_lr_find','check_val_every_n_epoch','val_check_interval',
                 'log_save_interval','row_log_interval','resume_from_checkpoint']
    trainer_args = dict((key, opts[key]) for key in optional_arguments if key in opts)

    trainer = Trainer(weights_summary='full', logger=logger, default_root_dir=opts['logs_path'], **trainer_args)

    return trainer


# TODO: implement additional trainer arguments, especially "resume_from_checkpoint"

"""
Additional Trainer args:
fast_dev_run = false (for debugging)
overfit_batches=0.01 (overfit over small datasets)
limit_train_batches=0.1, limit_val_batches=0.01, limit_test_batches=0.01 (use less data, to make epochs shorter)
progress_bar_refresh_rate (How often to refresh progress bar (in steps). Value 0 disables progress bar)
auto_lr_find (activate an algorithm to auto-find the learning rate)

check_val_every_n_epoch (Check val every n train epochs)
val_check_interval (How often within one training epoch to check the validation set)
log_save_interval (Writes logs to disk this often)
row_log_interval (How often to add logging rows (does not write to disk)
resume_from_checkpoint (to resume training from a specific checkpoint pass in the path here. This can be a URL.)
"""
