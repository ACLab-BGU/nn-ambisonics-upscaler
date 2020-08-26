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

    from src.options import trainer_optional_arguments
    trainer_args = dict((key, opts[key]) for key in trainer_optional_arguments if key in opts)

    trainer = Trainer(weights_summary='full', logger=logger, default_root_dir=opts['logs_path'], **trainer_args)

    return trainer