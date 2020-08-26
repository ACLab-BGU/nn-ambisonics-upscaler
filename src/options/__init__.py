import itertools
import json
import os
import sys
import warnings

import torch
import yaml

import numpy as np

from src.models import find_model_using_name

trainer_optional_arguments = ['max_epochs', 'gpus', 'fast_dev_run', 'overfit_batches', 'limit_train_batches',
                              'progress_bar_refresh_rate', 'auto_lr_find', 'check_val_every_n_epoch',
                              'val_check_interval',
                              'log_save_interval', 'row_log_interval', 'resume_from_checkpoint']

misc_arguments = ['port']

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


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


def read_json(path):
    assert os.path.exists(path)
    with open(path, "r") as f:
        return json.load(f)


def read_yaml(path):
    assert os.path.exists(path)
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def update_opts_with_defaults(opts, defaults):
    for k, v in defaults.items():
        if k not in opts:
            opts[k] = v
    return opts


def update_opts_with_flags(opts):
    if len(sys.argv) > 2:
        for flag in sys.argv[2:]:
            key_val_tuple = flag.split("=")
            if not len(key_val_tuple) == 2:
                continue
            key, value_str = key_val_tuple
            key = key.lstrip("-")
            try:
                value = json.loads(value_str)
            except:
                value = value_str
            print("setting parameter *{}* with value *{}*".format(key, value))
            opts[key] = value
    return opts


def prepare_opts(base_opts, opts=None, with_flags=True, print_flag=False):
    # open and parse in case of files
    base_opts = read_opts(base_opts)
    opts = read_opts(opts)

    # in case of no config input
    if opts is None:
        opts = base_opts

    # update config according to defaults and command-line flags
    opts = update_opts_with_defaults(opts, base_opts)
    if with_flags:
        opts = update_opts_with_flags(opts)

    # print
    if print_flag:
        print_opts(opts)

    return dict(opts)


def validate_opts(opts, print_flag=True):
    ''' validate that the given options are OK, and perform some automatic fixes if needed'''

    # check that all fields are legal
    default_opts = get_default_opts(opts)
    valid_keys = list(default_opts.keys()) + trainer_optional_arguments + misc_arguments
    for key in opts.keys():
        assert key in valid_keys, f"parameter {key} in opts is not valid"

    # GPU
    if (not torch.cuda.is_available()) and opts['gpus'] != 0:
        warnings.warn('GPU is not available, using CPU instead')
        opts['gpus'] = 0

    # Split data to train/validation
    assert np.sum(opts['train_val_split']) == 1, 'invalid split arguments'

    # print
    if print_flag:
        print_opts(opts)

    return opts


def get_default_opts(opts):
    ''' find the default options of the given model'''
    opts = read_opts(opts)
    _, default_opts = find_model_using_name(opts['model_name'])

    return default_opts


def read_opts(opts):
    # read config from yaml file / create basic config using model_name
    if type(opts) == str:
        if opts.endswith('yaml'):
            opts = EasyDict(read_yaml(opts))
        else:
            model_name = opts
            opts = {'model_name': model_name}

    return opts


def print_opts(opts):
    # print config
    print("Configuration Parameters: ")
    print("\n".join([k + ": " + str(v) for k, v in opts.items()]))


def get_opts_combs(opts):
    ''' get a list of combinations for different options'''
    keys = opts.keys()
    values = (opts[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations
