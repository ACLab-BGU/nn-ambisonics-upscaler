import json
import os
import sys
import warnings

import torch
import yaml

from src.models import find_model_using_name


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
            key, value_str = flag.split("=")
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
    if type(base_opts) == str:
        base_opts = EasyDict(read_yaml(base_opts))
    if type(opts) == str:
        opts = EasyDict(read_yaml(opts))
    # in case of no config input
    elif opts is None:
        opts = base_opts

    # update config according to defaults and command-line flags
    opts = update_opts_with_defaults(opts, base_opts)
    if with_flags:
        opts = update_opts_with_flags(opts)

    # print
    if print_flag:
        print_opts(opts)

    return dict(opts)

def validate_opts(opts,print_flag=True):
    ''' validate that the given options are OK, and perform some automatic fixes if needed'''

    # GPU
    if ~torch.cuda.is_available() and opts['gpus']!=0:
        warnings.warn('GPU is not available, using CPU instead')
        opts['gpus'] = 0

    # print
    if print_flag:
        print_opts(opts)

    return opts

def get_default_opts(opts):
    ''' find the default options of the given model'''
    if type(opts) == str:
        opts = EasyDict(read_yaml(opts))
    _,default_opts = find_model_using_name(opts['model_name'])
    return default_opts

def print_opts(opts):
    print("Configuration Parameters: ")
    print("\n".join([k + ": " + str(v) for k, v in opts.items()]))