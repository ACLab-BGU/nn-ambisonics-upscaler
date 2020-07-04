import json
import os
import sys

import yaml


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


def prepare_opts(default_config, config=None, with_flags=True, print_flag=True):
    # open and parse in case of files
    if type(default_config) == str:
        default_config = EasyDict(read_yaml(default_config))
    if type(config) == str:
        config = EasyDict(read_yaml(config))
    # in case of no config input
    elif config is None:
        config = default_config

    # update config according to defaults and command-line flags
    config = update_opts_with_defaults(config, default_config)
    if with_flags:
        config = update_opts_with_flags(config)

    # print
    if print_flag:
        print("Training parameters: ")
        print("\n".join([k + ": " + str(v) for k, v in config.items()]))

    return config
