import json
import os
import sys


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


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


def prepare_opts(defaults, config_path=None, with_flags=True, print_flag=True):
    # use config file
    if config_path is not None:
        assert os.path.exists(config_path)
        opts = EasyDict(read_json(config_path))
    else:
        opts = EasyDict(defaults)

    # update options
    opts = update_opts_with_defaults(opts, defaults)
    if with_flags:
        opts = update_opts_with_flags(opts)

    # print
    if print_flag:
        print("Training parameters: ")
        print("\n".join([k + ": " + str(v) for k, v in opts.items()]))

    return opts
