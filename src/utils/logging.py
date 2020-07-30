import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def get_scalars_from_tb_log(log_file, scalar_name):
    '''returns an array of scalar values from a tensorboard log file'''
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    scalar_vals = ea.Scalars(scalar_name)
    vals = np.zeros(len(scalar_vals))

    for i,scalar in enumerate(scalar_vals):
        vals[i] = scalar[2]

    return vals