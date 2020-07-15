import numpy as np


def grid(points, output_type='tuple_1d'):
    th_points = (np.floor(np.sqrt(points / 2))).astype(int)
    ph_points = (th_points * 2).astype(int)
    th = np.linspace(0, np.pi, th_points, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, ph_points, endpoint=False)

    if output_type == 'tuple_1d':
        return th, ph
    else:
        th = np.tile(th[:, np.newaxis], (1, ph_points))
        ph = np.tile(ph, (th_points, 1))
        if output_type == 'all_combs_1d':
            return np.stack([th.flatten(), ph.flatten()])
        elif output_type == 'tuple_2d':
            return th, ph
        elif output_type == 'all_combs_2d':
            return np.stack([th, ph])
        else:
            raise ValueError(f'illegal value for output_type: {output_type}')
