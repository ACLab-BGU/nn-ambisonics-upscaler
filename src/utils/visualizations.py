# %%
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


# from src.utils.sphere import sh
# from src.utils.sphere.sh import Q2N


def covariance_matrix(mat, ax=None, add_order_lines=True):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    im = ax.imshow(np.abs(mat))
    if add_order_lines:
        add_sh_order_lines(ax)
    ax.figure.colorbar(im, ax=ax)
    return im


def add_sh_order_lines(ax: plt.Axes, order=None, args_dict=None, x_flag=True, y_flag=True):
    if args_dict is None:
        args_dict = {}
    from src.utils.sphere import sh

    if order is None:
        order = sh.i2nm(np.floor(ax.get_xlim()[1]))[0]

    n = np.arange(order)
    m = n
    locs = sh.nm2i(n, m) + 0.5
    for loc in locs:
        if x_flag:
            ax.axvline(loc, color='red', **args_dict)
        if y_flag:
            ax.axhline(loc, color='red', **args_dict)


def power_map(cov: np.ndarray, points=5000, db=True, dynamic_range_db: Union[float, None] = 50., ax: plt.Axes = None, approx_nearest_PSD=False):
    from src.utils.sphere.sampling_schemes import grid
    from src.utils.sphere.sh import power_map as calc_power_map

    omega = grid(points, output_type='all_combs_2d', phi_zero_center=True)  # (2, X, Y)
    power = calc_power_map(cov, omega, approx_nearest_PSD=approx_nearest_PSD)
    if db:
        power = 10*np.log10(power)

    if dynamic_range_db is not None and db:
        vmax = np.max(power)
        vmin_by_data = np.min(power)
        vmin_by_dynamic_range = vmax-dynamic_range_db
        vmin = np.maximum(vmin_by_data, vmin_by_dynamic_range)
    else:
        vmin = None
        vmax = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    im = ax.imshow(power, extent=(-180, 180, 180, 0), origin='upper', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    fig.colorbar(im, ax=ax)
    return im


def compare_covs(input, output, expected, elementwise_normalization=False) -> plt.Figure:
    Q_in = input.shape[0]
    Q_out = output.shape[0]
    expected_norm = np.linalg.norm(expected, 'fro')
    output = output/expected_norm
    input = input/expected_norm
    expected = expected/expected_norm

    mats = [input, output, expected, output-expected]
    loss = np.linalg.norm(mats[-1], 'fro')
    if elementwise_normalization:
        mats[3] /= input

    fig, axes = plt.subplots(2, 2)
    im = []
    for i, (ax, mat, title) in enumerate(zip(axes.flat,
                              mats,
                              ["input", "output", "expected", "output - expected"])):
        im.append(covariance_matrix(mat, ax))
        ax.set_title(title)

    fig.suptitle(f"RMSE = {loss}")

    if elementwise_normalization:
        common_clim(im[:-1])
    else:
        common_clim(im)

    return fig


def compare_power_maps(input, output, expected) -> plt.Figure:
    Q_in = input.shape[0]
    Q_out = output.shape[0]

    fig, axes = plt.subplots(2, 3)
    im = []
    for ax, mat, title in zip(axes.flat,
                              [input, output, expected, output[:Q_in, :Q_in]],
                              ["input", "output", "expected", "output truncated"]):
        im.append(power_map(mat, ax=ax, approx_nearest_PSD=True))
        ax.set_title(title)
    common_clim([im[0], im[3]])
    common_clim([im[1], im[2]])

    im = []
    for ax, mat, title in zip(axes.flatten()[4:],
                              [output - expected, output[:Q_in, :Q_in]-input],
                              ["output-expected", "output truncated - input"]):
        im.append(power_map(mat, ax=ax, db=False, approx_nearest_PSD=False))
        ax.set_title(title)

    fig.show()
    return fig


def common_clim(images):
    vmin = []
    vmax = []
    for image in images:
        vmin.append(image.get_clim()[0])
        vmax.append(image.get_clim()[1])
    for image in images:
        image.set_clim(min(vmin), max(vmax))
