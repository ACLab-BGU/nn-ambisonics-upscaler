# %%
import matplotlib.pyplot as plt
import numpy as np

# from src.utils.sphere import sh
# from src.utils.sphere.sh import Q2N
from src.utils.sphere import sh


def covariance_matrix(mat, ax=None, add_order_lines=True):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    im = ax.imshow(np.abs(mat))
    if add_order_lines:
        add_sh_order_lines(ax)

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


def power_map(cov: np.ndarray, points=5000, db=True, dynamic_range_db=50, ax: plt.Axes = None):
    from src.utils.sphere.sampling_schemes import grid
    from src.utils.sphere.sh import mat as shmat

    omega = grid(points, output_type='all_combs_2d', phi_zero_center=True)  # (2, X, Y)
    from src.utils.sphere.sh import power_map as calc_power_map
    power = calc_power_map(cov, omega)
    if db:
        power = 10*np.log10(power)

    vmax = np.max(power)
    vmin_by_data = np.min(power)
    if db:
        vmin_by_dynamic_range = vmax-dynamic_range_db
    else:
        vmin_by_dynamic_range = vmax / (10. ** (dynamic_range_db/10.))
    vmin = np.maximum(vmin_by_data, vmin_by_dynamic_range)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    im = ax.imshow(power, extent=(-180, 180, 180, 0), origin='upper', vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    fig.colorbar(im, ax=ax)
    fig.show()
    return im


def compare_covs(input, output, expected) -> plt.Figure:
    Q_in = input.shape[0]
    Q_out = output.shape[0]

    fig, axes = plt.subplots(2, 2)
    for ax, mat, title in zip(axes.flat,
                              [input, output, expected, output[:Q_in, :Q_in]-input],
                              ["input", "output", "expected", "output truncated - input"]):
        covariance_matrix(mat, ax)
        ax.set_title(title)

    fig.show()
    return fig


def compare_power_maps(input, output, expected) -> plt.Figure:
    Q_in = input.shape[0]
    Q_out = output.shape[0]

    fig, axes = plt.subplots(2, 2)
    for ax, mat, title in zip(axes.flat,
                              [input, output, expected, output[:Q_in, :Q_in]],
                              ["input", "output", "expected", "output truncated"]):
        power_map(mat, ax=ax)
        ax.set_title(title)

    fig.show()
    return fig

#
# doa = np.array([[np.pi/3], [np.pi/2]])
# anm = np.conj(sh.mat(6, doa, is_transposed=True))
# cov = anm * anm.transpose().conj()
# power_map(cov)
#
# cov_in = cov[:25, :25]
# cov_out = np.eye(cov.shape[0])
#
# compare_covs(cov_in, cov_out, cov)
# compare_power_maps(cov_in, cov_out, cov)
