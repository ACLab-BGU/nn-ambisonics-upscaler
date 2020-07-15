# %%
import matplotlib.pyplot as plt
import numpy as np
# from src.utils.sphere import sh


def covariance_matrix(mat):
    fig = plt.figure()
    ax = fig.add_subplot()
    return ax.imshow(np.abs(mat))


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



h = covariance_matrix(np.arange(16*16).reshape((16, -1)))
ax = h.axes
add_sh_order_lines(ax, x_flag=False)
ax.figure.show()
