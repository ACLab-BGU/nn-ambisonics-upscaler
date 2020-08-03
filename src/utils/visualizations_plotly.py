import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def covariance_matrix(mat, ax=None, add_order_lines=False):
    # if ax is None:
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    im = go.Heatmap(z=mat)
    # if add_order_lines:
    #     add_sh_order_lines(ax)
    # ax.figure.colorbar(im, ax=ax)
    return im

def compare_covs(input, output, expected, normalize_diff=False):
    Q_in = input.shape[0]
    Q_out = output.shape[0]
    mats = [input, output[:Q_in, :Q_in]-input, output, expected]
    titles = ["input", "output truncated - input", "output", "expected"]

    if normalize_diff:
        mats[3] /= input

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles)
    rowcol = [(row,col) for row in range(1,3) for col in range(1,3)]
    for i,mat in enumerate(mats):
        row,col = rowcol[i]
        fig.add_trace(covariance_matrix(mat),row=row,col=col)

    # if normalize_diff:
    #     common_clim(im[:-1])
    # else:
    #     common_clim(im)

    return fig