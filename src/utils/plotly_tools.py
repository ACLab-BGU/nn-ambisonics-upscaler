

def axis_equal(fig):
    fig.for_each_yaxis(lambda axis: axis.update(scaleratio = 1, scaleanchor = axis.anchor))

def axis_tight(fig):
    fig.update_yaxes(constrain='domain')
    fig.update_xaxes(constrain='domain')

def set_colorbars_positions(fig):
    print(3)
