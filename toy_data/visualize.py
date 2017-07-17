import numpy as np
import bokeh.plotting as bp


def regression_1D(data_set, regressF=None, res=150, fig_width=500):
    p = bp.figure(
        plot_width=fig_width,
        plot_height=fig_width)
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'y'

    p.circle(data_set.X[:, 0], data_set.y[:, 0],
             color=data_set.color, alpha=0.5, line_alpha=0, size=8,
             legend="Training set"
             )
    x_mesh = np.linspace(data_set.X_range[0], data_set.X_range[1], res)

    p.line(x_mesh, data_set.model(x_mesh), color='green')
    if regressF:
        p.line(x_mesh, regressF(x_mesh), color='black')
    bp.show(p)