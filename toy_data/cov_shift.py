"""
Generates data with covariance shift
From toy_data:
    Module for toy-data generation for ML experiments
"""

import bokeh.plotting as bp
import numpy as np
import numpy.random as rnd

from toy_data import color
from toy_data import data_types

DataSet = data_types.DataSet


class Gaussian_Shift_1D:
    def __init__(self,
                 model,
                 tst_X_mean_shift=3,
                 tr_X_mean=0,
                 tr_X_sd=2,
                 tst_X_sd=1,
                 n_samples=200,
                 noise_sd=1,
                 tst_ratio=0.2):
        self.model = model

        n_tst = int(round(tst_ratio * n_samples))
        n_tr = n_samples - n_tst
        tr_X = rnd.normal(tr_X_mean, tr_X_sd, (n_tr, 1))
        tst_X_mean = tr_X_mean + tst_X_mean_shift
        tst_X = rnd.normal(tst_X_mean, tst_X_sd, (n_tst, 1))

        def model_noisy(x):
            return model(x) + rnd.normal(0, noise_sd, (x.shape[0], 1))

        self.tr = DataSet.from_X(tr_X, model_noisy)
        self.tst = DataSet.from_X(tst_X, model_noisy)
        self.X = np.vstack((tr_X, tst_X))
        self.X_range = [np.min(self.X), np.max(self.X)]

        self.y_ = self.model(self.X)
        self.y = np.vstack((self.tr.y, self.tst.y))

        colors = color.get_N_by_hue(3)
        self.color, self.tr.color, self.tst.color = colors


def visualize_1D_regression(data, regressF=None, res=150, fig_width=500):
    p = bp.figure(
        plot_width=fig_width,
        plot_height=fig_width)
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'y'

    p.circle(data.tr.X[:, 0], data.tr.y[:, 0],
             color=data.tr.color, alpha=0.5, line_alpha=0, size=8,
             legend="Training set"
             )
    p.circle(data.tst.X[:, 0], data.tst.y[:, 0],
             color=data.tst.color, alpha=0.5, line_alpha=0, size=8,
             legend="Testing set"
             )
    x_mesh = np.linspace(data.X_range[0], data.X_range[1], res)

    p.line(x_mesh, data.model(x_mesh), color='green')
    if regressF:
        p.line(x_mesh, regressF(x_mesh), color='black')
    bp.show(p)


def visualize_1D_regression_with_tr_weights(data, weights, regressF=None, res=150, fig_width=500):
    p = bp.figure(
        plot_width=fig_width,
        plot_height=fig_width)
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'y'

    print(data.tr.color)
    tr_color = color.map_color(np.ravel(weights).tolist(), data.tr.color)
    print(data.tst.color)
    p.scatter(data.tr.X[:, 0], data.tr.y[:, 0],
              color=tr_color, alpha=0.5, line_alpha=0, size=8,
              legend="Training set"
              )
    p.circle(data.tst.X[:, 0], data.tst.y[:, 0],
             color=data.tst.color, alpha=0.5, line_alpha=0, size=8,
             legend="Testing set"
             )
    x_mesh = np.linspace(data.X_range[0], data.X_range[1], res)

    p.line(x_mesh, data.model(x_mesh), color='green')
    if regressF:
        p.line(x_mesh, regressF(x_mesh), color='black')
    bp.show(p)
