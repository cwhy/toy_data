"""
Includes Sine Wave Generator
From toy_data:
    Module for toy-data generation for ML experiments
"""

import bokeh.plotting as bp
import numpy as np
import numpy.random as rnd

from toy_data import color
from toy_data import data_types

DataSet = data_types.DataSet


def split(_fts, _lbs, test_ratio):
    indices = range(_fts.shape[0])
    i_test = rnd.choice(indices, size=int(round(test_ratio * _fts.shape[0])))
    i_train = np.array([i for i in indices if i not in i_test])
    return (_fts[i_train, :], _lbs[i_train, :]), (_fts[i_test, :], _lbs[i_test, :])


class SineWave:
    def __init__(self,
                 X_range=(0, 20),
                 A=3,
                 y_offset=0,
                 phase=0,
                 frequency=1,
                 n_samples=200,
                 sigma=1,
                 split_ratio=0.2):
        """

        """
        self.frequency = frequency
        self.phase = phase
        self.y_offset = y_offset
        self.A = A
        self.X_range = X_range

        self.X = rnd.uniform(X_range[0], X_range[1], n_samples)
        self.X = self.X.reshape((n_samples, 1))

        self.model = lambda x: y_offset + self.A * np.sin(self.frequency * x + self.phase)
        self.y_sine = self.model(self.X)
        self.y = self.y_sine + rnd.normal(0, sigma, (n_samples, 1))
        self.y = self.y.reshape((n_samples, 1))

        (tr_X, tr_y), (tst_X, tst_y) = split(self.X, self.y, split_ratio)
        self.tr = DataSet(tr_X, tr_y)
        self.tst = DataSet(tst_X, tst_y)

        colors = color.get_N_by_hue(3)
        self.color, self.tr.color, self.tst.color = colors


def visualize_1D_regression(data, regressF=None, res=150, fig_width=500):
    p = bp.figure(
        plot_width=fig_width,
        plot_height=fig_width)
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'y'

    p.circle(data.X[:, 0], data.y[:, 0], color=data.color, alpha=0.5, line_alpha=0, size=8)
    x_mesh = np.linspace(data.X_range[0], data.X_range[1], res)

    p.line(x_mesh, data.model(x_mesh), color='green')
    if regressF:
        p.line(x_mesh, regressF(x_mesh), color='black')
    bp.show(p)
