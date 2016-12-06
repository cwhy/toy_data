"""
Generates 1D data with covariance shift
From toy_data:
    Module for toy-data generation for ML experiments
"""

import bokeh as bk
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
            return model(x + rnd.normal(0, noise_sd, x.shape))

        self.tr = DataSet.from_X(tr_X, model_noisy)
        self.tst = DataSet.from_X(tst_X, model_noisy)
        self.X = np.vstack((tr_X, tst_X))
        self.X_range = [np.min(self.X), np.max(self.X)]

        self.y_ = self.model(self.X)

        self.y = np.vstack((self.tr.y, self.tst.y))

        colors = color.get_N_by_hue(3)
        self.color, self.tr.color, self.tst.color = colors


class Gaussian_Shift_2D_BinaryClassification:
    def __init__(self,
                 model,
                 tst_X_mean_shift=(2, 1),
                 tr_X_mean=(-0.5, -0.3),
                 tr_X_cov=((3, 0.2), (-0.8, 3)),
                 tst_X_cov=((1, 0.1), (-0.4, 1)),
                 n_samples=200,
                 noise_sd=0.5,
                 tst_ratio=0.2):
        self.model = model

        n_tst = int(round(tst_ratio * n_samples))
        n_tr = n_samples - n_tst
        tr_X = rnd.multivariate_normal(tr_X_mean, tr_X_cov, n_tr)
        tst_X_mean = np.array(tr_X_mean) + np.array(tst_X_mean_shift)
        tst_X = rnd.multivariate_normal(tst_X_mean, tst_X_cov, n_tst)

        def model_noisy(x):
            return model(x + rnd.normal(0, noise_sd, x.shape))

        self.tr = DataSet.from_X(tr_X, model_noisy)
        self.tst = DataSet.from_X(tst_X, model_noisy)
        self.X = np.vstack((tr_X, tst_X))
        self.X_range = [np.min(self.X, axis=0), np.max(self.X, axis=0)]

        self.y_ = self.model(self.X)

        self.y = np.vstack((self.tr.y, self.tst.y))

        colors = color.get_N_by_hue(2, v=0.8)
        self.tr.color = colors
        self.tst.color = list(map(color.darker, colors))
        self.tr.name = "Training set"
        self.tst.name = "Testing set"
        self.tr.alpha = 0.3
        self.tst.alpha = 0.7


def visualize_2D_classification(data, classifyF=None, res=150, fig_width=500):
    _xr = np.array(data.X_range).T

    p = bp.figure(x_range=_xr[0, :].tolist(),
                  y_range=_xr[1, :].tolist(),
                  plot_width=fig_width,
                  plot_height=fig_width)
    p.xaxis.axis_label = 'x0'
    p.yaxis.axis_label = 'x1'
    if classifyF:
        x_mesh = np.linspace(_xr[0, 0], _xr[0, 1], res)
        y_mesh = np.linspace(_xr[1, 0], _xr[1, 1], res)
        _xx, _yy = np.meshgrid(x_mesh, y_mesh)
        y_hats = classifyF(np.vstack((np.ravel(_xx), np.ravel(_yy))).T)
        for _c in (True, False):
            if _c:
                _y_hats_xy = y_hats.reshape((res, res)) / 4
            else:
                _y_hats_xy = (1 - y_hats).reshape((res, res)) / 4
            _c_col = np.tile(data.tr.color[_c], (res, res, 1))
            _rgba = np.dstack((_c_col, np.round(255 * _y_hats_xy))).astype(np.uint8)
            img = np.squeeze(_rgba.view(np.uint32))
            p.image_rgba(image=[img], x=[_xr[0, 0]], y=[_xr[1, 0]],
                         dw=[_xr[0, 1] - _xr[0, 0]],
                         dh=[_xr[1, 1] - _xr[1, 0]])

    for _data in (data.tr, data.tst):
        for _c in (True, False):
            _col = color.int2hex(_data.color[_c])
            _y = np.ravel(_data.y)
            p.circle(x=_data.X[_y == _c, 0],
                     y=_data.X[_y == _c, 1],
                     color=_col,
                     alpha=_data.alpha,
                     line_alpha=0,
                     size=8,
                     legend=_data.name + ", " + str(_c))
    bp.show(p)


def visualize_2D_classification_with_tr_weights(data, weights, classifyF=None, res=150, fig_width=500):
    _xr = np.array(data.X_range).T

    p = bp.figure(x_range=_xr[0, :].tolist(),
                  y_range=_xr[1, :].tolist(),
                  plot_width=fig_width,
                  plot_height=fig_width)
    p.xaxis.axis_label = 'x0'
    p.yaxis.axis_label = 'x1'
    if classifyF:
        x_mesh = np.linspace(_xr[0, 0], _xr[0, 1], res)
        y_mesh = np.linspace(_xr[1, 0], _xr[1, 1], res)
        _xx, _yy = np.meshgrid(x_mesh, y_mesh)
        y_hats = classifyF(np.vstack((np.ravel(_xx), np.ravel(_yy))).T)
        for _c in (True, False):
            if _c:
                _y_hats_xy = y_hats.reshape((res, res)) / 5
            else:
                _y_hats_xy = (1 - y_hats).reshape((res, res)) / 5
            _c_col = np.tile(data.tr.color[_c], (res, res, 1))
            _rgba = np.dstack((_c_col, np.round(255 * _y_hats_xy))).astype(np.uint8)
            img = np.squeeze(_rgba.view(np.uint32))
            p.image_rgba(image=[img], x=[_xr[0, 0]], y=[_xr[1, 0]],
                         dw=[_xr[0, 1] - _xr[0, 0]],
                         dh=[_xr[1, 1] - _xr[1, 0]])

    for _data in (data.tr, data.tst):
        for _c in (True, False):
            _y = np.ravel(_data.y)
            if _data.name == "Training set":
                _weights = np.ravel(weights[_y == _c]).tolist()
                _col = color.map_color(_weights, data.tr.color[_c])
            else:
                _col = color.int2hex(_data.color[_c])
            p.circle(x=_data.X[_y == _c, 0],
                     y=_data.X[_y == _c, 1],
                     color=_col,
                     alpha=_data.alpha,
                     line_alpha=0,
                     size=8,
                     legend=_data.name + ", " + str(_c))
    bp.show(p)


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

    tr_color = color.map_color(np.ravel(weights).tolist(), data.tr.color)
    p.circle(data.tst.X[:, 0], data.tst.y[:, 0],
             color=data.tst.color, alpha=0.5, line_alpha=0, size=8,
             legend="Testing set"
             )
    p.circle(data.tr.X[:, 0], data.tr.y[:, 0],
             color=tr_color, alpha=0.8, line_alpha=0, size=8,
             legend="Training set"
             )
    x_mesh = np.linspace(data.X_range[0], data.X_range[1], res)

    p.line(x_mesh, np.ravel(data.model(x_mesh)), color='green')
    if regressF:
        p.line(x_mesh, regressF(x_mesh), color='black')
    bp.show(p)
