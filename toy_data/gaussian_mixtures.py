"""
Includes Gaussian Mixture Generator
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


def mix_shuffle(_Xs):
    _Xs_lb = []
    _Nclass = len(_Xs)
    for i, _X in enumerate(_Xs):
        _label = np.zeros((_X.shape[0], _Nclass))
        _label[:,i] = 1
        _X = np.hstack((_X, _label))
        _Xs_lb.append(_X)

    _mixed = np.vstack(_Xs_lb)
    rnd.shuffle(_mixed)
    _ft_all = _mixed[:, :-_Nclass]
    _lb_all = _mixed[:, -_Nclass:].astype(int)
    return _ft_all, _lb_all


def split(_fts, _lbs, test_ratio):
    indices = range(_fts.shape[0])
    i_test = rnd.choice(indices, size=int(round(test_ratio * _fts.shape[0])))
    i_train = np.array([i for i in indices if i not in i_test])
    return (_fts[i_train, :], _lbs[i_train, :]), (_fts[i_test, :], _lbs[i_test, :])


def visualize_2D(_Classes, _colors, classifyF=None, res=150, fig_width=500):
    _xmins = []
    _xmaxs = []
    for _C in _Classes:
        _xmins.append(np.min(_C, axis=0))
        _xmaxs.append(np.max(_C, axis=0))
    _xmin = np.min(np.array(_xmins), axis=0)
    _xmax = np.max(np.array(_xmaxs), axis=0)
    _xr = np.vstack((_xmin, _xmax)).T
    p = bp.figure(x_range=_xr[0, :].tolist(),
                  y_range=_xr[1, :].tolist(),
                  plot_width=fig_width,
                  plot_height=fig_width)
    if classifyF:
        x_mesh = np.linspace(_xr[0, 0], _xr[0, 1], res)
        y_mesh = np.linspace(_xr[1, 0], _xr[1, 1], res)
        _xx, _yy = np.meshgrid(x_mesh, y_mesh)
        y_hats = classifyF(np.vstack((np.ravel(_xx), np.ravel(_yy))).T)
        for _c in range(len(_Classes)):
            _y_hats_xy = y_hats[:, _c].reshape((res, res))/4
            _c_col = np.tile(_colors[_c], (res, res, 1))
            # print(_colors[_c])
            _rgba = np.dstack((_c_col, np.round(255*_y_hats_xy))).astype(np.uint8)
            img = np.squeeze(_rgba.view(np.uint32))
            p.image_rgba(image=[img], x=[_xr[0, 0]], y=[_xr[1, 0]],
                         dw=[_xr[0, 1] - _xr[0, 0]],
                         dh=[_xr[1, 1] - _xr[1, 0]])

    for i, _C in enumerate(_Classes):
        _col = bk.colors.RGB(_colors[i][0],
                             _colors[i][1],
                             _colors[i][2]).to_hex()
        p.circle(x=_C[:, 0], y=_C[:, 1], color=_col)
    bp.show(p)


class GaussianMixture:
    def __init__(self,
                 dim=2,
                 n_class=2,
                 n_samples=100,
                 mu_diff=(3, 4),
                 sigma_range=(1.5, 2.5),
                 split_ratio=0.2):
        """

        :type dim: int
        """
        self.xs_mu = []
        self.xs_sigma = []
        self.Classes = []
        self.class_colors = []
        _mu = rnd.uniform(mu_diff[0], mu_diff[-1], dim)
        for _c in range(n_class):
            _mu += rnd.choice([-1, 1], dim)*rnd.uniform(mu_diff[0], mu_diff[-1], dim)
            _sigma = rnd.uniform(sigma_range[0], sigma_range[-1], dim)
            self.xs_mu.append(_mu)
            self.xs_sigma.append(_sigma)
            self.Classes.append(rnd.normal(_mu, _sigma, (n_samples, dim)))
        self.class_colors = color.get_N_by_hue(n_class)

        self.X, self.y = mix_shuffle(self.Classes)
        (tr_X, tr_y), (tst_X, tst_y) = split(self.X, self.y, split_ratio)
        self.tr = DataSet(tr_X, tr_y)
        self.tst = DataSet(tst_X, tst_y)
