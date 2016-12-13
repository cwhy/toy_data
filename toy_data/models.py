"""
Define various mathematical models
From toy_data:
    Module for toy-data generation for ML experiments
"""

import numpy as np
import numpy.random as rnd
from scipy import interpolate


def nullspace(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol * s[0]).sum()
    return v[rank:].copy()


def append1(_x):
    return np.hstack((_x, np.ones((_x.shape[0], 1))))


def Sine_1D(
        A=3,
        y_offset=0,
        phase=0,
        frequency=1):
    return lambda x: y_offset + A * np.sin(frequency * x + phase)


def LinearBinary(
        dim=2):
    _var_dot = 1
    dots = rnd.normal(0, _var_dot, (dim, dim))
    dots1 = append1(dots)
    W = nullspace(dots1)

    def _f(x):
        if x.shape[1] != dim:
            raise Exception("X must be in " + str(dim) + "D")
        return (W.dot(append1(x.reshape(-1, dim)).T) > 0).T

    return _f


def spline2D(
        boundary_points=((0, 5), (1, 2), (-4, -1)),
        decision_points=((1, 1), (2, 2))
):
    """
   Inputs several points to get a decision boundary using B-spline
    :param boundary_points:
    :param drift:
    :return:
    """
    raise NotImplementedError("It should be a good idea but I don't have time to implement.")


def rotatedSine2D(
        rotation=np.pi / 4.5,
        A=1,
        y_offset=0,
        phase=0,
        frequency=1.5
):
    def _f(rotated_x):
        r = -rotation
        cos_r = np.cos(r)
        sin_r = np.sin(r)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        x = rotated_x.dot(R.T)
        x1_bar = y_offset + A * np.sin(frequency * x[:, 0] + phase)
        return (x[:, 1] > x1_bar).reshape(-1, 1)

    return _f
