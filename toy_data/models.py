"""
Define various mathematical models
From toy_data:
    Module for toy-data generation for ML experiments
"""

import numpy as np
import numpy.random as rnd


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol * s[0]).sum()
    return v[rank:].copy()


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
    append1 = lambda _x: np.hstack((_x, np.ones((_x.shape[0], 1))))
    dots1 = append1(dots)
    W = null(dots1)

    def _f(x):
        if x.shape[1] != dim:
            raise Exception("X must be in " + str(dim) + "D")
        return (W.dot(append1(x.reshape(-1, dim)).T) > 0).T
    return _f
