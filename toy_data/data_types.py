import toy_data.color as color
import numpy as np


class DataSet:
    def __init__(self, X, y, model=None):
        self.X = X
        self.y = y
        self.model = model
        self.X_range = [np.min(self.X), np.max(self.X)]
        self.color = color.color_loop[0]

    @classmethod
    def from_X(cls, X, model):
        y = model(X)
        return cls(X, y, model)

    def if_y_is(self, _c):
        idx = self.y == _c
        return self.X[idx, :]
