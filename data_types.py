class DataSet:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def if_y_is(self, _c):
        idx = self.y == _c
        return self.X[idx, :]


