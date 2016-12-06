class DataSet:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @classmethod
    def from_X(cls, X, model):
        y = model(X)
        return cls(X, y)

    def if_y_is(self, _c):
        idx = self.y == _c
        return self.X[idx, :]
