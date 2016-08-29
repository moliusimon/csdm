from regressor import Regressor
import numpy as np


class RegressorLinear(Regressor):
    def __init__(self):
        Regressor.__init__(self)
        self.weights, self.bias = (None, None)

    def learn(self, inputs, targets):
        inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1), dtype=np.float32)), axis=1)
        reg = np.dot(np.linalg.pinv(inputs), targets)
        self.weights = reg[:-1, :]
        self.bias = reg[None, -1, :]
        return np.dot(inputs, reg)

    def apply(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
