from regressor import Regressor
from linear import RegressorLinear
import numpy as np


class RegressorGausslinear(Regressor):
    def __init__(self):
        Regressor.__init__(self)
        self.n_features, self.n_targets = (None, None)
        self.m_weights, self.m_bias = (None, None)

    def learn(self, f_projs, inputs, targets):
        self.n_features = inputs.shape[1]
        self.n_targets = targets.shape[1]

        # Learn regressor for each instance
        # -----------------------------------------------------------

        t_fact = - 1.0 / (2 * (1 ** 2) * f_projs.shape[1])
        regressors = [RegressorLinear() for _ in range(len(inputs))]
        for i in range(len(inputs)):
            w = np.exp(t_fact * np.sum((f_projs - f_projs[i, :]) ** 2, axis=1, keepdims=True))
            regressors[i].learn(inputs * w, targets * w)

        # Learn 'meta-regressor'
        # -----------------------------------------------------------

        meta = np.dot(
            np.linalg.pinv(np.concatenate((f_projs, np.ones((len(f_projs), 1), dtype=np.float32)), axis=1)),
            np.concatenate([np.concatenate((r.weights, r.bias), axis=0).reshape(1, -1) for r in regressors], axis=0)
        )

        self.m_weights = meta[:-1, :]
        self.m_bias = meta[None, -1, :]
        return self.apply(f_projs, inputs)

    def apply(self, f_projs, inputs):
        n_inst = f_projs.shape[0]

        ret = np.empty((n_inst, self.n_targets), dtype=np.float32)
        for i in range(n_inst):
            regressor = (np.dot(f_projs[None, i, :], self.m_weights) + self.m_bias).reshape((self.n_features+1, -1))
            ret[i, :] = np.dot(inputs[None, i, :], regressor[:-1, :]) + regressor[None, -1, :]

        return ret
