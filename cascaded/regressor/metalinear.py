from regressor import Regressor
import numpy as np
from scipy import linalg
# import theano
# import theano.tensor as T


class RegressorMetalinear(Regressor):
    def __init__(self):
        Regressor.__init__(self)
        self.n_features, self.n_targets, self.n_bases = (None, None, None)
        self.m_weights = None

    def learn(self, f_projs, inputs, targets, indices):
        self.n_features, self.n_targets, self.n_bases = inputs.shape[1], targets.shape[1], f_projs.shape[1]
        khatri_rprod = self._khatri_rao(inputs, f_projs)
        self.m_weights = np.linalg.lstsq(khatri_rprod, targets)[0]
        return np.dot(khatri_rprod, self.m_weights)

    def apply(self, f_projs, inputs):
        khatri_rprod = self._khatri_rao(inputs, f_projs[:, :self.n_bases])
        return np.dot(khatri_rprod, self.m_weights)

    @staticmethod
    def _khatri_rao(features, bases):
        n_inst, n_features, n_bases = features.shape[0], features.shape[1], bases.shape[1]
        ret = np.empty((n_inst, (n_features+1)*(n_bases+1)), dtype=np.float32)

        ret[:, n_bases*(n_features+1):-1] = features
        ret[:, -1] = 1
        for i in range(n_bases):
            ret[:, i*(n_features+1):(i+1)*(n_features+1)-1] = features * bases[:, i][:, None]
            ret[:, (i+1)*(n_features+1)-1] = bases[:, i]

        return ret
