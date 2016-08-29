from metalinear import RegressorMetalinear
import numpy as np


class RegressorMetalinearFsen(RegressorMetalinear):
    def __init__(self):
        RegressorMetalinear.__init__(self)

    def learn(self, f_projs, inputs, targets, indices):
        self.n_features, self.n_targets = inputs.shape[1], targets.shape[1]
        n_inst, n_folds = np.max(indices)+1, 25
        b_size = n_inst / n_folds

        # Allocate fit results
        r_prev, r_new = np.empty(targets.shape, dtype=np.float32), np.zeros(targets.shape, dtype=np.float32)

        self.n_bases, error = 0, 1 * (10 ** 8)
        for i in range(f_projs.shape[1]+1):
            khatri_rprod = self._khatri_rao(inputs, f_projs[:, :i])
            for j in range(n_folds):
                # Remove validation set
                i_valid = np.in1d(indices, np.array(range(j*b_size, ((j+1)*b_size))))
                c_i, c_t = khatri_rprod[i_valid, :], targets[i_valid, :]
                khatri_rprod[i_valid, :], targets[i_valid, :] = 0, 0

                # Predict validation set and restore validation entries
                r_new[i_valid, :] = np.dot(c_i, np.linalg.lstsq(khatri_rprod, targets)[0])
                khatri_rprod[i_valid, :], targets[i_valid, :] = c_i, c_t

            # If error worsened, exit loop
            del khatri_rprod
            error_new = np.mean((r_new - targets) ** 2)
            if error_new > error:
                break

            # Specify best
            self.n_bases, error = i, error_new
            r_prev[:, :] = r_new

        # Calculate regressor
        khatri_rprod = self._khatri_rao(inputs, f_projs[:, :self.n_bases])
        self.m_weights = np.linalg.lstsq(khatri_rprod, targets)[0]
        return r_prev

