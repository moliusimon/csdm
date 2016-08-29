from metalinear import RegressorMetalinear
import numpy as np


class RegressorMetalinearFs(RegressorMetalinear):
    def __init__(self):
        RegressorMetalinear.__init__(self)

    def learn(self, f_projs, inputs, targets, indices):
        self.n_features, self.n_targets = inputs.shape[1], targets.shape[1]
        n_folds, b_size = 25, (np.max(indices) + 1) / 25

        errors = np.zeros((n_folds,), dtype=np.float32)
        self.n_bases, error = 0, 1 * (10 ** 8)
        for i in range(f_projs.shape[1]+1):
            khatri_rprod = self._khatri_rao(inputs, f_projs[:, :i])
            for j in range(n_folds):
                # Remove validation set
                i_valid = np.in1d(indices, np.array(range(j*b_size, ((j+1)*b_size))))
                c_i, c_t = khatri_rprod[i_valid, :], targets[i_valid, :]
                khatri_rprod[i_valid, :], targets[i_valid, :] = 0, 0

                # Predict validation set and restore validation entries
                errors[j] = np.mean((np.dot(c_i, np.linalg.lstsq(khatri_rprod, targets)[0]) - c_t) ** 2)
                khatri_rprod[i_valid, :], targets[i_valid, :] = c_i, c_t

            # If error worsened, exit loop
            del khatri_rprod
            if np.mean(errors) > error:
                break

            # Specify best
            self.n_bases, error = i, np.mean(errors)

        # Calculate regressor
        khatri_rprod = self._khatri_rao(inputs, f_projs[:, :self.n_bases])
        self.m_weights = np.linalg.lstsq(khatri_rprod, targets)[0]
        return self.apply(f_projs, inputs)

