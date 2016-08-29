from sdm import CascadeSdm
from toolkit.pca import variation_modes
import numpy as np


class CascadeCsdm(CascadeSdm):
    def __init__(self, regressor='metalinear', descriptor='sift', num_bases=5, variance=0.9):
        CascadeSdm.__init__(self, regressor, descriptor)
        self.num_bases = num_bases
        self.variance = variance

    # Parent class overrides
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        CascadeSdm._initialize_method(self, images, ground_truth)

    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        n_b = self.num_bases[i] if isinstance(self.num_bases, list) else self.num_bases
        var = self.variance[i] if isinstance(self.variance, list) else self.variance

        # Prepare shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        # Find features compression
        mean_features, pca_transform, _ = variation_modes(features, min_variance=var)
        sbs_transform = pca_transform[:, :n_b]
        f_projs = np.dot(features - mean_features[None, :], sbs_transform)
        f_feats = np.dot(features - mean_features[None, :], pca_transform)
        del features

        # Prepare targets
        targets = ground_truth[mapping, ...] - params

        # Learn regressor
        regressor = self.regressor()
        tr_preds = regressor.learn(f_projs, f_feats, targets, mapping)
        return {'regressor': regressor, 'descriptor': {
            'descriptor': descriptor,
            'mean_features': mean_features,
            'pca_transform': pca_transform,
            'sbs_transform': sbs_transform
        }}, tr_preds

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        descriptor = self.steps[i]['descriptor']

        # Prepare shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        features = features if features is not None else descriptor['descriptor'].extract(
            images,
            shapes[:, :, :2],
            mapping,
            args={'rotations': self._get_angles(shapes)}
        )[0]

        f_projs = np.dot(features - descriptor['mean_features'][None, :], descriptor['sbs_transform'])
        f_feats = np.dot(features - descriptor['mean_features'][None, :], descriptor['pca_transform'])

        # Apply regressor
        preds = self.steps[i]['regressor'].apply(f_projs, f_feats)
        return params + preds, f_projs
