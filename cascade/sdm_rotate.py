from sdm import CascadeSdm
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeSdmRotate(CascadeSdm):
    def __init__(self, regressor='linear', descriptor='sift'):
        CascadeSdm.__init__(self, regressor, descriptor)

    # Parent class overrides
    # --------------------------------------------------

    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        # Calculate shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        mean_features, pca_transform, _ = variation_modes(features, min_variance=0.97)
        features = np.dot(features - mean_features[None, :], pca_transform)

        # Prepare rotated targets
        targets = ground_truth[mapping, ...] - params
        targets = self._encode_parameters(self._apply_rotations(
            self._decode_parameters(targets),
            rotations, center=False)
        )

        # Learn regressor
        regressor = self.regressor()
        tr_preds = regressor.learn(features, targets)
        return {'regressor': regressor, 'descriptor': {
            'descriptor': descriptor,
            'mean_features': mean_features,
            'pca_transform': pca_transform,
        }}, self._encode_parameters(self._apply_rotations(self._decode_parameters(tr_preds), -rotations, center=False))

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        descriptor = self.steps[i]['descriptor']

        # Calculate shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        features = features if features is not None else np.dot(descriptor['descriptor'].extract(
            images,
            shapes[:, :, :2],
            mapping,
            args={'rotations': self._get_angles(shapes)}
        )[0].reshape((len(mapping), -1)) - descriptor['mean_features'][None, :], descriptor['pca_transform'])

        # Apply regressor
        return params + self._encode_parameters(self._apply_rotations(self._decode_parameters(
            self.steps[i]['regressor'].apply(features)
        ), -rotations, center=False))
