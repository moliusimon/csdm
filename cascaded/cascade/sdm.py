from cascade import Cascade
from ..toolkit.procrustes import procrustes_generalized, procrustes
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeSdm(Cascade):
    def __init__(self, regressor='linear', descriptor='sift'):
        Cascade.__init__(self, regressor, descriptor)
        self.mean_shape = None

    # Parent class overrides
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        self.mean_shape, _ = procrustes_generalized(ground_truth)
        self.mean_shape += np.mean(np.mean(ground_truth, axis=0), axis=0)[None, :]

    def _initialize_instances(self, images):
        return np.tile(self.mean_shape.reshape((1, -1)), (images.shape[0], 1))

    def _encode_parameters(self, decoded):
        return decoded.reshape((decoded.shape[0], self.num_landmarks * self.num_dimensions))

    def _decode_parameters(self, encoded):
        return encoded.reshape((encoded.shape[0], self.num_landmarks, self.num_dimensions))

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

        # Prepare targets
        targets = ground_truth[mapping, ...] - params

        # Learn regressor
        regressor = self.regressor()
        tr_preds = regressor.learn(features, targets)
        return {'regressor': regressor, 'descriptor': {
            'descriptor': descriptor,
            'mean_features': mean_features,
            'pca_transform': pca_transform,
        }}, tr_preds

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
        return params + self.steps[i]['regressor'].apply(features)

    # SDM-Specific methods
    # --------------------------------------------------

    def _get_angles(self, shapes):
        # Prepare unit vector to rotate
        vector = np.zeros((1, shapes.shape[2]), dtype=np.float32)
        vector[0, 0] = 1

        # Calculate rotation angle of the face wrt. the screeen plane, for each shape
        angles = np.empty((shapes.shape[0],), dtype=np.float32)
        for i, s in enumerate(shapes):
            _, _, tfm = procrustes(self.mean_shape, s-np.mean(s, axis=0, keepdims=True))
            vect = np.dot(vector, tfm['rotation'][:, :2])
            angles[i] = np.arctan2(vect[0, 1], vect[0, 0])

        return angles

    def _apply_rotations(self, shapes, angles, center=True):
        means = np.mean(shapes, axis=1)[:, None, :]
        shapes = np.copy(shapes-means if center else shapes)

        for i, (s, r) in enumerate(zip(shapes, angles)):
            shapes[i, ...] = np.dot(s, np.array([[np.cos(r), np.sin(r)], [-np.sin(r),  np.cos(r)]]))

        return shapes
