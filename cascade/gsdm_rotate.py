from sdm import CascadeSdm
from ..toolkit.pca import variation_modes
import numpy as np


class CascadeGsdmRotate(CascadeSdm):
    def __init__(self, regressor='linear', descriptor='sift', nb_shape=2, nb_feats=1):
        CascadeSdm.__init__(self, regressor, descriptor)
        self.nb_shape = nb_shape
        self.nb_feats = nb_feats

    # Parent class overrides
    # --------------------------------------------------

    def _initialize_method(self, images, ground_truth):
        CascadeSdm._initialize_method(self, images, ground_truth)

    def _train_step(self, images, ground_truth, params, mapping, i, args=None):
        nb_shape = self.nb_shape if args is None else args.get('nb_shape', self.nb_shape)
        nb_feats = self.nb_feats if args is None else args.get('nb_feats', self.nb_feats)

        # Get shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        descriptor = self.descriptor()
        descriptor.initialize(images, shapes[:, :, :2], mapping, args={'rotations': rotations})
        features, visibility = descriptor.extract(images, shapes[:, :, :2], mapping, args={'rotations': rotations})

        # Find features compression
        mean_features, pca_transform, _ = variation_modes(features, min_variance=0.90)
        features = np.dot(features - mean_features[None, :], pca_transform)

        # Prepare rotated targets
        targets = ground_truth[mapping, ...] - params
        targets = self._encode_parameters(self._apply_rotations(
            self._decode_parameters(targets),
            rotations, center=False)
        )

        # Find two principal components of shape deltas
        mean_dshape, pca_dshape, _ = variation_modes(targets, n_bases=nb_shape)
        bases = np.concatenate((
            np.dot(targets - mean_dshape, pca_dshape),
            features[:, :nb_feats]
        ), axis=1)

        # Train individual regressors
        reg_instances = np.sum((bases > 0) * (2 ** np.arange(nb_feats+nb_shape))[None, ::-1], axis=1)
        regressors = [None] * (2 ** bases.shape[1])
        tr_preds = np.zeros((len(mapping), ground_truth.shape[1]), dtype=np.float32)
        for i in range(2 ** bases.shape[1]):
            regressors[i] = self.regressor()
            tr_preds[reg_instances == i, :] = regressors[i].learn(
                features[reg_instances == i, :],
                targets[reg_instances == i, :]
            )

        # Save regressor structure
        return {'regressors': regressors, 'descriptor': {
            'descriptor': descriptor,
            'mean_features': mean_features,
            'pca_transform': pca_transform,
            'mean_dshape': mean_dshape,
            'pca_dshape': pca_dshape,
            'nb_feats': nb_feats,
        }}, self._encode_parameters(self._apply_rotations(self._decode_parameters(tr_preds), -rotations, center=False))

    def _align_step(self, images, params, mapping, i, features=None, args=None):
        descriptor = self.steps[i]['descriptor']
        args = {} if args is None else args

        # Get shapes and rotations
        shapes = params.reshape((-1, self.num_landmarks, self.num_dimensions))
        rotations = self._get_angles(shapes)

        # Extract features
        features = features if features is not None else np.dot(descriptor['descriptor'].extract(
            images,
            shapes[:, :, :2],
            mapping,
            args={'rotations': rotations}
        )[0].reshape((len(mapping), -1)) - descriptor['mean_features'][None, :], descriptor['pca_transform'])

        # Rotate targets if present
        deltas = None if descriptor['pca_dshape'].shape[1] <= 0 else self._apply_rotations(
            args['target'][mapping, ...] - params, rotations, center=False
        )

        # Get bases
        bases = np.dot(
            deltas - descriptor['mean_dshape'], descriptor['pca_dshape']
        ) if descriptor['pca_dshape'].shape[1] > 0 else np.zeros((features.shape[0], 0), dtype=np.float32)
        bases = np.concatenate((bases, features[:, :descriptor['nb_feats']]), axis=1)

        # Apply specific regressor to each instance
        reg_instances = np.sum((bases > 0) * (2 ** np.arange(bases.shape[1]))[None, ::-1], axis=1)
        for j in range(2 ** bases.shape[1]):
            params[reg_instances == j, :] += self._encode_parameters(self._apply_rotations(
                self._decode_parameters(self.steps[i]['regressors'][j].apply(features[reg_instances == j, :])),
                -rotations[reg_instances == j], center=False
            ))

        return params, bases