import numpy as np
from sklearn.decomposition import PCA


def impute_variables(bases, instances, masks):
    r_data = (instances * masks).astype(np.float32, copy=False)
    tfm = np.dot(bases, np.transpose(bases))
    n_masks = ~masks

    for i in range(10):
        r_data[n_masks] = np.dot(r_data, tfm)[n_masks]

    return r_data


def variation_modes(data, mean=None, min_variance=0.95, n_bases=None, mask=None):
    # If not considering missing values, use fast PCA implementation from sklearn
    if mask is None:
        mean = np.mean(data, axis=0) if mean is None else mean
        pca = PCA(n_components=min_variance if n_bases is None else n_bases).fit(data - mean[None, :])
        return mean, np.transpose(pca.components_), pca.explained_variance_

    # Check number of samples and variables, set missing data to 0
    mask = np.ones(tuple(data.shape), dtype=np.bool) if mask is None else mask
    f_mask = mask.astype(np.float32)
    data[~mask] = 0

    # Calculate and subtract mean from samples
    mean = np.sum(data, axis=0) / np.sum(f_mask, axis=0) if mean is None else mean
    data = data - mean[None, :]
    data[~mask] = 0

    # Calculate covariances matrix and decompose
    covariances = np.dot(np.transpose(data), data) / (np.dot(np.transpose(f_mask), f_mask) - 1)
    u, s, _ = np.linalg.svd(covariances)

    n_bases = np.where(np.cumsum(s / np.sum(s)) >= min_variance)[0][0]+1 if n_bases is None else n_bases
    return mean, u[:, :n_bases], s[:n_bases]
