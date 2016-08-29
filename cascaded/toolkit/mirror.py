import numpy as np


def mirror_instances(images, geometry, lmk_left, lmk_right):
    n_inst = images.shape[0]
    i_inst = 2 * np.arange(n_inst)

    # Create normal + mirrored images
    r_images = np.empty((2*n_inst,)+images.shape[1:], dtype=images.dtype)
    r_images[i_inst, ...] = images
    r_images[i_inst+1, ...] = images[:, :, ::-1] if len(images.shape) == 3 else images[:, :, ::-1, :]

    # Prepare mirrored geometry
    geometry_t = np.copy(geometry)
    geometry_t[:, lmk_right, :] = geometry[:, lmk_left, :]
    geometry_t[:, lmk_left, :] = geometry[:, lmk_right, :]
    geometry_t[:, :, 1] = images.shape[2] - geometry_t[:, :, 1]

    # Create normal + mirrored geometry
    r_geometry = np.empty((2*n_inst,)+geometry.shape[1:], dtype=geometry.dtype)
    r_geometry[i_inst, ...] = geometry
    r_geometry[i_inst+1, ...] = geometry_t

    return r_images, r_geometry
