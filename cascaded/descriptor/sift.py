from descriptor import Descriptor

import numpy as np
from skimage.feature import hog

class DescriptorSift(Descriptor):
    def __init__(self):
        Descriptor.__init__(self)

    def _extract(self, images, coords, mapping, args):
        assert images.shape[1] == images.shape[2]
        n_inst = coords.shape[0]

        nb = args.get('num_bins', 8)
        win_sizes = args.get('window_sizes', 32)
        win_sizes = win_sizes if isinstance(win_sizes, np.ndarray) else np.ones((n_inst,), dtype=np.int32) * win_sizes

        # Prepare descriptors
        descriptors = np.zeros(tuple(coords.shape[:2])+(nb*4*4,), dtype=np.float32)

        # Fill descriptors
        coords, vis = np.copy(coords), np.zeros(coords.shape[:2], dtype=np.bool)
        for i, (c, mp, ws) in enumerate(zip(coords, mapping, win_sizes)):
            hsize, qsize = ws/2, ws/4

            # Pad image, set landmarks visibility
            im, c = np.pad(images[mp, ...], ((hsize, hsize), (hsize, hsize)), 'constant', constant_values=0), c+hsize
            ims = im.shape[0] - hsize
            vis[i, :] = (c[:, 0] >= hsize) & (c[:, 1] >= hsize) & (c[:, 0] < ims) & (c[:, 1] < ims)

            # Extract descriptors from each interest window
            for j, (jc, jv) in enumerate(zip(c, vis[i, :])):
                descriptors[i, j, :] = hog(
                    im[jc[0]-hsize:jc[0]+hsize, jc[1]-hsize:jc[1]+hsize],
                    orientations=nb,
                    pixels_per_cell=(qsize, qsize),
                    cells_per_block=(1, 1)
                ) if jv else 0

        # Normalize descriptors, return extracted information
        return descriptors.reshape((len(mapping), -1)), vis
