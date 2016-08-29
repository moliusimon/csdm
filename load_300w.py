import os
from PIL import Image
import numpy as np
from scipy.misc import imresize
import cPickle


def load_300w(path):
    if os.path.exists(path+'database.pkl'):
        return cPickle.load(open(path+'database.pkl', 'rb'))

    ret = {
        'train': load_subparts([
            path+'afw/',
            path+'lfpw/trainset/',
            path+'helen/trainset/',
        ]), 'test': load_subparts([
            path+'ibug/',
            path+'lfpw/testset/',
            path+'helen/testset/',
        ])
    }

    cPickle.dump(ret, open(path+'database.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    return ret


def load_subparts(paths):
    gt_files = [p+f for p in paths for f in os.listdir(p) if f.endswith('.jpg') or f.endswith('.png')]
    geometry = np.zeros((len(gt_files), 68, 2), dtype=np.float32)
    images = np.zeros((len(gt_files), 200, 200), dtype=np.uint8)

    for i, f_gt in enumerate(gt_files):
        # Load image
        im = np.array(Image.open(f_gt))
        im = im if len(im.shape) < 3 else np.mean(im, axis=2).astype(np.uint8)
        imsz = im.shape

        # Load geometry
        landmarks = [[float(e) for e in l.split(' ')[:2]] for l in open(f_gt.split('.')[0]+'.pts').readlines()[3:-1]]

        # Calculate geometry center and window size
        c = np.mean(landmarks, axis=0)
        sz = int(5 * np.max(np.std(landmarks - c[None, :], axis=0)) / 2)
        c[1] -= int(0.08*sz)
        sc = 200.0 / (2 * sz)

        u, d, l, r = (int(c[0]-sz), int(c[0]+sz), int(c[1]-sz), int(c[1]+sz))
        mu, md, ml, mr = (max(0, -u), max(0, d-imsz[1]), max(0, -l), max(0, r-imsz[0]))
        mu, md, ml, mr = (int(mu*sc), int(200-md*sc), int(ml*sc), int(200-mr*sc))
        u, d, l, r = (max(0, u), min(imsz[1], d), max(0, l), min(imsz[0], r),)

        # Crop image
        geometry[i, ...] = sc * (landmarks - (c - sz))
        images[i, ml:mr, mu:md] = imresize(im[l:r, u:d], size=(mr-ml, md-mu))

    # Pass coordinates from (width, height) to (height, width)
    geometry = geometry[:, :, [1, 0]]

    return {'images': images, 'landmarks': geometry}

if __name__ == '__main__':
    path = '/mnt/Storage/Datasets/facial_landmarks/300-w/'
    data = {
        'train': load_subparts([
            path+'afw/',
            path+'lfpw/trainset/',
            path+'helen/trainset/',
        ]), 'test': load_subparts([
            path+'ibug/',
            path+'lfpw/testset/',
            path+'helen/testset/',
        ])
    }

    cPickle.dump(data, open(path+'database.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)