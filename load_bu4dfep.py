import scipy.io as sio
import numpy as np
import cPickle
import os


def load_bu4dfep(path):
    if os.path.exists(path+'database_0.pkl'):
        ret = cPickle.load(open(path+'database_0.pkl', 'rb'))
        ret['train']['images'] = np.concatenate((
            ret['train']['images'],
            cPickle.load(open(path+'database_1.pkl', 'rb'))
        ), axis=0)

        return ret

    ret = {'train': prepare_data(path+'train/'), 'test': prepare_data(path+'test/')}
    tmp = ret['train']['images'][35000:, ...]
    ret['train']['images'] = ret['train']['images'][:35000, ...]
    cPickle.dump(ret, open(path+'database_0.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(tmp, open(path+'database_1.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    return ret


def prepare_data(path):
    # List files of interest
    ind_files = [f for f in os.listdir(path) if f.endswith('.mat')]
    n_inds = len(ind_files)
    inds = [None] * n_inds

    images = [None] * n_inds
    landmarks = [None] * n_inds
    metadata = [None] * n_inds

    # Parse all files one by one
    for i, f in enumerate(ind_files):
        # Read samples from file
        t_samp = sio.loadmat(path+f)
        t_samp = (t_samp['augperson'] if 'augperson' in t_samp.keys() else t_samp['pers']).flatten()
        n_samp = len(t_samp)

        # Prepare buffers
        images[i] = np.empty((n_samp, 200, 200), dtype=np.float32)
        landmarks[i] = np.empty((n_samp, 83, 3), dtype=np.float32)
        metadata[i] = [None] * n_samp

        # Capture useful data
        for j, e in enumerate(t_samp):
            mdata = e[2][0][0]
            images[i][j, ...] = np.cast[np.uint8](e[0])
            landmarks[i][j, ...] = e[1]
            metadata[i][j] = {
                'expression': str(mdata[1][0]),
                'roll': float(mdata[2][0, 0][2][0][0]),
                'pitch': float(mdata[2][0, 0][1][0][0]),
                'yaw': float(mdata[2][0, 0][0][0][0])
            }

        # Invert X and Y coordinates (to correspond with image coordinates)
        landmarks[i] = landmarks[i][:, :, np.array([1, 0, 2])]

    return {
        'images': np.cast[np.uint8](np.concatenate(tuple(images), axis=0)),
        'landmarks': np.cast[np.float32](np.concatenate(tuple(landmarks), axis=0)),
        'metadata': sum(metadata, []),
    }


if __name__ == '__main__':
    load_bu4dfep('/mnt/Storage/Datasets/facial_landmarks/bu4dfe+/')
