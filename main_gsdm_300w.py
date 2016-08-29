from load_300w import load_300w
from eval import evaluate_results
from cascaded.toolkit.mirror import mirror_instances
from cascaded.cascade.gsdm import CascadeGsdm as CascadeMethod
import numpy as np
import cPickle


def augmenter(images, inits, cascade, n_augs=25):
    inits = np.tile(cascade._decode_parameters(inits), (n_augs, 1, 1))
    angles = np.random.uniform(low=-np.pi/4.0, high=np.pi/4.0, size=len(inits))
    disps = np.random.uniform(low=0.95, high=1.05, size=(len(inits), 2))
    scales = np.random.uniform(low=0.9, high=1.1, size=len(inits))
    mapping = np.tile(np.array(range(len(images)), dtype=np.int32), (n_augs,))
    for i in range(len(inits)):
        an, sc, dx, dy = angles[i], scales[i], disps[i][0], disps[i][1]
        mn = np.mean(inits[i, ...], axis=0)[None, :]
        inits[i, ...] = np.dot(
            inits[i, ...] - mn,
            sc * np.array([[np.cos(an), -np.sin(an)], [np.sin(an), np.cos(an)]], dtype=np.float32)
        ) + mn * [dx, dy]

    return cascade._encode_parameters(inits), mapping


def train_model(fpath, data, n_steps=5, savefile=None):
    data['images'], data['landmarks'] = mirror_instances(
        data['images'], data['landmarks'],
        [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41, 31, 32, 50, 61, 67, 58, 49, 60, 59, 48],
        [16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 45, 44, 43, 42, 47, 46, 35, 34, 52, 63, 65, 56, 53, 64, 55, 54]
    )

    model = CascadeMethod(
        nb_shape=0,
        nb_feats=2
    ) if savefile is None else cPickle.load(open(savefile, 'rb'))

    model.train(
        data['images'],
        data['landmarks'],
        n_steps=n_steps,
        augmenter=augmenter,
        n_augs=25,
        save_as=fpath,
        continue_previous=(savefile is not None),
    )

    # Save model to file
    cPickle.dump(model, open(fpath, 'wb'), cPickle.HIGHEST_PROTOCOL)
    return model


def validate_model(mpath, rpath, data, steps=None):
    model = cPickle.load(open(mpath, 'rb'))
    predictions = model.align(data['images'], num_steps=steps, save_all=True)
    cPickle.dump(predictions, open(rpath, 'wb'), cPickle.HIGHEST_PROTOCOL)
    return predictions


def evaluate(mpath, rpath, data):
    predictions = cPickle.load(open(rpath, 'rb'))
    model = cPickle.load(open(mpath, 'rb'))

    for preds in predictions[0]:
        print "MEE: " + str(evaluate_results(model, data['images'], data['landmarks'], preds[0], lmk_l=[36, 39], lmk_r=[42, 45]))


if __name__ == '__main__':
    # Load data and partition indices
    model_file, results_file = ('global_300w.pkl', 'global_300w_results.pkl')
    path = '/home/cvc/moliu/Datasets/300-w/'
    data = load_300w(path)

    # Train model, validate, evaluate results
    train_model(path+model_file, data['train'], n_steps=5, savefile=None)
    validate_model(path+model_file, path+results_file, data['test'])
    evaluate(path+model_file, path+results_file, data['test'])
