from load_bu4dfep import load_bu4dfep
from eval import evaluate_results

from cascaded.cascade.gsdm import CascadeGsdm as CascadeMethod
from cascaded.toolkit.mirror import mirror_instances
import numpy as np
import cPickle


def augmenter(images, inits, cascade, n_augs=5):
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
        range(16,26) + range(0,8) + range(36, 42) + [60, 61, 67, 59, 58, 48, 49, 50] + range(68, 75),
        range(26,36) + range(8,16) + range(47,41,-1) + [64, 63, 65, 55, 56, 54, 53, 52] + range(82,75,-1)
    )

    model = CascadeMethod(
        nb_shape=0,
        nb_feats=2
    ) if savefile is None else cPickle.load(open(savefile, 'rb'))

    model.train(
        data['images'],
        data['landmarks'][:, :, :2],
        n_steps=n_steps,
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
        print "MEE: " + str(evaluate_results(model, data['images'], data['landmarks'], preds[0], lmk_l=[0, 4], lmk_r=[8, 12]))


if __name__ == '__main__':
    # Load data and partition indices
    model_file, results_file = ('global_bu4dfep.pkl', 'global_bu4dfep_results.pkl')
    path = '/home/cvc/moliu/Datasets/bu4dfe+/'
    data = load_bu4dfep(path)

    # Train model, validate, evaluate results
    train_model(path+model_file, data['train'], n_steps=5, savefile=None)
    validate_model(path+model_file, path+results_file, data['test'])
    evaluate(path+model_file, path+results_file, data['test'])
