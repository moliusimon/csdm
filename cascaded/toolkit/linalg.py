import numpy as np
import math


def transform_shapes(shapes, transforms, inverse=False):
    aligned = np.empty(tuple(shapes.shape), dtype=np.float32)

    if inverse:
        for i, (s, t) in enumerate(zip(shapes, transforms)):
            aligned[i, ...] = np.dot(s - t['translation'][None, :], np.transpose(t['rotation']) / t['scale'])
    else:
        for i, (s, t) in enumerate(zip(shapes, transforms)):
            aligned[i, ...] = np.dot(s, t['scale']*t['rotation']) + t['translation'][None, :]

    return aligned


def build_rotation_matrix(roll, pitch, yaw):
    return np.dot(np.dot([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1],
    ], [
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ]), [
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)],
    ])