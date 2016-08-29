import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_image(image, shape, shape_gt):
    implot = plt.imshow(image, cmap=cm.Greys_r)
    plt.scatter(shape_gt[:, 1], shape_gt[:, 0], color='blue')
    plt.scatter(shape[:, 1], shape[:, 0], color='green')
    plt.show()


def evaluate_results(model, gt_images, gt_shapes, pr_params, lmk_l=2, lmk_r=11):
    pr_shapes = model._decode_parameters(pr_params)
    gt_params = model._encode_parameters(gt_shapes[:, :, :pr_shapes.shape[2]])

    # Calculate intra-ocular distance for all samples
    n_dist = np.sqrt(np.sum(
        (np.mean(gt_shapes[:, lmk_l, :], axis=1) - np.mean(gt_shapes[:, lmk_r, :], axis=1)) ** 2,
        axis=1
    ))

    mees = np.mean(np.sqrt(np.sum((pr_shapes-gt_shapes[:, :, :pr_shapes.shape[2]]) ** 2, axis=2)), axis=1) / n_dist[:, None, None]
    mee = np.mean(mees[~np.isnan(mees)])

    # for i in range(9000):
    #     plot_image(gt_images[i, ...], pr_shapes[i, ...], gt_shapes[i, ...])
    #     # plot_image(gt_images[i, ...], pr_shapes[i, lmk_l+lmk_r, :])

    return mee
