import numpy as np
from PIL import Image

from torchvision.datasets import CIFAR10


import torch


def corrupt_label(training_labels):
    """
    training_labels: clean training labels
    return: corrupted labels
    """

    np.random.seed(1)
    num_annotator = 5
    label_ = int(len(training_labels)/num_annotator)
    noise_ratio = [0, 0.2, 0.4, 1, -1]  # -1 means total adversarial
    noisy = np.array(noise_ratio) * label_

    normal_ind = []
    normal_ind.extend(range(label_))

    for i in range(num_annotator):
        if noise_ratio[i] > 0:
            training_labels[i * label_ : i * label_ + int(noisy[i])] = np.random.choice(10, int(noisy[i]))
            training_labels[i * label_ : (i+1) * label_] = [l + i * 100 for l in training_labels[i * label_ : (i+1) * label_]]
            normal_ind.extend(range(i * label_ + int(noisy[i]), (i+1)*label_))
        elif noise_ratio[i] == -1:
            # adversarial
            training_labels[i * label_: (i + 1) * label_] = [(l+1) % 10 + i * 100 for l in
                                                             training_labels[i * label_: (i + 1) * label_]]

    return training_labels, np.array(normal_ind).flatten()


