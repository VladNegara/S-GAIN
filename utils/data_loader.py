"""Dataset loader for S-GAIN:

(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np

from utils.utils import binary_sampler
from keras.datasets import mnist, fashion_mnist, cifar10


def data_loader(dataset, miss_rate, miss_modality, seed=None):
    """Load a dataset and introduce missing elements.

    Todo: other miss modalities [MAR, MNAR, AI_upscaler, square]

    :param dataset: the dataset to use
    :param miss_rate: the probability of missing elements in the data
    :param miss_modality: the modality of missing data [MCAR, MAR, MNAR, AI_upscaler, square]
    :param seed: the seed used to introduce missing elements in the data

    :return:
    - data_x: the original data (without missing values)
    - miss_data_x: the data with missing values
    - data_mask: the indicator matrix for missing elements
    """

    # Load the data
    if dataset in ['health', 'letter', 'spam']:
        file_name = f'datasets/{dataset}.csv'
        data_x = np.loadtxt(file_name, delimiter=',', skiprows=1)
    elif dataset == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'fashion_mnist':
        (data_x, _), _ = fashion_mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28 * 28]).astype(float)
    elif dataset == 'cifar10':
        (data_x, _), _ = cifar10.load_data()
        data_x = np.reshape(np.asarray(data_x), [50000, 32 * 32 * 3]).astype(float)
    else:  # This should not happen
        print(f'Invalid dataset: "{dataset}". Exiting the program.')
        return None

    # Introduce missing elements in the data
    if miss_modality == 'MCAR':
        no, dim = data_x.shape
        data_mask = binary_sampler(1 - miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan
    else:  # This should not happen
        print(f'Invalid miss modality: "{miss_modality}". Exiting the program.')
        return None

    return data_x, miss_data_x, data_mask
