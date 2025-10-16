"""Dataset loader for S-GAIN:

(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np

from utils.utils import binary_sampler, remove_square_image
from keras.datasets import mnist, fashion_mnist, cifar10


def data_loader(dataset, miss_rate, miss_modality, seed=None):
    """Load a dataset and introduce missing elements.

    Todo: other miss modalities (MAR, MNAR, others for image data?)

    :param dataset: the dataset to use
    :param miss_rate: the probability of missing elements in the data
    :param miss_modality: the modality of missing data (MCAR, MAR, MNAR)
    :param seed: the seed used to introduce missing elements in the data

    :return:
    - data_x: the original data (without missing values)
    - miss_data_x: the data with missing values
    - data_mask: the indicator matrix for missing elements
    """

    image_datasets = ['fashion_mnist', 'cifar10']

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
    if miss_modality == 'MCAR' and dataset not in image_datasets:
        no, dim = data_x.shape
        data_mask = binary_sampler(1 - miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan
    elif miss_modality == 'MAR' and dataset not in image_datasets:
        print('MAR not yet implemented. Exiting the program.')
        return None
    elif miss_modality == 'MNAR' and dataset not in image_datasets:
        print('MNAR not yet implemented. Exiting the program.')
        return None
    elif dataset in image_datasets:
        no, dim = data_x.shape
        data_mask = remove_square_image(miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan
    else:
        print('Invalid miss modality. Exiting the program.')
        return None

    return data_x, miss_data_x, data_mask
