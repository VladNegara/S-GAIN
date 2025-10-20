"""Dataset loader for S-GAIN:

(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np

from utils.utils import binary_sampler, remove_square_image
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
    if miss_modality == 'MCAR':
        no, dim = data_x.shape
        data_mask = binary_sampler(1 - miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan
    elif miss_modality == 'MAR':
        N, d = data_x.shape

        # Uniform p_m
        p_m = np.full((d,), miss_rate)

        # Array to memoize sums in exponents in the formula
        # Cell [n][i] holds the sum over j<i
        exponent_terms = np.zeros(shape=(N,d+1))

        # Array to memoize the denominator in the formula
        denominators = np.zeros(shape=(d+1,))
        denominators[0] = N

        if seed: np.random.seed(seed)

        w = np.random.uniform(0., 1., size=d)
        b = np.random.uniform(0., 1., size=d)

        data_mask = np.ones(shape=(N,d))
        miss_data_x = data_x.copy()

        for i in range(d):
            for n in range(N):
                if data_mask[n][i] == 1:
                    exponent_terms[n][i+1] = exponent_terms[n][i] + w[i] * miss_data_x[n][i]
                else:
                    exponent_terms[n][i+1] = exponent_terms[n][i] + b[i]
                
                denominators[i+1] += np.exp(-exponent_terms[n][i+1])
                
                numerator_exponent = exponent_terms[n][i]

                denominator = denominators[i]

                P = p_m[i] * N * np.exp(-numerator_exponent) / denominator

                uniform_random_value = np.random.uniform()

                if uniform_random_value < P:
                    # The value is missing
                    data_mask[n][i] = 0
                    miss_data_x[n][i] = np.nan
    elif miss_modality == 'MNAR':
        N, d = data_x.shape

        # Uniform p_m
        p_m = np.full((d,), miss_rate)

        if seed: np.random.seed(seed)

        w = np.random.uniform(0., 1., size=d)

        # Array to memoize the denominator in the formula
        denominators = np.zeros(shape=(d,))
        for i in range(d):
            for n in range(N):
                denominators[i] += np.exp(-w[i] * data_x[n][i])

        data_mask = np.ones(shape=(N,d))
        miss_data_x = data_x.copy()

        for i in range(d):
            for n in range(N):
                P = p_m[i] * N * np.exp(-w[i] * data_x[n][i]) / denominators[i]

                uniform_random_value = np.random.uniform()

                if uniform_random_value < P:
                    # The value is missing
                    data_mask[n][i] = 0
                    miss_data_x[n][i] = np.nan
    elif dataset in image_datasets:
        no, dim = data_x.shape
        data_mask = remove_square_image(miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()
        miss_data_x[data_mask == 0] = np.nan
    else:
        print('Invalid miss modality. Exiting the program.')
        return None
    
    # actual_missing_rates = np.count_nonzero(np.isnan(miss_data_x), axis=0) / N

    # print('Missing rates per column:')
    # print(actual_missing_rates)
    # print(f'Average missing rate: {actual_missing_rates.mean()}')

    # np.savetxt("temp/miss_data.csv", miss_data_x, delimiter=',')

    return data_x, miss_data_x, data_mask
