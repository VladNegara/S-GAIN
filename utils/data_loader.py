"""Dataset loader for S-GAIN:

(1) data_loader: load a dataset and introduce missing elements
"""

import numpy as np

from utils.utils import binary_sampler, remove_square_image
from keras.datasets import mnist, fashion_mnist, cifar10


def data_loader(dataset, miss_rate, miss_modality, seed=None):
    """Load a dataset and introduce missing elements.

    Returns `None` if the miss modality is incompatible with the dataset.

    Todo: other miss modalities [AI_upscaler]

    :param dataset: the dataset to use
    :param miss_rate: the probability of missing elements in the data
    :param miss_modality: the modality of missing data [MCAR, MAR, MNAR, SQUARE]
    :param seed: the seed used to introduce missing elements in the data

    :return:
    - data_x: the original data (without missing values)
    - miss_data_x: the data with missing values
    - data_mask: the indicator matrix for missing elements
    """

    image_dataset = True

    # Load the data
    if dataset in ['health', 'letter', 'spam']:
        image_dataset = False
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

        # The first denominator is always equal to N
        denominators[0] = N

        # Set the seed
        if seed: np.random.seed(seed)

        # Initialize random weights with the U(0,1) distribution
        w = np.random.uniform(0., 1., size=d)

        # Initialize random biases with the U(0,1) distribution
        b = np.random.uniform(0., 1., size=d)

        # Initialize the mask and the data with missingness
        data_mask = np.ones(shape=(N,d))
        miss_data_x = data_x.copy()

        # Normalize data using min-max scaling
        data_x_min = data_x.min(axis=0)
        data_x_max = data_x.max(axis=0)
        data_x_normalized = (data_x.copy() - data_x_min) / (data_x_max - data_x_min)

        # Iterate over the features, then the rows
        for i in range(d):
            for n in range(N):
                # Extract the memoized exponent in the numerator of the formula
                numerator_exponent = exponent_terms[n][i]

                # Extract the memoized denominator of the formula
                denominator = denominators[i]

                # Compute the probability of missingness using the formula
                P = p_m[i] * N * np.exp(-numerator_exponent) / denominator

                # Generate a random value between 0 and 1 to check against the
                # probability
                uniform_random_value = np.random.uniform()
                if uniform_random_value < P:
                    # The value is missing
                    data_mask[n][i] = 0
                    miss_data_x[n][i] = np.nan

                    # Add the bias of this feature to the memoized numerator
                    # exponent for the next feature
                    exponent_terms[n][i+1] = exponent_terms[n][i] + b[i]
                else:
                    # Add the weighted value of this feature to the memoized
                    # numerator exponent for the next feature
                    exponent_terms[n][i+1] = exponent_terms[n][i] + w[i] * data_x_normalized[n][i]

                # Add the numerator exponent for the next feature to its
                # memoized denominator
                denominators[i+1] += np.exp(-exponent_terms[n][i+1])

    elif miss_modality == 'MNAR':
        N, d = data_x.shape

        # Uniform p_m
        p_m = np.full((d,), miss_rate)

        # Set the seed
        if seed: np.random.seed(seed)

        # Initialize random weights with the U(0,1) distribution
        w = np.random.uniform(0., 1., size=d)

        # Normalize data using min-max scaling
        data_x_min = data_x.min(axis=0)
        data_x_max = data_x.max(axis=0)
        data_x_normalized = (data_x.copy() - data_x_min) / (data_x_max - data_x_min)

        # Array to memoize the denominator in the formula
        denominators = np.zeros(shape=(d,))
        for i in range(d):
            for n in range(N):
                denominators[i] += np.exp(-w[i] * data_x_normalized[n][i])

        # Initialize the mask and the data with missingness
        data_mask = np.ones(shape=(N,d))
        miss_data_x = data_x.copy()

        # Iterate over the features, then the rows
        for i in range(d):
            for n in range(N):
                # Extract the memoized denominator of the formula
                denominator = denominators[i]

                # Compute the probability of missingness using the formula
                P = p_m[i] * N * np.exp(-w[i] * data_x_normalized[n][i]) / denominator

                # Generate a random value between 0 and 1 to check against the
                # probability
                uniform_random_value = np.random.uniform()
                if uniform_random_value < P:
                    # The value is missing
                    data_mask[n][i] = 0
                    miss_data_x[n][i] = np.nan

    elif miss_modality == 'SQUARE':

        # Square miss modality only works if the dataset is an image, it would not make sense for other types of data
        if not image_dataset:
            print('SQUARE miss modality is only valid for image datasets. Exiting the program.')
            return None
        
        no, dim = data_x.shape
        data_mask = missing_square_masks(miss_rate, no, dim, seed)
        miss_data_x = data_x.copy()

        # Use the data mask to make values nan for the model
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
