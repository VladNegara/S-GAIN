# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for S-GAIN:

Samplers:
(1) uniform_sampler: sample uniform random variables
(2) binary_sampler: sample binary random variables

Other functions:
(3) sample_batch_index: sample index of the mini-batch
(4) normalization: normalize the data in [0, 1] range
(5) renormalization: re-normalize data from [0, 1] range to the original range
(6) rounding: round the imputed data for categorical variables
"""

import numpy as np


# -- Samplers ---------------------------------------------------------------------------------------------------------

def uniform_sampler(low, high, rows, cols, seed=None):
    """Sample uniform random variables.

    :param low: the low limit
    :param high: the high limit
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - uniform_random_matrix: a uniform random matrix
    """

    # Fix seed for run-to-run consistency
    if seed is not None: np.random.seed(seed)

    uniform_random_matrix = np.random.uniform(low, high, size=(rows, cols))
    return uniform_random_matrix


def binary_sampler(p, rows, cols, seed=None):
    """Sample binary random variables.

    :param p: the probability of 1
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - binary_random_matrix: a binary random matrix
    """

    uniform_random_matrix = uniform_sampler(0., 1., rows, cols, seed)
    binary_random_matrix = 1 * (uniform_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols, seed=None):
    """Sample uniform random variables.

    :param low: the low limit
    :param high: the high limit
    :param rows: the number of rows
    :param cols: the number of columns

    :return:
    - uniform_random_matrix: a uniform random matrix
    """

    if seed is not None: 
        np.random.seed(seed=seed)

    uniform_random_matrix = np.random.uniform(low, high, size=(rows, cols))
    return uniform_random_matrix

def missing_square_masks(miss_rate, rows, cols, seed):
    """For a list of flattened images, create a list of masks that remove a
    square from each image.

    The function assumes that each flattened image was originally square.

    :param miss_rate: the ratio between the size of the missing square and the
    size of the image
    :param rows: the number of images
    :param cols: the number of pixels in each (flattened) image
    :param seed: the seed

    :return:
    - mask_arr: an array of the size of the original dataset with values of 0 or 1 depending on if the values should be included    
    """
    seed = np.random.seed(seed)
    mask = []

    # Loop over flattened images
    for _ in range(rows):
        # Size of the image is the square root of the number of pixels
        # We want to unflatten the image
        image_size = int(cols**0.5)
        temp_mask = np.ones((image_size, image_size))

        square_size = int((miss_rate**0.5) * image_size)
        
        # The max_pos is how far the square can be from the top left corner
        max_pos = image_size - square_size

        # Left and upper edges of the square
        square_left_x = np.random.randint(0, max_pos)
        square_upper_y = np.random.randint(0, max_pos)

        # Right and lower edges of the square
        square_right_x = square_left_x + square_size
        square_lower_y = square_upper_y + square_size

        # Set values in the square to 0
        temp_mask[square_left_x:square_right_x, square_upper_y:square_lower_y] = 0

        # Flatten the mask to match original dataset
        mask.append(temp_mask.flatten())

    mask_arr = np.array(mask)
    return mask_arr

def sample_batch_index(total, batch_size):
    """Sample index of the mini-batch.

    :param total: the total number of samples
    :param batch_size: the batch size

    Returns:
    - batch_idx: the batch index
    """

    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def normalization(data_x, norm_parameters=None):
    """Normalize the data in [0, 1] range.

    :param data_x: the original data

    :return:
    - norm_data_x: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data_x.shape
    norm_data_x = data_x.copy()

    if norm_parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):  # Todo: run on GPU?
            min_val[i] = np.nanmin(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] - np.nanmin(norm_data_x[:, i])
            max_val[i] = np.nanmax(norm_data_x[:, i])
            norm_data_x[:, i] = norm_data_x[:, i] / (np.nanmax(norm_data_x[:, i]) + 1e-7)

        norm_parameters = {'min_val': min_val, 'max_val': max_val}

    else:
        min_val = norm_parameters['min_val']
        max_val = norm_parameters['max_val']

        for i in range(dim):  # Todo: run on GPU?
            norm_data_x[:, i] = norm_data_x[:, i] - min_val[i]
            norm_data_x[:, i] = norm_data_x[:, i] / (max_val[i] + 1e-7)

    return norm_data_x, norm_parameters


def renormalization(norm_data_x, norm_parameters):
    """Re-normalize data from [0, 1] range to the original range.

    :param norm_data_x: the normalized data
    :param norm_parameters: the min_val and max_val for each feature for renormalization

    :returns:
    - renorm_data_x: the re-normalized data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data_x.shape
    renorm_data_x = norm_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        renorm_data_x[:, i] = renorm_data_x[:, i] * (max_val[i] + 1e-7)
        renorm_data_x[:, i] = renorm_data_x[:, i] + min_val[i]

    return renorm_data_x


def rounding(imputed_data_x, miss_data_x):
    """Round the imputed data for categorical variables.

    :param imputed_data_x: the imputed data
    :param miss_data_x: the data with missing values

    Returns:
    - rounded_data_x: the rounded data
    """

    _, dim = miss_data_x.shape
    rounded_data_x = imputed_data_x.copy()

    for i in range(dim):  # Todo: run on GPU?
        temp = miss_data_x[~np.isnan(miss_data_x[:, i]), i]

        # Only for the categorical variables
        if len(np.unique(temp)) < 20:
            rounded_data_x[:, i] = np.round(rounded_data_x[:, i])

    return rounded_data_x
