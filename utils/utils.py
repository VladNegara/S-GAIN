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

(1) binary_sampler: sample binary random variables
(2) uniform_sampler: sample uniform random variables
(3) sample_batch_index: sample index of the mini-batch
(3) normalization: normalize the data in [0, 1] range
(4) renormalization: re-normalize data from [0, 1] range to the original range
(5) rounding: round the imputed data for categorical variables
"""

import numpy as np


def binary_sampler(p, rows, cols, seed=None):
    """Sample binary random variables.

    :param p: the probability of 1
    :param rows: the number of rows
    :param cols: the number of columns
    :param seed: the random seed

    :return:
    - binary_random_matrix: a binary random matrix
    """

    # Fix seed for run-to-run consistency
    if seed: np.random.seed(seed)

    uniform_random_matrix = np.random.uniform(0., 1., size=(rows, cols))
    binary_random_matrix = 1 * (uniform_random_matrix < p)
    return binary_random_matrix

def mar_sampler(X, p_m, seed=None, standardize=True, w_scale=0.1):
    """
    Optimized MAR sampler with numerically stable softmax-like formula.
    Returns mask M (1 observed, 0 missing).
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.asarray(X, dtype=float)
    rows, cols = X.shape
    M = np.ones((rows, cols), dtype=int)
    
    # Handle p_m input
    p_m = np.broadcast_to(np.asarray(p_m, dtype=float), cols)
    
    # Optionally standardize features
    if standardize:
        std = X.std(axis=0, ddof=0)
        std = np.where(std == 0, 1.0, std)
        X = (X - X.mean(axis=0)) / std
    
    # Generate random weights and biases once
    w = np.random.uniform(size=cols) 
    b = np.random.uniform(size=cols)
    
    # Process each column
    for i in range(cols):
        if i == 0:
            # First column: uniform probabilities
            probs = np.ones(rows) / rows
        else:
            # Compute logits using vectorized operations
            prev_cols = slice(0, i)
            observed_term = (w[prev_cols] * X[:, prev_cols] * M[:, prev_cols]).sum(axis=1)
            missing_term = (b[prev_cols] * (1 - M[:, prev_cols])).sum(axis=1)
            logits = -(observed_term + missing_term)
            
            # Numerically stable softmax
            logits_max = logits.max()
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum()
        
        # Sample missing entries
        n_missing = np.random.binomial(rows, p_m[i])
        if n_missing > 0:
            missing_rows = np.random.choice(rows, size=n_missing, replace=False, p=probs)
            M[missing_rows, i] = 0
    
    return M

def uniform_sampler(low, high, rows, cols):
    """Sample uniform random variables.

    :param low: the low limit
    :param high: the high limit
    :param rows: the number of rows
    :param cols: the number of columns

    :return:
    - uniform_random_matrix: a uniform random matrix
    """

    uniform_random_matrix = np.random.uniform(low, high, size=(rows, cols))
    return uniform_random_matrix


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
