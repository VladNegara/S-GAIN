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

"""Contains standardization functions for S-GAIN.

Helper functions:
(1) standardize: standardize value(s) based on a key

Standardizers:
(2) standardize_dataset: standardize the dataset
(3) standardize_miss_modality: standardize the miss modality
(4) standardize_version: standardize the version
(5) standardize_init: standardize the sparsity and the initialization
(6) standardize_pruner: standardize the pruner
(7) standardize_regrower: standardize the regrower
(8) standardize_strategy: standardize the strategy
"""


# -- Helper functions -------------------------------------------------------------------------------------------------

def standardize(key, value):
    """Standardize value(s) based on a key.

    :param key: The key to standardize on
    :param value: The value(s) to standardize

    :return: The standardized value(s)
    """

    if key == 'dataset': return standardize_dataset(value)
    if key == 'miss_modality': return standardize_miss_modality(value)
    if key == 'version': return standardize_version(value)
    if 'initialization' in key:
        if value.lower == 'dense': return 'dense'
        return standardize_init(value, 1)[0]
    if 'pruner' in key: return standardize_pruner(value)
    if 'regrower' in key: return standardize_regrower(value)
    if 'strategy' in key: return standardize_strategy(value)
    return value


# -- Standardizers ----------------------------------------------------------------------------------------------------

def standardize_dataset(dataset):
    """Standardize the dataset.

    :param dataset: the dataset

    :return: the standardized dataset
    """

    if not dataset: return None
    if dataset.lower() in ['spam', 'letter', 'health']: return dataset.lower()
    if dataset.lower() in ['mnist', 'cifar10']: return dataset.upper()
    if dataset.lower() == 'fashion_mnist': return 'Fashion_MNIST'
    return dataset


def standardize_miss_modality(miss_modality):
    """Standardize the miss modality.

    :param miss_modality: the miss modality

    :return: the standardized miss modality
    """

    if not miss_modality: return None
    if miss_modality.upper() in ['MCAR', 'MAR', 'MNAR']: return miss_modality.upper()
    if miss_modality.lower() == 'ai_upscaler': return 'AI_upscaler'
    if miss_modality.lower() == 'square': return 'square'
    return miss_modality


def standardize_version(version):
    """Standardize the version.

    :param version: the version

    :return: the standardized version
    """

    if not version: return None
    if version.upper() == 'TFV1_FP32': return 'TFv1_FP32'
    if version.upper() == 'TFV2_INT8': return 'TFv2_INT8'
    return version


def standardize_init(init, sparsity):
    """Standardize the initialization and sparsity.

    :param init: the initialization
    :param sparsity: the sparsity

    :return: the standardized initialization and sparsity
    """

    if not init: return None, None
    if sparsity == 0 or init.lower() == 'dense': return init.lower(), 0
    if init.lower() in ['normal_random', 'normal', 'nr', 'n']: return 'NR', sparsity
    if init.lower() in ['uniform_normal', 'uniform', 'ur', 'u']: return 'UR', sparsity
    if init.lower() in ['erdos_renyi', 'er']: return 'ER', sparsity
    if init.lower() in ['erdos_renyi_kernel', 'erk']: return 'ERK', sparsity
    if init.lower() in ['erdos_renyi_normal_random', 'erdos_renyi_normal', 'ernr', 'ern']: return 'ERNR', sparsity
    if init.lower() in ['erdos_renyi_uniform_random', 'erdos_renyi_uniform', 'erur', 'eru']: return 'ERUR', sparsity
    ERKNR = ['erdos_renyi_kernel_normal_random', 'erdos_renyi_kernel_normal', 'erknr', 'erkn']
    if init.lower() in ERKNR: return 'ERKNR', sparsity
    ERKUR = ['erdos_renyi_kernel_uniform_random', 'erdos_renyi_kernel_uniform', 'erkur', 'erku']
    if init.lower() in ERKUR: return 'ERKUR', sparsity
    if init.upper() == 'SNIP': return 'SNIP', sparsity
    if init.upper() == 'GRASP': return 'GraSP', sparsity
    if init.lower() == 'rsensitivity': return 'RSensitivity', sparsity
    return init, sparsity


def standardize_pruner(pruner):
    """Standardize the pruner.

    :param pruner: the pruner

    :return: the standardized pruner
    """

    if not pruner: return None
    if pruner.lower() in ['random', 'magnitude']: return pruner.lower()
    return pruner


def standardize_regrower(regrower):
    """Standardize the regrower.

    :param regrower: the regrower

    :return: the standardized regrower
    """

    if not regrower: return None
    if regrower.lower() == 'random': return 'random'
    return regrower


def standardize_strategy(strategy):
    """Standardize the strategy.

    Todo implement complete training strategies

    :param strategy: the strategy

    :return: the standardized strategy
    """

    return strategy
