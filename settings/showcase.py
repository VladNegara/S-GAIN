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

"""Settings file for S-GAIN.

These are showcase settings.
The experiments are intended to complete quickly in order to give a demonstration of the system.

All possible combination of settings are run for every listed item. Nonsense is de facto excluded,
i.e. dense initialization with > 0% sparsity or other initializations with 0% sparsity.

Settings:
(1) Data preparation: every setting related to introducing missing values in the data
(2) S-GAIN: every setting related to the S-GAIN data imputer
(3) Generator: every setting related to the Generator model
(4) Discriminator: every setting related to the Discriminator model
(5) Output: every setting related to the output
(6) Monitor: every setting related to the Monitor
(7) Run: every setting related to running the experiments, incl. data preparation, analysis and auto shutdown
(8) Analysis: every setting related to the analysis
(9) Inclusions: a list of dictionaries of settings to modify in the config and then include in the experiments
(10) Exclusions: a list of dictionaries of settings to exclude from the experiments
(11) Options: other options: verbose, no_system_information, auto_shutdown

* loop_until_complete only works when: retry_failed_experiments = True and ignore_existing_files = False
"""

# -- Settings ---------------------------------------------------------------------------------------------------------

# -- Data preparation settings ----------------------------------------------------------------------------------------

dataset = ['letter']
# The dataset to use.
# Options: ['spam', 'letter', 'health', 'mnist', 'fashion_mnist', 'cifar10']
# Default: ['spam', 'letter']

miss_rate = [0.2]
# The probability of missing elements in the data.
# Float (0, 1)
# Default: [0.2]

miss_modality = ['MCAR']
# The modality of missing data.
# Options: ['MCAR', 'MAR', 'MNAR', 'upscaler', 'square']
# Default: ['MCAR']

seed = [0]
# The seed used to introduce missing elements in the data.
# * Use None for a random seed.
# Int [0, 2^31)
# Default: [0]

prepared_datasets_folder = 'prepared_datasets_showcase'
# The folder to store the prepared datasets in.
# Default: 'datasets/prepared'

store_prepared_dataset = True
# Whether to store the prepared dataset.
# * Useful for time intensive miss modalities, i.e. MAR, or for analysis.
# Default: True


# -- S-GAIN settings --------------------------------------------------------------------------------------------------

version = ['TFv1_FP32']
# The S-GAIN version.
# Options: ['TFv1_FP32', 'TFv1_INT8']
# Default: ['TFv1_FP32']

batch_size = [128]
# The number of samples in the mini-batch.
# Default: [128]

hint_rate = [0.9]
# The hint probability.
# Default: [0.9]

alpha = [100]
# The hyperparameter.
# Default: [100]

iterations = [10000]
# The number of training iterations.
# Default: [10000]

clipping = True
# Enable clipping for D_prob.
# Default: True


# -- Generator settings -----------------------------------------------------------------------------------------------

generator_initialization = ['dense', 'random', 'ERNR']
# the initialization strategy of the generator.
# Options: ['dense', 'random', 'ER', 'ERNR'] Todo add what your students added
# Default: ['dense']

generator_sparsity = [0, 0.5, 0.99]
# The probability of sparsity in the generator.
# Float [0, 1)
# Default: [0]

generator_pruner = [None]
# The pruning strategy of the generator.
# Options: ['random', 'magnitude', None]
# Default: [None]

generator_prune_rate = [0]
# The probability of pruning a non-zero weight in the generator,
# based on the number of non-zero weights at initialization.
# Float [0, 1)
# Default: [0]

generator_prune_period = [0]
# The number of iterations before pruning the generator,
# after initialization or previous pruning.
# Default: [0]

generator_regrower = [None]
# The regrowing strategy of the generator.
# Options: ['random', None]
# Default: [None]

generator_regrow_rate = [0]
# The probability of regrowing a zero weight in the generator,
# based on the number of non-zero weights at initialization.
# Float [0, 1)
# Default: [0]

generator_regrow_period = [0]
# The number of iterations before regrowing the generator,
# after initialization or previous regrowing.
# Float [0, 1)
# Default: [0]

generator_strategy = [None]
# Options:  [None] Todo add what your students added
# The training strategy of the generator.
# Default: [None]

generator_use_strategy = False
# Use a complete training strategy for the generator, instead of
# separate initialization, pruning and regrowing strategies.
# Default: False


# -- Discriminator settings -------------------------------------------------------------------------------------------

discriminator_initialization = ['dense', 'random', 'ERNR']
# The initialization strategy of the discriminator.
# Options: ['dense', 'random', 'ER', 'ERNR'] Todo add what your students added
# Default: ['dense']

discriminator_sparsity = [0, 0.2, 0.5]
# The probability of sparsity in the discriminator.
# Float [0, 1)
# Default: [0]

discriminator_pruner = [None]
# The pruning strategy of the discriminator.
# Options: ['random', 'magnitude', None]
# Default: [None]

discriminator_prune_rate = [0]
# The probability of pruning a non-zero weight in the discriminator,
# based on the number of non-zero weights at initialization.
# Float [0, 1)
# Default: [0]

discriminator_prune_period = [0]
# The number of iterations before pruning the discriminator,
# after initialization or previous pruning.
# Float [0, 1)
# Default: [0]

discriminator_regrower = [None]
# The regrowing strategy of the discriminator.
# Options: ['random', None]
# Default: [None]

discriminator_regrow_rate = [0]
# The probability of regrowing a zero weight in the discriminator,
# based on the number of non-zero weights at initialization.
# Float [0, 1)
# Default: [0]

discriminator_regrow_period = [0]
# The number of iterations before regrowing the discriminator,
# after initialization or previous regrowing.
# Float [0, 1)
# Default: [0]

discriminator_strategy = [None]
# The training strategy of the discriminator.
# Options: [None] Todo add what your students added
# Default: [None]

discriminator_use_strategy = False
# Use a complete training strategy for the discriminator, instead of
# separate initialization, pruning and regrowing strategies.
# Default: False


# -- Output settings --------------------------------------------------------------------------------------------------

output_folder = 'output_showcase'
# The folder to save experiments to.
# Default: 'output'

no_imputation = False
# Don't save the imputed data.
# Default: False

no_log = False
# Turn off the logging of metrics.
# * Disables graphs.
# Default: False (also disables graphs)

no_graphs = False
# Don't plot the graphs after training.
# Default: False

no_model = False
# Don't save the trained model.
# Default: False


# -- Monitor settings -------------------------------------------------------------------------------------------------

enable_rmse_monitor = True
# Enable monitoring of the RMSE.
# Default: True

enable_imputation_time_monitor = True
# Enable monitoring of the imputation time.
# Default: True

enable_memory_usage_monitor = False
# Enable monitoring of the memory usage. Todo implement
# Default: True

enable_energy_consumption_monitor = False
# Enable monitoring of the energy consumption. Todo implement
# Default: True

enable_sparsity_monitor = True
# Enable monitoring of the sparsity of both models.
# Default: True

enable_FLOPs_monitor = False
# Enable monitoring of the FLOPs of both models.
# * Takes significantly more time.
# Default: False

enable_loss_monitor = True
# Enable monitoring of the losses (cross entropy and MSE).
# Default: True


# -- Run settings -----------------------------------------------------------------------------------------------------

n_runs = 3
# The number of times each experiment should be performed.
# Default: 10

retry_failed_experiments = True
# Retry a failed experiment until it successfully completes n_runs times
# or reaches max_failed_experiments.
# Default: True

max_failed_experiments = 9
# The maximum number of times the experiment can fail.
# * Used to prevent infinite loops.
# Default: 40 (success_rate < 20%)

ignore_existing_files = False
# Ignore the existing files in the output folder.
# * Disables retry_failed_experiments, a random seed will usually ignore
#   existing files.
# Default: False


# -- Analysis settings ------------------------------------------------------------------------------------------------

analysis_folder = 'analysis_showcase'
# The folder to save the analysis to.
# Default: 'analysis'

perform_analysis = False
# Automatically analyze the experiments after completion.
# Default: True

compile_metrics = True
# Compile the metrics.
# Default: True

plot_rmse = True
# Plot the RMSE graphs.
# Default: True

plot_success_rate = True
# Plot the success rate graphs.
# Default: True

plot_imputation_time = True
# Plot the imputation time graphs.
# Default: True

plot_memory_usage = False
# Plot the memory usage graphs. Todo implement
# Default: True

plot_energy_consumption = False
# Plot the energy consumption graphs. Todo implement
# Default: True


# -- Inclusions -------------------------------------------------------------------------------------------------------

inclusions = []
# An inclusion is a dictionary of settings. It overwrites the base
# config and adds the newly specified experiments. The config reloads
# before each inclusion and previously made changes don't carry over.
# This ensures each inclusion is independent of any previous inclusion.
#
# Example: [{
#     'n_runs': 1,
#     'enable_FLOPs_monitor': True,
#     'output_folder': 'output_FLOPs',
#     'perform_analysis': False
# }]


# -- Exclusions -------------------------------------------------------------------------------------------------------

exclusions = []
# An exclusion is a dictionary of settings. It removes experiments
# with this combination of settings. It overwrites the inclusions.
# Each exclusion is independent of any previous exclusion.
#
# Example: [{
#     'dataset': ['health', 'spam', 'letter'],
#     'miss_modality': ['upscaler', 'square']
# }, {
#     'dataset': ['fashion_mnist'],
#     'miss_modality': ['MAR', 'MNAR']
# }]


# -- Options ----------------------------------------------------------------------------------------------------------

verbose = True
# Enable verbose output to the console.
# Default: True

no_system_information = False
# Don't log system information.
# Default: False

auto_shutdown = False
# Automatically shutdown the computer after running the experiments
# and performing the analyses.
# Default: False
