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

These are the settings used for B.P. van Oers, I. Baysal Erez, M. van Keulen, "Sparse GAIN: Imputation Methods to Handle
Missing Values with Sparse Initialization", IDEAL conference, 2025.

* loop_until_complete only works when: retry_failed_experiments = True and ignore_existing_files = False
"""

# Data preparation settings
dataset = ['spam', 'letter', 'health', 'fashion_mnist']
miss_rate = [0.2]
miss_modality = ['MCAR']
seed = [0]
store_prepared_dataset = False

# S-GAIN settings
version = ['TFv1_FP32']
batch_size = [128]
hint_rate = [0.9]
alpha = [100]
iterations = [10000]

# Generator settings
generator_sparsity = [0, 0.6, 0.8, 0.9, 0.95, 0.99]
generator_initialization = ['dense', 'random', 'ER', 'ERRW']
generator_regrower = [None]
generator_regrow_rate = [None]
generator_regrow_period = [None]
generator_pruner = [None]
generator_prune_rate = [None]
generator_prune_period = [None]
generator_enable_clipping = [False]
generator_use_strategy = [False]

# Discriminator settings
discriminator_sparsity = [0]
discriminator_initialization = ['dense']
discriminator_regrower = [None]
discriminator_regrow_rate = [None]
discriminator_regrow_period = [None]
discriminator_pruner = [None]
discriminator_prune_rate = [None]
discriminator_prune_period = [None]
discriminator_enable_clipping = [False]
discriminator_use_strategy = [False]

# Monitor settings
enable_rmse_monitor = True
enable_imputation_time_monitor = False
enable_memory_usage_monitor = False
enable_energy_consumption_monitor = False
enable_sparsity_monitor = False
enable_FLOPs_monitor = False
enable_loss_monitor = False

# Output settings
output_folder = 'output'
analysis_folder = 'analysis'
verbose = True
no_log = False
no_graph = False
no_model = False
no_save = False
no_system_information = True

# Run settings
n_runs = 10
ignore_existing_files = False
retry_failed_experiments = True
loop_until_complete = True
perform_analysis = False
auto_shutdown = False

# Inclusions
inclusions = [{
    'n_runs': 1,
    'enable_FLOPs_monitor': True,
    'output_folder': 'output_FLOPs',
    'dataset': ['health', 'fashion_mnist'],
}]

# Exclusions (overwrites inclusions)
exclusions = []
