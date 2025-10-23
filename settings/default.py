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

These are the default settings.

* loop_until_complete only works when: retry_failed_experiments = True and ignore_existing_files = False
"""

# -- Settings ---------------------------------------------------------------------------------------------------------

# Data preparation settings
dataset = ['spam', 'letter']   # Options: ['spam', 'letter', 'health', 'mnist', 'fashion_mnist', 'cifar10']
miss_rate = [0.2]
miss_modality = ['MCAR']       # Options: ['MCAR', 'MAR', 'MNAR']
seed = [0]                     # Use None for random seed
store_prepared_dataset = True  # Default: True

# S-GAIN settings
version = ['TFv1_FP32']  # Options: ['TFv1_FP32', 'TFv1_INT8']
batch_size = [128]       # Default: [128]
hint_rate = [0.9]        # Default: [0.9]
alpha = [100]            # Default: [100]
iterations = [10000]     # Default: [10000]

# Generator settings
generator_initialization = ['dense']  # Options: ['dense', 'random', 'ER', 'ERRW']
generator_sparsity = [0]
generator_pruner = [None]  # Todo list options
generator_prune_rate = [0]
generator_prune_period = [0]
generator_regrower = [None]  # Todo list options
generator_regrow_rate = [0]
generator_regrow_period = [0]
generator_enable_clipping = False
generator_strategy = [None]
generator_use_strategy = False

# Discriminator settings
discriminator_initialization = ['dense']  # Options: ['dense', 'random', 'ER', 'ERRW']
discriminator_sparsity = [0]
discriminator_pruner = [None]  # Todo list options
discriminator_prune_rate = [0]
discriminator_prune_period = [0]
discriminator_regrower = [None]  # Todo list options
discriminator_regrow_rate = [0]
discriminator_regrow_period = [0]
discriminator_enable_clipping = False
discriminator_strategy = [None]
discriminator_use_strategy = False

# Output settings
output_folder = 'output'  # Default: 'output'
no_imputation = False     # Default: False
no_log = False            # Default: False
no_graphs = False         # Default: False
no_model = False          # Default: False

# Monitor settings
enable_rmse_monitor = True                 # Default: True
enable_imputation_time_monitor = True      # Default: True
enable_memory_usage_monitor = False        # Default: True
enable_energy_consumption_monitor = False  # Default: True
enable_sparsity_monitor = True             # Default: True
enable_FLOPs_monitor = False               # Default: False (takes significantly more time)
enable_loss_monitor = True                 # Default: True

# Run settings
n_runs = 10                      # Default: 10
retry_failed_experiments = True  # Default: True
max_failed_experiments = 40      # Default: 40 (success_rate < 20%)
ignore_existing_files = False    # Default: False

# Analysis settings
analysis_folder = 'analysis'     # Default: 'analysis'
perform_analysis = True          # Default: True
compile_metrics = True           # Default: True
plot_rmse = True                 # Default: True
plot_success_rate = True         # Default: True
plot_imputation_time = True      # Default: True
plot_memory_usage = True         # Default: True
plot_energy_consumption = True   # Default: True

# Inclusions (modify the settings and run again)
inclusions = [{
    'n_runs': 1,
    'enable_FLOPs_monitor': True,
    'output_folder': 'output_FLOPs',
    'perform_analysis': False
}]

# Exclusions (overwrites inclusions)
exclusions = []

# Options
verbose = True                 # Default: True
no_system_information = False  # Default: False
auto_shutdown = False          # Default: False
