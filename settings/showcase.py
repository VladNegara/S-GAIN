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

These are showcase settings. These are intended to complete quickly in order to give a demonstration of the system.

* loop_until_complete only works when: retry_failed_experiments = True and ignore_existing_files = False
"""

# -- Settings ---------------------------------------------------------------------------------------------------------

# Data preparation settings
dataset = ['letter']
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
generator_initialization = ['dense', 'random', 'ER', 'ERRW']
generator_sparsity = [0, 0.6, 0.8, 0.99]
generator_pruner = [None]
generator_prune_rate = [0]
generator_prune_period = [0]
generator_regrower = [None]
generator_regrow_rate = [0]
generator_regrow_period = [0]
generator_enable_clipping = False
generator_strategy = [None]
generator_use_strategy = False

# Discriminator settings
discriminator_initialization = ['dense', 'random', 'ERRW']
discriminator_sparsity = [0, 0.2, 0.4, 0.6]
discriminator_pruner = [None]
discriminator_prune_rate = [0]
discriminator_prune_period = [0]
discriminator_regrower = [None]  # Todo list options
discriminator_regrow_rate = [0]
discriminator_regrow_period = [0]
discriminator_enable_clipping = False
discriminator_strategy = [None]
discriminator_use_strategy = False

# Output settings
output_folder = 'output_showcase'
no_imputation = False
no_log = False
no_graphs = False
no_model = False

# Monitor settings
enable_rmse_monitor = True
enable_imputation_time_monitor = True
enable_memory_usage_monitor = False
enable_energy_consumption_monitor = False
enable_sparsity_monitor = True
enable_FLOPs_monitor = False
enable_loss_monitor = True

# Run settings
n_runs = 3
retry_failed_experiments = True
max_failed_experiments = 9
ignore_existing_files = False

# Analysis settings
analysis_folder = 'analysis_showcase'
perform_analysis = True
compile_metrics = True
plot_rmse = True
plot_success_rate = True
plot_imputation_time = True
plot_memory_usage = True
plot_energy_consumption = True

# Inclusions (modify the settings and run again)
inclusions = []

# Exclusions (overwrites inclusions)
exclusions = [{
    'generator_initialization': 'ER',
    'generator_sparsity': 0.6
}]

# Options
verbose = True
no_system_information = False
auto_shutdown = False
