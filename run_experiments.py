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

"""Run all the specified experiments consecutively:

(1) update_experiments: return the experiments to run
"""

import os
import subprocess

import pandas as pd

from time import time
from datetime import timedelta

from utils.load_store import get_experiments, read_bin

"""
Settings used for B.P. van Oers, I. Baysal Erez, M. van Keulen, "Sparse GAIN: Imputation Methods to Handle Missing
Values with Sparse Initialization", IDEAL conference, 2025.
"""
# datasets = ['spam', 'letter', 'health', 'fashion_mnist']
# miss_rates = [0.2]
# miss_modalities = ['MCAR']
# seeds = [0]
# batch_sizes = [128]
# hint_rates = [0.9]
# alphas = [100]
# iterations_s = [10000]
# generator_sparsities = [0, 0.6, 0.8, 0.9, 0.95, 0.99]
# generator_modalities = ['dense', 'random', 'ER', 'ERRW']
# discriminator_sparsities = [0]
# discriminator_modalities = ['dense']
# n_runs = 10

# Settings
datasets = ['health', 'fashion_mnist']  # ['spam', 'letter', 'health', 'mnist', 'fashion_mnist', 'cifar10']
miss_rates = [0.2]
miss_modalities = ['MCAR']  # ['MCAR', 'MAR', 'MNAR']
seeds = [0]
batch_sizes = [128]
hint_rates = [0.9]
alphas = [100]
iterations_s = [10000]
generator_sparsities = [0, 0.6, 0.8, 0.9, 0.95, 0.99]
generator_modalities = ['dense', 'random']  # ['dense', 'random', 'ER', 'ERRW']
discriminator_sparsities = [0, 0.2, 0.4, 0.6, 0.8]
discriminator_modalities = ['dense', 'random']  # ['dense', 'random', 'ER', 'ERRW']
output_folder = 'output'  # Default: 'output'
n_runs = 5
ignore_existing_files = False  # Default: False
retry_failed_experiments = True  # Default: True
loop_until_complete = True  # Only works when retry_failed_experiments = True and ignore_existing_files = False
verbose = True  # Default: True
no_log = False  # Default: False
no_graph = False  # Default: False
no_model = False  # Default: False
no_save = False  # Default: False
no_system_information = False  # Default: False
analyze = True  # Automatically analyze the experiments after completion
analysis_folder = 'analysis'  # Default: 'analysis'
auto_shutdown = True  # Default: False


def update_experiments():
    """Return the experiments to run.

    :return: a list of experiments formatted as executable commands
    """

    return get_experiments(
        datasets, miss_rates=miss_rates, miss_modalities=miss_modalities, seeds=seeds, batch_sizes=batch_sizes,
        hint_rates=hint_rates, alphas=alphas, iterations_s=iterations_s, generator_sparsities=generator_sparsities,
        generator_modalities=generator_modalities, discriminator_sparsities=discriminator_sparsities,
        discriminator_modalities=discriminator_modalities, folder=output_folder, n_runs=n_runs,
        ignore_existing_files=ignore_existing_files, retry_failed_experiments=retry_failed_experiments,
        verbose=verbose, no_log=True, no_graph=True, no_model=no_model, no_save=no_save,
        no_system_information=no_system_information, get_commands=True
    )


if __name__ == '__main__':
    # Get the experiments
    experiments = update_experiments()

    # Report initial progress
    i = 0
    total = len(experiments)
    start_time = time()
    print(f'\nProgress: 0% completed (0/{total}) 0:00:00\n')

    # Run all experiments
    while len(experiments) > 0:
        for experiment in experiments:
            # Run experiment
            print(experiment)
            os.system(experiment)

            # Compile logs and plot graphs
            command = f'python log_and_graphs.py' \
                      f'{" --no_graph" if no_graph else ""}' \
                      f'{" --no_system_information" if no_system_information else ""}' \
                      f'{" --verbose" if verbose else ""}'

            if verbose: print(f'\n{command}')
            if not no_log: os.system(command)

            # Increase counter
            rmse = read_bin('temp/exp_bins/rmse.bin')[-1]
            if ignore_existing_files or not retry_failed_experiments or pd.notna(rmse): i += 1

            # Report progress
            elapsed_time = int(time() - start_time)
            time_to_completion = int(elapsed_time / i * (total - i)) if i > 0 else 0
            estimated = f' (estimated left: {timedelta(seconds=time_to_completion)})' if time_to_completion > 0 else ''
            print(f'\nProgress: {int(i / total * 100)}% completed ({i}/{total}) {timedelta(seconds=elapsed_time)}'
                  f'{estimated}\n')

        # Update the experiments
        if loop_until_complete and not ignore_existing_files and retry_failed_experiments:
            experiments = update_experiments()
        else:
            break

    if analyze: os.system(f'python analyze.py --all -in {output_folder} -out {analysis_folder} --save --verbose')

    if auto_shutdown and total > 0:
        if verbose: print(f'Processes finished.\nShutting down...')
        subprocess.run(['shutdown', '-s'])
