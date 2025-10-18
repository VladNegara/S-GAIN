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
(2) run_experiments: run all the experiments
"""

import os
import subprocess

import pandas as pd

from time import perf_counter
from datetime import timedelta

from analyze import analyze
from config import *

from utils.load_store import get_experiments, read_bin


def update_experiments():
    """Return the experiments to run.

    Todo inclusions and exclusions

    :return: a list of experiments formatted as executable commands
    """

    return get_experiments(
        dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity,
        generator_initialization, discriminator_sparsity, discriminator_initialization, folder=output_folder,
        n_runs=n_runs, ignore_existing_files=ignore_existing_files, retry_failed_experiments=retry_failed_experiments,
        verbose=verbose, no_log=True, no_graph=True, no_model=no_model, no_save=no_imputation,
        no_system_information=no_system_information, get_commands=True
    )


def run_experiments():
    """Run all the experiments."""

    # Get the experiment (and log_and_graphs) commands
    experiment_commands = update_experiments()
    log_and_graphs_command = f'python log_and_graphs.py' \
                             f'{" --no_graph" if no_graphs else ""}' \
                             f'{" --no_system_information" if no_system_information else ""}' \
                             f'{" --verbose" if verbose else ""}'

    # Report initial progress
    i = 0
    total = len(experiment_commands)
    start_time = perf_counter()
    print(f'\nProgress: 0% completed (0/{total}) 0:00:00\n')

    # Run all experiments
    while len(experiment_commands) > 0:
        for experiment_command in experiment_commands:
            # Run experiment
            print(experiment_command)
            os.system(experiment_command)

            # Compile logs and plot graphs
            if verbose: print(f'\n{log_and_graphs_command}')
            if not no_log: os.system(log_and_graphs_command)

            # Increase counter
            rmse = read_bin('temp/exp_bins/rmse.bin')[-1]
            if ignore_existing_files or not retry_failed_experiments or pd.notna(rmse): i += 1

            # Report progress
            elapsed_time = int(perf_counter() - start_time)
            time_to_completion = int(elapsed_time / i * (total - i)) if i > 0 else 0
            estimated = f' (estimated left: {timedelta(seconds=time_to_completion)})' if time_to_completion > 0 else ''
            print(f'\nProgress: {int(i / total * 100)}% completed ({i}/{total}) {timedelta(seconds=elapsed_time)}'
                  f'{estimated}\n')

        # Update the experiments
        if loop_until_complete and not ignore_existing_files and retry_failed_experiments:
            experiment_commands = update_experiments()
        else:
            break

    # Analyze experiments
    if perform_analysis: analyze()

    # Auto shutdown
    if auto_shutdown and total > 0:
        if verbose: print(f'Processes finished.\nShutting down...')
        subprocess.run(['shutdown', '-s'])
