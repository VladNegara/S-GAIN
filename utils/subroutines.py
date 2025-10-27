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

"""The subroutines for the S-GAIN testing framework.

Subroutines:
(1) settings_subroutine: show settings, load settings, store current settings or delete settings
(3) run_experiments: run all the experiments
(4) analyze: compile the metrics and plot the graphs
"""

import json
import os
import subprocess
from datetime import timedelta

from os import listdir, remove, makedirs
from os.path import isfile, isdir
from time import perf_counter

import pandas as pd

from utils.analysis import extract_log_info, compile_metrics, plot_rmse, plot_success_rate, plot_imputation_time
from utils.load_store import parse_files, system_information, get_experiments_from_config, get_completed_experiments, \
    read_bin


# -- Subroutines ------------------------------------------------------------------------------------------------------

def settings(parser, operation, filename, information):
    """Show settings, load settings, store current settings or delete settings.

    Todo set settings through terminal

    :param parser: the settings parser
    :param operation: show, load, store or delete
    :param filename: the filename of the settings
    :param information: show additional information about the settings
    """

    # Show settings
    file = f'settings/{filename}.py' if filename else 'config.py'
    if operation in ['show', 'list', 'ls']:
        if isfile(file):
            with open(file, 'r') as f:
                s = f.read().split('---\n', 1)[-1]

            if information:
                print(s)
            else:
                s = s.split('\n')
                for line in s:
                    if line and not line.startswith('#'): print(line)
                    if line.startswith('# --'):
                        line = line.replace('--', '')
                        line = line.replace('# ', '#')
                        line = line.replace(' -', '')
                        print(f'\n{line}')
                print()
        else:
            print(f'{filename} settings not found!')

    # Load, store or delete settings
    elif file != 'config.py':

        # Load settings
        if operation in ['load', 'l']:
            if isfile(file):
                with open(file, 'r') as f:
                    cfg = f.read()
                with open('config.py', 'w') as f:
                    f.write(cfg)
                print(f'{filename} settings loaded.')
                return

        # Store current settings
        elif operation in ['store', 'save', 's']:

            # Check if file exists
            store = True
            if isfile(file):
                if filename == 'default':
                    choice = input('Are you sure you want to overwrite the default settings? [y/n]: ')
                else:
                    choice = input(f'{filename} already exists. Do you want to overwrite these settings? [y/n]: ')

                if choice.lower() not in ['yes', 'y', '1']: store = False

            # Store settings
            if store:
                with open('config.py', 'r') as f:
                    cfg = f.read()
                with open(file, 'w') as f:
                    f.write(cfg)
                print(f'Current settings stored as {filename}.')
            else:
                print(f'Current settings not stored.')

            return

        # Delete settings
        else:  # operation in ['delete', 'del', 'remove', 'rm']
            if isfile(file):

                # Don't delete default settings
                if filename == 'default':
                    print('Cannot delete default settings!')
                else:
                    remove(file)
                    print(f'{filename} settings removed.')

                return

        # Settings not found
        print(f'{filename} settings not found!')

    else:
        parser.print_help()


def run_experiments(config):
    """Run all the experiments.

    :param config: the configuration file
    """

    experiments = get_experiments_from_config(config)

    # Report initial progress
    i = 0
    total = experiments['n_runs'].sum()
    start_time = perf_counter()
    print(f'\nProgress: 0% completed 0:00:00\n')

    # Parameters
    output_folder = None
    analysis_folder = None
    perform_analysis = None
    completed_experiments = None

    # Run experiments and analysis
    if not isdir('temp'): makedirs('temp')
    for _, experiment in experiments.iterrows():

        # Perform analysis and remove completed experiments
        if output_folder and analysis_folder:
            if output_folder != experiment['output_folder']:
                if perform_analysis: analyze(output_folder, analysis_folder)

                # Update parameters
                output_folder = experiment['output_folder']
                analysis_folder = experiment['analysis_folder']
                perform_analysis = experiment['perform_analysis']
                completed_experiments = get_completed_experiments(output_folder)
        else:
            # Update parameters
            output_folder = experiment['output_folder']
            analysis_folder = experiment['analysis_folder']
            perform_analysis = experiment['perform_analysis']
            completed_experiments = get_completed_experiments(output_folder)

        # Run experiments
        if not (experiment['no_imputation'] and experiment['no_log'] and experiment['no_graphs']
                and experiment['no_model']):

            # Calculate how often the experiment should run
            if experiment['ignore_existing_files']:
                n_runs = experiment['n_runs']
                max_failed_experiments = experiment['max_failed_experiments']
            else:
                # Get the experiment in completed experiments
                keys = completed_experiments.columns.values.tolist()[:-2]
                ce = completed_experiments
                for key in keys:
                    value = experiment[key] if experiment[key] is not None else 'None'
                    ce = ce.loc[ce[key] == value]

                if ce.empty:  # experiment not in completed_experiments
                    n_runs = experiment['n_runs']
                    max_failed_experiments = experiment['max_failed_experiments']
                else:
                    decrement = ce.iloc[0]['successes']
                    n_runs = experiment['n_runs'] - decrement
                    max_failed_experiments = experiment['max_failed_experiments'] - ce.iloc[0]['failures']
                    total -= decrement

            # Store the experiment to run as run_config.json
            if n_runs > 0 and max_failed_experiments > 0:
                experiment = experiment.to_dict()
                with open('temp/run_config.json', 'w') as f:
                    f.write(json.dumps(experiment))

            # Run experiment
            while n_runs > 0 and max_failed_experiments > 0:
                os.system(f'python main.py'
                          f'{" --verbose" if config.verbose else ""}'
                          f'{" --no_system_information" if config.no_system_information else ""}')

                # Decrease counter
                rmse = read_bin('temp/exp_bins/rmse.bin')[-1]
                if pd.notna(rmse):
                    n_runs -= 1
                    i += 1
                else:
                    max_failed_experiments -= 1

                # Report progress
                percent_complete = int(i / total * 100) if total > 0 else 100
                elapsed_time = int(perf_counter() - start_time)
                time_to_complete = int(elapsed_time / i * (total - i)) if i > 0 else 0
                estimated = f' (estimated left: {timedelta(seconds=time_to_complete)})' if time_to_complete > 0 else ''
                print(f'\nProgress: {percent_complete}% completed {timedelta(seconds=elapsed_time)}{estimated}\n')

    # Analysis
    if perform_analysis and total > 0: analyze(output_folder, analysis_folder)

    # Auto shutdown
    if config.auto_shutdown and total > 0:
        if config.verbose: print(f'Processes finished.\nShutting down...')
        subprocess.run(['shutdown', '-s'])


def analyze(config, experiments_folder, analysis_folder):
    """Compile the metrics and plot the graphs.

    :param config: the configuration file
    :param experiments_folder: the folder the experiments are saved in
    :param analysis_folder: the folder to save the analysis to
    """

    # Parameters
    if not experiments_folder: experiments_folder = config.output_folder
    if not analysis_folder: analysis_folder = config.analysis_folder

    # Get all log files
    if config.verbose: print('Loading experiments...')
    logs = [file for file in listdir(experiments_folder) if file.endswith('log.json')]
    experiments = parse_files(files=logs)
    sys_info = system_information(print_ready=True) if not config.no_system_information else None

    # Drop failed experiments
    logs = [file for file in logs if 'nan' not in file]

    # Get experiments info
    experiments_info = extract_log_info(logs, folder=experiments_folder)

    # Analyze (non-compiled) experiments
    if config.verbose: print('Analyzing experiments...')
    if config.compile_metrics:
        compile_metrics(experiments, experiments_info, folder=analysis_folder, verbose=config.verbose)
    if config.plot_rmse: plot_rmse(experiments, sys_info=sys_info, folder=analysis_folder, verbose=config.verbose)
    if config.plot_success_rate:
        plot_success_rate(experiments, sys_info=sys_info, folder=analysis_folder, verbose=config.verbose)

    # Analyze experiments information
    if config.plot_imputation_time:
        plot_imputation_time(experiments_info, sys_info=sys_info, folder=analysis_folder, verbose=config.verbose)

    # Todo the rest of the analysis

    if config.verbose: print('Finished.')