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

"""The main interface to interact with S-GAIN.

Subroutines:
(1) settings_subroutine: show settings, load settings, store current settings or delete settings
(2) analysis_subroutine: compile the metrics and plot the graphs

Todo other functions:
(1) print_settings: print the specified settings to the console
(2) print_experiments: print the experiments the current settings would run to the console
"""

import argparse

from os import listdir, remove
from os.path import isfile

import config

from utils.analysis import extract_log_info, compile_metrics, plot_rmse, plot_success_rate, plot_imputation_time
from utils.data_loader import data_loader
from utils.load_store import parse_files, system_information


# -- Subroutines ------------------------------------------------------------------------------------------------------

def settings_subroutine(operation, filename):
    """Show settings, load settings, store current settings or delete settings.

    Todo set settings through terminal

    :param operation: show, load, store or delete
    :param filename: the filename of the settings
    """

    # Show settings
    file = f'settings/{filename}.py' if filename else 'config.py'
    if operation in ['show', 'list', 'ls']:
        if isfile(file):
            with open(file, 'r') as f:
                print(f.read().split('---\n')[-1])
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
        settings.print_help()


def prepare_datasets(dataset, miss_rate, miss_modality, seed):
    """Prepare the datasets.

    :param dataset: the dataset to prepare
    :param miss_rate: the miss rate
    :param miss_modality: the miss modality
    :param seed: the seed
    """

    if config.verbose: print("Preparing dataset(s)...")

    # Set parameters
    if not dataset: dataset = config.dataset
    if not miss_rate: miss_rate = config.miss_rate
    if not miss_modality: miss_modality = config.miss_modality

    # Set seed
    if seed is None:  # No seed specified, use config
        seed = config.seed
    elif not seed:  # Random seed
        seed = None

    # Turn parameters into lists
    if type(dataset) is not list: dataset = [dataset]
    if type(miss_rate) is not list: miss_rate = [miss_rate]
    if type(miss_modality) is not list: miss_modality = [miss_modality]
    if type(seed) is not list: seed = [seed]

    [   # Call data_loader
        data_loader(d, mr, mm, s, store_prepared_dataset=True, verbose=config.verbose)
        for d in dataset
        for mr in miss_rate
        for mm in miss_modality
        for s in seed
    ]


def run_experiments():
    pass


def analyze():
    """Compile the metrics and plot the graphs."""

    # Get parameters
    output_folder = args.input_folder if args.input_folder else config.output_folder
    analysis_folder = args.output_folder if args.output_folder else config.analysis_folder

    # Get all log files
    if config.verbose: print('Loading experiments...')
    logs = [file for file in listdir(output_folder) if file.endswith('log.json')]
    experiments = parse_files(files=logs)
    sys_info = system_information(print_ready=True) if not config.no_system_information else None

    # Drop failed experiments
    logs = [file for file in logs if 'nan' not in file]

    # Get experiments info
    experiments_info = extract_log_info(logs, folder=output_folder)

    # Analyze (non-compiled) experiments
    if config.verbose: print('Analyzing experiments...')
    if config.compile_metrics: compile_metrics(experiments, experiments_info=experiments_info, folder=analysis_folder)
    if config.plot_rmse: plot_rmse(experiments, sys_info=sys_info, folder=analysis_folder)
    if config.plot_success_rate: plot_success_rate(experiments, sys_info=sys_info, folder=analysis_folder)

    # Analyze experiments information
    if config.plot_imputation_time: plot_imputation_time(experiments_info, sys_info=sys_info, folder=analysis_folder)

    # Todo the rest of the analysis

    if config.verbose: print(f'Finished.')


# -- Other functions --------------------------------------------------------------------------------------------------

def print_settings(filename):
    """Print the specified settings to the console.

    :param filename: the name of the settings file
    """
    pass


def print_experiments():  # Todo show experiments
    """Print the experiments the current settings would run to the console."""

    print('print_experiments is not implemented yet')  # Todo show to run and completed


# -- Main -------------------------------------------------------------------------------------------------------------

def main(args):
    """The main interface to interact with S-GAIN."""

    # Select subroutine
    subroutine = args.subroutine
    if subroutine == 'settings':
        settings_subroutine(args.operation, args.filename)
    elif subroutine == 'prepare':
        prepare_datasets(args.dataset, args.miss_rate, args.miss_modality, args.seed)
    elif subroutine == 'run':
        # run_experiments()
        pass
    elif subroutine == 'analyze':
        analyze()
    else:
        parser.print_help()


if __name__ == '__main__':
    # Parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='subroutine',
        title='subroutine',
        help='change settings, run or analyze experiments'
    )

    # Settings
    settings = subparsers.add_parser(
        'settings',
        help='show, load, store or delete settings'
    )
    settings.add_argument(
        'operation',
        choices=['show', 'list', 'ls', 'load', 'l', 'store', 'save', 's', 'delete', 'del', 'remove', 'rm'],
        help='show settings, load settings (default, IDEAL2025, ...), store current settings or delete settings',
        type=str
    )
    settings.add_argument(
        'filename',
        help='the name of the settings file to show, load, store or delete (shows the current settings if left blank)',
        nargs='?',
        type=str
    )

    # Todo prepare datasets: only call the dataloader
    prepare = subparsers.add_parser(
        'prepare',
        help='prepare the datasets specified in config.py'
    )
    prepare.add_argument(
        'dataset',
        help='the dataset to prepare (overwrites config)',
        nargs='?',
        type=str
    )
    prepare.add_argument(
        '--datasets', '-ds',
        help='the datasets to prepare (overwrites config)',
        nargs='+',
        type=str
    )
    prepare.add_argument(
        'miss_rate',
        help='the miss rate to prepare the datasets with (overwrites config)',
        nargs='?',
        type=float
    )
    prepare.add_argument(
        '--miss_rate', '--miss_rates', '-mr',
        help='the miss rate to prepare the datasets with (overwrites config)',
        nargs='+',
        type=float
    )
    prepare.add_argument(
        'miss_modality',
        help='the miss modality to prepare the datasets with (overwrites config)',
        choices=['MCAR', 'MAR', 'MNAR', 'AI_upscaler', 'square'],
        nargs='?',
        type=str
    )
    prepare.add_argument(
        '--miss_modality', '--miss_modalities', '-mm',
        help='the miss modality to prepare the datasets with (overwrites config)',
        choices=['MCAR', 'MAR', 'MNAR', 'AI_upscaler', 'square'],
        nargs='+',
        type=str
    )
    prepare.add_argument(
        'seed',
        help='the seeds to prepare the datasets with (overwrites config)',
        nargs='?',
        type=int
    )
    prepare.add_argument(
        '--seed', '--seeds', '-s',
        help='the seeds to prepare the datasets with (overwrites config). random seed if left blank',
        default=[],
        nargs='*',
        type=int
    )

    # Run experiments
    run = subparsers.add_parser(
        'run',
        help='run the experiments specified in config.py'
    )

    # Analysis
    analysis = subparsers.add_parser(
        'analyze',
        help='analyze the completed experiments'
    )
    analysis.add_argument(
        'input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-in', '--input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        type=str
    )
    analysis.add_argument(
        'output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-out', '--output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        type=str
    )

    # Call main
    args = parser.parse_args()
    main(args)
