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


"""Analyze the experiments:

(1) extract_log_info: extract information from the experiment logs
(2) analyze: compile the metrics and plot the graphs
"""

import json

from os import listdir

from config import *

from utils.load_store import parse_files, system_information, parse_experiment
from utils.analysis import compile_metrics, plot_rmse, plot_success_rate, plot_imputation_time


def extract_log_info(logs, input_folder='output'):
    """Extract information from the experiment logs.

    :param logs: a list of logs
    :param input_folder: the folder containing the experiment logs

    :return: a dictionary with the experiment log information
    """

    exps = {}
    for log in logs:
        # Parse the experiment
        d, mr, mm, s, bs, hr, a, i, gs, gm, ds, dm, _, _, _ = parse_experiment(log, file=True)
        experiment = (d, mr, mm, s, bs, hr, a, i, gs, gm, ds, dm)

        # Read the log
        with open(f'{input_folder}/{log}', 'r') as f:
            data = json.load(f)

        # Get imputation time
        it = data['imputation_time']
        it_total = it['total']
        it_preparation = it['preparation']
        it_s_gain = it['s_gain']
        it_finalization = it['finalization']

        # Add experiment to dictionary
        if experiment not in exps:
            exps.update({
                experiment: {
                    'imputation_time': {
                        'total': [it_total],
                        'preparation': [it_preparation],
                        's_gain': [it_s_gain],
                        'finalization': [it_finalization]
                    }
                }
            })
        else:  # Experiment already in dictionary (append)
            exps[experiment]['imputation_time']['total'].append(it_total)
            exps[experiment]['imputation_time']['preparation'].append(it_preparation)
            exps[experiment]['imputation_time']['s_gain'].append(it_s_gain)
            exps[experiment]['imputation_time']['finalization'].append(it_finalization)

    return exps


def analyze():
    """Compile the metrics and plot the graphs."""

    # Get all log files
    if verbose: print('Loading experiments...')
    logs = [file for file in listdir(output_folder) if file.endswith('log.json')]
    experiments = parse_files(logs)
    sys_info = system_information(print_ready=True) if not no_system_information else None

    # Drop failed experiments
    logs = [file for file in logs if 'nan' not in file]

    # Get experiments info
    experiments_info = extract_log_info(logs, input_folder=output_folder)

    # Analyze (non-compiled) experiments
    if verbose: print('Analyzing experiments...')
    if compile_metrics: compile_metrics(experiments, experiments_info=experiments_info, folder=analysis_folder,
                                        verbose=verbose)
    if plot_rmse: plot_rmse(experiments, sys_info=sys_info, folder=analysis_folder, verbose=verbose)
    if plot_success_rate: plot_success_rate(experiments, sys_info=sys_info,folder=analysis_folder, verbose=verbose)

    # Analyze experiments information
    if plot_imputation_time: plot_imputation_time(experiments_info, sys_info=sys_info, folder=analysis_folder,
                                                  verbose=verbose)

    # Todo the rest of the analysis

    if verbose: print(f'Finished.')