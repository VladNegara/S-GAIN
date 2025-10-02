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

(1) extract_log_info: extract information from the experiment logs.
"""

import argparse
import json

from os import listdir

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
        f = open(f'{input_folder}/{log}', 'r')
        data = json.load(f)
        f.close()

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


def main(args):
    """Compile the metrics and plot the graphs.

    :param args:
    - all: plot all graphs
    - rmse: plot the RMSE
    - success_rate: plot the success rate
    - save: save the plots
    - input: the folder the experiments were saved to
    - output: the folder to save the analysis to
    - no_system_information: don't log system information
    - verbose: enable verbose output to console
    """

    # Get the parameters
    plot_all = args.all
    rmse = args.rmse
    success_rate = args.success_rate
    imputation_time = args.imputation_time
    save = args.save
    input_folder = args.input
    output_folder = args.output
    no_system_information = args.no_system_information
    verbose = args.verbose

    # Get all log files
    if verbose: print('Loading experiments...')
    logs = [file for file in listdir(input_folder) if file.endswith('log.json')]
    experiments = parse_files(logs)
    sys_info = system_information(print_ready=True) if not no_system_information else None

    # Drop failed experiments
    logs = [file for file in logs if 'nan' not in file]

    # Get experiments info
    experiments_info = extract_log_info(logs, input_folder=input_folder)

    # Analyze (non-compiled) experiments
    if verbose: print('Analyzing experiments...')
    compile_metrics(experiments, experiments_info=experiments_info, save=save, folder=output_folder, verbose=verbose)

    if verbose: print('Plotting RMSE graphs...')
    if plot_all or rmse: plot_rmse(experiments, sys_info=sys_info, save=save, folder=output_folder, verbose=verbose)

    if verbose: print('Plotting success rate graphs...')
    if plot_all or success_rate: plot_success_rate(experiments, sys_info=sys_info, save=save, folder=output_folder,
                                                   verbose=verbose)

    # Analyze experiments information
    if verbose: print('Plotting imputation time graphs...')
    if plot_all or imputation_time: plot_imputation_time(experiments_info, sys_info=sys_info, save=save,
                                                         folder=output_folder, verbose=verbose)

    # Todo the rest of the analysis

    if verbose: print(f'Finished.')


if __name__ == '__main__':
    # Inputs for the analysis function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--all',
        help="plot all the graphs",
        action='store_true')
    parser.add_argument(
        '-rmse', '--rmse',
        help="plot the RMSE graphs",
        action='store_true')
    parser.add_argument(
        '-sr', '--success_rate',
        help="plot the success rate graphs",
        action='store_true')
    parser.add_argument(
        '-it', '--imputation_time',
        help="plot the imputation time graphs",
        action='store_true')
    parser.add_argument(
        '-s', '--save',
        help="save the analysis",
        action='store_true')
    parser.add_argument(
        '-in', '--input', '--experiments',
        help='the folder the experiments were saved to (optional) [default: output]',
        default='output',
        type=str)
    parser.add_argument(
        '-out', '--output', '--analysis',
        help='save the analysis to a different folder (optional) [default: analysis]',
        default='analysis',
        type=str)
    parser.add_argument(
        '-nsi', '--no_system_information',
        help="don't log system information",
        action='store_true')
    parser.add_argument(
        '-v', '--verbose',
        help='enable verbose logging',
        action='store_true')
    args = parser.parse_args()

    main(args)
