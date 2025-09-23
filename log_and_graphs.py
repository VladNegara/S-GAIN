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

"""Compile the logs and plot the graphs."""

import argparse
import json

from utils.graphs2 import plot_graphs
from utils.load_store import parse_experiment, read_bin, system_information


def main(args):
    """Compile and save the logs to a json file and plot the graphs.

    :param args:
    - directory: the directory of the temporary files
    - no_graph: don't plot the graphs (log only)
    - no_system_information: don't log system information
    - verbose: enable verbose output to console
    """

    # Get the parameters
    directory = args.directory
    no_graph = args.no_graph
    no_system_information = args.no_system_information
    verbose = args.verbose

    # Read run data
    f = open('temp/run_data', 'r')
    experiment, filepath_imputed_data, filepath_log, filepath_graphs, _ = f.read().split('\n')
    f.close()

    # Compile and save the logs
    if verbose: print('Saving logs...')
    sys_info = system_information() if not no_system_information else None
    rmse_log, imputation_time_log, memory_usage_log, energy_consumption_log, sparsity_log, sparsity_G_log, \
        sparsity_G_W1_log, sparsity_G_W2_log, sparsity_G_W3_log, sparsity_D_log, sparsity_D_W1_log, sparsity_D_W2_log, \
        sparsity_D_W3_log, flops_log, flops_G_log, flops_D_log, loss_G_log, loss_D_log, loss_MSE_log, exp \
        = save_logs(filepath_log, experiment, directory, sys_info)

    if not no_graph:
        if verbose: print('Plotting graphs...')
        sys_info = system_information(print_ready=True) if not no_system_information else None
        title = filepath_imputed_data.split('/')[-1].replace('.csv', '')
        plot_graphs(filepath_graphs, rmse_log, imputation_time_log, memory_usage_log, energy_consumption_log,
                    [sparsity_log, sparsity_G_log, sparsity_G_W1_log, sparsity_G_W2_log, sparsity_G_W3_log,
                     sparsity_D_log, sparsity_D_W1_log, sparsity_D_W2_log, sparsity_D_W3_log],
                    [flops_log, flops_G_log, flops_D_log], [loss_G_log, loss_D_log, loss_MSE_log],
                    experiment=exp, sys_info=sys_info, title=title)

    if verbose: print('Finished.')


def save_logs(filepath, experiment=None, directory='temp/exp_bins', sys_info=None):
    """Compile and save the logs to a json file.

    :param filepath: the filepath to save the logs to
    :param experiment: the name of the experiment
    :param directory: the directory of the temporary files
    :param sys_info: the system information

    :return:
    - RMSE: the RMSE log
    - imputation_time: the imputation time log
    - memory_usage: the memory usage log
    - energy_consumption: the energy consumption log
    - sparsity: the sparsity log (total)
    - sparsity_G: the sparsity log for the generator
    - sparsity_G_W1: the sparsity log for the first layer of the generator
    - sparsity_G_W2: the sparsity log for the second layer of the generator
    - sparsity_G_W3: the sparsity log for the third layer of the generator
    - sparsity_D: the sparsity log for the discriminator
    - sparsity_D_W1: the sparsity log for the first layer of the discriminator
    - sparsity_D_W2: the sparsity log for the second layer of the discriminator
    - sparsity_D_W3: the sparsity log for the third layer of the discriminator
    - FLOPs: the FLOPs log (total)
    - FLOPs_G: the FLOPs log for the generator
    - FLOPs_D: the FLOPs log for the discriminator
    - loss_G: the loss log for the generator (cross entropy)
    - loss_D: the loss log for the discriminator (cross entropy)
    - loss_MSE: the loss log (MSE)
    - exp: a dictionary containing the experiment
    """

    # Read the log files
    RMSE = read_bin(f'{directory}/RMSE.bin')
    imputation_time = read_bin(f'{directory}/imputation_time.bin')
    memory_usage = [0]  # read_bin(f'{directory}/memory_usage.bin')
    energy_consumption = []  # read_bin(f'{directory}/energy_consumption.bin')
    sparsity_G = read_bin(f'{directory}/sparsity_G.bin')
    sparsity_G_W1 = read_bin(f'{directory}/sparsity_G_W1.bin')
    sparsity_G_W2 = read_bin(f'{directory}/sparsity_G_W2.bin')
    sparsity_G_W3 = read_bin(f'{directory}/sparsity_G_W3.bin')
    sparsity_D = read_bin(f'{directory}/sparsity_D.bin')
    sparsity_D_W1 = read_bin(f'{directory}/sparsity_D_W1.bin')
    sparsity_D_W2 = read_bin(f'{directory}/sparsity_D_W2.bin')
    sparsity_D_W3 = read_bin(f'{directory}/sparsity_D_W3.bin')
    FLOPs_G = []  # read_bin(f'{directory}/flops_G.bin')
    FLOPs_D = []  # read_bin(f'{directory}/flops_D.bin')
    loss_G = read_bin(f'{directory}/loss_G.bin')
    loss_D = read_bin(f'{directory}/loss_D.bin')
    loss_MSE = read_bin(f'{directory}/loss_MSE.bin')

    # Totals
    sparsity = [(sparsity_G[i] + sparsity_D[i]) / 2 for i in range(len(sparsity_G))]
    FLOPs = [FLOPs_G[i] + FLOPs_D[i] for i in range(len(FLOPs_G))]

    logs, exp = {}, None
    if experiment is not None:
        dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity, \
            generator_modality, discriminator_sparsity, discriminator_modality \
            = parse_experiment(experiment, file=False)

        exp = {
            'dataset': dataset,
            'miss_rate': miss_rate,
            'miss_modality': miss_modality,
            'seed': seed,
            'batch_size': batch_size,
            'hint_rate': hint_rate,
            'alpha': alpha,
            'iterations': iterations,
            'generator_sparsity': generator_sparsity,
            'generator_modality': generator_modality,
            'discriminator_sparsity': discriminator_sparsity,
            'discriminator_modality': discriminator_modality
        }
        logs.update({'experiment': exp})

    if sys_info: logs.update({'system_information': sys_info})

    logs.update({
        'rmse': {
            'final': RMSE[-1],
            'log': RMSE,
        },
        'imputation_time': {
            'total': sum(imputation_time),
            'log': imputation_time
        },
        'memory_usage': {
            'maximum': max(memory_usage),
            'average': sum(memory_usage) / len(memory_usage),
            'log': memory_usage
        },
        'energy_consumption': {
            'total': sum(energy_consumption),
            'log': energy_consumption
        },
        'sparsity': {
            'initial': sparsity[0],
            'final': sparsity[-1],
            'log': sparsity,
            'generator': {
                'initial': sparsity_G[0],
                'final': sparsity_G[-1],
                'log': sparsity_G,
                'G_W1': {
                    'initial': sparsity_G_W1[0],
                    'final': sparsity_G_W1[-1],
                    'log': sparsity_G_W1
                },
                'G_W2': {
                    'initial': sparsity_G_W2[0],
                    'final': sparsity_G_W2[-1],
                    'log': sparsity_G_W2
                },
                'G_W3': {
                    'initial': sparsity_G_W3[0],
                    'final': sparsity_G_W3[-1],
                    'log': sparsity_G_W3
                }
            },
            'discriminator': {
                'initial': sparsity_D[0],
                'final': sparsity_D[-1],
                'log': sparsity_D,
                'D_W1': {
                    'initial': sparsity_D_W1[0],
                    'final': sparsity_D_W1[-1],
                    'log': sparsity_D_W1
                },
                'D_W2': {
                    'initial': sparsity_D_W2[0],
                    'final': sparsity_D_W2[-1],
                    'log': sparsity_D_W2
                },
                'D_W3': {
                    'initial': sparsity_D_W3[0],
                    'final': sparsity_D_W3[-1],
                    'log': sparsity_D_W3
                }
            }
        },
        'flops': {
            'total': sum(FLOPs),
            'log': FLOPs,
            'generator': {
                'total': sum(FLOPs_G),
                'log': FLOPs_G
            },
            'discriminator': {
                'total': sum(FLOPs_D),
                'log': FLOPs_D
            }
        },
        'loss': {
            'cross_entropy': {
                'generator': {
                    'initial': loss_G[0],
                    'total': loss_G[-1],
                    'log': loss_G
                },
                'discriminator': {
                    'initial': loss_D[0],
                    'final': loss_D[-1],
                    'log': loss_D
                }
            },
            'MSE': {
                'initial': loss_MSE[0],
                'final': loss_MSE[-1],
                'log': loss_MSE
            }
        }
    })

    f_logs = open(filepath, 'w')
    f_logs.write(json.dumps(logs))
    f_logs.close()

    return RMSE, imputation_time, memory_usage, energy_consumption, sparsity, sparsity_G, sparsity_G_W1, \
        sparsity_G_W2, sparsity_G_W3, sparsity_D, sparsity_D_W1, sparsity_D_W2, sparsity_D_W3, FLOPs, FLOPs_G, \
        FLOPs_D, loss_G, loss_D, loss_MSE, exp


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '-dir', '--directory', '-f', '--folder',
        help='the directory of the temporary files',
        default='temp/exp_bins',
        type=str)
    parser.add_argument(
        '-ng', '--no_graph',
        help="don't plot the graphs (log only)",
        action='store_true')
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
