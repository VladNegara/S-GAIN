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

"""Load and store operations for S-GAIN:

(1) get_filepaths: create the necessary directory and return the appropriate filepaths
(2) save_imputation: save the imputed data to a csv file
(3) parse_experiment: parse the experiment
(4) parse_files: parse the output files to a pandas dataframe
(5) parse_log: parse the log file to a list
(6) get_experiments: get a dictionary (or a list of strings) of the experiments to run
(7) read_bin: read a (temporary) binary file
(8) system_info: get the system information
"""

import cpuinfo
import json
import platform
import psutil
import struct
import subprocess
import wmi

import pandas as pd

from os import makedirs, listdir
from os.path import isdir, isfile


def get_filepaths(directory, experiment, rmse):
    """Create the necessary directory and return the appropriate filepaths.

    :param directory: the directory to save to
    :param experiment: the name of the experiment
    :param rmse: the Root Mean Squared Error

    :return:
    - filepath_imputed_data: the filepath for the imputed data
    - filepath_log: the filepath for the log
    - filepath_model: the filepath for the (trained) model
    - filepath_rmse: the filepath for the RMSE graph
    - filepath_imputation_time: the filepath for the imputation time graph
    - filepath_memory_usage: the filepath for the memory usage graph
    - filepath_energy_consumption: the filepath for the energy consumption graph
    - filepath_sparsity: the filepath for the sparsity graph
    - filepath_flops: the filepath for the FLOPs graph
    - filepath_loss: the filepath for the loss graph
    """

    if not isdir(directory): makedirs(directory)
    temp_filepath = f'{directory}/{experiment}_RMSE_{rmse}'

    # Avoid overwriting if RMSE is the same
    if (isfile(f'{temp_filepath}.csv')
            or isfile(f'{temp_filepath}_log.json')
            or isfile(f'{temp_filepath}_model.json')
            or isfile(f'{temp_filepath}_graphs.png')
    ):
        i = 1
        while (isfile(f'{temp_filepath}_{i}.csv')
               or isfile(f'{temp_filepath}_{i}_log.json')
               or isfile(f'{temp_filepath}_{i}_model.json')
               or isfile(f'{temp_filepath}_{i}_graphs.png')
        ): i += 1
        temp_filepath = f'{temp_filepath}_{i}'

    filepath_imputed_data = f'{temp_filepath}.csv'
    filepath_log = f'{temp_filepath}_log.json'
    filepath_model = f'{temp_filepath}_model.json'
    filepath_graphs = f'{temp_filepath}_graphs.png'

    return filepath_imputed_data, filepath_log, filepath_model, filepath_graphs


def save_imputation(filepath, imputed_data_x):
    """Save the imputed data to a CSV file.

    :param filepath: the filepath to save to
    :param imputed_data_x: the imputed data
    """

    # Save the imputation
    imputed_data_x = pd.DataFrame(imputed_data_x)
    imputed_data_x.to_csv(filepath, index=False)


def parse_experiment(experiment, file=False):
    """Parse the experiment.

    :param experiment: the name of the experiment
    :param file: whether parsing a file or not

    :return:
    - False: if the experiment is not in S-GAIN format
    - dataset: the dataset used
    - miss_rate: the probability of missing elements in the data
    - miss_modality: the modality of missing data (MCAR, MAR, MNAR)
    - seed: the seed used to introduce missing elements in the data
    - batch_size: the number of samples in mini-batch
    - hint_rate: the hint probability
    - alpha: the hyperparameter
    - iterations (epochs): the number of training iterations (epochs)
    - generator_sparsity: the probability of sparsity in the generator
    - generator_modality: the initialization and pruning and regrowth strategy of the generator
    - discriminator_sparsity: the probability of sparsity in the discriminator
    - discriminator_modality: the initialization and pruning and regrowth strategy of the discriminator
    - rmse: the RMSE (if parsing a file)
    - index: the index of the experiment (if parsing a file)
    - filetype: the type of file (imputed_data, log, model or graphs)
    """

    # Check if the experiment belongs to S-GAIN
    if not experiment.startswith('S-GAIN'): return False

    # Remove the file extension
    if file: experiment = experiment.rsplit('.', 1)[0]

    # Parse experiment
    _, rest = experiment.split('S-GAIN_')
    dataset, rest = rest.split('_MR_')
    miss_rate, rest = rest.split('_MM_')
    miss_rate = float(miss_rate)
    miss_modality, rest = rest.split('_S_')
    seed, rest = rest.split('_BS_')
    seed = int(seed, 16)
    batch_size, rest = rest.split('_HR_')
    batch_size = int(batch_size)
    hint_rate, rest = rest.split('_a_')
    hint_rate = float(hint_rate)
    alpha, rest = rest.split('_i_')
    alpha = float(alpha)
    iterations, rest = rest.split('_GS_')
    iterations = int(iterations)
    generator_sparsity, rest = rest.split('_GM_')
    generator_sparsity = float(generator_sparsity)
    generator_modality, rest = rest.split('_DS_')
    discriminator_sparsity, rest = rest.split('_DM_')
    discriminator_sparsity = float(discriminator_sparsity)

    if file:  # rmse, index, filetype
        discriminator_modality, rest = rest.split('_RMSE_')

        rests = rest.split('_', 1)
        rmse = float(rests[0])

        if len(rests) == 1:
            index = 0
            filetype = 'imputed_data'
        else:  # index, filetype
            index_filetype = rests[1].split('_', 1)
            if len(index_filetype) == 1:  # index or filetype
                if index_filetype[0].isdigit():  # index
                    index = int(index_filetype[0])
                    filetype = 'imputed_data'
                else:  # filetype
                    index = 0
                    filetype = index_filetype[0]
            elif index_filetype[0].isdigit():  # index and filetype
                index = int(index_filetype[0])
                filetype = index_filetype[1]
            else:  # filetype
                index = 0
                filetype = rests[1]

        return dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity, \
            generator_modality, discriminator_sparsity, discriminator_modality, rmse, index, filetype

    else:
        discriminator_modality = rest

        return dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, \
            generator_sparsity, generator_modality, discriminator_sparsity, discriminator_modality


def parse_files(files=None, filepath='output', filetype=None):
    """Parse the output files to a pandas dataframe.

    :param files: a list of files to parse (optional)
    :param filepath: the output filepath
    :param filetype: only return files of this type (imputed_data, log, model, etc.)

    :return:
    - df_files: a Pandas DataFrame with all the experiments
    """

    files = [parse_experiment(file, file=True) for file in files] if files \
        else [parse_experiment(file, file=True) for file in listdir(filepath)] if isdir(filepath) \
        else []

    # Remove files that don't belong to S-GAIN (parse_experiment returns False for non S-GAIN files)
    files = [file for file in files if file]

    header = ['dataset', 'miss_rate', 'miss_modality', 'seed', 'batch_size', 'hint_rate', 'alpha', 'iterations',
              'generator_sparsity', 'generator_modality', 'discriminator_sparsity', 'discriminator_modality', 'rmse',
              'index', 'filetype']

    # Only keep selected filetype
    if filetype: files = [file for file in files if file[-1] == filetype]

    df_files = pd.DataFrame(files, columns=header)
    return df_files


def parse_log(filepath_log):
    """Parse the log file to a list.

    :param filepath_log: the filepath for the log file

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
    - loss_MSE: the los log (MSE)
    """

    # Read the log file
    with open(filepath_log, 'r') as f:
        log = json.loads(f.read())

    # Retrieve the logs
    RMSE = log['rmse']['log']
    imputation_time = log['imputation_time']['log']
    memory_usage = log['memory_usage']['log']
    energy_consumption = log['energy_consumption']['log']
    sparsity = log['sparsity']['log']
    sparsity_G = log['sparsity']['generator']['log']
    sparsity_G_W1 = log['sparsity']['generator']['G_W1']['log']
    sparsity_G_W2 = log['sparsity']['generator']['G_W2']['log']
    sparsity_G_W3 = log['sparsity']['generator']['G_W3']['log']
    sparsity_D = log['sparsity']['discriminator']['log']
    sparsity_D_W1 = log['sparsity']['discriminator']['D_W1']['log']
    sparsity_D_W2 = log['sparsity']['discriminator']['D_W2']['log']
    sparsity_D_W3 = log['sparsity']['discriminator']['D_W3']['log']
    FLOPs = log['flops']['log']
    FLOPs_G = log['flops']['generator']['log']
    FLOPs_D = log['flops']['discriminator']['log']
    loss_G = log['loss']['cross_entropy']['generator']['log']
    loss_D = log['loss']['cross_entropy']['discriminator']['log']
    loss_MSE = log['loss']['MSE']['log']

    return RMSE, imputation_time, memory_usage, energy_consumption, sparsity, sparsity_G, sparsity_G_W1, \
        sparsity_G_W2, sparsity_G_W3, sparsity_D, sparsity_D_W1, sparsity_D_W2, sparsity_D_W3, FLOPs, FLOPs_G, \
        FLOPs_D, loss_G, loss_D, loss_MSE


def get_experiments(datasets, miss_rates, miss_modalities, seeds, batch_sizes, hint_rates, alphas, iterations_s,
                    generator_sparsities, generator_modalities, discriminator_sparsities, discriminator_modalities,
                    folder='output', n_runs=10, ignore_existing_files=False, retry_failed_experiments=True,
                    include=None, exclude=None, verbose=False, no_log=False, no_graph=False, no_model=False,
                    no_save=False, no_system_information=False, get_commands=False):
    """Get a dictionary (or a list of commands) of the experiments to run.

    :param datasets: which datasets to use
    :param miss_rates: the probabilities of missing elements in the data
    :param miss_modalities: the modalities of missing data (MCAR, MAR, MNAR)
    :param seeds: the seeds used to introduce missing elements in the data (optional)
    :param batch_sizes: a list of the number of samples in mini-batch
    :param hint_rates: the hint probabilities
    :param alphas: the hyperparameters
    :param iterations_s: a list of the number of training iterations (epochs)
    :param generator_sparsities: the probabilities of sparsity in the generator
    :param generator_modalities: the initializations and pruning and regrowth strategies of the generator
    :param discriminator_sparsities: the probabilities of sparsity in the discriminator
    :param discriminator_modalities: the initializations and pruning and regrowth strategies of the discriminator
    :param folder: the output folder (change if saving to a different folder)
    :param n_runs: the number of times each experiment should run (default: 10)
    :param ignore_existing_files: ignore existing files in the output folder (always run the experiment n times)
    :param retry_failed_experiments: retry failed experiments (default: True)
    :param include: include these keys in the experiments to run
    :param exclude: exclude these keys from the experiments to run
    :param verbose: enable verbose output to console
    :param no_log: turn off the logging of metrics (also disables graphs and model)
    :param no_graph: don't plot graphs after training
    :param no_model: don't save the trained model
    :param no_save: don't save the imputation
    :param no_system_information: don't log system information
    :param get_commands: get a list of ready to run commands instead of a dictionary

    :return:
    - experiments: a dictionary (or a list of commands) of the experiments to run
    """

    # Standardization
    def sparsities_modalities(sparsities, modalities):
        sparsity_modality = [
            (0, 'dense')
            for sparsity in sparsities if sparsity == 0
            for modality in modalities if modality == 'dense'
        ]
        sparsity_modality += [
            (sparsity, 'random')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality == 'random'
        ]
        sparsity_modality += [
            (sparsity, 'ER')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ER', 'erdos_renyi')
        ]
        sparsity_modality += [
            (sparsity, 'ERK')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERK', 'erdos_renyi_kernel')
        ]
        sparsity_modality += [
            (sparsity, 'ERRW')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERRW', 'erdos_renyi_random_weight')
        ]
        sparsity_modality += [
            (sparsity, 'ERKRW')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERKRW', 'erdos_renyi_kernel_random_weight')
        ]
        return sparsity_modality

    generator_sparsity_modality = sparsities_modalities(generator_sparsities, generator_modalities)
    discriminator_sparsity_modality = sparsities_modalities(discriminator_sparsities, discriminator_modalities)

    # Get the experiments
    experiments = {}
    experiments.update({
        (dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity,
         generator_modality, discriminator_sparsity, discriminator_modality): n_runs
        for dataset in datasets
        for miss_rate in miss_rates
        for miss_modality in miss_modalities
        for seed in seeds
        for batch_size in batch_sizes
        for hint_rate in hint_rates
        for alpha in alphas
        for iterations in iterations_s
        for generator_sparsity, generator_modality in generator_sparsity_modality
        for discriminator_sparsity, discriminator_modality in discriminator_sparsity_modality
    })

    # Add inclusions
    if include is not None: experiments.update({
        key: n_runs for key in include
    })

    # Remove completed experiments
    if not ignore_existing_files:
        df = parse_files(filepath=folder)
        df.drop('filetype', axis=1, inplace=True)  # ignore filetype
        df.drop_duplicates(inplace=True)  # remove duplicates
        if retry_failed_experiments: df.dropna(inplace=True)  # remove failed experiments from the count
        df.drop(['rmse', 'index'], axis=1, inplace=True)  # ignore rmse and index
        df = df.groupby(df.columns.tolist(), as_index=False).size()  # count
        completed = {tuple(exp[:-1]): exp[-1] for exp in df.values}

        for experiment, n_completed in completed.items():
            if experiment in experiments:
                if experiments[experiment] - n_completed <= 0:
                    experiments.pop(experiment)
                else:
                    experiments[experiment] = n_runs - n_completed

    # Remove exclusions
    if exclude is not None:
        for key in exclude:
            if key in experiments: experiments.pop(key)

    # Convert to executable strings
    if get_commands:
        commands = [
            f'python main.py {dataset} --miss_rate {miss_rate} --miss_modality {miss_modality} --seed {seed} '
            f'--batch_size {batch_size} --hint_rate {hint_rate} --alpha {alpha} --iterations {iterations} '
            f'--generator_sparsity {generator_sparsity} --generator_modality {generator_modality} '
            f'--discriminator_sparsity {discriminator_sparsity} --discriminator_modality {discriminator_modality} '
            f'--folder {folder}{" --verbose" if verbose else ""}{" --no_log" if no_log else ""}'
            f'{" --no_graph" if no_graph else ""}{" --no_model" if no_model else ""}{" --no_save" if no_save else ""}'
            f'{" --no_system_information" if no_system_information else ""}'

            for [dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity,
                 generator_modality, discriminator_sparsity, discriminator_modality], n in experiments.items()
            for _ in range(n)
        ]
        return commands

    return experiments


def read_bin(filepath):
    """Read a (temporary) binary file.

    :param filepath: the filepath

    :return: the unpacked data from the file
    """

    # Read binary data
    with open(filepath, 'rb') as f:
        data = f.read()

    # Unpack the data
    fmt = '<%df' % (len(data) // 4)
    data = list(struct.unpack(fmt, data))

    return data


def system_information(directory='temp', print_ready=False):
    """Get the system information.

    :param directory: the temporary directory
    :param print_ready: return a list of print ready strings instead of a dictionary

    :return:
    - sys_info: the system information
    """

    filepath = f'{directory}/sys_info.json'

    # Load system information
    if isfile(filepath):
        with open(filepath, 'r') as f:
            sys_info = json.load(f)

    # Get and store system information
    else:
        sys_info = {
            'platform': platform.system(),
            'version': platform.version(),
            'cpu': cpuinfo.get_cpu_info()['brand_raw'],
            'memory': f'{psutil.virtual_memory().total / (1024 ** 3):.1f} GB'
        }

        # Todo multiple gpu support (log the one used for running the experiment only)
        #  and more memory detail (speed, timings or identifier)
        if sys_info['platform'] == 'Linux':
            # Todo Get GPU
            sys_info['gpu'] = 'unable to identify GPU (no support for Linux yet)'

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Linux yet)'

            # Todo Update OS and log motherboard
            sys_info['motherboard'] = 'unable to identify motherboard (no support for Linux yet)'

        elif sys_info['platform'] == 'Windows':
            # Get GPU
            sys_info['gpu'] = wmi.WMI().Win32_VideoController()[0].name

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Windows yet)'

            # Update OS and log motherboard
            for x in subprocess.check_output(['systeminfo']).decode('utf-8').split('\n'):
                if x.startswith('OS Name'):
                    sys_info['version'] = x.split(':')[1].strip()
                elif x.startswith('OS Version'):
                    sys_info['version'] += f' {x.split(":")[1].strip()}'
                elif x.startswith('System Manufacturer'):
                    sys_info['motherboard'] = x.split(':')[1].strip()
                elif x.startswith('System Model'):
                    sys_info['motherboard'] += f' {x.split(":")[1].strip()}'
                    break

        elif sys_info['platform'] == 'Darwin':
            # Todo Get GPU
            sys_info['gpu'] = 'unable to identify GPU (no support for Mac OS yet)'

            # Todo Get disk info
            sys_info['disk'] = 'unable to identify disk (no support for Mac OS yet)'

            # Todo Update OS and log motherboard
            sys_info['motherboard'] = 'unable to identify motherboard (no support for Mac OS yet)'

        else:
            # Log GPU, disk and motherboard
            sys_info['gpu'] = f'unable to identify GPU ({sys_info["platform"]} unsupported)'
            sys_info['disk'] = f'unable to identify disk ({sys_info["platform"]} unsupported)'
            sys_info['motherboard'] = f'unable to identify motherboard ({sys_info["platform"]} unsupported)'

        # Store the system information
        if not isdir(directory): makedirs(directory)
        with open(filepath, 'w') as f:
            f.write(json.dumps(sys_info))

    # Convert dictionary to print ready strings
    if print_ready:
        info = 'System information'
        version = f'OS: {sys_info["version"]}'
        cpu = f'CPU: {sys_info["cpu"]}'
        memory = f'Memory: {sys_info["memory"]}'
        gpu = f'GPU: {sys_info["gpu"]}'
        disk = f'Disk: {sys_info["disk"]}'
        motherboard = f'Motherboard: {sys_info["motherboard"]}'
        sys_info = [info, version, cpu, memory, gpu, disk, motherboard]

    return sys_info
