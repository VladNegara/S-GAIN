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

"""Monitor class for S-GAIN:

Todo: run in separate thread

Initialization:
(1) init_monitors: initialize the temporary folder

Start monitors:
(2) start_rmse_monitor: open the RMSE log file
(3) start_imputation_time_monitor: open the imputation time log file
(4) start_memory_usage_monitor: open the memory usage log file
(5) start_energy_consumption_monitor: open the energy consumption log file
(6) start_sparsity_monitor: open the sparsity log files
(7) start_flops_monitor: open the FLOPs log files
(8) start_loss_monitor: open the loss log files
(9) start_all_monitors: start all the monitors

Log metrics:
(10) log_rmse: log the RMSE
(11) log_imputation_time: log the imputation time
(12) log_memory_usage: log the memory usage
(13) log_energy_consumption: log the energy consumption
(14) log_sparsity: log the sparsity
(15) log_flops: log the FLOPs
(16) log_loss: log the loss
(17) log_all_monitors: log the all monitors

Stop monitors:
(18) stop_rmse_monitor: close the RMSE log file
(19) stop_imputation_time_monitor: close the imputation time log file
(20) stop_memory_usage_monitor: close the memory usage log file
(21) stop_energy_consumption_monitor: close the energy consumption log file
(22) stop_sparsity_monitor: close the sparsity log files
(23) stop_flops_monitor: close the FLOPs log files
(24) stop_loss_monitor: close the loss log files
(25) stop_all_monitors: close all the monitors

Store trained model:
(26) set_model: set the (trained) model, so it can be saved later
(27) save_model: save the (trained) model to a json file
"""

import json
import struct

from os import makedirs
from os.path import isdir
from shutil import rmtree
from time import perf_counter

from utils.metrics import get_rmse


class Monitor:

    # -- Initialization -----------------------------------------------------------------------------------------------

    def __init__(self, data_x, data_mask, enable_rmse_monitor=True, enable_imputation_time_monitor=True,
                 enable_memory_usage_monitor=False, enable_energy_consumption_monitor=False,
                 enable_sparsity_monitor=True, enable_FLOPs_monitor=False, enable_loss_monitor=True, experiment=None,
                 verbose=False):
        """Initialize the monitor.

        :param data_x: the original data (without missing values)
        :param data_mask: the indicator matrix for missing elements
        :param enable_rmse_monitor: enable the RMSE monitor
        :param enable_imputation_time_monitor: enable the imputation time monitor
        :param enable_memory_usage_monitor: enable the memory usage monitor
        :param enable_energy_consumption_monitor: enable the energy consumption monitor
        :param enable_sparsity_monitor: enable the sparsity monitor
        :param enable_FLOPs_monitor: enable the FLOPs monitor
        :param enable_loss_monitor: enable the loss monitor
        :param experiment: the name of the experiment (optional)
        :param verbose: enable verbose output to console
        """

        # Parameters
        self.data_x = data_x
        self.data_mask = data_mask
        self.experiment = experiment
        self.verbose = verbose

        # Log variables
        self.imputation_time = None

        # Log files
        self.f_RMSE = enable_rmse_monitor
        self.f_imputation_time = enable_imputation_time_monitor
        self.f_memory_usage = enable_memory_usage_monitor
        self.f_energy_consumption = enable_energy_consumption_monitor
        self.f_sparsity_G, self.f_sparsity_G_W1, self.f_sparsity_G_W2, self.f_sparsity_G_W3, self.f_sparsity_D, \
            self.f_sparsity_D_W1, self.f_sparsity_D_W2, self.f_sparsity_D_W3 = [enable_sparsity_monitor] * 8
        self.f_FLOPs_G, self.f_FLOPs_D = [enable_FLOPs_monitor] * 2
        self.f_loss_G, self.f_loss_D, self.f_loss_MSE = [enable_loss_monitor] * 3

        # Model
        self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3 = [None] * 6
        self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3 = [None] * 6

    def init_monitor(self):
        """Initialize the temporary folder."""

        # Clear the logs and create temporary directory
        if self.verbose: print('Initializing monitor...')
        if isdir('temp/exp_bins'): rmtree('temp/exp_bins')
        makedirs('temp/exp_bins')

    # -- Start monitors -----------------------------------------------------------------------------------------------

    def start_rmse_monitor(self):
        """Open the RMSE log file and start monitoring.

        :return: True
        """

        if self.f_RMSE:
            self.f_RMSE = open('temp/exp_bins/rmse.bin', 'ab')
            if self.verbose: print('Monitoring RMSE...')
            return True

        return False

    def start_imputation_time_monitor(self):
        """Open the imputation time log file and start monitoring.

        :return: True
        """

        if self.f_imputation_time:
            self.f_imputation_time = open('temp/exp_bins/imputation_time.bin', 'ab')
            self.imputation_time = perf_counter()
            if self.verbose: print('Monitoring imputation time...')
            return True

        return False

    def start_memory_usage_monitor(self):
        """Open the memory usage log file and start monitoring.

        :return: True
        """

        if self.f_memory_usage:
            self.f_memory_usage = open('temp/exp_bins/memory_usage.bin', 'ab')
            if self.verbose: print('Monitoring memory usage...')
            return True

        return False

    def start_energy_consumption_monitor(self):
        """Open the energy consumption log file and start monitoring.

        :return: True
        """

        if self.f_energy_consumption:
            self.f_energy_consumption = open('temp/exp_bins/energy_consumption.bin', 'ab')
            if self.verbose: print('Monitoring energy consumption...')
            return True

        return False

    def start_sparsity_monitor(self):
        """Open the sparsity log files and start monitoring.

        :return: True
        """

        if self.f_sparsity_G:
            self.f_sparsity_G = open('temp/exp_bins/sparsity_G.bin', 'ab')
            self.f_sparsity_G_W1 = open('temp/exp_bins/sparsity_G_W1.bin', 'ab')
            self.f_sparsity_G_W2 = open('temp/exp_bins/sparsity_G_W2.bin', 'ab')
            self.f_sparsity_G_W3 = open('temp/exp_bins/sparsity_G_W3.bin', 'ab')
            self.f_sparsity_D = open('temp/exp_bins/sparsity_D.bin', 'ab')
            self.f_sparsity_D_W1 = open('temp/exp_bins/sparsity_D_W1.bin', 'ab')
            self.f_sparsity_D_W2 = open('temp/exp_bins/sparsity_D_W2.bin', 'ab')
            self.f_sparsity_D_W3 = open('temp/exp_bins/sparsity_D_W3.bin', 'ab')

            if self.verbose: print('Monitoring sparsity...')
            return True

        return False

    def start_flops_monitor(self):
        """Open the FLOPs log files and start monitoring.

        :return: True
        """

        if self.f_FLOPs_G:
            self.f_FLOPs_G = open('temp/exp_bins/flops_G.bin', 'ab')
            self.f_FLOPs_D = open('temp/exp_bins/flops_D.bin', 'ab')
            if self.verbose: print('Monitoring FLOPs...')
            return True

        return False

    def start_loss_monitor(self):
        """Open the loss log files and start monitoring.

        :return: True
        """

        if self.f_loss_G:
            self.f_loss_G = open('temp/exp_bins/loss_G.bin', 'ab')
            self.f_loss_D = open('temp/exp_bins/loss_D.bin', 'ab')
            self.f_loss_MSE = open('temp/exp_bins/loss_MSE.bin', 'ab')
            if self.verbose: print('Monitoring loss...')
            return True

        return False

    def start_all_monitors(self):
        """Start all the monitors.

        :return: True
        """

        if self.verbose: print('Starting monitors...')
        self.start_rmse_monitor()
        self.start_imputation_time_monitor()
        self.start_memory_usage_monitor()
        self.start_energy_consumption_monitor()
        self.start_sparsity_monitor()
        self.start_flops_monitor()
        self.start_loss_monitor()

        return True

    # -- Log metrics --------------------------------------------------------------------------------------------------

    def log_rmse(self, imputed_data):
        """Log the RMSE.

        :param imputed_data: The imputed data

        :return:
        - RMSE: the Root Mean Square Error
        """

        if self.f_RMSE:
            RMSE = get_rmse(self.data_x, imputed_data, self.data_mask)
            self.f_RMSE.write(struct.pack('f', RMSE))
            return RMSE

        return None

    def log_imputation_time(self):
        """Log the imputation time.

        :return:
        - step_time: the time (in seconds) between the previous step and now
        """

        if self.f_imputation_time:
            current_time = perf_counter()
            step_time = current_time - self.imputation_time
            self.f_imputation_time.write(struct.pack('f', step_time))
            self.imputation_time = current_time
            return step_time

        return None

    def log_memory_usage(self):
        """Log the memory usage.

        :return: True
        """

        if self.f_memory_usage:
            # Todo
            self.f_memory_usage.write()
            return True

        return None

    def log_energy_consumption(self):
        """Log the energy consumption.

        :return: True
        """

        if self.f_energy_consumption:
            # Todo
            self.f_energy_consumption.write()
            return True

        return None

    def log_sparsity(self, G_sparsities, D_sparsities):
        """Log the sparsity.

        :param G_sparsities: the sparsities of the Generator [Total, W1, W2, W3]
        :param D_sparsities: the sparsities of the Discriminator [Total, W1, W2, W3]

        :return: True
        """

        if self.f_sparsity_G:
            G_sparsity, G_W1_sparsity, G_W2_sparsity, G_W3_sparsity = G_sparsities
            D_sparsity, D_W1_sparsity, D_W2_sparsity, D_W3_sparsity = D_sparsities

            self.f_sparsity_G.write(struct.pack('f', G_sparsity))
            self.f_sparsity_G_W1.write(struct.pack('f', G_W1_sparsity))
            self.f_sparsity_G_W2.write(struct.pack('f', G_W2_sparsity))
            self.f_sparsity_G_W3.write(struct.pack('f', G_W3_sparsity))
            self.f_sparsity_D.write(struct.pack('f', D_sparsity))
            self.f_sparsity_D_W1.write(struct.pack('f', D_W1_sparsity))
            self.f_sparsity_D_W2.write(struct.pack('f', D_W2_sparsity))
            self.f_sparsity_D_W3.write(struct.pack('f', D_W3_sparsity))

            return True

        return None

    def log_flops(self):
        """Log the FLOPs.

        :return: True
        """

        if self.f_FLOPs_G:
            # Todo
            self.f_FLOPs_G.write()
            self.f_FLOPs_D.write()
            return True

        return None

    def log_loss(self, loss_G, loss_D, loss_MSE):
        """Log the loss.

        :param loss_G: the loss of the generator (cross entropy)
        :param loss_D: the loss of the discriminator (cross entropy)
        :param loss_MSE: the loss (MSE)

        :return: True
        """

        if self.f_loss_G:
            self.f_loss_G.write(struct.pack('f', loss_G))
            self.f_loss_D.write(struct.pack('f', loss_D))
            self.f_loss_MSE.write(struct.pack('f', loss_MSE))
            return True

        return None

    def log_all(self, imputed_data, G_sparsities, D_sparsities, loss_G, loss_D, loss_MSE):
        """Log the all monitors.

        :param imputed_data: The imputed data
        :param G_sparsities: the sparsities of the Generator [Total, W1, W2, W3]
        :param D_sparsities: the sparsities of the Discriminator [Total, W1, W2, W3]
        :param loss_G: the loss of the generator (cross entropy)
        :param loss_D: the loss of the discriminator (cross entropy)
        :param loss_MSE: the loss (MSE)

        :return: True
        """

        # Todo
        self.log_rmse(imputed_data)
        self.log_imputation_time()
        self.log_memory_usage()
        self.log_energy_consumption()
        self.log_sparsity(G_sparsities, D_sparsities)
        self.log_flops()
        self.log_loss(loss_G, loss_D, loss_MSE)

        return True

    # -- Stop monitors ------------------------------------------------------------------------------------------------

    def stop_rmse_monitor(self):
        """Close the RMSE log file and stop monitoring.

        :return: False
        """

        if self.f_RMSE:
            self.f_RMSE.close()
            if self.verbose: print('Stopped monitoring RMSE.')

        return False

    def stop_imputation_time_monitor(self):
        """Close the imputation time log file and stop monitoring.

        :return: False
        """

        if self.f_imputation_time:
            self.f_imputation_time.close()
            if self.verbose: print('Stopped monitoring imputation time.')

        return False

    def stop_memory_usage_monitor(self):
        """Close the memory usage log file and stop monitoring.

        :return: False
        """

        if self.f_memory_usage:
            self.f_memory_usage.close()
            if self.verbose: print('Stopped monitoring memory usage.')

        return False

    def stop_energy_consumption_monitor(self):
        """Close the energy consumption log file and stop monitoring.

        :return: False
        """

        if self.f_energy_consumption:
            self.f_energy_consumption.close()
            if self.verbose: print('Stopped monitoring energy consumption.')

        return False

    def stop_sparsity_monitor(self):
        """Close the sparsity log files and stop monitoring.

        :return: False
        """

        if self.f_sparsity_G:
            self.f_sparsity_G.close()
            self.f_sparsity_G_W1.close()
            self.f_sparsity_G_W2.close()
            self.f_sparsity_G_W3.close()
            self.f_sparsity_D.close()
            self.f_sparsity_D_W1.close()
            self.f_sparsity_D_W2.close()
            self.f_sparsity_D_W3.close()

            if self.verbose: print('Stopped monitoring sparsity.')

        return False

    def stop_flops_monitor(self):
        """Close the FLOPs log files and stop monitoring.

        :return: False
        """

        if self.f_FLOPs_G:
            self.f_FLOPs_G.close()
            self.f_FLOPs_D.close()
            if self.verbose: print('Stopped monitoring FLOPs.')

        return False

    def stop_loss_monitor(self):
        """Close the loss log files and stop monitoring.

        :return: False
        """

        if self.f_loss_G:
            self.f_loss_G.close()
            self.f_loss_D.close()
            self.f_loss_MSE.close()
            if self.verbose: print('Stopped monitoring loss (cross entropy and MSE).')

        return False

    def stop_all_monitors(self):
        """Stop all the monitors.

        :return: False
        """

        self.stop_rmse_monitor()
        self.stop_imputation_time_monitor()
        self.stop_memory_usage_monitor()
        self.stop_energy_consumption_monitor()
        self.stop_sparsity_monitor()
        self.stop_flops_monitor()
        self.stop_loss_monitor()

        if self.verbose: print('Stopped monitors.')
        return False

    # -- Store trained model ------------------------------------------------------------------------------------------

    def set_model(self, theta_G, theta_D):
        """Set the (trained) model, so it can be saved later.

        :param theta_G: the generator variables: G_W1, G_W2, G_W3, G_b1, G_b2, G_b3
        :param theta_D: the discriminator variables: D_W1, D_W2, D_W3, D_b1, D_b2, D_b3
        """

        self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3 = theta_G
        self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3 = theta_D

    def save_model(self, filepath):
        """Save the (trained) model to a json file.

        :param filepath: the filepath to save the model to
        """

        model = json.dumps({
            'theta_G': {
                'G_W1': self.G_W1.tolist(),
                'G_W2': self.G_W2.tolist(),
                'G_W3': self.G_W3.tolist(),
                'G_b1': self.G_b1.tolist(),
                'G_b2': self.G_b2.tolist(),
                'G_b3': self.G_b3.tolist()
            },
            'theta_D': {
                'D_W1': self.D_W1.tolist(),
                'D_W2': self.D_W2.tolist(),
                'D_W3': self.D_W3.tolist(),
                'D_b1': self.D_b1.tolist(),
                'D_b2': self.D_b2.tolist(),
                'D_b3': self.D_b3.tolist()
            }
        })

        with open(filepath, 'w') as f:
            f.write(model)
