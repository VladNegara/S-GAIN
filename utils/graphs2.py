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

"""Graphing functions for S-GAIN:

(1) get_sizing: helper function to calculate the different sizes for the plot
(2) plot_info: helper function to plot the experiment and system information
(3) plot_graphs: load and plot the graphs (RMSE, imputation time, memory usage, energy_consumption, loss and FLOPs)
"""

import matplotlib.pyplot as plt

from datetime import timedelta
from matplotlib import ticker


def get_sizing(ncols, nrows, ax_width, ax_height, w_space=1.28, h_space=1.2):
    """Calculates the different sizes for the plot.

    :param ncols: the number of columns in the plot
    :param nrows: the number of rows in the plot
    :param ax_width: the width of the subplots
    :param ax_height: the height of the subplots
    :param w_space: the width of the whitespace
    :param h_space: the height of the whitespace

    :return:
    - fig_width: total width of the figure
    - fig_height: total height of the figure
    - left: left margin of the figure
    - right: right margin of the figure
    - top: top margin of the figure
    - bottom: bottom margin of the figure
    - wspace: the horizontal padding between two subplots
    - hspace: the vertical padding between two subplots
    - title: the (relative) position of the title in the figure
    """

    # Margins (absolute)
    left_abs = w_space
    right_abs = 0.52
    top_abs = 2
    bottom_abs = 1
    title_abs = 0.76

    # Subplots (absolute)
    ax_width_total = ax_width * ncols
    ax_height_total = ax_height * nrows

    # Padding (absolute)
    wspace_abs = w_space
    hspace_abs = h_space
    wspace_total = wspace_abs * (ncols - 1)
    hspace_total = hspace_abs * (nrows - 1)

    # Figure (absolute)
    fig_width = left_abs + ax_width_total + wspace_total + right_abs
    fig_height = top_abs + ax_height_total + hspace_total + bottom_abs

    # Margins (relative)
    left = left_abs / fig_width
    right = 1 - right_abs / fig_width
    top = 1 - top_abs / fig_height
    bottom = bottom_abs / fig_height
    title = (fig_height - title_abs) / fig_height

    # Padding (relative)
    wspace = wspace_abs / ax_width
    hspace = hspace_abs / ax_height

    return fig_width, fig_height, left, right, top, bottom, wspace, hspace, title


def plot_info(ax, text, x=0.0, y=0.97):
    """Plot the experiment and system information.

    :param ax: the subplot to print the experiment and/or experiment to
    :param text: a list of strings to print
    :param x: the x coordinate to start printing from
    :param y: the y coordinate to start printing from
    """

    for txt in text:
        if txt in ('Experiment', 'Experiments', ' ', 'System information'):
            ax.text(x, y, txt, fontsize=13, weight='bold')
            y -= .06
        else:
            ax.text(x, y, txt, fontsize=12)
            y -= .05


def plot_graphs(filepath, rmse_log=None, imputation_time_log=None, memory_usage_log=None, energy_consumption_log=None,
                sparsity_logs=None, flops_logs=None, loss_logs=None, logs=None, experiment=None, sys_info=None,
                title=None):
    """Load and plot the graphs.

    :param filepath: the filepath for the graphs
    :param rmse_log: the RMSE log
    :param imputation_time_log: the imputation time log
    :param memory_usage_log: the memory usage log
    :param energy_consumption_log: the energy consumption log
    :param sparsity_logs: the sparsity logs [S-GAIN, G, G_W1, G_W2, G_W3, D, D_W1, D_W2, D_W3]
    :param flops_logs: the flops logs [S-GAIN, Generator, Discriminator]
    :param loss_logs: the loss logs [Generator, Discriminator, MSE]
    :param logs: a list of all logs (optional)
    :param experiment: the experiment
    :param sys_info: the system info (in print ready format)
    :param title: the title (optional)
    """

    # Todo implement
    rmse_log, memory_usage_log, flops_logs = None, None, []

    if logs:  # Get all the logs from the list
        rmse_log, imputation_time_log, memory_usage_log, energy_consumption_log, sparsity_log, \
            sparsity_G_log, sparsity_G_W1_log, sparsity_G_W2_log, sparsity_G_W3_log, \
            sparsity_D_log, sparsity_D_W1_log, sparsity_D_W2_log, sparsity_D_W3_log, \
            flops_log, flops_G_log, flops_D_log, loss_G_log, loss_D_log, loss_MSE_log = logs

        sparsity_logs = [sparsity_log, sparsity_G_log, sparsity_G_W1_log, sparsity_G_W2_log, sparsity_G_W3_log,
                         sparsity_D_log, sparsity_D_W1_log, sparsity_D_W2_log, sparsity_D_W3_log]
        flops_logs = [flops_log, flops_G_log, flops_D_log]
        loss_logs = [loss_G_log, loss_D_log, loss_MSE_log]

    # Get nrows required
    nrows = (1 if experiment or sys_info else 0) + (1 if rmse_log else 0) + (1 if imputation_time_log else 0) \
            + (1 if memory_usage_log else 0) + (1 if energy_consumption_log else 0) + (2 if sparsity_logs else 0) \
            + (1 if flops_logs else 0) + (2 if loss_logs else 0)

    # Stop if no logs are provided
    if nrows == 0: return

    # New plot
    width, height, left, right, top, bottom, wspace, hspace, y_title = get_sizing(1, nrows, 12.8, 4.8)
    fig, axs = plt.subplots(nrows, figsize=(width, height))

    index = 0
    if experiment or sys_info:
        text = []
        if experiment:
            exp = 'Experiment'
            dataset = f'Dataset: {experiment["dataset"]}'
            miss_rate = f'Miss rate: {int(experiment["miss_rate"] * 100)}%'
            miss_modality = f'Miss modality: {experiment["miss_modality"]}'
            seed = f'Seed: {hex(experiment["seed"])}'
            batch_size = f'Batch size: {experiment["batch_size"]}'
            hint_rate = f'Hint rate: {experiment["hint_rate"]}'
            alpha = f'Alpha: {experiment["alpha"]}'
            iterations = f'Iterations: {experiment["iterations"]}'
            generator_sparsity = f'Generator sparsity: {experiment["generator_sparsity"]}'
            generator_modality = f'Generator modality: {experiment["generator_modality"]}'
            discriminator_sparsity = f'Discriminator sparsity: {experiment["discriminator_sparsity"]}'
            discriminator_modality = f'Discriminator modality: {experiment["discriminator_modality"]}'

            text += [exp, dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations,
                     generator_sparsity, generator_modality, discriminator_sparsity, discriminator_modality, ' ']

        if sys_info: text += sys_info

        # Plot info
        plot_info(axs[index], text)
        axs[index].set_axis_off()

        # Increase index
        index += 1

    if rmse_log:  # Plot RMSE
        axs[index].plot(rmse_log, label=f'{rmse_log[-1]:.4f}')

        len_log = len(rmse_log)

        # RMSE parameters
        axs[index].set_title('RMSE per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('RMSE', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Final RMSE', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if imputation_time_log:  # Plot imputation time
        label = f'{timedelta(seconds=round(sum(imputation_time_log)))}'
        axs[index].plot(imputation_time_log, label=label, color='black')

        len_log = len(imputation_time_log)

        # Plot parameters
        axs[index].set_title('Imputation time per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Time (in seconds)', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total imputation time', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if memory_usage_log:  # Plot memory usage
        axs[index].plot(memory_usage_log)
        # Todo: plot the graph (use the legend to display the maximum)

        len_log = len(memory_usage_log)

        # Plot parameters
        axs[index].set_title('Memory usage per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Memory usage (in MB)', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total memory usage', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if energy_consumption_log:  # Plot energy consumption
        axs[index].plot(energy_consumption_log)
        # Todo: plot the graph (use the legend to display the total)

        len_log = len(energy_consumption_log)

        # Plot parameters
        axs[index].set_title('Energy consumption per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Energy consumption (in joule)', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total energy consumption', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if sparsity_logs:  # Plot sparsity
        sparsity_log, sparsity_G_log, sparsity_G_W1_log, sparsity_G_W2_log, sparsity_G_W3_log, sparsity_D_log, \
            sparsity_D_W1_log, sparsity_D_W2_log, sparsity_D_W3_log = sparsity_logs

        # Labels Todo add average and maximum?
        label_S_GAIN = f'S-GAIN: {sparsity_log[0] * 100:.1f}% | {sparsity_log[-1] * 100:.1f}% | {min(sparsity_log) * 100:.1f}%'

        label_G = f'Overall: {sparsity_G_log[0] * 100:.1f}% | {sparsity_G_log[-1] * 100:.1f}% | {min(sparsity_G_log) * 100:.1f}%'
        label_G_W1 = f'Layer 1: {sparsity_G_W1_log[0] * 100:.1f}% | {sparsity_G_W1_log[-1] * 100:.1f}% | {min(sparsity_G_W1_log) * 100:.1f}%'
        label_G_W2 = f'Layer 2: {sparsity_G_W2_log[0] * 100:.1f}% | {sparsity_G_W2_log[-1] * 100:.1f}% | {min(sparsity_G_W2_log) * 100:.1f}%'
        label_G_W3 = f'Layer 3: {sparsity_G_W3_log[0] * 100:.1f}% | {sparsity_G_W3_log[-1] * 100:.1f}% | {min(sparsity_G_W3_log) * 100:.1f}%'

        label_D = f'Overall: {sparsity_D_log[0] * 100:.1f}% | {sparsity_D_log[-1] * 100:.1f}% | {min(sparsity_D_log) * 100:.1f}%'
        label_D_W1 = f'Layer 1: {sparsity_D_W1_log[0] * 100:.1f}% | {sparsity_D_W1_log[-1] * 100:.1f}% | {min(sparsity_D_W1_log) * 100:.1f}%'
        label_D_W2 = f'Layer 2: {sparsity_D_W2_log[0] * 100:.1f}% | {sparsity_D_W2_log[-1] * 100:.1f}% | {min(sparsity_D_W2_log) * 100:.1f}%'
        label_D_W3 = f'Layer 3: {sparsity_D_W3_log[0] * 100:.1f}% | {sparsity_D_W3_log[-1] * 100:.1f}% | {min(sparsity_D_W3_log) * 100:.1f}%'

        # Plots
        axs[index].plot(sparsity_log, label=label_S_GAIN, color='black')
        axs[index].plot(sparsity_G_log, label=label_G, color='navy')
        axs[index].plot(sparsity_G_W1_log, label=label_G_W1, color='blue')
        axs[index].plot(sparsity_G_W2_log, label=label_G_W2, color='dodgerblue')
        axs[index].plot(sparsity_G_W3_log, label=label_G_W3, color='deepskyblue')

        axs[index + 1].plot(sparsity_log, label=label_S_GAIN, color='black')
        axs[index + 1].plot(sparsity_D_log, label=label_D, color='darkred')
        axs[index + 1].plot(sparsity_D_W1_log, label=label_D_W1, color='tab:red')
        axs[index + 1].plot(sparsity_D_W2_log, label=label_D_W2, color='lightcoral')
        axs[index + 1].plot(sparsity_D_W3_log, label=label_D_W3, color='pink')

        len_log = len(sparsity_log)

        # Generator parameters
        axs[index].set_title('Generator sparsity per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Sparsity', size=13)
        axs[index].set_ylim(0, 1)
        axs[index].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Sparsity: Initial | Final | Minimum', prop={'size': 13})
        axs[index].grid(True)

        # Discriminator parameters
        axs[index + 1].set_title('Discriminator sparsity per epoch')
        axs[index + 1].title.set_size(16)
        axs[index + 1].set_ylabel('Sparsity', size=13)
        axs[index + 1].set_ylim(0, 1)
        axs[index + 1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))
        axs[index + 1].set_xlabel('Epochs', size=13)
        axs[index + 1].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index + 1].tick_params(labelsize=12)
        lgnd = axs[index + 1].legend(fontsize=12)
        lgnd.set_title(title='Sparsity: Initial | Final | Minimum', prop={'size': 13})
        axs[index + 1].grid(True)

        # Increase index
        index += 2

    if flops_logs:  # Plot FLOPs
        flops_log, flops_G_log, flops_D_log = flops_logs

        axs[index].plot(flops_log, label=f'S-GAIN: {sum(flops_log)}', color='black')
        axs[index].plot(flops_G_log, label=f'Generator: {sum(flops_G_log)}', color='tab:blue')
        axs[index].plot(flops_D_log, label=f'Discriminator: {sum(flops_D_log)}', color='tab:red')

        len_log = len(flops_log)

        # Plot parameters
        axs[index].set_title('FLOPs per epoch')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('FLOPs', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        lgnd = axs[index].legend(fontsize=12)
        lgnd.set_title(title='Total FLOPs', prop={'size': 13})
        axs[index].grid(True)

        # Increase index
        index += 1

    if loss_logs:  # Plot losses (Cross Entropy and MSE)
        loss_G_log, loss_D_log, loss_MSE_log = loss_logs

        axs[index].plot(loss_G_log, label='Generator loss', color='tab:blue')
        axs[index].plot(loss_D_log, label='Discriminator loss', color='tab:red')

        axs[index + 1].plot(loss_MSE_log, label='MSE loss', color='black')

        len_log = len(loss_G_log)

        # Cross Entropy parameters
        axs[index].title.set_text('Learning curves')
        axs[index].title.set_size(16)
        axs[index].set_ylabel('Cross Entropy', size=13)
        axs[index].set_xlabel('Epochs', size=13)
        axs[index].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index].tick_params(labelsize=12)
        axs[index].legend(fontsize=12)
        axs[index].grid(True)

        # MSE parameters
        axs[index + 1].title.set_text('Learning curves')
        axs[index + 1].title.set_size(16)
        axs[index + 1].set_ylabel('MSE', size=13)
        axs[index + 1].set_xlabel('Epochs', size=13)
        axs[index + 1].set_xlim(-len_log * 0.01, len_log * 1.01)
        axs[index + 1].tick_params(labelsize=12)
        axs[index + 1].legend(fontsize=12)
        axs[index + 1].grid(True)

        # Increase index
        index += 2

    # Plot parameters
    m = int(len(title) / 2)
    title = title[:m] + title[m:].replace('_', '\n_', 1)
    plt.suptitle(title, size=22, y=y_title)
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    # Save plot
    plt.savefig(filepath, format='png')
