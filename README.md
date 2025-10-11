# Codebase of S-GAIN

When using this code, please cite the following paper:

Paper: B.P. van Oers, I. Baysal Erez, M. van Keulen, "Sparse GAIN: Imputation Methods to Handle Missing Values with
Sparse Initialization", IDEAL conference, 2025.

Authors: Brian Patrick van Oers, Işıl Baysal Erez, Maurice van Keulen

Release: This paper is associated with [pre-release v0.1.0-alpha](
https://github.com/BrianPvanOers/S-GAIN/releases/tag/v0.1.0-alpha) *

Contact: b.p.vanoers@student.utwente.nl

\* Alternatively one may import s_gain from models.IDEAL2025.s_gain_TFv1_FP32_init_only in main.py to run the
experiments associated with this paper with the current tools for analysis. The settings used for the experiments
discussed in this paper are also available in run_experiments.py for easy replication.

---

## About the project

We adapted the original GAIN code for our work: [J. Yoon, J. Jordon and M. van der Schaar, "GAIN: Missing Data
Imputation using Generative Adversarial Nets," International Conference on Machine Learning (ICML), 2018.](
https://github.com/jsyoon0823/GAIN)

We created a framework for (automated) testing and implemented sparse initialization approaches to improve computational
efficiency and therefore energy consumption, memory usage and imputation time, and possibly increase performance and
reduce failure rates. We plan to turn these initializations into full Dynamic Sparse Training strategies, rebuild the
model in TensorFlow 2.x using Sparse Tensors and INT8 precision, and then run it on the GPU to speed up the experiments.

We ran our experiments using python 3.11, earlier or later versions might have package conflicts.

Compiling the log and plotting the graphs may not function correctly when called from main.py. It is advised to run
log_and_graphs.py after each experiment.

One may use run_experiments.py, which does not have aforementioned problem, to run multiple experiments in sequence,
automatically analyze them and if needed shutdown the computer after wards.

---

## How to run the command

### Explanation of parameters

####

- **dataset:** the dataset to use [spam, letter, health, mnist, fashion_mnist, cifar10]
- **miss_rate:** the probability of missing elements in the data (default: 0.2)
- **miss_modality:** the modality of missing data [MCAR, MAR, MNAR] (default: 'MCAR')
- **seed:** the seed used to introduce missing elements in the data (optional)

####

- **batch_size:** the number of samples in mini-batch (default: 128)
- **hint_rate:** the hint probability (default: 0.9)
- **alpha:** the hyperparameter (default: 100)
- **iterations (epochs):** the number of training iterations (epochs) (default: 10000)
- **generator_sparsity:** the probability of sparsity in the generator (default: 0)
- **generator_modality:** the initialization and pruning and regrowth strategy of the generator [dense, random,
  erdos_renyi, ERRW] (default: 'dense')
- **discriminator_sparsity:** the probability of sparsity in the discriminator (default: 0)
- **discriminator_modality:** the initialization and pruning and regrowth strategy of the discriminator (default: 'dense')

####

- **folder (directory):** save the imputed data to a different folder (optional, default: 'output')
- **verbose:** enable verbose logging
- **no_log:** turn off the logging of metrics (also disables graphs)
- **no_graph:** don't plot graphs after training
- **no_model:** don't save the trained model
- **no_save:** don't save the imputation
- **no_system_information:** don't log system information

### Example commands (default parameters)

###### Minimum required parameters (dataset only, random seed)

```shell
$ python main.py spam
```

###### Fully specified parameters (fixed seed)

```shell
$ python main.py spam --miss_rate 0.2 --miss_modality MCAR --seed 0 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000 --generator_sparsity 0 --generator_modality dense --discriminator_sparsity 0 --discriminator_modality dense --folder output
```

###### Fully specified parameters with abbreviated flags (fixed seed)

```shell
$ python main.py spam -mr 0.2 -mm MCAR -s 0 -bs 128 -hr 0.9 -a 100 -i 10000 -gs 0 -gm dense -ds 0 -dm dense -f output
```

###### Set flags

```shell
$ python main.py spam --verbose --no_log --no_graph --no_model --no_save --no_system_information
```

### Outputs

- **imputed_data_x:** the imputed data
- **rmse:** Root Mean Squared Error

---

### Log and graphs

- **folder (directory):** the directory of the temporary files
- **no_graph:** don't plot the graphs (log only)
- **no_system_information:** don't log system information
- **verbose:** enable verbose logging

```shell
$ python log_and_graphs.py --verbose
```

---

### Analyze

- **all:** plot all the graphs
- **rmse:** plot the RMSE graphs
- **success_rate:** plot the success rate graphs
- **imputation_time:** plot the imputation time graphs
- **save:** save the analysis
- **input (experiments):** the folder where the experiments were saved to (optional, default: 'output')
- **output (analysis):** save the analysis to a different folder (optional, default: 'analysis')
- **no_system_information:** don't log system information
- **verbose:** enable verbose logging

```shell
$ python analyze.py --all --save --experiments output --analysis analysis --verbose
```

- **exps:** a Pandas DataFrame with the computed metrics (RMSE mean, std and improvement, successes, total, success
  rate and imputation times (total, preparation, S-GAIN and finalization steps))

---

### Run_experiments

One may use this file to run multiple experiments in sequence, automatically analyze them and if needed shutdown the
computer after wards (if no experiments will be run; auto_shutdown is ignored). The settings are given as lists.
run_experiments.py will run all possible combinations of these settings for n_runs times. Nonsense is ignored, i.e.
dense initialization with > 0% sparsity and non-dense initializations with 0% sparsity won't be run. Below you find
additional settings not already explained in prior sections:

- **n_runs:** the amount of times each experiment should be performed (default: 10)
- **ignore_existing_files:** ignore the existing files in the output folder (disables loop_until_complete, default:
  False)
- **retry_failed_experiments:** retry the failed experiments (enables loop_until_complete, default: True)
- **loop_until_complete:** loop until each experiment successfully completes n_runs times (default: True)
- **analyze:** automatically analyze all the experiments after completion (default: True)
- **analysis_folder:** the output folder of the analysis (default: 'analysis')
- **auto_shutdown:** automatically shutdown the computer after running the experiments and performing the analysis
  (default: False)


---

## Folders and files

####

- **datasets:** Contains (some of) the datasets to run the S-GAIN imputer on. A dataset must be complete, have a header
  and the labels and index must be removed. These datasets serve as x_train. (Todo: test with labels to test its
  classifier performance)
- **datasets/health.csv:** Ahmed, M. (2020). Maternal Health Risk [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5DP5D.
- **datasets/letter.csv:** Slate, D. (1991). Letter Recognition [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5ZP40.
- **datasets/spam.csv:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C53G6X.

####

- **models:** Contains the different models. Currently only contains S-GAIN.
- **models/IDEAL2025/s_gain_TFv1_FP32_init_only.py:** The S-GAIN imputer. This version is associated with the IDEAL 2025
  paper, it uses TensorFlow 1.x, FP32 precision and only applies sparse initialization.
- **models/s_gain_TFv2_INT8.py:** The S-GAIN imputer. This version uses TensorFlow 2.x and INT8 precision.
- **models/s_gain_TFv1_FP32.py:** The S-GAIN imputer. This version uses TensorFlow 1.x and FP32 precision.

####

- **monitors:** Contains the monitor file. Used for measuring things.
- **monitors/monitor.py:** The monitor file. Runs in a separate thread as to not interfere with S-GAIN.

####

- **output:** The output folder for the experiments.
- **output/[experiment].csv:** The imputed data for the experiment.
- **output/[experiment]_graphs.png:** A single png file containing all the graphs: RMSE, imputation time, memory usage,
  energy consumption, sparsity, FLOPs and loss (cross entropy and MSE).
- **output/[experiment]_log.json:** A log file of all measurements taken throughout the experiment.
- **output/[experiment]_model.json:** The imputed data for the specified experiment.

####

- **temp:** Contains temporary files.
- **temp/exp_bins:** Contains binary files used for logging measurements throughout the experiment.
- **temp/run_data:** Stores the experiment and filepaths. Used to automate running logs_and_graphs.py.
- **temp/sys_info.json:** Caches the system information.

####

- **utils:** Contains different utility files.
- **utils/flops:** Contains code to calculate FLOPs. (copied from Google Research)
- **utils/inits:** Contains files for initialization strategies.
- **utils/inits/s_gain_TFv2_INT8.py:** Contains all the different initialization strategies for the s_gain_TFv2_INT8
  version.
- **utils/inits/s_gain_TFv1_FP32.py:** Contains all the different initialization strategies for the s_gain_TFv1_FP32
  version.
- **modes:** Contains files for advanced training strategies (modalities).
- **utils/modes/s_gain_TFv2_INT8.py:** Contains all the different advanced training strategies (modalities) for the
  s_gain_TFv2_INT8 version.
- **utils/modes/s_gain_TFv1_FP32.py:** Contains all the different advanced training strategies (modalities) for the
  s_gain_TFv1_FP32 version.
- **utils/pruners:** Contains files for pruning strategies.
- **utils/pruners/s_gain_TFv2_INT8.py:** Contains all the different pruning strategies for the s_gain_TFv2_INT8 version.
- **utils/pruners/s_gain_TFv1_FP32.py:** Contains all the different pruning strategies for the s_gain_TFv1_FP32 version.
- **utils/regrowers:** Contains files for regrowing strategies.
- **utils/regrowers/s_gain_TFv2_INT8.py:** Contains all the different regrowing strategies for the s_gain_TFv2_INT8
  version.
- **utils/regrowers/s_gain_TFv1_FP32.py:** Contains all the different regrowing strategies for the s_gain_TFv1_FP32
  version.
- **utils/analysis.py:** Contains functions to analyze the experiments.
- **utils/data_loader.py:** Loads the datasets and introduces missingness in the data.
- **utils/graphs2.py:** An updated version of graphs.py: Plot all the relevant graphs to the same file.
- **utils/load_store.py:** Loads and stores files.
- **utils/metrics.py:** Calculates all the relevant metrics.
- **utils/utils.py:** Contains other utilities.

####

- **analyze.py:** This file is used to run the analysis of the experiments.
- **log_and_graphs.py:** This file is used to compile the temporary files into a single log file and to plot the
  corresponding graphs.
- **main.py:** The main file from which the experiments are run.
- **run_experiments.py:** This file enables running multiple experiments consecutively.