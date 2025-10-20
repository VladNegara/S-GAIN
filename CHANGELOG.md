# Changelog

This file marks the changes since the submission of the manuscript to the IDEAL 2025 conference.

## Pre-release v0.1.3-alpha (in development)

This version is currently being developed and seeks to implement full monitoring of system resources:
- Memory usage.
- Energy consumption (possible integration with HWMonitor on Windows to detect power draw).
- CPU utilization.
- CPU temperatures (to check for throttling).
- GPU utilization.
- GPU temperatures (to check for throttling).
- GPU VRAM usage.

Current major changes:
- Implemented imputation time analysis.
- Implemented auto shutdown for run_experiments.py.
- Improvements to the analysis.
- Immediately terminate failed experiments to speed up testing.
- Switched to a config based project structure.
- Other improvements to the project structure.
- Fixed bugs related to calling log_and_graphs.py from main.py or from the README.
- Better explanation of run_experiments.py in the README.

## Pre-release v0.1.2-alpha (11-09-2025)

This version saw major improvements to the analysis:
- Implemented loss monitoring (cross entropy and MSE).
- Now plots all graphs to a single file along with experiment information and system details.
- Plot sizing is now consistent for all number of subplots.
- Added verbosity to analyze.py and log_and_graphs.py.

## Pre-release v0.1.1-alpha (28-08-2025)

This version saw a major overhaul of the testing framework:
- Now specifies all experiment settings for the output files for easy manipulation of data and replication of results.
- Implemented imputation time monitoring.
- run_experiments.py replaces loop_main.py and solves the issue of TensorFlow not restarting.
- Converted the jupyter notebook to pure python and allow for automatic analysis.
- Restructured the project to improve comprehension.
- Significantly improved README.

## Pre-release v0.1.0-alpha (25-08-2025)

This version is associated with: B.P. van Oers, I. Baysal Erez, M. van Keulen, "Sparse GAIN: Imputation Methods to
Handle Missing Values with Sparse Initialization", IDEAL conference, 2025. And marks the beginning of the changelog.