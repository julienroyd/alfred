# alfred

Just some boilerplate code for loggers, plots and the like as well as a collection of useful scripts for preparing and launching hyperparameter searches and aggregating results. We use `alfred` for machine learning experiments and try to keep it as project-agnostic as possible.

![good_old_alfred](alfred.jpg)

## Installation

  > git clone https://github.com/julienroyd/alfred.git
  
  > pip install -e .
  
## Useful aliases

To make using alfred as seamless as possible, add the followings to your `.bachrc`:
```
alias alprep='python -m alfred.prepare_schedule'
alias allaunch='python -m alfred.launch_schedule'
alias alclean='python -m alfred.clean_interrupted'
alias alplot='python -m alfred.make_comparative_plots'
alias alretrain='python -m alfred.create_retrainbest'
alias albench='python -m alfred.benchmark'
alias alsync='python -m alfred.sync_wandb'
alias alcopy='python -m alfred.copy_config'
```

## Content

    ├─── alfred
    │
    │    └─── benchmark.py
    │    └─── clean_interrupted.py
    │    └─── copy_config.py
    │    └─── create_retrainbest.py
    │    └─── launch_schedule.py
    │    └─── make_plot_arrays.py
    │    └─── prepare_schedule.py
    │    └─── synch_wandb.py
    │
    │    └─── schedules_examples
    |
    |         └─── gridSearch_example1
    │              └─── grid_schedule_example1.py
    |         └─── randomSearch_example1
    │              └─── random_schedule_example1.py
    │
    │    └─── utils
    |
    │         └─── config.py
    │         └─── directory_tree.py
    │         └─── misc.py
    │         └─── plots.py
    │         └─── recorder.py
    │         └─── stats.py

This repository contains two different group of files: 

* Experiment management scripts directly under `alfred`. They are meant to help manage folder creation, experiment launching and results aggregation. See next section for usage. We refer to them as << alfred's scripts >>.
* Common functions for plots, directory trees, loggers, argparsers and the like, located under `alfred.utils`. We refer to them as << alfred's utils >>.


## Usage

### alfred's utils

Simply use as any other package, e.g:

> from alfred.utils import *

### alfred's scripts

There are some structural requirements that `alfred` expects in order to be able to interact with your machine learning codebase. Say my main folder is called `my_ml_project`, it should contain:

  1. a file called `main.py`
  2. a function `main.get_run_args(overwritten_cmd_line)` that defines the hyperparameters for this project
  3. a function `main.main(config, dir_tree, logger, pbar)` that launches an experiment with the specified hyperparameters
  4. a folder named `schedules` that contains two files (`grid_schedule.py` and `random_schedule.py`) that specify what combinations of hyperparameters to vary over for these two types of searches. These are project-specific and therefore should be located in `my_ml_project`. However, an example of such files is included in `alfred/schedule_examples/`.

To use any of the scripts, simply call it from `my_ml_project`. For example:

> python -m alfred.prepare_schedule --schedule_file=schedules/gridSearchExample/grid_schedule_gridSearchExample.py --desc=abc

For a description of their purpose and their arguments, please refer to the help command, e.g:

> python -m alfred.prepare_schedule --help

## Typical usage

**1. Create the search folders:**

```
python -m alfred.prepare_schedule --schedule_file=schedules/benchmarkExample/random_schedule_benchmarkExample.py 
                                  --desc benchmarkExample
```

**2. Launch the searches:**

```
python -m alfred.launch_schedule --from_file schedules/benchmarkExample/list_searches_benchmarkExample.txt
```

**3. Create the folders to retrain the best configuration from the search:**

```
python -m alfred.create_retrainbest --from_file schedules/benchmarkExample/list_searches_benchmarkExample.txt
```

**4. Launch the retrainbest:**

```
python -m alfred.launch_schedule --from_file schedules/benchmarkExample/list_retrains_benchmarkExample.txt
```

**5. Benchmark the searches:**
```
python -m alfred.benchmark --from_file schedules/benchmarkExample/list_searches_benchmarkExample.txt 
                           --benchmark_type compare_searches
```

**6. Benchmark the retrains:**
```
python -m alfred.benchmark --from_file schedules/benchmarkExample/list_retrains_benchmarkExample.txt 
                           --benchmark_type compare_models
```

## Key mechanisms used by alfred

The spirit of this codebase is to have project-agnostic scripts that can be called from anywhere, to create folders, launch experiments in parallel and communicate asynchronously through FLAG-files in order to know which experiments are completed, which ones are left to run and which ones have crashed. This framework uses the fact that the directory-tree is known from `alfred` (see `alfred.utils.directory_tree.py`). 

### Directory Tree

The directory-tree used by alfred is defined in the class `alfred.utils.directory_tree.DirectoryTree`.

TODO: add description

### FLAG-files

#### FLAG-files for seed-directories

There are two main flag files present in seed-directories: 
  * `UNHATCHED`: signals that this run has not been launched yet
  * `COMPLETED`: signals that this run has reached termination without crash
  * `CRASH.txt`: signals that the run from this config has crashed and contains the error message

A seed-directory that does not contain any FLAG-file can be explained in two ways:
  1. It is currently being runned (a process is executing this config and hasn't finished yet)
  2. The process running this config has been killed (e.g. by a cluster's slurm system) without having completed its task

Such a seed-directory (containing no FLAG-file) will be identified as `MYSTERIOUSLY STOPPED` by `alfred.clean_interrupted.py` and will be cleaned to its initial state.

#### FLAG-files for summaries

Other flag files are used by `alfred.launch_schedule.py` to signal that a storage-directory is being summarised by a process, so that other process can move on to another storage-directory. These are:
  * `PLOT_ARRAYS_ONGOING`, `PLOT_ARRAYS_COMPLETED`
  * `SUMMARY_ONGOING`, `SUMMARY_COMPLETED`
