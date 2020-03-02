# alfred

Just some boilerplate code for loggers, plots and the like as well as a collection of useful scripts that I use for preparing and launching hyperparameter searches and aggregating results. I use it for machine learning experiments and try to keep it as project-agnostic as possible.

![good_old_alfred](alfred.jpg)

## Requirements

`python >= 3.6`, `matplotlib`, `seaborn`, `tqdm`.

## Installation

  > git clone https://github.com/julienroyd/alfred.git
  
  > pip install -e .

## Content

    ├─── alfred
    │
    │    └─── benchmark.py
    │    └─── clean_interrupted.py
    │    └─── copy_experiment.py
    │    └─── create_retrain_best.py
    │    └─── make_plot_array.py
    │    └─── prepare_schedule.py
    │    └─── run_schedule.py
    │
    │    └─── schedules_examples
    │
    │         └─── grid_schedule.py
    │         └─── random_schedule.py
    │
    │    └─── utils
    |
    │         └─── config.py
    │         └─── directory_tree.py
    │         └─── misc.py
    │         └─── plots.py
    │         └─── recorder.py

This repository contains two different type of boilerplate code: 

* Scripts directly under `alfred`. They are pretty much standalone and meant to help manage folder creation, experiment launching and results aggregation. See next section for usage. 
* Common functions for plots, directory trees, loggers, argparsers and the like are located under `alfred.utils`. 


## Usage


### To use the boilerplate code

Simply use as any other package, e.g:

> from alfred.utils import *

### To use the experiment management scripts

There are some structural requirements that `alfred` expects in order to be able to interact with your machine learning codebase. Say your main folder is called `my_ml_project`, it should contain:

  1. a file called `main.py`
  2. a function `main.get_main_args(overwritten_cmd_line)` that defines the hyperparameters for this project
  3. a function `main.main(config, dir_tree, logger, pbar)` that launches an experiment with the specified hyperparameters
  
  4. a folder named `schedules` containing two files (`grid_schedule.py` and `random_schedule.py`) that specify what combinations of hyperparameters to vary over for these two types of searches. These are project-specific and therefore should be located in `my_ml_project`. However, an example of such files is included in `alfred/schedule_examples/`.

To use any of the scripts, simply call it from `my_ml_project`. For example:

> python -m alfred.prepare_schedule --desc abc_xyz

For a description of their purpose and their arguments, please refer to the help command, e.g:

> python -m alfred.prepare_schedule --help

## More detailed documentation

The spirit of this codebase is to have project-agnostic that can be called independently, from anywhere, to create folders, launch experiments in parallel and communicate asynchronously through FLAG-files in order to know which experiments are completed, left to run or have crashed. This framework uses the fact that the directory-tree is known from `alfred` (see `alfred.utils.directory_tree.py`). 

### Directory Tree

TODO: add description

### FLAG-files

TODO: add description

### Main Experiment Management scripts

* **prepare_schedule.py**

TODO: add description

* **run_schedule.py**

TODO: add description

### Other Experiment Management scripts

* **clean_interrupted.py**

TODO: add description

* **copy_config.py**

TODO: add description

* **make_comparative_plots.py**

TODO: add description
