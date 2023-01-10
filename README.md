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
alias alplot='python -m alfred.make_plot_arrays'
alias alretrain='python -m alfred.create_retrainbest'
alias albench='python -m alfred.benchmark'
alias alsync='python -m alfred.sync_wandb'
alias alcopy='python -m alfred.copy_config'
alias alupdate='python -m alfred.update_config_unique'
```

## Content

    ├─── alfred
    │
    |    └─── defaults.py
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
* Some important default configurations are defined in `alfred.defaults.py` as global variables and can be overwritten on the ML side by simply reassigning them inside a function called `main.set_up_alfred()`.


## Usage

### alfred's utils

Simply use as any other package, e.g:

> from alfred.utils import *

### alfred's scripts

There are some structural requirements that `alfred` expects in order to be able to interact with your machine learning codebase. Say my main folder is called `my_ml_project`, it should contain:

  1. a file called `main.py`
  2. a function `main.get_run_args(overwritten_cmd_line)` that defines the hyperparameters for this project
  3. a function `main.main(config, dir_tree, logger, pbar)` that launches an experiment with the specified hyperparameters
  4. a folder named `schedules`. A schedule is just a folder that contains everything that defines a hyperparameter-search, mainly, its schedule_file (e.g. `random_schedule_mySearch.py`) but also text files listing which result directories belong to that search, json files for defining some markers, colors and labels for the algorithms in the search, etc. See `alfred/schedule_examples/`.
  5. [OPTIONAL] a function `main.set_up_alfred()` which sets the default values used by alfred (see in alfred/defaults.py)

That being in place, you can use alfred's scripts to prepare, launch, clean these hyperparameter searches. To use any of the scripts, simply call it from `my_ml_project`. For example:

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

The directory-tree used by alfred is defined in the class `alfred.utils.directory_tree.DirectoryTree`. An example of how it could be laid out for a Reinforcement Learning experiment is shown below. Note that all these files would be automatically created either by `alfred's scripts` or by `my_ml_project`.

```
    ├─── root_dir
    │
    │    └─── Ju1_f7b375e-58332a7_ppo_cartpole_random_benchmarkv1
    │    └─── Ju2_f7b375e-58332a7_ppo_mountaincar_random_benchmarkv1
    │    └─── Ju3_f7b375e-58332a7_sac_cartpole_random_benchmarkv1
    |         └─── experiment1
    |         └─── experiment2
    |              └─── seed123
    |                   └─── config.json
    |                   └─── config_unique.json
    |                   └─── UNHATCHED
    |                   └─── model.pt
    |              └─── seed456
    |                   └─── config.json
    |                   └─── config_unique.json
    |                   └─── COMPLETED
    |                   └─── logger.out
    |                   └─── graph.png
    |                   └─── recorders
    |                        └─── train_recorder.pkl
    |                        └─── other_data.pkl
    |              └─── seed789
    |                   └─── config.json
    |                   └─── config_unique.json
    |                   └─── CRASH.txt
    |                   └─── logger.out
    |         └─── experiment3
    |         └─── experiment4
    |         └─── experiment5
    |         └─── eval_return_over_episodes.png
    |         └─── PLOT_ARRAYS_COMPLETED
    │    └─── Ju4_f7b375e-58332a7_sac_mountaincar_random_benchmarkv1
```
The whole directory-tree is a result of `alfred.prepare_schedule`. It uses a file defining your search and creates the experiment directories accordingly (see `alfred/schedules_examples` for an example of such files).

* `root_dir`: Root-directory. By default it uses `DirectoryTree.default_root`. This default can be overwrited when importing alfred in `my_ml_project/main.set_up_alfred()`, or the `--root_dir` can be passed in argument to all `alfred's scripts`.
* `Ju1_f7b375e-58332a7_ppo_cartpole_random_benchmarkv3`: Storage-directory. It is composed of:
  * `Ju1`: the storage-id (defined automatically from git-username and ordinal numbering)
  * `f7b375e-58332a7`: git-hashes of packages being tracked by alfred. These are defined in `my_ml_project` by giving the path to the .git file to alfred in your function main.set_up_alfred(), e.g: `alfred.defaults.DEFAULT_DIRECTORY_TREE_GIT_REPOS_TO_TRACK['mlProject'] = str(Path(__file__).absolute().parents[0])`.
  * `ppo`: Algorithm-name. Defined in schedule-file and `my_ml_project`.
  * `cartpole`: Task-name. Defined in schedule-file and `my_ml_project`.
  * `random`: Search-type. Defined in `alfred.prepare_schedule` from the provided `schedule_file`.
  * `benchmarkv1`: Description. Passed as argument to `alfred.prepare_schedule`.
* `experiment1`: Experiment-directory. All leaves of an experiment-dir have the same `config.json` except for the `seed`.
* `seed123`: Seed-directory. The folder for each particular (unique) run. See it as an egg ready to hatch. These eggs are prepared by `alfred.prepare_schedule`, and they will be executed by `alfred.launch_schedule`. In this example, we see that `seed123` has not been run yet, `seed456` has completed and `seed789` has crashed.

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
