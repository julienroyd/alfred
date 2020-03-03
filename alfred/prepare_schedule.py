# USAGE
# TODO: write description of how to use it (maybe in the help from argparse?)

# ASSUMPTIONS: alfred.prepare_schedule assumes the following structure:
# 1. a folder named 'schedules' containing two files: grid_schedule.py and random_schedule.py
# 2. in each of these folders, a function named get_run_args() that returns the hyperparameters to be used

import logging
import sys
import itertools
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alfred.utils.directory_tree import DirectoryTree, get_root
from alfred.utils.config import save_dict_to_json, load_dict_from_json, save_config_to_json, config_to_str, parse_bool
from alfred.utils.plots import plot_sampled_hyperparams
from alfred.utils.misc import create_logger


def get_prepare_schedule_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default=None)
    parser.add_argument('--search_type', type=str, choices=['grid', 'random'], required=True)
    parser.add_argument('--n_experiments', type=int, default=50)
    parser.add_argument('--root_dir', default=None, type=str)
    parser.add_argument('--add_to_folder', type=str, default=None)
    parser.add_argument('--resample', type=parse_bool, default=True,
                        help="If true we resample a configuration for each task*alg combination")

    return parser.parse_args()


def extract_schedule_grid():
    try:
        from schedules.grid_schedule import VARIATIONS, ALG_NAMES, TASK_NAMES, SEEDS, get_run_args
    except ImportError as e:
        raise ImportError(
            f"alfred.prepare_schedule assumes the following structure (see alfred/schedules_examples):"
            f"\n\t1. a folder named 'schedules' containing two files: grid_schedule.py and random_schedule.py"
            f"\n\t2. in each of these folders, a function named get_run_args() that returns the hyperparameters to be used"
        )

    # Transforms our dictionary of lists (key: list of values) into a list of lists of tuples (key, single_value)

    VARIATIONS_LISTS = []
    sorted_keys = sorted(VARIATIONS.keys(), key=lambda item: (len(VARIATIONS[item]), item), reverse=True)
    for key in sorted_keys:
        VARIATIONS_LISTS.append([(key, VARIATIONS[key][j]) for j in range(len(VARIATIONS[key]))])

    # Creates a list of combinations of hyperparams given in VARIATIONS (to grid-search over)

    experiments = list(itertools.product(*VARIATIONS_LISTS))

    # Convert to list of dicts

    experiments = [dict(experiment) for experiment in experiments]

    # Checks which hyperparameter are actually varied

    varied_params = [k for k in VARIATIONS.keys() if len(VARIATIONS[k]) > 1]

    return VARIATIONS, ALG_NAMES, TASK_NAMES, SEEDS, experiments, varied_params, get_run_args


def extract_schedule_random(n_experiments):
    try:
        from schedules.random_schedule import sample_experiment, ALG_NAMES, TASK_NAMES, SEEDS, get_run_args
    except ImportError as e:
        raise ImportError(
            f"alfred.prepare_schedule assumes the following structure (see alfred/schedules_examples):"
            f"\n\t1. a folder named 'schedules' containing two files: grid_schedule.py and random_schedule.py"
            f"\n\t2. in each of these folders, a function named get_run_args() that returns the hyperparameters to be used"
        )

    # Samples all experiments' hyperparameters

    experiments = [sample_experiment().items() for _ in range(n_experiments)]

    # Convert to list of dicts

    experiments = [dict(experiment) for experiment in experiments]

    # Checks which hyperparams are actually varied

    param_samples = {param_name: [] for param_name in experiments[0].keys()}
    for experiment in experiments:
        for param_name in experiment.keys():
            param_samples[param_name].append(experiment[param_name])

    non_varied_params = []
    for param_name in param_samples.keys():
        if len(param_samples[param_name]) == param_samples[param_name].count(param_samples[param_name][0]):
            non_varied_params.append(param_name)

    for param_name in non_varied_params:
        del param_samples[param_name]
    varied_params = list(param_samples.keys())

    return param_samples, ALG_NAMES, TASK_NAMES, SEEDS, experiments, varied_params, get_run_args


def create_experiment_dir(desc, alg_name, task_name, param_dict, varied_params, storage_name_id, SEEDS,
                          root_dir, get_run_args, git_hashes=None):
    # Creates dictionary pointer-access to a training config object initialized by default

    config = get_run_args(overwritten_cmd_line="")
    config_dict = vars(config)

    # Modifies the config for this particular experiment

    for param_name in param_dict.keys():
        if param_name not in config_dict.keys():
            raise ValueError(f"'{param_name}' taken from the schedule is not a valid hyperparameter "
                             f"i.e. it cannot be found in the Namespace returned by get_run_args().")
        else:
            config_dict[param_name] = param_dict[param_name]

    tmp_dir_tree = DirectoryTree(id=storage_name_id, alg_name=alg_name, task_name=task_name, desc=desc, seed=1,
                                 git_hashes=git_hashes, root=root_dir)

    experiment_num = int(tmp_dir_tree.experiment_dir.name.strip('experiment'))

    # For each seed in these experiments, creates a directory

    for seed in SEEDS:
        config.seed = seed
        config.alg_name = alg_name
        config.task_name = task_name
        config.desc = desc

        # Creates the experiment directory

        dir_tree = DirectoryTree(id=storage_name_id,
                                 alg_name=config.alg_name,
                                 task_name=config.task_name,
                                 desc=config.desc,
                                 seed=config.seed,
                                 experiment_num=experiment_num,
                                 git_hashes=git_hashes,
                                 root=root_dir)

        dir_tree.create_directories()

        # Saves the set of unique variations as json file (to easily identify the uniqueness of this experiment)

        config_unique_dict = {k: v for k, v in param_dict.items() if k in varied_params}
        config_unique_dict['alg_name'] = alg_name
        config_unique_dict['task_name'] = task_name
        config_unique_dict['seed'] = seed
        save_dict_to_json(config_unique_dict, filename=str(dir_tree.seed_dir / 'config_unique.json'))

        # Saves the config as json file (to be run later)

        save_config_to_json(config, filename=str(dir_tree.seed_dir / 'config.json'))

        # Creates empty file UNHATCHED meaning that the experiment is ready to be run

        open(str(dir_tree.seed_dir / 'UNHATCHED'), 'w+').close()

    return dir_tree


def prepare_schedule(desc, add_to_folder, search_type, n_experiments, ask_for_validation, resample, logger, root_dir):
    # Gets experiments parameters

    if search_type == 'grid':

        VARIATIONS, ALG_NAMES, TASK_NAMES, SEEDS, experiments, varied_params, get_run_args = extract_schedule_grid()

    elif search_type == 'random':

        param_samples, ALG_NAMES, TASK_NAMES, SEEDS, experiments, varied_params, get_run_args = extract_schedule_random(n_experiments)
        experiments = [experiments]
        param_samples = [param_samples]
        n_combinations = len(ALG_NAMES) * len(TASK_NAMES)

        if resample:
            assert not add_to_folder
            for i in range(n_combinations - 1):
                param_sa, _, _, _, expe, varied_pa, get_run_args = extract_schedule_random(n_experiments)
                experiments.append(expe)
                param_samples.append(param_sa)

        else:
            experiments = experiments * n_combinations
            param_samples = param_samples * n_combinations

    else:
        raise NotImplementedError

    # Creates a list of alg_agent and task_name unique combinations

    if desc is not None:
        assert add_to_folder is None, "If --desc is defined, a new storage_dir folder will be created." \
                                      "No --add_to_folder should be provided."

        desc = f"{search_type}_{desc}"
        agent_task_combinations = list(itertools.product(ALG_NAMES, TASK_NAMES))
        mode = "NEW_STORAGE"

    elif add_to_folder is not None:
        assert (get_root(root_dir) / add_to_folder).exists(), f"{add_to_folder} does not exist."
        assert desc is None, "If --add_to_folder is defined, new experiments will be added to the existing folder." \
                             "No --desc should be provided."

        storage_name_id, git_hashes, alg_name, task_name, desc = \
            DirectoryTree.extract_info_from_storage_name(add_to_folder)

        agent_task_combinations = list(itertools.product([alg_name], [task_name]))
        mode = "EXISTING_STORAGE"

    else:
        raise NotImplementedError

    # Printing summary of schedule_xyz.py

    info_str = f"\n\nPreparing a {search_type.upper()} search over {len(experiments)} experiments, {len(SEEDS)} seeds"
    info_str += f"\nALG_NAMES: {ALG_NAMES}"
    info_str += f"\nTASK_NAMES: {TASK_NAMES}"
    info_str += f"\nSEEDS: {SEEDS}"

    if search_type == "grid":
        info_str += f"\n\nVARIATIONS:"
        for key in VARIATIONS.keys():
            info_str += f"\n\t{key}: {VARIATIONS[key]}"
    else:
        info_str += f"\n\nParams to be varied over: {varied_params}"

    info_str += f"\n\nDefault {config_to_str(get_run_args(overwritten_cmd_line=''))}\n"

    logger.debug(info_str)

    # Asking for user validation

    if ask_for_validation:

        if mode == "NEW_STORAGE":
            git_hashes = DirectoryTree.get_git_hashes()

            string = "\n"
            for alg_name, task_name in agent_task_combinations:
                string += f"\n\tID(to be determined)_{git_hashes}_{alg_name}_{task_name}_{desc}"
            logger.debug(f"\n\nAbout to create {len(agent_task_combinations)} storage directories, "
                         f"each with {len(experiments)} experiments:"
                         f"{string}")

        else:
            n_existing_experiments = len([path for path in get_root(root_dir) / add_to_folder.iterdir()
                                          if path.name.startswith('experiment')])

            logger.debug(f"\n\nAbout to add {len(experiments)} experiment folders in the following directory"
                         f" (there are currently {n_existing_experiments} in this folder):"
                         f"\n\t{add_to_folder}")

        answer = input("\nShould we proceed? [y or n]")
        if answer.lower() not in ['y', 'yes']:
            logger.debug("Aborting...")
            sys.exit()

    logger.debug("Starting...")

    # For each storage_dir to be created

    for i, (alg_name, task_name) in enumerate(agent_task_combinations):

        # Creates the experiment directories...

        if mode == "NEW_STORAGE":

            # ... in a new storage_dir

            tmp_dir_tree = DirectoryTree(alg_name=alg_name, task_name=task_name, desc=desc, seed=1, root=root_dir)
            storage_name_id = tmp_dir_tree.storage_dir.name.split('_')[0]

            for param_dict in experiments:
                dir_tree = create_experiment_dir(desc, alg_name, task_name, param_dict,
                                                 varied_params, storage_name_id, SEEDS,
                                                 root_dir, get_run_args, git_hashes)

        else:

            # ... in an existing storage_dir

            for param_dict in experiments:
                dir_tree = create_experiment_dir(desc, alg_name, task_name, param_dict,
                                                 varied_params, storage_name_id, SEEDS,
                                                 root_dir, get_run_args)

        # Saves VARIATIONS in the storage directory

        first_experiment_created = int(dir_tree.current_experiment.strip('experiment')) - len(experiments) + 1
        last_experiment_created = first_experiment_created + len(experiments) - 1

        if search_type == 'grid':

            VARIATIONS['alg_name'] = ALG_NAMES
            VARIATIONS['task_name'] = TASK_NAMES
            VARIATIONS['seed'] = SEEDS

            key = f'{first_experiment_created}-{last_experiment_created}'

            if (dir_tree.storage_dir / 'variations.json').exists():
                variations_dict = load_dict_from_json(filename=str(dir_tree.storage_dir / 'variations.json'))
                assert key not in variations_dict.keys()
                variations_dict[key] = VARIATIONS
            else:
                variations_dict = {key: VARIATIONS}

            save_dict_to_json(variations_dict, filename=str(dir_tree.storage_dir / 'variations.json'))
            open(str(dir_tree.storage_dir / 'GRID_SEARCH'), 'w+').close()

        elif search_type == 'random':

            fig, ax = plt.subplots(len(param_samples[i]), 1, figsize=(6, 2 * len(param_samples[i])))
            if not hasattr(ax, '__iter__'):
                ax = [ax]

            plot_sampled_hyperparams(ax, param_samples[i])

            j = 1
            while True:
                if (dir_tree.storage_dir / f'variations{j}.png').exists():
                    j += 1
                else:
                    break
            fig.savefig(str(dir_tree.storage_dir / f'variations{j}.png'))
            plt.close(fig)

            open(str(dir_tree.storage_dir / 'RANDOM_SEARCH'), 'w+').close()

        # Printing summary

        logger.info(f'Created directories '
                    f'{str(dir_tree.storage_dir)}/experiment{first_experiment_created}-{last_experiment_created}')

    logger.info(f"\nEach of these experiments contain directories for the following seeds: {SEEDS}")


if __name__ == '__main__':
    logger = create_logger(name="PREPARE_SCHEDULE - MAIN", loglevel=logging.DEBUG)
    kwargs = vars(get_prepare_schedule_args())
    prepare_schedule(**kwargs, logger=logger, ask_for_validation=True)
