import logging
import sys
from math import floor, log10
import re
from tqdm import tqdm
from pathlib import Path

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

from alfred.utils.config import config_to_str
from alfred.utils.directory_tree import DirectoryTree

COMMENTING_CHAR_LIST = ['#']


def create_logger(name, loglevel, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='a'))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def create_new_filehandler(logger_name, logfile):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(logger_name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setFormatter(formatter)

    return file_handler


def keep_two_signif_digits(x):
    try:
        if x == 0.:
            return x
        else:
            return round(x, -int(floor(log10(abs(x))) - 1))
    except:
        return x


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_management_objects(dir_tree, logger, pbar, config):
    # Creates directory tres

    if dir_tree is None:
        dir_tree = DirectoryTree(alg_name=config.alg_name,
                                 task_name=config.task_name,
                                 desc=config.desc,
                                 seed=config.seed,
                                 root=config.root_dir)

        dir_tree.create_directories()

    # Creates logger and prints config

    if logger is None:
        logger = create_logger('MASTER', config.log_level, dir_tree.seed_dir / 'logger.out')
    logger.debug(config_to_str(config))

    # Creates a progress-bar

    if pbar == "default_pbar":
        pbar = tqdm()

    if pbar is not None:
        pbar.n = 0
        pbar.desc += f'{dir_tree.storage_dir.name}/{dir_tree.experiment_dir.name}/{dir_tree.seed_dir.name}'
        pbar.total = config.max_episodes

    return dir_tree, logger, pbar


def check_params_defined_twice(keys):
    counted_keys = {key: keys.count(key) for key in keys}
    for key in counted_keys.keys():
        if counted_keys[key] > 1:
            raise ValueError(f'Parameter "{key}" appears {counted_keys[key]} times in the schedule.')


def is_commented(str_line, commenting_char_list):
    return str_line[0] in commenting_char_list


def select_storage_dirs(from_file, storage_name, root_dir):
    if from_file is not None:
        assert storage_name is None, "If launching --from_file, no storage_name should be provided"
        assert Path(from_file).suffix == '.txt', f"The provided --from_file should be a text file listing " \
                                                 f"storage_name's of directories to act on. " \
                                                 f"You provided '--from_file={from_file}'"

    if storage_name is not None:
        assert from_file is None, "Cannot launch --from_file if --storage_name"

    if from_file is not None:
        with open(from_file, "r") as f:
            storage_names = f.readlines()
        storage_names = [sto_name.strip('\n') for sto_name in storage_names]

        # drop the commented lignes in the .txt
        storage_names = [sto_name for sto_name in storage_names if not is_commented(sto_name, COMMENTING_CHAR_LIST)]

        storage_dirs = [root_dir / sto_name for sto_name in storage_names]

    elif storage_name is not None:

        storage_dirs = [root_dir / storage_name]

    else:
        raise NotImplementedError(
            "storage_dirs to operate over must be specified either by --from_file or --storage_name")

    return storage_dirs


def formatted_time_diff(total_time_seconds):
    n_hours = int(total_time_seconds // 3600)
    n_minutes = int((total_time_seconds - n_hours * 3600) // 60)
    n_seconds = int(total_time_seconds - n_hours * 3600 - n_minutes * 60)
    return f"{n_hours}h{str(n_minutes).zfill(2)}m{str(n_seconds).zfill(2)}s"


def uniquify(newfile_path):
    """
    Appends a number ID to newfile_path if files with same name (but different ID) already exist
    :param newfile_path (pathlib.Path): Full path to new file
    """
    assert isinstance(newfile_path, Path), f"newfile_path should be a pathlib.Path. Instead got {type(newfile_path)}"
    max_num = -1
    for existing_file in newfile_path.parent.iterdir():
        if newfile_path.stem in existing_file.stem and newfile_path.suffix == existing_file.suffix:
            str_end = str(existing_file.stem).split('_')[-1]
            if str_end.isdigit():
                num = int(str_end)
                if num > max_num:
                    max_num = num

    return newfile_path.parent / (newfile_path.stem + f"_{max_num + 1}" + newfile_path.suffix)


def get_95_confidence_interval(samples, method):
    if method == "stderr":
        mean = samples.mean(-1)
        number_of_samples = len(samples)
        samples_std = samples.std(-1)
        samples_stderr = 1.96 * samples_std / (number_of_samples ** 0.5)
        err_up = samples_stderr
        err_down = samples_stderr

    elif method == "bootstrapped_CI":
        bootstrapped_result = bs.bootstrap(samples, stat_func=bs_stats.mean)
        mean = bootstrapped_result.value
        err_up = bootstrapped_result.upper_bound - bootstrapped_result.value
        err_down = bootstrapped_result.value - bootstrapped_result.lower_bound

    else:
        raise NotImplementedError(method)

    return mean, err_up, err_down


def get_95_confidence_interval_of_sequence(list_of_samples, method):
    # list_of_samples must be of shape (n_time_steps, n_samples)
    means = []
    err_ups = []
    err_downs = []
    for samples in list_of_samples:
        mean, err_up, err_down = get_95_confidence_interval(samples=samples, method=method)
        means.append(mean)
        err_ups.append(err_up)
        err_downs.append(err_down)
    return np.asarray(means), np.asarray(err_ups), np.asarray(err_downs)


def plot_sampled_hyperparams(ax, param_samples, log_params):
    cm = plt.cm.get_cmap('viridis')
    for i, param in enumerate(param_samples.keys()):
        args = param_samples[param], np.zeros_like(param_samples[param])
        kwargs = {'linestyle': '', 'marker': 'o', 'label': param, 'alpha': 0.2,
                  'color': cm(float(i) / float(len(param_samples)))}
        if param in log_params:
            ax[i].semilogx(*args, **kwargs)
        else:
            ax[i].plot(*args, **kwargs)
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[i].get_yaxis().set_ticks([])
        ax[i].legend(loc='upper right')
