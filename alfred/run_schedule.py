# USAGE
# TODO: write description of how to use it (maybe in the help from argparse?)

# ASSUMPTION: this module (alfred) assumes that the directory from which it is called contains:
# 1. a file named 'main.py'
# 2. a function 'main.get_run_args()' that defines the hyperparameters for this project
try:
    from main import main, get_run_args
except ImportError as e:
    raise ImportError(
        f"{e.msg}\n"
        f"This module (alfred) assumes that the directory from which it is called contains:"
        f"\n\t1. a file named 'main.py'"
        f"\n\t2. a function 'main.get_run_args(overwritten_cmd_line)' that defines the hyperparameters for this project"
        f"\n\t3. a function 'main.main(config, dir_tree, logger, pbar)' that runs the project with the specified hyperparameters"
    )

# other imports
import numpy as np
import traceback
import datetime
import argparse
from multiprocessing import Process
import time
import logging
from tqdm import tqdm

from alfred.utils.config import load_config_from_json, parse_bool
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, create_new_filehandler
from alfred.make_comparative_plots import create_comparative_figure
from alfred.clean_interrupted import clean_interrupted
from alfred.benchmark import summarize_search


def get_run_schedule_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_name', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--n_experiments_per_proc', type=int, default=np.inf)
    parser.add_argument('--use_pbar', type=parse_bool, default=False)
    parser.add_argument('--check_hash', type=parse_bool, default=True)
    parser.add_argument('--run_over_tasks', type=parse_bool, default=False,
                        help="If true, subprocesses will look for unhatched seeds in all storage_dir"
                             "that have the same hashes, 'alg_name', 'desc' but different 'task_name'")
    parser.add_argument('--run_clean_interrupted', type=parse_bool, default=False,
                        help="Will clean mysteriously stopped seeds to be re-runned, but not crashed experiments")
    parser.add_argument('--root_dir', default=None, type=str)

    return parser.parse_args()


def _run_schedule(storage_dirs, n_processes, n_experiments_per_proc, use_pbar, logger, root_dir, process_i=0):
    call_i = 0

    try:

        time.sleep(np.random.uniform(low=0., high=1.5))

        # For all storage_dirs...

        for storage_dir in storage_dirs:

            # Gets unhatched seeds directories for the current storage_dir

            unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')

            while len(unhatched_seeds) > 0:

                # Checks if that process didn't exceed its number of experiments to run

                if call_i >= n_experiments_per_proc:
                    logger.info(f"I reached my limit of {n_experiments_per_proc} experiments.")
                    break

                # Select the next seed directory

                unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')

                if len(unhatched_seeds) > 0:
                    seed_dir = unhatched_seeds[0]
                else:
                    logger.info(f"{storage_dir.name}: No more unhatched seeds")
                    break

                # Removes its unhatched flag

                try:
                    os.remove(str(seed_dir / 'UNHATCHED'))
                except:
                    logger.info(f"{seed_dir}: Already hatched")
                    continue

                # Load the config and try to train the model

                try:
                    config = load_config_from_json(str(seed_dir / 'config.json'))
                    dir_tree = DirectoryTree.init_from_seed_path(seed_dir, root=root_dir)

                    experiment_logger = create_logger(
                        name=f'PROCESS{process_i}:'
                             f'{dir_tree.storage_dir.name}/'
                             f'{dir_tree.experiment_dir.name}/'
                             f'{dir_tree.seed_dir.name}',
                        loglevel=logging.INFO,
                        logfile=dir_tree.seed_dir / 'logger.out',
                        streamHandle=not (use_pbar)
                    )

                    if use_pbar:
                        pbar = tqdm(position=process_i + (1 + n_processes) * call_i)
                        pbar.desc = f"PROCESS{process_i}:"
                    else:
                        pbar = None

                    logger.info(
                        f"{dir_tree.storage_dir.name}/{dir_tree.experiment_dir.name}/{dir_tree.seed_dir.name} - "
                        f"Launching...")

                    main(config=config, dir_tree=dir_tree, logger=experiment_logger, pbar=pbar)

                    open(str(seed_dir / 'COMPLETED'), 'w+').close()
                    call_i += 1

                except Exception as e:
                    with open(str(seed_dir / 'CRASH.txt'), 'w+') as f:
                        f.write(f'Crashed at : {datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
                        f.write(str(e))
                        f.write(traceback.format_exc())

            # If all experiments have been completed (or at least crashed but have been attempted)...

            all_seeds = get_all_seeds(storage_dir)
            unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
            crashed_seeds = get_some_seeds(storage_dir, file_check='CRASH.txt')
            completed_seeds = get_some_seeds(storage_dir, file_check='COMPLETED')

            if len(unhatched_seeds) == 0 and len(crashed_seeds) == 0 and len(completed_seeds) == len(all_seeds):

                # Creates comparative plots

                if not (storage_dir / 'MAKING_COMPARATIVE_PLOTS').exists():
                    open(str(storage_dir / 'MAKING_COMPARATIVE_PLOTS'), 'w+').close()
                    logger.info(f"{storage_dir.name}: MAKING COMPARATIVE PLOTS")

                    try:
                        create_comparative_figure(storage_dir, logger=logger)

                    except Exception as e:
                        logger.info(f"{type(e)}: unable to plot comparative graphs"
                                    f"\n\n{e}\n{traceback.format_exc()}")

                    os.remove(str(storage_dir / 'MAKING_COMPARATIVE_PLOTS'))

                # If all experiments are completed benchmark them

                if all_seeds == completed_seeds and not (storage_dir / "summary" / "SUMMARY_ONGOING").exists():

                    if not (storage_dir / "summary" / "SUMMARY_ONGOING").exists() \
                            and not (storage_dir / "summary" / "SUMMARY_COMPLETED").exists():
                        os.makedirs(str(storage_dir / "summary"), exist_ok=True)
                        open(str(storage_dir / "summary" / 'SUMMARY_ONGOING'), 'w+').close()
                        logger.info(f"{storage_dir.name}: SUMMARIZING SEARCH")

                        try:
                            summarize_search(storage_name=storage_dir.name,
                                             x_metric="episode",
                                             y_metric="eval_return",
                                             n_eval_runs=None,
                                             performance_metric="avg_eval_return",
                                             performance_aggregation="last",
                                             re_run_if_exists=False,
                                             make_performance_chart=True,
                                             make_learning_plots=True,
                                             logger=logger,
                                             root_dir=root_dir)

                            os.remove(str(storage_dir / "summary" / 'SUMMARY_ONGOING'))
                            open(str(storage_dir / "summary" / 'SUMMARY_COMPLETED'), 'w+').close()

                        except Exception as e:
                            logger.info(f"{type(e)}: unable to run 'summarize_search'"
                                        f"\n{e}\n{traceback.format_exc()}")

                            os.remove(str(storage_dir / "summary" / 'SUMMARY_ONGOING'))
                            open(str(storage_dir / "summary" / 'SUMMARY_FAILED'), 'w+').close()

        logger.info(f"Done. Shutting down.")

    except Exception as e:
        logger.info(f"The process CRASHED with the following error:\n{e}")

    return call_i


def launch_run_schedule(storage_name, n_processes, n_experiments_per_proc, use_pbar, check_hash, run_over_tasks,
                        run_clean_interrupted, root_dir):
    # Creates logger

    storage_dir = get_root(root_dir) / storage_name
    master_logger = create_logger(name=f'RUN_SCHEDULE - MASTER', loglevel=logging.INFO,
                                  logfile=storage_dir / 'run_schedule_logger.out', streamHandle=True)

    # Log some info

    master_logger.info(f"\n\n{'=' * 200}\n"
                       f"\nRunning schedule for:\n"
                       f"\nstorage_name={storage_name}"
                       f"\nn_processes={n_processes}"
                       f"\nn_experiments_per_proc={n_experiments_per_proc}"
                       f"\nuse_pbar={use_pbar}"
                       f"\ncheck_hash={check_hash}"
                       f"\nrun_over_tasks={run_over_tasks}"
                       f"\nroot={get_root(root_dir)}"
                       f"\n")

    # Checks if code hash matches the folder to be run_schedule

    if check_hash:

        current_git_hashes = {}
        for name, repo in DirectoryTree.git_repos_to_track.items():
            current_git_hashes[name] = get_git_hash(path=repo)

        _, storage_name_git_hashes, _, _, _ = DirectoryTree.extract_info_from_storage_name(storage_name)
        storage_name_git_hash_list = storage_name_git_hashes.split("-")

        for i, (name, hash) in enumerate(current_git_hashes.items()):
            if hash not in storage_name_git_hash_list:
                raise ValueError(f"Repository '{name}' hash is: {hash}. "
                                 f"You are running a storage_dir with '{name}' hash: {storage_name_git_hash_list[i]}")

    # Select storage_dirs to run over

    storage_dir = get_root(root_dir) / storage_name

    if run_over_tasks:
        storage_dirs = get_storage_dirs_across_tasks(storage_dir, root_dir=root_dir)
    else:
        storage_dirs = [storage_dir]

    master_logger.info("Storage Directories to be runned:")
    for storage_dir in storage_dirs:
        master_logger.info(storage_dir)

    # Clean the storage_dirs if asked to

    if run_clean_interrupted:
        for storage_dir in storage_dirs:
            clean_interrupted(storage_name=storage_dir.name,
                              clean_crashes=False,
                              clean_over_tasks=False,
                              ask_for_validation=False,
                              logger=master_logger,
                              root_dir=root_dir)

    # Launches multiple processes

    if n_processes > 1:
        ## TODO: Logger is not supported with multiprocess (should use queues and all)
        n_calls = None  # for now we only return n_calls != None if running with one process only

        processes = []

        for i in range(n_processes):

            # Creates process logger

            logger = create_logger(name=f'RUN_SCHEDULE - PROCESS_{i}', loglevel=logging.INFO,
                                   logfile=storage_dir / 'run_schedule_logger.out', streamHandle=True)

            # Adds logfiles to logger if multiple storage_dirs
            if len(storage_dirs) > 1:
                for storage_dir in storage_dirs[1:]:
                    file_handler = create_new_filehandler(logger.name, logfile=storage_dir / 'run_schedule_logger.out')
                    logger.addHandler(file_handler)

            # Creates process

            processes.append(Process(target=_run_schedule, args=(storage_dirs,
                                                                 n_processes,
                                                                 n_experiments_per_proc,
                                                                 use_pbar,
                                                                 logger,
                                                                 root_dir,
                                                                 i)))
        try:
            # start processes

            for p in processes:
                p.start()
                time.sleep(0.5)

            # waits for all processes to end

            dead_processes = []
            while any([p.is_alive() for p in processes]):

                # check if some processes are dead

                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in dead_processes:
                        master_logger.info(f'PROCESS_{i} has died.')
                        dead_processes.append(i)

                time.sleep(3)

        except KeyboardInterrupt:
            master_logger.info("KEYBOARD INTERRUPT. Killing all processes")

            # terminates all processes

            for process in processes:
                process.terminate()

        master_logger.info("All processes are done. Closing '__main__'\n\n")

    # No additional processes

    else:
        n_calls = _run_schedule(storage_dirs,
                                n_processes,
                                n_experiments_per_proc,
                                use_pbar,
                                master_logger,
                                root_dir=root_dir)

    return n_calls


if __name__ == '__main__':
    time.sleep(np.random.uniform(low=0., high=1.5))
    kwargs = vars(get_run_schedule_args())
    launch_run_schedule(**kwargs)
