# This script does three things:

# (1) calls run_schedule on a random-search given by 'storage_name'
#     once the search is done, it gets summarized and the best config is identified

# (2) calls prepare_schedule for that configuration to creates storage_dir
#     to retrain the best config for each task and calls run_schedule on them

# (3) benchmark the result against all storage_dirs with the same description

import argparse
import numpy as np
from copy import deepcopy
import logging

from alfred.create_retrainbest import create_retrain_best
from alfred.run_schedule import launch_schedule
from alfred.utils.config import parse_bool
from alfred.utils.misc import create_logger


def get_full_cycle_args():
    parser = argparse.ArgumentParser()

    # arguments identical to run_schedule.py

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

    # arguments unique to full_cycle.py

    parser.add_argument('--n_retrain_seeds', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':

    cycle_logger = create_logger(name="FULL_CYCLE", loglevel=logging.INFO)
    full_cycle_args = get_full_cycle_args()

    # 1: RUNNING THE SCHEDULE ---

    cycle_logger.info("\nSTARTING PART (1) - Running run_schdule on searches")

    # Launches run_schedule

    run_schedule_args = deepcopy(vars(full_cycle_args))
    del run_schedule_args['n_retrain_seeds']

    n_calls = launch_schedule(**run_schedule_args)

    if full_cycle_args.n_processes == 1 and full_cycle_args.n_experiments_per_proc == 1 and n_calls >= 1:
        full_cycle_args.n_experiments_per_proc = 0

    # 2: CREATING RETRAIN DIRECTORIES FOR BEST CONFIGS OF EVERY SEARCH ---

    cycle_logger.info("\nSTARTING PART (2) - Creating retrainBest directories")

    retrainBest_storage_dirs = create_retrain_best(storage_name=full_cycle_args.storage_name,
                                                   run_over_tasks=full_cycle_args.run_over_tasks,
                                                   n_retrain_seeds=full_cycle_args.n_retrain_seeds)

    # 3: RUNNING THE SCHEDULE FOR RETRAIN BESTS ---

    cycle_logger.info("\nSTARTING PART (3) - Running run_schdule on retrainBest directories")

    # Since different retrain-directories have different descriptions, we cannot simply launch run_schedule with
    # --run_over_tasks=True to launch them all sequentially. We have to launch run_schedule separately on each of
    # the retrain-directory. However, since the number of seeds used for retrain is typically smaller than n_processes
    # we can launch multiple run_schedule in parallel in different subprocesses.

    run_schedule_args = deepcopy(vars(full_cycle_args))
    del run_schedule_args['n_retrain_seeds']
    run_schedule_args['run_over_tasks'] = False

    n_parallel_run_schedule = full_cycle_args.n_processes // full_cycle_args.n_retrain_seeds

    if n_parallel_run_schedule <= 1 or True:  # TODO: implement 'else' and remove 'or True'

        run_schedule_args['n_processes'] = full_cycle_args.n_processes
        for retrainBest_storage_dir in retrainBest_storage_dirs:
            run_schedule_args['storage_name'] = retrainBest_storage_dir.name
            launch_schedule(**run_schedule_args)

    # else:
    #
    #     # Create processes
    #
    #     processes = []

    # TODO: implement automatic benchmarking
    # cycle_logger.info("STARTING PART (4) - Benchmarking the retrainBest directories against benchmark_list.json")
