import logging
import traceback
import argparse

from alfred.utils.misc import create_logger
from alfred.utils.directory_tree import DirectoryTree, get_storage_dirs_across_tasks, get_root
from alfred.utils.config import parse_bool, load_dict_from_json
from alfred.prepare_schedule import create_experiment_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--storage_name', type=str, required=True)
    parser.add_argument('--run_over_tasks', type=parse_bool, default=False,
                        help="If true, subprocesses will look for unhatched seeds in all storage_dir"
                             "that have the same hashes, 'alg_name', 'desc' but different 'task_name'")
    parser.add_argument('--n_retrain_seeds', type=int, default=10)
    parser.add_argument('--root_dir', type=str, default=None)

    return parser.parse_args()


def create_retrain_best(storage_name, run_over_tasks, n_retrain_seeds, root_dir):
    logger = create_logger(name="CREATE_RETRAIN", loglevel=logging.INFO)
    logger.info("\nCREATING retrainBest directories")

    storage_dir = get_root(root_dir) / storage_name

    if run_over_tasks:
        storage_dirs = get_storage_dirs_across_tasks(storage_dir, root_dir)
    else:
        storage_dirs = [storage_dir]

    # Creates retrainBest directories

    retrainBest_storage_dirs = []
    for storage_dir in storage_dirs:

        try:

            # Checks if a retrainBest directory already exists for this search

            search_storage_id = storage_dir.name.split('_')[0]
            corresponding_retrain_directories = [path for path in get_root(root_dir).iterdir()
                                                 if f"retrainBest{search_storage_id}" in path.name.split('_')]

            if len(corresponding_retrain_directories) > 0:
                assert len(corresponding_retrain_directories) == 1
                retrainBest_dir = corresponding_retrain_directories[0]

                logger.info(f"Existing retrainBest\n"
                            f"{storage_dir.name} -> {retrainBest_dir.name}")

                retrainBest_storage_dirs.append(retrainBest_dir)
                continue

            else:

                # The retrainBest directory will contain one experiment with bestConfig from the search

                best_config = [path for path in (storage_dir / "summary").iterdir()
                               if path.name.startswith("bestConfig")]

                assert len(best_config) == 1 and type(best_config) is list
                config_dict = load_dict_from_json(filename=str(best_config[0]))

                # Retrain experiments run for twice as long

                config_dict['max_episodes'] *= 2

                # Gets new storage_name_id

                tmp_dir_manager = DirectoryTree(alg_name="", task_name="", desc="", seed=1)
                retrain_storage_id = tmp_dir_manager.storage_dir.name.split('_')[0]

                # Creates the new storage_dir for retrainBest
                if "random" in config_dict['desc'] or "grid" in config_dict['desc']:
                    new_desc = config_dict['desc'] \
                        .replace("random", f"retrainBest{search_storage_id}") \
                        .replace("grid", f"retrainBest{search_storage_id}")
                else:
                    new_desc = config_dict['desc'] + f"_retrainBest{search_storage_id}"

                dir_manager = create_experiment_dir(storage_name_id=retrain_storage_id,
                                                    desc=new_desc,
                                                    alg_name=config_dict['alg_name'],
                                                    task_name=config_dict['task_name'],
                                                    param_dict=config_dict,
                                                    varied_params=[],
                                                    root_dir=root_dir,
                                                    check_param_in_main=False,
                                                    SEEDS=[i * 10 for i in range(n_retrain_seeds)])

                retrainBest_storage_dirs.append(dir_manager.storage_dir)

                logger.info(f"New retrainBest:\n"
                            f"{storage_dir.name} -> {dir_manager.storage_dir.name}")

        except Exception as e:
            logger.info(f"Could not create retrainBest-storage_dir {storage_dir}")
            logger.info(f"\n\n{e}\n{traceback.format_exc()}")

    info_str = "The following retrainBest directories will be runned over:\n"
    for retrainBest_storage_dir in retrainBest_storage_dirs:
        info_str += f"\n\t{retrainBest_storage_dir}"
    logger.info(info_str + "\n")
    return retrainBest_storage_dirs


if __name__ == "__main__":
    args = get_args()
    create_retrain_best(**args.__dict__)
