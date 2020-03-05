import logging
import traceback
import argparse

from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.utils.directory_tree import DirectoryTree, get_root
from alfred.utils.config import parse_bool, load_dict_from_json
from alfred.prepare_schedule import create_experiment_dir

try:
    from schedules import grid_schedule  # just to get DirectoryTree.git_repos_to_track configured
except ImportError:
    pass


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to create retrainBests")

    parser.add_argument('--storage_name', type=str, required=True)
    parser.add_argument('--over_tasks', type=parse_bool, default=False,
                        help="If true, subprocesses will create retrainBests for all storage_dir "
                             "that have the same hashes, 'alg_name', 'desc' but different 'task_name'")

    parser.add_argument('--n_retrain_seeds', type=int, default=10)

    parser.add_argument('--root_dir', type=str, default=None)

    return parser.parse_args()


def create_retrain_best(from_file, storage_name, over_tasks, n_retrain_seeds, root_dir):
    logger = create_logger(name="CREATE_RETRAIN", loglevel=logging.INFO)
    logger.info("\nCREATING retrainBest directories")

    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, over_tasks, root_dir)

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

                logger.info(f"Existing retrainBest\n\n"
                            f"\t{storage_dir.name} -> {retrainBest_dir.name}")

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

                # Updates the description

                if "random" in config_dict['desc'] or "grid" in config_dict['desc']:
                    new_desc = config_dict['desc'] \
                        .replace("random", f"retrainBest{search_storage_id}") \
                        .replace("grid", f"retrainBest{search_storage_id}")
                else:
                    new_desc = config_dict['desc'] + f"_retrainBest{search_storage_id}"

                config_dict['desc'] = new_desc

                # Creates config Namespace with loaded config_dict

                config = argparse.ArgumentParser().parse_args("")
                config_pointer = vars(config)
                config_pointer.update(config_dict)  # updates config

                config_unique_dict = {}
                config_unique_dict['alg_name'] = config.alg_name
                config_unique_dict['task_name'] = config.task_name
                config_unique_dict['seed'] = config.seed

                # Gets new storage_name_id

                tmp_dir_tree = DirectoryTree(alg_name="", task_name="", desc="", seed=1)
                retrain_storage_id = tmp_dir_tree.storage_dir.name.split('_')[0]

                # Creates the new storage_dir for retrainBest

                dir_tree = create_experiment_dir(storage_name_id=retrain_storage_id,
                                                 config=config,
                                                 config_unique_dict=config_unique_dict,
                                                 SEEDS=[i * 10 for i in range(n_retrain_seeds)],
                                                 root_dir=root_dir,
                                                 git_hashes=DirectoryTree.get_git_hashes())

                retrainBest_storage_dirs.append(dir_tree.storage_dir)

                logger.info(f"New retrainBest:\n\n"
                            f"\t{storage_dir.name} -> {dir_tree.storage_dir.name}")

        except Exception as e:
            logger.info(f"Could not create retrainBest-storage_dir {storage_dir}")
            logger.info(f"\n\n{e}\n{traceback.format_exc()}")

    return retrainBest_storage_dirs


if __name__ == "__main__":
    kwargs = vars(get_args())
    create_retrain_best(**kwargs)
