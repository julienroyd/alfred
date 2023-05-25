from alfred.utils.directory_tree import get_some_seeds, get_all_seeds, sanity_check_exists
from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.utils.config import parse_bool

import argparse
import logging
import shutil
import os


def get_clean_interrupted_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--from_file', type=str, default=None,
                        help="Path containing all the storage_names to clean")

    parser.add_argument('-s', '--storage_name', type=str, default=None)

    parser.add_argument('--clean_opened', action='store_true', default=False)
    parser.add_argument('--clean_crashed', action='store_true', default=False)
    parser.add_argument('--ask_for_validation', type=parse_bool, default=True)

    parser.add_argument('-r', '--root_dir', default=None, type=str)
    return parser.parse_args()


def clean_interrupted(from_file, storage_name, clean_opened, clean_crashed, ask_for_validation, logger, root_dir):
    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    # For all storage_dirs...

    for storage_dir in storage_dirs:

        all_seeds = get_all_seeds(storage_dir)
        unhatched_seeds = get_some_seeds(storage_dir, file_check='UNHATCHED')
        opened_seeds = get_some_seeds(storage_dir, file_check='OPENED')
        completed_seeds = get_some_seeds(storage_dir, file_check='COMPLETED')
        crashed_seeds = get_some_seeds(storage_dir, file_check='CRASH.txt')
        assert set(all_seeds) == set(unhatched_seeds + opened_seeds + completed_seeds + crashed_seeds)

        # Prints some info

        logger.info(f"All seed_dir status in {storage_dir}:\n"
                    f"\nNumber of seeds:\t\t{len(all_seeds)}"
                    f"\n{'-'*30}"
                    f"\nNumber of seeds UNHATCHED:\t{len(unhatched_seeds)}"
                    f"\nNumber of seeds OPENED: \t{len(opened_seeds)}"
                    f"\nNumber of seeds CRASHED:\t{len(crashed_seeds)}"
                    f"\nNumber of seeds COMPLETED:\t{len(completed_seeds)}"
                    f"\n\nclean_opened={clean_opened}"
                    f"\nclean_crashed={clean_crashed}"
                    f"\n"
                    )

        # Check what should be cleaned

        seeds_to_clean = []

        if clean_opened:
            seeds_to_clean += [seed_dir for seed_dir in all_seeds if seed_dir in opened_seeds]

        if clean_crashed:
            seeds_to_clean += [seed_dir for seed_dir in all_seeds if seed_dir in crashed_seeds]

        if len(seeds_to_clean) != 0:
            logger.info(f'{len(seeds_to_clean)} seeds about to be cleaned:')
            for seed_dir in seeds_to_clean:
                logger.info(f'--- {seed_dir}')

        if ask_for_validation:

            # Asks for validation to clean this storage_dir

            answer = input("\nShould we proceed? [y or n]")
            if answer.lower() not in ['y', 'yes']:
                logger.debug("Aborting...")
                continue

            logger.debug("Starting...")

            # Clean each seed_directory

            for seed_dir in seeds_to_clean:
                logger.info(f"Cleaning {seed_dir}")

                for path in seed_dir.iterdir():
                    if path.name not in ["config.json", "config_unique.json"]:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    else:
                        continue

                open(str(seed_dir / 'UNHATCHED'), 'w+').close()
            logger.info('Done')

        else:
            logger.info('No seed_dir to clean.')

if __name__ == '__main__':
    kwargs = vars(get_clean_interrupted_args())
    logger = create_logger(name="CLEAN_INTERRUPTED - MAIN", loglevel=logging.INFO)
    clean_interrupted(**kwargs, logger=logger)
