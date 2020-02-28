import argparse
import logging

from alfred.utils.config import parse_bool
from alfred.utils.misc import create_logger
from alfred.utils.directory_tree import *


def get_clean_interrupted_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default=None, type=str,
                        help="The starting directory, the script will try to sync all child directory"
                             " that matches the tag")
    parser.add_argument('--tag', type=str, default="",
                        help="Will try to push only the child directories which name contains the tag"
                             " if default then try to push all child directories")
    parser.add_argument('--env_activation', type=str, default="",
                        help="Command line to activate the corresponding python env (e.g. 'activate irl'")
    parser.add_argument('--ask_for_validation', type=parse_bool, default=True)

    return parser.parse_args()


def sync_wandb(root_dir, tag, ask_for_validation, env_activation, logger):

    # Define sync command line

    if not env_activation == "":
        command_line = f"{env_activation} && wandb sync wandb/"

    else:
        command_line = "wandb sync wandb/"

    if not os.name == "posix":
        command_line = command_line.split(" ")

    # Select the root dir

    root = get_root(root_dir)

    child_dirs = [child for child in root.iterdir() if tag in child.name]

    info_string = "Folders to be synced : \n"

    for child in child_dirs:
        info_string += str(child) + "\n"

    logger.info(info_string)

    if ask_for_validation:

        # Asks for validation to sync the storages

        answer = input("\nShould we proceed? [y or n]")
        if answer.lower() not in ['y', 'yes']:
            logger.debug("Aborting...")
            return

        logger.debug("Starting...")

    for child in child_dirs:

        # get all wandb folders

        wandb_dirs = child.glob('**/wandb')

        for to_sync in wandb_dirs:
            subprocess.run(command_line, shell=True, cwd=str(to_sync.parent))

        logger.info(f'Storage {child} has been synced \n')


if __name__ == '__main__':
    kwargs = vars(get_clean_interrupted_args())
    logger = create_logger(name="SYNCH TO WANDB", loglevel=logging.INFO)
    sync_wandb(**kwargs, logger=logger)
