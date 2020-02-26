import os
import subprocess
from pathlib import Path
import os.path as osp
from collections import OrderedDict

class DirectoryTree(object):
    root = Path("./storage")
    git_repos_to_track = OrderedDict()

    def __init__(self, alg_name, task_name, desc, seed,
                 experiment_num=None, git_hashes=None, id=None):

        # Creates the root folder (if doesn't already exist)

        os.makedirs(str(DirectoryTree.root), exist_ok=True)

        # Defines storage_name id

        if id is not None:
            id = id
        else:
            git_name_short = get_git_name()[:2]
            exst_ids_numbers = [int(folder.name.split('_')[0][2:]) for folder in DirectoryTree.root.iterdir()
                                if folder.is_dir() and folder.name.split('_')[0].startswith(git_name_short)]

            if len(exst_ids_numbers) == 0:
                id = f'{git_name_short}1'
            else:
                id = f'{git_name_short}{(max(exst_ids_numbers) + 1)}'

        # Adds code versions git-hash for tracked projects

        if git_hashes is not None:
            git_hashes = git_hashes
        else:
            git_hashes = DirectoryTree.get_git_hashes()

        # Defines folder name

        storage_name = f"{id}_{git_hashes}_{alg_name}_{task_name}_{desc}"

        # Level 1: storage_dir

        self.storage_dir = DirectoryTree.root / storage_name

        # Level 1.5: summary_dir and benchmark_dir

        self.summary_dir = self.storage_dir / 'summary'
        self.benchmark_dir = self.storage_dir / 'benchmark'

        # Defines experiment number

        if experiment_num is not None:
            self.current_experiment = f'experiment{experiment_num}'
        else:
            if not self.storage_dir.exists():
                self.current_experiment = 'experiment1'
            else:
                exst_run_nums = [int(str(folder.name).split('experiment')[1]) for folder in
                                 self.storage_dir.iterdir() if
                                 str(folder.name).startswith('experiment')]
                if len(exst_run_nums) == 0:
                    self.current_experiment = 'experiment1'
                else:
                    self.current_experiment = f'experiment{(max(exst_run_nums) + 1)}'

        # Level 2: experiment_dir

        self.experiment_dir = self.storage_dir / self.current_experiment

        # Level 3: seed_dir

        self.seed_dir = self.experiment_dir / f"seed{seed}"

        # Level 4: recorders_dir and incrementals_dir

        self.recorders_dir = self.seed_dir / 'recorders'
        self.incrementals_dir = self.seed_dir / 'incrementals'

        # Level 4.5: images

        self.random_results_dir = self.recorders_dir / 'random_results'
        self.fixed_results_dir = self.recorders_dir / 'fixed_results'

    def create_directories(self):
        os.makedirs(str(self.seed_dir))

    @staticmethod
    def get_all_experiments(storage_dir):
        all_experiments = [path for path in storage_dir.iterdir()
                           if path.is_dir() and str(path.stem).startswith('experiment')]

        return sorted(all_experiments, key=lambda item: (int(str(item.stem).strip('experiment')), item))

    @staticmethod
    def get_all_seeds(experiment_dir):
        all_seeds = [path for path in experiment_dir.iterdir()
                     if path.is_dir() and str(path.stem).startswith('seed')]

        return sorted(all_seeds, key=lambda item: (int(str(item.stem).strip('seed')), item))

    @classmethod
    def init_from_seed_path(cls, seed_path):
        assert isinstance(seed_path, Path)

        id, git_hashes, alg_name, task_name, desc = \
            DirectoryTree.extract_info_from_storage_name(seed_path.parents[1].name)

        instance = cls(id=id,
                       git_hashes=git_hashes,
                       alg_name=alg_name,
                       task_name=task_name,
                       desc=desc,
                       experiment_num=seed_path.parents[0].name.strip('experiment'),
                       seed=seed_path.name.strip('seed'))

        return instance

    @classmethod
    def extract_info_from_storage_name(cls, storage_name):

        id = storage_name.split("_")[0]
        git_hashes = storage_name.split("_")[1]
        alg_name = storage_name.split("_")[2]
        task_name = storage_name.split("_")[3]
        desc = "_".join(storage_name.split("_")[4:])

        return id, git_hashes, alg_name, task_name, desc

    @classmethod
    def get_git_hashes(cls):
        git_hashes = []
        for name, repo in cls.git_repos_to_track.items():
            git_hashes.append(get_git_hash(path=repo))

        git_hashes = '-'.join(git_hashes)
        return git_hashes


def get_some_seeds(storage_dir, file_check='UNHATCHED'):
    # Finds all seed directories containing an UNHATCHED file and sorts them numerically

    sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

    some_seed_dirs = []
    for experiment_dir in sorted_experiments:
        some_seed_dirs += [seed_path for seed_path
                           in DirectoryTree.get_all_seeds(experiment_dir)
                           if (seed_path / file_check).exists()]

    return some_seed_dirs


def get_all_seeds(storage_dir):
    # Finds all seed directories and sorts them numerically

    sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

    all_seeds_dirs = []
    for experiment_dir in sorted_experiments:
        all_seeds_dirs += [seed_path for seed_path
                           in DirectoryTree.get_all_seeds(experiment_dir)]

    return all_seeds_dirs


def get_storage_dirs_across_envs(storage_dir):
    # Finds all storage directories that are identical (hash, alg_name, desc) but for the environment

    all_storage_dirs = sorted([path for path in DirectoryTree.root.iterdir() if path.is_dir()])

    similar_storage_dirs = []
    name_elements_current = storage_dir.name.split('_')

    for dir in all_storage_dirs:
        name_elements_dir = dir.name.split('_')
        if all([str_1 == str_2 for i, (str_1, str_2)
                in enumerate(zip(name_elements_dir, name_elements_current))
                if i not in [0, 4]]):
            similar_storage_dirs.append(dir)

    # Moves initial storage_dir in front of the list

    similar_storage_dirs.insert(0, similar_storage_dirs.pop(similar_storage_dirs.index(storage_dir)))

    return similar_storage_dirs


def get_git_hash(path):
    try:
        return subprocess.check_output(
            ["git", "--git-dir", os.path.join(path, '.git'), "rev-parse", "--short", "HEAD"]).decode(
            "utf-8").strip()

    except subprocess.CalledProcessError:
        return 'NoGitHash'


def get_git_name():
    try:
        return subprocess.check_output(["git", "config", "user.name"]).decode("utf-8").strip()

    except subprocess.CalledProcessError:
        return 'NoGitUsr'
