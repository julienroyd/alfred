from alfred.utils.config import *
from alfred.utils.directory_tree import *
from alfred.utils.misc import create_logger, select_storage_dirs


def my_type_func(add_arg):
    name, val_type = add_arg.split("=", 1)
    val, typ = val_type.split(",", 1)
    if typ == 'float':
        val = float(val)
    elif val == "None":
        val = None
    elif typ == 'str':
        val = str(val)
    else:
        raise NotImplementedError
    return name, val


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_file', type=str, default=None,
                        help="Path containing all the storage_names for which to create retrainBests")
    parser.add_argument('--storage_name', type=str, required=True)
    parser.add_argument('--over_tasks', type=parse_bool, default=False,
                        help="If true, subprocesses will create retrainBests for all storage_dir "
                             "that have the same hashes, 'alg_name', 'desc' but different 'task_name'")
    parser.add_argument('--new_desc', type=str, default=None)
    parser.add_argument("--additional_param", action='append',
                        type=my_type_func, dest='additional_params',
                        help='To add two params p1 and p2 with values v1 and v2 of type t1 and t2 do : --additional_param p1=v1,t1 '
                             '--additional_param p2=v2,t2')
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def copy_configs(from_file, storage_name, over_tasks, new_desc, additional_params, root_dir):

    logger = create_logger(name="COPY CONFIG", loglevel=logging.INFO)
    logger.info("\nCOPYING Config")

    # Select storage_dirs to run over

    storage_dirs = select_storage_dirs(from_file, storage_name, over_tasks, root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    for storage_to_copy in storage_dirs:
        seeds_to_copy = get_all_seeds(storage_to_copy)
        config_path_list = []
        config_unique_path_list = []

        # find the path to all the configs files

        for dir in seeds_to_copy:
            config_path_list.append(dir / 'config.json')
            config_unique_path_list.append(dir / 'config_unique.json')

        # extract storage name info

        _, _, _, _, old_desc = \
            DirectoryTree.extract_info_from_storage_name(str(storage_to_copy))

        # overwrites it

        tmp_dir_tree = DirectoryTree(alg_name="nope", task_name="nap", desc="nip", seed=1, root=root_dir)
        storage_name_id, git_hashes, _, _, _ = \
            DirectoryTree.extract_info_from_storage_name(str(tmp_dir_tree.storage_dir.name))

        desc = old_desc if new_desc is None else new_desc

        # creates the new folders with loaded config from which we overwrite the task_name

        dir = None
        for config_path, config_unique_path in zip(config_path_list, config_unique_path_list):

            config = load_config_from_json(str(config_path))
            config.desc = desc
            expe_name = config_path.parents[1].name
            experiment_num = int(''.join([s for s in expe_name if s.isdigit()]))

            config_unique_dict = load_dict_from_json(str(config_unique_path))

            if additional_params is not None:

                for (key, value) in additional_params:
                    config.__dict__[key] = value
                    config_unique_dict[key] = value

            dir = DirectoryTree(id=storage_name_id,
                                alg_name=config.alg_name,
                                task_name=config.task_name,
                                desc=config.desc,
                                seed=config.seed,
                                experiment_num=experiment_num,
                                git_hashes=git_hashes,
                                root=root_dir)

            dir.create_directories()
            print(f"Creating {str(dir.seed_dir)}\n")
            save_config_to_json(config, filename=str(dir.seed_dir / "config.json"))
            validate_config_unique(config, config_unique_dict)
            save_dict_to_json(config_unique_dict, filename=str(dir.seed_dir / "config_unique.json"))
            open(str(dir.seed_dir / 'UNHATCHED'), 'w+').close()

        open(str(dir.seed_dir.parents[1] / f'config_copied_from_{str(storage_to_copy.name)}'), 'w+').close()


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    copy_configs(from_file=args.from_file,
                 storage_name=args.storage_name,
                 over_tasks=args.over_tasks,
                 new_desc=args.new_desc,
                 additional_params=args.additional_params,
                 root_dir=args.root_dir)
