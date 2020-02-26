from alfred.utils.config import *
from alfred.utils.directory_tree import *


def my_type_func(add_arg):
    name, val_type = add_arg.split("=", 1)
    val, typ = val_type.split(",", 1)
    if typ == 'float':
        val = float(val)
    elif val == "None":
        val = None
    else:
        raise NotImplementedError
    return name, val


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_name', type=str, required=True)
    parser.add_argument('--new_task', type=str, required=True)
    parser.add_argument('--new_desc', type=str, default=None)
    parser.add_argument("--additional_param", action='append',
                        type=my_type_func, dest='additional_params',
                        help='To add two params p1 and p2 with values v1 and v2 of type t1 and t2 do : --additional_param p1=v1,t1 '
                             '--additional_param p2=v2,t2')
    parser.add_argument("--root_dir", default=None, type=str)
    return parser.parse_args()


def copy_configs(storage_name, new_task, new_desc, additional_params, root_dir):
    storage_to_copy = get_root(root_dir) / storage_name
    seeds_to_copy = get_all_seeds(storage_to_copy)
    config_list = []
    config_unique_list = []

    test = Path('.')
    # find the path to all the configs files

    for dir in seeds_to_copy:
        config_list.append(dir / 'config.json')
        config_unique_list.append(dir / 'config_unique.json')

    # extract storage name info

    _, _, _, _, old_desc = \
        DirectoryTree.extract_info_from_storage_name(str(storage_to_copy))

    # overwrites it

    tmp_dir_manager = DirectoryTree(alg_name="nope", task_name="nap", desc="nip", seed=1, root=root_dir)
    storage_name_id, git_hashes, _, _, _ = \
        DirectoryTree.extract_info_from_storage_name(str(tmp_dir_manager.storage_dir.name))

    desc = old_desc if new_desc is None else new_desc

    # creates the new folders with loaded config from which we overwrite the env name
    dir = None
    for config, config_unique in zip(config_list, config_unique_list):

        config = load_config_from_json(str(config))
        config.task_name = new_task
        config.desc = desc
        expe_name = config.parents[1].name
        experiment_num = int(''.join([s for s in expe_name if s.isdigit()]))

        config_unique = load_config_from_json(str(config_unique))
        config_unique.task_name = new_task

        if additional_params is not None:

            for (key, value) in additional_params:
                config.__dict__[key] = value

        dir = DirectoryTree.init_from_training_param(id=storage_name_id,
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
        save_config_to_json(config_unique, filename=str(dir.seed_dir / "config_unique.json"))
        open(str(dir.seed_dir / 'UNHATCHED'), 'w+').close()

    open(str(dir.seed_dir.parents[1] / f'config_copied_from_{str(storage_to_copy.name)}'), 'w+').close()


if __name__ == "__main__":
    args = get_args()
    print(args.__dict__)
    copy_configs(args.storage_name, args.new_env, args.new_desc, args.additional_params, args.root_dir)
