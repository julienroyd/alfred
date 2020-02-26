import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alfred.utils.misc import create_logger
from alfred.utils.config import load_dict_from_json
from alfred.utils.recorder import TrainingRecorder
from alfred.utils.plots import plot_curves
from alfred.utils.directory_tree import DirectoryTree


def get_make_plots_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_name', type=str, required=True)
    return parser.parse_args()


def create_comparative_figure(storage_dir, logger):
    """
    Creates and and saves comparative figure containing a plot of total reward for each different experiment
    :param storage_dir: pathlib.Path object of the model directory containing the experiments to compare
    :param plots_to_make: list of strings indicating which comparative plots to make
    :return: None
    """
    assert isinstance(storage_dir, Path)
    PLOTS_TO_MAKE = [
        'return',
        'eval_return',
        'actor_loss',
        'critic_loss',
    ]

    # Get all experiment directories and sorts them numerically
    sorted_experiments = DirectoryTree.get_all_experiments(storage_dir)

    all_seeds_dir = []
    for experiment in sorted_experiments:
        all_seeds_dir = all_seeds_dir + DirectoryTree.get_all_seeds(experiment)

    # Determines what type of search was done
    if (storage_dir / 'GRID_SEARCH').exists():
        search_type = 'grid'
    elif (storage_dir / 'RANDOM_SEARCH').exists():
        search_type = 'random'
    else:
        search_type = 'unknown'

    # Determines row and columns of subplots
    if search_type == 'grid':
        variations = load_dict_from_json(filename=str(storage_dir / 'variations.json'))
        # experiment_groups account for the fact that all the experiment_dir in a storage_dir may have been created
        # though several runs of prepare_schedule.py, and therefore, many "groups" of experiments have been created
        experiment_groups = {key: {} for key in variations.keys()}
        for group_key, properties in experiment_groups.items():
            properties['variations'] = variations[group_key]
            properties['variations_lengths'] = {k: len(properties['variations'][k]) for k in
                                                properties['variations'].keys()}

            i_max = sorted(properties['variations_lengths'].values())[-1]
            j_max = int(np.prod(sorted(properties['variations_lengths'].values())[:-1]))

            if i_max < 4 and j_max == 1:
                # If only one hyperparameter was varied over, we order plots on a line
                j_max = i_max
                i_max = 1
                ax_array_dim = 1

            elif i_max >= 4 and j_max == 1:
                # ... unless there are 4 or more variations, then we put them in a square-ish fashion
                j_max = int(np.sqrt(i_max))
                i_max = int(np.ceil(float(i_max) / float(j_max)))
                ax_array_dim = 2

            else:
                ax_array_dim = 2

            properties['ax_array_shape'] = (i_max, j_max)
            properties['ax_array_dim'] = ax_array_dim

    else:
        experiment_groups = {"all": {}}
        for group_key, properties in experiment_groups.items():
            i_max = int(np.ceil(np.sqrt(len(all_seeds_dir))))
            j_max = i_max
            ax_array_dim = 2
            properties['ax_array_shape'] = (i_max, j_max)
            properties['ax_array_dim'] = ax_array_dim

    for group_key, properties in experiment_groups.items():
        logger.debug(f"\n===========================\nPLOTS FOR EXPERIMENT GROUP: {group_key}")
        i_max, j_max = properties['ax_array_shape']
        ax_array_dim = properties['ax_array_dim']

        first_exp = group_key.split('-')[0] if group_key != "all" else 0
        if first_exp != 0:
            for seed_idx, seed_dir in enumerate(all_seeds_dir):
                if seed_dir.parent.stem.strip('experiment') == first_exp:
                    first_seed_idx = seed_idx
                    break
        else:
            first_seed_idx = 0

        for current_comparative_plot in PLOTS_TO_MAKE:
            logger.debug(f'\n{current_comparative_plot.upper()}:')

            # Creates the subplots
            fig, ax_array = plt.subplots(i_max, j_max, figsize=(10 * j_max, 6 * i_max))

            for i in range(i_max):
                for j in range(j_max):

                    if ax_array_dim == 1 and i_max == 1 and j_max == 1:
                        current_ax = ax_array
                    elif ax_array_dim == 1 and (i_max > 1 or j_max > 1):
                        current_ax = ax_array[j]
                    elif ax_array_dim == 2:
                        current_ax = ax_array[i, j]
                    else:
                        raise Exception('ax_array should not have more than two dimensions')

                    try:
                        seed_dir = all_seeds_dir[first_seed_idx + (i * j_max + j)]
                        if group_key != 'all' \
                                and (int(str(seed_dir.parent).split('experiment')[1]) < int(group_key.split('-')[0]) \
                                     or int(str(seed_dir.parent).split('experiment')[1]) > int(
                                    group_key.split('-')[1])):
                            raise IndexError
                        logger.debug(str(seed_dir))
                    except IndexError as e:
                        logger.debug(f'experiment{i * j_max + j} does not exist')
                        current_ax.text(0.2, 0.2, "no experiment\n found",
                                        transform=current_ax.transAxes, fontsize=24, fontweight='bold', color='red')
                        continue

                    logger.debug(seed_dir)

                    # Writes unique hyperparameters on plot
                    config_unique = load_dict_from_json(filename=str(seed_dir / 'config_unique.json'))

                    if search_type == 'grid':
                        sorted_keys = sorted(config_unique.keys(),
                                             key=lambda item: (properties['variations_lengths'][item], item),
                                             reverse=True)

                    else:
                        sorted_keys = config_unique

                    info_str = f'{seed_dir.parent.stem}\n' + '\n'.join(
                        [f'{k} = {config_unique[k]}' for k in sorted_keys])
                    bbox_props = dict(facecolor='gray', alpha=0.1)
                    current_ax.text(0.05, 0.95, info_str, transform=current_ax.transAxes, fontsize=12,
                                    verticalalignment='top', bbox=bbox_props)

                    # Skip cases of UNHATCHED or CRASHED experiments
                    if (seed_dir / 'UNHATCHED').exists():
                        logger.debug('UNHATCHED')
                        current_ax.text(0.2, 0.2, "UNHATCHED",
                                        transform=current_ax.transAxes, fontsize=24, fontweight='bold', color='blue')
                        continue

                    if (seed_dir / 'CRASH.txt').exists():
                        logger.debug('CRASHED')
                        current_ax.text(0.2, 0.2, "CRASHED",
                                        transform=current_ax.transAxes, fontsize=24, fontweight='bold', color='red')
                        continue

                    # Plots data for one experiment
                    try:
                        loaded_recorder = TrainingRecorder.init_from_pickle_file(
                            filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

                        if current_comparative_plot == 'return':
                            plot_curves(current_ax,
                                        ys=np.vstack(loaded_recorder.tape['total_reward']).T,
                                        colors=loaded_recorder.tape['agent_colors'],
                                        labels=[f"agent {i}" for i in range(len(loaded_recorder.tape['agent_colors']))],
                                        xlabel="Episodes", title="Return")

                        elif current_comparative_plot == 'eval_return':
                            plot_curves(current_ax,
                                        xs=[loaded_recorder.tape['eval_episodes'] for _ in loaded_recorder.tape['agent_colors']],
                                        ys=np.stack(loaded_recorder.tape['eval_total_reward']).mean(axis=1).T,
                                        stds=np.stack(loaded_recorder.tape['eval_total_reward']).std(axis=1).T,
                                        colors=loaded_recorder.tape['agent_colors'],
                                        labels=[f"agent {i}" for i in range(len(loaded_recorder.tape['agent_colors']))],
                                        xlabel="Episodes", title="Return")

                        elif current_comparative_plot == 'actor_loss':
                            plot_curves(current_ax,
                                        ys=np.vstack(loaded_recorder.tape['actor_loss']).T,
                                        colors=loaded_recorder.tape['agent_colors'],
                                        labels=[f"agent {i}" for i in range(len(loaded_recorder.tape['agent_colors']))],
                                        xlabel="Updates", title="Actor Loss")

                        elif current_comparative_plot == 'critic_loss':
                            plot_curves(current_ax,
                                        ys=np.vstack(loaded_recorder.tape['critic_loss']).T,
                                        colors=loaded_recorder.tape['agent_colors'],
                                        labels=[f"agent {i}" for i in range(len(loaded_recorder.tape['agent_colors']))],
                                        xlabel="Updates", title="Critic Loss")

                        else:
                            raise Warning(f"Unrecognized current_comparative_plot='{current_comparative_plot}'")

                    except FileNotFoundError:
                        logger.debug('Training recorder not found')
                        current_ax.text(0.2, 0.2, "'train_recorder'\nnot found",
                                        transform=current_ax.transAxes, fontsize=24, fontweight='bold', color='red')
                        continue

            fig.savefig(str(storage_dir / f'{group_key}_comparative_{current_comparative_plot}.png'))
            plt.close(fig)


if __name__ == '__main__':
    logger = create_logger("PLOTS", logging.DEBUG, logfile=None)
    args = get_make_plots_args()
    storage_dir = DirectoryTree.root / args.storage_name
    create_comparative_figure(storage_dir, logger)
