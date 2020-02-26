from evaluate import evaluate, get_evaluation_args  # TODO: deal with that

# TODO: eventually, I should:
# TODO:    1. create a file common.py that has some general functions used by both summaries (intra storage_dir) and benchmarks (across storage_dir's)
# TODO:    2. separate 'summarise.py' and 'benchamark.py'. Only 'summarise.py' would be called at the end of run_schedule

import numpy as np
import argparse
import pickle
import os
import logging
from shutil import copyfile
from collections import OrderedDict
import seaborn as sns
sns.set()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.plots import create_fig, plot_curves, bar_chart  # TODO: deal with that
from alfred.utils.config import load_dict_from_json, save_dict_to_json, parse_bool
from alfred.utils.recorder import TrainingRecorder
from alfred.utils.misc import create_logger
from schedules import official_benchmark_lists


def get_benchmark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_type', type=str, choices=['compare_models', 'compare_searches'], required=True)
    parser.add_argument('--storage_names', type=str, nargs='+', default=None)
    parser.add_argument('--agent_i', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--re_run_if_exists', type=parse_bool, default=False)
    return parser.parse_args()


# utility functions for curves (should not be called alone) -------------------------------------------------------

def _run_performance_benchmark(storage_dir, n_episodes, group_key, bar_key, re_run_if_exists, save_dir, logger):
    if (storage_dir / save_dir / f"{save_dir}_performance_data.pkl").exists() and not re_run_if_exists:
        logger.info(f" SKIPPING {storage_dir} - {save_dir}_performance.pkl already exists")
        return

    assert group_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']
    assert bar_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']

    # Initialize container

    scores = OrderedDict()

    # Get all experiment directories

    all_experiments = DirectoryTree.get_all_experiments(storage_dir=storage_dir)

    for experiment_dir in all_experiments:

        # For that experiment, get all seed directories

        experiment_seeds = DirectoryTree.get_all_seeds(experiment_dir=experiment_dir)

        # Initialize container

        all_seeds_return = []

        for i, seed_dir in enumerate(experiment_seeds):
            # Prints which seed directory is being treated

            logger.debug(f"{seed_dir}")

            # Loads training config

            config_dict = load_dict_from_json(str(seed_dir / "config.json"))

            # Selects how data will be identified

            keys = {
                "task_name": config_dict["task_name"],
                "storage_name": seed_dir.parents[1].name,
                "alg_name": config_dict["alg_name"],
                "experiment_num": seed_dir.parents[0].name.strip('experiment')
            }

            outer_key = keys[bar_key]
            inner_key = keys[group_key]

            # Sets config for evaluation phase

            eval_config = get_evaluation_args(overwritten_args="")
            eval_config.storage_name = seed_dir.parents[1].name
            eval_config.experiment_num = int(seed_dir.parents[0].name.strip("experiment"))
            eval_config.seed_num = int(seed_dir.name.strip("seed"))
            eval_config.render = False
            eval_config.n_episodes = n_episodes

            # Evaluates agent and stores the return

            returns = evaluate(eval_config)
            all_seeds_return.append(returns)

        if outer_key not in scores.keys():
            scores[outer_key] = OrderedDict()

        scores[outer_key][inner_key] = np.stack(all_seeds_return)

    os.makedirs(storage_dir / save_dir, exist_ok=True)
    with open(storage_dir / save_dir / f"{save_dir}_performance_data.pkl", "wb") as f:
        pickle.dump(scores, f)

    return


def _make_benchmark_performance_figure(storage_dirs, agent_i, save_dir, logger, normalize_with_first_model=True,
                                       sort_bars=False, std_error=True, quantile=0.10):
    # Initialize containers

    scores_means = OrderedDict()
    scores_err_up = OrderedDict()
    scores_err_down = OrderedDict()

    # Loads performance benchmark data

    individual_scores = OrderedDict()
    for storage_dir in storage_dirs:
        with open(storage_dir / save_dir / f"{save_dir}_performance_data.pkl", "rb") as f:
            individual_scores[storage_dir.name] = pickle.load(f)

    # Print keys so that user can verify all these benchmarks make sense to compare (e.g. same envs)

    for storage_name, idv_score in individual_scores.items():
        logger.debug(storage_name)
        for outer_key in idv_score.keys():
            logger.debug(f"{outer_key}: {list(idv_score[outer_key].keys())}")
        logger.debug(f"\n")

    # Reorganize all individual_scores in a single dictionary and extract the right agent's results

    scores = OrderedDict()
    for storage_name, idv_score in individual_scores.items():
        for outer_key in idv_score:
            if outer_key not in list(scores.keys()):
                scores[outer_key] = OrderedDict()
            for inner_key in idv_score[outer_key]:
                if inner_key not in list(scores.keys()):
                    scores[outer_key][inner_key] = OrderedDict()
                _, _, _, task_name, _ = DirectoryTree.extract_info_from_storage_name(storage_name)
                scores[outer_key][inner_key] = idv_score[outer_key][inner_key].mean(axis=2)

    # First storage_dir will serve as reference if normalize_with_first_model is True

    reference_key = list(scores.keys())[0]
    reference_means = OrderedDict()
    for inner_key in scores[reference_key].keys():
        if normalize_with_first_model:
            reference_means[inner_key] = scores[reference_key][inner_key].mean()
        else:
            reference_means[inner_key] = 1.

    # Sorts inner_keys (bars among groups)

    sorted_inner_keys = list(reversed(sorted(reference_means.keys(),
                                             key=lambda item: (scores[reference_key][item].mean(), item))))

    if sort_bars:
        inner_keys = sorted_inner_keys
    else:
        inner_keys = scores[reference_key].keys()

    # Computes means and error bars

    for inner_key in inner_keys:
        for outer_key in scores.keys():
            if outer_key not in scores_means.keys():
                scores_means[outer_key] = OrderedDict()
                scores_err_up[outer_key] = OrderedDict()
                scores_err_down[outer_key] = OrderedDict()

            scores_means[outer_key][inner_key] = np.mean(
                scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8))

            if std_error:
                scores_err_down[outer_key][inner_key] = np.std(
                    scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8)) / len(
                    scores[outer_key][inner_key]) ** 0.5
                scores_err_up[outer_key][inner_key] = scores_err_down[outer_key][inner_key]

            else:
                scores_err_down[outer_key][inner_key] = np.abs(
                    np.quantile(a=scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8), q=0. + quantile) \
                    - scores_means[outer_key][inner_key])
                scores_err_up[outer_key][inner_key] = np.abs(
                    np.quantile(a=scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8), q=1. - quantile) \
                    - scores_means[outer_key][inner_key])

    # Creates the graph

    n_bars_per_group = len(scores_means.keys())
    n_groups = len(scores_means[reference_key].keys())
    fig, ax = create_fig((1, 1), figsize=(n_bars_per_group * n_groups, n_groups))

    bar_chart(ax,
              scores=scores_means,
              err_up=scores_err_up,
              err_down=scores_err_down,
              group_names=scores_means[reference_key].keys(),
              title="Average Return"
              )

    n_training_seeds = scores[reference_key][list(scores_means[reference_key].keys())[0]].shape[0]
    n_evaluation_runs = scores[reference_key][list(scores_means[reference_key].keys())[0]].shape[1]
    info_str = f"agent{agent_i}\n{n_training_seeds} training seeds\n{n_evaluation_runs} evaluation runs"

    ax.text(0.05, 0.95, info_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.1))

    # Saves storage_dirs from which the graph was created for traceability

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)
        fig.savefig(storage_dir / save_dir / f'{save_dir}_performance.png')
        save_dict_to_json({'sources': str(storage_dir) in storage_dirs,
                           'agent_i': agent_i,
                           'n_training_seeds': n_training_seeds,
                           'n_evaluation_runs': n_evaluation_runs,
                           },
                          storage_dir / save_dir / f'{save_dir}_performance_sources.json')

    plt.close(fig)

    return sorted_inner_keys


def _gather_experiments_training_curves(storage_dir, agent_i, graph_key, curve_key, logger, episodes=None, curves=None):
    assert graph_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']
    assert curve_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']

    # Initialize containers

    if episodes is None:
        episodes = OrderedDict()
    else:
        assert type(episodes) is OrderedDict

    if curves is None:
        curves = OrderedDict()
    else:
        assert type(curves) is OrderedDict

    # Get all experiment directories

    all_experiments = DirectoryTree.get_all_experiments(storage_dir=storage_dir)

    for experiment_dir in all_experiments:

        # For that experiment, get all seed directories

        experiment_seeds = DirectoryTree.get_all_seeds(experiment_dir=experiment_dir)

        for i, seed_dir in enumerate(experiment_seeds):

            # Prints which seed directory is being treated

            logger.debug(f"{seed_dir}")

            # Loads training config

            config_dict = load_dict_from_json(str(seed_dir / "config.json"))

            # Selects how data will be identified

            keys = {
                "task_name": config_dict["task_name"],
                "storage_name": seed_dir.parents[1],
                "alg_name": config_dict["alg_name"],
                "experiment_num": seed_dir.parent.stem.strip('experiment')
            }

            outer_key = keys[graph_key]  # number of graphs to be made
            inner_key = keys[curve_key]  # number of curves per graph

            # Loads training data

            loaded_recorder = TrainingRecorder.init_from_pickle_file(
                filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

            # Stores the data

            if outer_key not in curves.keys():
                episodes[outer_key] = OrderedDict()
                curves[outer_key] = OrderedDict()

            if inner_key not in curves[outer_key].keys():
                episodes[outer_key][inner_key] = []
                curves[outer_key][inner_key] = []

            episodes[outer_key][inner_key].append(loaded_recorder.tape['eval_episodes'])

            _, _, _, task_name, _ = DirectoryTree.extract_info_from_storage_name(storage_dir.name)
            if task_name == 'compromise':
                curves[outer_key][inner_key].append(np.stack(loaded_recorder.tape['eval_total_reward']).mean(axis=2))
            else:
                curves[outer_key][inner_key].append(np.stack(loaded_recorder.tape['eval_total_reward'])[:, :, agent_i])

    return episodes, curves


def _make_benchmark_learning_figure(episodes, curves, storage_dirs, save_dir, n_labels=np.inf):
    # Initialize containers

    curves_means = OrderedDict()
    curves_stds = OrderedDict()
    labels = OrderedDict()

    for outer_key in curves:
        curves_means[outer_key] = OrderedDict()
        curves_stds[outer_key] = OrderedDict()

    # Initialize figure

    n_graphs = len(curves.keys())

    if n_graphs > 1:
        axes_shape = (2, int(np.ceil(len(curves.keys()) / 2.)))
    else:
        axes_shape = (1, 1)

    fig, axes = create_fig(axes_shape)

    # Compute means and stds for all inner_key curve from raw data

    for i, outer_key in enumerate(curves.keys()):

        for inner_key in curves[outer_key].keys():
            episodes[outer_key][inner_key] = episodes[outer_key][inner_key][0]
            curves_means[outer_key][inner_key] = np.stack(curves[outer_key][inner_key], axis=2).mean(axis=(1, 2))
            curves_stds[outer_key][inner_key] = np.stack(curves[outer_key][inner_key], axis=2).std(axis=(1, 2))

        labels[outer_key] = list(curves_means[outer_key].keys())

        # Limits the number of labels to be displayed (only displays labels of n_labels best experiments)

        if n_labels < np.inf:
            mean_over_entire_curves = np.array([array.mean() for array in curves_means[outer_key].values()])
            n_max_idxs = (-mean_over_entire_curves).argsort()[:n_labels]

            for ii in range(len(labels[outer_key])):
                if ii in n_max_idxs:
                    continue
                else:
                    labels[outer_key][i] = None

        # Selects right ax object

        if axes_shape == (1, 1):
            current_ax = axes
        elif any(np.array(axes_shape) == 1):
            current_ax = axes[i]
        else:
            current_ax = axes[i // axes_shape[1], i % axes_shape[1]]

        # Plots the curves

        plot_curves(current_ax,
                    xs=list(episodes[outer_key].values()),
                    ys=np.stack(list(curves_means[outer_key].values())),
                    stds=np.stack(list(curves_stds[outer_key].values())),
                    labels=labels[outer_key],
                    xlabel="Episodes",
                    title=outer_key)

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)
        fig.savefig(storage_dir / save_dir / f'{save_dir}_learning.png')

    plt.close(fig)


def _make_vertical_densities_figure(storage_dirs, agent_i, save_dir, logger):
    # Initialize container

    all_means = OrderedDict()

    # Gathers data

    for storage_dir in storage_dirs:
        logger.debug(storage_dir)

        # Loads the scores saved by summarize_search

        with open(str(storage_dir / save_dir / f"{save_dir}_performance_data.pkl"), "rb") as f:
            scores = pickle.load(f)

        # Taking the mean across seeds and evaluation-runs for each experiment

        x = list(scores.keys())[0]
        storage_name = storage_dir.name

        # Adding task_name if first time it is encountered

        task_name = storage_name.split("_")[4]
        if task_name not in list(all_means.keys()):
            all_means[task_name] = OrderedDict()

        # Taking the mean across evaluations and seeds for agent_i

        _, _, _, task_name, _ = DirectoryTree.extract_info_from_storage_name(storage_dir.name)
        all_means[task_name][storage_name] = [array.mean() for array in scores[x].values()]

    # Initialize figure

    n_graphs = len(all_means.keys())

    if n_graphs > 1:
        axes_shape = (2, int(np.ceil(len(all_means.keys()) / 2.)))
    else:
        axes_shape = (1, 1)

    fig, axes = create_fig(axes_shape)

    for i, task_name in enumerate(all_means.keys()):

        # Selects right ax object

        if axes_shape == (1, 1):
            current_ax = axes
        elif any(np.array(axes_shape) == 1):
            current_ax = axes[i]
        else:
            current_ax = axes[i // axes_shape[1], i % axes_shape[1]]

        # Makes the plots

        plot_vertical_densities(current_ax, all_means[task_name], title=task_name, ylabel="average performance")

    # Saves the figure

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)
        fig.savefig(storage_dir / save_dir / f'{save_dir}_vertical_densities.png')
        save_dict_to_json([str(storage_dir) in storage_dirs],
                          storage_dir / save_dir / f'{save_dir}_vertical_densities_sources.json')

    plt.close(fig)


# benchmark interface ---------------------------------------------------------------------------------------------

def compare_models(storage_names, agent_i, n_benchmark_episodes, re_run_if_exists, logger, make_performance_chart=True,
                   make_learning_plots=True):
    assert type(storage_names) is list

    if make_learning_plots:
        logger.debug(f'\n{"benchmark_learning".upper()}:')

        episodes = OrderedDict()
        curves = OrderedDict()
        storage_dirs = []

        for storage_name in storage_names:
            episodes, curves = _gather_experiments_training_curves(
                storage_dir=DirectoryTree.root / storage_name,
                agent_i=agent_i,
                graph_key="task_name",
                curve_key="storage_name",
                logger=logger,
                episodes=episodes,
                curves=curves)

            storage_dirs.append(DirectoryTree.root / storage_name)

        _make_benchmark_learning_figure(episodes=episodes,
                                        curves=curves,
                                        storage_dirs=storage_dirs,
                                        n_labels=np.inf,
                                        save_dir="benchmark")

    if make_performance_chart:
        logger.debug(f'\n{"benchmark_performance".upper()}:')

        storage_dirs = []

        for storage_name in storage_names:
            _run_performance_benchmark(storage_dir=DirectoryTree.root / storage_name,
                                       n_episodes=n_benchmark_episodes,
                                       group_key="task_name",
                                       bar_key="storage_name",
                                       re_run_if_exists=re_run_if_exists,
                                       save_dir="benchmark",
                                       logger=logger)

            storage_dirs.append(DirectoryTree.root / storage_name)

        _make_benchmark_performance_figure(storage_dirs=storage_dirs,
                                           agent_i=agent_i,
                                           logger=logger,
                                           normalize_with_first_model=True,
                                           sort_bars=False,
                                           std_error=True,
                                           quantile=0.10,
                                           save_dir="benchmark")

    return


def summarize_search(storage_name, agent_i, n_benchmark_episodes, re_run_if_exists, logger, make_performance_chart=True,
                     make_learning_plots=True):
    assert type(storage_name) is str

    storage_dir = DirectoryTree.root / storage_name

    if make_learning_plots:
        logger.debug(f'\n{"benchmark_learning".upper()}:')

        episodes, curves = _gather_experiments_training_curves(storage_dir=storage_dir,
                                                               agent_i=agent_i,
                                                               graph_key="task_name",
                                                               curve_key="experiment_num",
                                                               logger=logger)

        _make_benchmark_learning_figure(episodes=episodes,
                                        curves=curves,
                                        storage_dirs=[storage_dir],
                                        n_labels=10,
                                        save_dir="summary")

    if make_performance_chart:
        logger.debug(f'\n{"benchmark_performance".upper()}:')

        _run_performance_benchmark(storage_dir=storage_dir,
                                   n_episodes=n_benchmark_episodes,
                                   group_key="experiment_num",
                                   bar_key="storage_name",
                                   re_run_if_exists=re_run_if_exists,
                                   save_dir="summary",
                                   logger=logger)

        sorted_inner_keys = _make_benchmark_performance_figure(storage_dirs=[storage_dir],
                                                               agent_i=agent_i,
                                                               logger=logger,
                                                               normalize_with_first_model=False,
                                                               sort_bars=True,
                                                               quantile=0.10,
                                                               std_error=True,
                                                               save_dir="summary")

        best_experiment_num = sorted_inner_keys[0]
        seed_dirs_for_best_exp = [path for path in (storage_dir / f"experiment{best_experiment_num}").iterdir()]
        copyfile(src=seed_dirs_for_best_exp[0] / "config.json",
                 dst=storage_dir / "summary" / f"bestConfig_exp{best_experiment_num}.json")

    return


def compare_searches(storage_names, agent_i, re_run_if_exists, logger):
    assert type(storage_names) is list

    logger.debug(f'\n{"benchmark_vertical_densities".upper()}:')

    storage_dirs = []
    for storage_name in storage_names:
        storage_dirs.append(DirectoryTree.root / storage_name)

    for storage_dir in storage_dirs:
        if not (storage_dir / "summary" / f"summary_performance_data.pkl").exists() or re_run_if_exists:
            summarize_search(storage_name=storage_dir.name,
                             agent_i=0,
                             n_benchmark_episodes=100,
                             re_run_if_exists=re_run_if_exists,
                             make_performance_chart=True,
                             make_learning_plots=True,
                             logger=logger)

    _make_vertical_densities_figure(storage_dirs, agent_i=agent_i, save_dir="summary", logger=logger)

    return


if __name__ == '__main__':
    benchmark_args = get_benchmark_args()
    logger = create_logger(name="BENCHMARK - MAIN", loglevel=logging.DEBUG)
    if benchmark_args.storage_names is None:
        if benchmark_args.benchmark_type == 'compare_models':
            benchmark_args.storage_names = official_benchmark_lists.retrains_list
        elif benchmark_args.benchmark_type == 'compare_searches':
            benchmark_args.storage_names = official_benchmark_lists.searches_list
        else:
            raise ValueError

    if benchmark_args.benchmark_type == "compare_models":
        compare_models(storage_names=benchmark_args.storage_names,
                       agent_i=benchmark_args.agent_i,
                       n_benchmark_episodes=benchmark_args.n_episodes,
                       make_performance_chart=True,
                       make_learning_plots=True,
                       re_run_if_exists=benchmark_args.re_run_if_exists,
                       logger=logger)

    elif benchmark_args.benchmark_type == "compare_searches":
        compare_searches(storage_names=benchmark_args.storage_names,
                         agent_i=benchmark_args.agent_i,
                         re_run_if_exists=benchmark_args.re_run_if_exists,
                         logger=logger)

    else:
        raise NotImplemented