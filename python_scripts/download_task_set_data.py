import argparse
import json
import os
import urllib

from concurrent import futures
import numpy as np
import tensorflow as tf # for gfile.
import tensorflow_io as tfio
import tqdm


parser = argparse.ArgumentParser(
    description='Prepare hyperparameter candidates from the taskset task',
)
parser.add_argument(
    '--task_id',
    help='The task index to retrieve from all the TaskSet tasks',
    type=int,
    default=0,
)
parser.add_argument(
    '--output_dir',
    help='The output directory where the validation curves and hyperparameter configurations will be saved',
    type=str,
    default='./taskset',
)

gfile = tf.io.gfile


def load_joint_cache(task, opt_set_name):
    """Loads the learning curves for the given task and opt_set_name."""
    base_dir = "gs://gresearch/task_set_data/"
    p = os.path.join(base_dir, task,
                     "%s_10000_replica5.npz" % (opt_set_name))
    cc = np.load(gfile.GFile(p, "rb"))
    return cc["optimizers"], cc["xs"], cc["ys"]


def threaded_tqdm_map(threads, func, data):
    """Helper that does a map on multiple threads."""
    future_list = []
    with futures.ThreadPoolExecutor(threads) as executor:
        for l in tqdm.tqdm(data, position=0):
            future_list.append(executor.submit(func, l))
        return [x.result() for x in tqdm.tqdm(future_list, position=0)]


def load_tasks(tasks):
    """Multi threaded loading of all data for each task.
    Args:
      tasks: list of task names
    Returns:
      A dictionary mapping taks name to tuples of:
      (optimizer names, x data points, and y data points)
    """

    def load_one(t):
        adam8p = load_joint_cache(t, "adam8p_wide_grid_1k")
        adam6p = load_joint_cache(t, "adam6p_wide_grid_1k")
        adam4p = load_joint_cache(t, "adam4p_wide_grid_1k")
        adam1p = load_joint_cache(t, "adam1p_wide_grid_1k")
        nadamw = load_joint_cache(t, "nadamw_grid_1k")
        return {
            "adam8p_wide_grid_1k": adam8p,
            "adam6p_wide_grid_1k": adam6p,
            "adam4p_wide_grid_1k": adam4p,
            "adam1p_wide_grid_1k": adam1p,
            "nadamw": nadamw,
        }

    results = threaded_tqdm_map(100, load_one, tasks)

    for k, v in zip(tasks, results):
        if v is None:
            print("No data found for task: %s" % k)

    return {k: v for k, v in zip(tasks, results) if v is not None}


def get_task_names():
    content = gfile.GFile("gs://gresearch/task_set_data/task_names.txt").read()
    return sorted(content.strip().split("\n"))


args = parser.parse_args()

task_id = args.task_id
task_names = get_task_names()

for task_name in task_names:
    if task_name.startswith('FixedTextRNNClassification'):

        results = load_tasks([task_name])

        # For each task, there is then a dictionary of optimizer families.
        optimizer_families = results[task_name].keys()
        # hardcode to only the search space with 8 hyperparameters now
        optimizer_name = 'adam8p_wide_grid_1k'
        optimizer_names, x, y = results[task_name][optimizer_name]

        nr_seeds = y.shape[1]
        nr_optimizations = y.shape[0]
        train_curves = []
        val_curves = []
        test_curves = []

        for hp_index in range(nr_optimizations):

            train_seed_curve = []
            valid_seed_curve = []
            test_seed_curve = []

            for seed_index in range(nr_seeds):
                train_seed_curve.append(y[hp_index, seed_index, :, 0])
                valid_seed_curve.append(y[hp_index, seed_index, :, 1])
                test_seed_curve.append(y[hp_index, seed_index, :, 2])

            train_seed_curve = np.mean(train_seed_curve, axis=0)
            valid_seed_curve = np.mean(valid_seed_curve, axis=0)
            test_seed_curve = np.mean(test_seed_curve, axis=0)

            train_curves.append(train_seed_curve.tolist())
            val_curves.append(valid_seed_curve.tolist())
            test_curves.append(test_seed_curve.tolist())

        os.makedirs(args.output_dir, exist_ok=True)

        path = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs/adam8p_wide_grid.json"
        configs = json.loads(urllib.request.urlopen(path).read())
        hparam_dicts = [configs[optname.decode("utf8")][0] for optname in optimizer_names]

        all_results = []

        for hp_index, hp_config in enumerate(hparam_dicts):

            hp_config_result = {
                'hp': hp_config,
                'train': {'loss': train_curves[hp_index]},
                'valid': {'loss': val_curves[hp_index]},
                'test': {'loss': test_curves[hp_index]},
            }

            all_results.append(hp_config_result)

        result_path = os.path.join(
            args.output_dir,
            f'{task_name}_0_{nr_optimizations}.json',
        )

        with open(result_path, 'w') as file_handle:
            json.dump(all_results, file_handle)

    else:
        continue