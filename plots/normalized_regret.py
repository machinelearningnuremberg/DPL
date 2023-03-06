import json
import os
from typing import Tuple, List

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import seaborn as sns

from benchmarks.benchmark import BaseBenchmark
from benchmarks.lcbench import LCBench
from benchmarks.taskset import TaskSet
from benchmarks.hyperbo import PD1


sns.set_style('white')

sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 31,
        'axes.titlesize': 31,
        'axes.labelsize': 31,
        'xtick.labelsize': 31,
        'ytick.labelsize': 31,
        'legend.fontsize': 31,
    },
    style="white"
)


result_path = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'results',
    )
)

project_folder = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'PhD',
        'Projekte',
        'DeepRegret',
    )
)

method_names = [
    'power_law',
    'ahb',
    'bohb',
    'dehb',
    'dragonfly',
    'hyperband',
    'random',
    'smac',
]

pretty_names = {
    'power_law': 'DPL',
    'ahb': 'ASHA',
    'bohb': 'BOHB',
    'dehb': 'DEHB',
    'dragonfly': 'Dragonfly',
    'hyperband': 'Hyperband',
    'random': 'Random',
    'smac': 'SMAC',
}


def get_method_dataset_regret_time_performance(
        benchmark: BaseBenchmark,
        dataset_name: str,
        method_name: str,
        benchmark_surrogate_results: str,
        benchmark_name: str = 'lcbench',
) -> Tuple[List, np.ndarray]:
    """Retrieve the time taken and the incumbent curve
    of a method for a particular dataset.

    Retrieves the time taken and the incumbent curve over the
    HPO budget for a given method, for a particular dataset from
    a benchmark.

    Args:
        benchmark: BaseBenchmark
            The benchmark object.
        dataset_name: str
            The dataset for which the results will be generated.
        method_name: str
            The name of the method for which the results will be generated.
        benchmark_surrogate_results: str
            The path there the results are stored for all methods.
        benchmark_name: str
            The name of the benchmark.

    Returns:
        iterations_overhead, baseline_incumbent_curve: Tuple
            The time taken for every HPO step and the baseline incumbent curve over
            every HPO step.
    """
    gap_performance = benchmark.get_gap_performance()

    total_overhead = 0
    iterations_overhead = []
    baseline_incumbent_curves = []
    baseline_overhead = []
    configuration_eval_overhead = []

    repeat_nrs = [nr for nr in range(0, 10)]

    longest_run = 0

    for repeat_nr in repeat_nrs:

        dataset_result_file = os.path.join(benchmark_surrogate_results, method_name, f'{dataset_name}_{repeat_nr}.json')
        try:
            with open(dataset_result_file, 'r') as fp:
                result_info = json.load(fp)
        except FileNotFoundError:
            # no results for this repeat/seed, move on to the next
            continue

        baseline_incumbent_curve = result_info['curve']
        if len(baseline_incumbent_curve) < 1000:
            continue
        elif len(baseline_incumbent_curve) > 1000:
            baseline_incumbent_curve = baseline_incumbent_curve[0:1000]

        baseline_incumbent_curves.append(baseline_incumbent_curve)

        if 'overhead' in result_info:
            baseline_overhead.append(result_info['overhead'])

        evaluated_hps = result_info['hp']
        budgets_evaluated = result_info['epochs']

        config_times = []
        for hp_index, budget_eval in zip(evaluated_hps, budgets_evaluated):
            config_time = benchmark.get_step_cost(
                hp_index=hp_index,
                budget=budget_eval,
            )
            config_times.append(config_time)
        if len(config_times) > longest_run:
            longest_run = len(config_times)
        configuration_eval_overhead.append(config_times)

    if len(baseline_incumbent_curves) == 0:
        return [], []

    mean_config_eval_values = []

    for curve_point in range(0, len(baseline_incumbent_curves[0])):
        mean_config_point_values = []
        for curve_nr in range(0, len(configuration_eval_overhead)):
            config_curve = configuration_eval_overhead[curve_nr]
            if len(config_curve) > curve_point:
                mean_config_point_values.append(config_curve[curve_point])
            else:
                continue
        mean_config_eval_values.append(np.mean(mean_config_point_values))

    # take the mean value over all repeats
    baseline_incumbent_curve = np.mean(baseline_incumbent_curves, axis=0)
    best_incumbent_curve = benchmark.get_incumbent_curve()
    if benchmark_name == 'lcbench':
        incumbent_best_performance = max(best_incumbent_curve)
    else:
        incumbent_best_performance = min(best_incumbent_curve)

    worst_performance = benchmark.get_worst_performance()

    if benchmark_name == 'lcbench':
        # convert the incumbent iteration performance to the normalized regret for every iteration
        baseline_incumbent_curve = [(incumbent_best_performance - incumbent_it_performance) / gap_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]
    elif benchmark_name == 'taskset':
        if method_name != 'power_law' and method_name != 'smac':
            baseline_incumbent_curve = [worst_performance - curve_element for curve_element in baseline_incumbent_curve]

        baseline_incumbent_curve = [(incumbent_it_performance - incumbent_best_performance) / gap_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]
    elif benchmark_name == 'pd1':
        if method_name == 'dragonfly' or method_name == 'ahb':
            baseline_incumbent_curve = [1.0 - curve_element for curve_element in baseline_incumbent_curve]
            # convert the incumbent iteration performance to the normalized regret for every iteration
        baseline_incumbent_curve = [(incumbent_it_performance - incumbent_best_performance) / gap_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]

    if len(baseline_overhead) > 0:

        mean_baseline_eval_values = []
        for curve_point in range(0, len(baseline_incumbent_curve)):
            mean_config_point_values = []
            for curve_nr in range(0, len(baseline_overhead)):
                config_curve = baseline_overhead[curve_nr]
                if len(config_curve) > curve_point:
                    mean_config_point_values.append(config_curve[curve_point])
                else:
                    continue
            mean_baseline_eval_values.append(np.mean(mean_config_point_values))

        for baseline_overhead_it, config_overhead in zip(mean_baseline_eval_values, mean_config_eval_values):
            total_overhead = total_overhead + baseline_overhead_it + config_overhead
            iterations_overhead.append(total_overhead)
    else:
        for config_overhead in mean_config_eval_values:
            total_overhead = total_overhead + config_overhead
            iterations_overhead.append(total_overhead)

    return iterations_overhead, baseline_incumbent_curve


def get_method_dataset_regret_epoch_performance(
        benchmark: BaseBenchmark,
        dataset_name: str,
        method_name: str,
        benchmark_surrogate_results: str,
        benchmark_name: str = 'lcbench',
) -> Tuple[List, np.ndarray, np.ndarray]:
    """Retrieve the epochs taken and the incumbent curve
    of a method for a particular dataset.

    Retrieves the epochs taken and the incumbent curve over the
    HPO budget for a given method, for a particular dataset from
    a benchmark.

    Args:
        benchmark: BaseBenchmark
            The benchmark object.
        dataset_name: str
            The dataset for which the results will be generated.
        method_name: str
            The name of the method for which the results will be generated.
        benchmark_surrogate_results: str
            The path there the results are stored for all methods.
        benchmark_name: str
            The name of the benchmark.

    Returns:
        iteration_cost, baseline_incumbent_curve, baseline_incumbent_std: Tuple
            The epochs taken for every HPO step, the baseline incumbent curve over
            every HPO step and the baseline incumbent std over the HPO steps.
    """
    gap_performance = benchmark.get_gap_performance()

    baseline_epochs = []
    baseline_incumbent_curves = []

    repeat_nrs = [nr for nr in range(0, 10)]

    for repeat_nr in repeat_nrs:
        repeat_epochs_cost = []
        configs_evaluated = dict()
        dataset_result_file = os.path.join(benchmark_surrogate_results, method_name, f'{dataset_name}_{repeat_nr}.json')
        try:
            with open(dataset_result_file, 'r') as fp:
                result_info = json.load(fp)
        except FileNotFoundError:
            # no results for this repeat/seed, move on to the next
            continue

        baseline_incumbent_curve = result_info['curve']

        if len(baseline_incumbent_curve) < 1000:
            continue
        elif len(baseline_incumbent_curve) > 1000:
            baseline_incumbent_curve = baseline_incumbent_curve[0:1000]
        baseline_incumbent_curves.append(baseline_incumbent_curve)

        evaluated_hps = result_info['hp']
        budgets_evaluated = result_info['epochs']

        for evaluated_hp, budget_evaluated in zip(evaluated_hps, budgets_evaluated):
            if evaluated_hp in configs_evaluated:
                budgets = configs_evaluated[evaluated_hp]
                max_budget = max(budgets)
                cost = budget_evaluated - max_budget
                repeat_epochs_cost.append(cost)
                configs_evaluated[evaluated_hp].append(budget_evaluated)
            else:
                repeat_epochs_cost.append(budget_evaluated)
                configs_evaluated[evaluated_hp] = [budget_evaluated]

        baseline_epochs.append(repeat_epochs_cost)

    if len(baseline_incumbent_curves) == 0:
        return [], []

    mean_cost_values = []
    try:
        for curve_point in range(0, len(baseline_incumbent_curves[0])):
            mean_config_point_values = []
            for curve_nr in range(0, len(baseline_epochs)):
                config_curve = baseline_epochs[curve_nr]
                if len(config_curve) > curve_point:
                    mean_config_point_values.append(config_curve[curve_point])
                else:
                    continue
            if len(mean_config_point_values) > 0:
                mean_cost_values.append(np.mean(mean_config_point_values))
    except Exception:
        return [], []

    if len(mean_cost_values) < len(baseline_incumbent_curves[0]):
        mean_cost_values = [1 for _ in range(1, len(baseline_incumbent_curves[0]) + 1)]
    iteration_cost = []

    total_iteration_cost = 0
    for iteration_cost_value in mean_cost_values:
        total_iteration_cost += iteration_cost_value
        iteration_cost.append(total_iteration_cost)

    # take the mean value over all repeats and the standard deviation
    baseline_incumbent_curve = np.mean(baseline_incumbent_curves, axis=0)
    baseline_incumbent_std = np.std(baseline_incumbent_curves, axis=0)

    incumbent_best_performance = benchmark.get_best_performance()

    worst_performance = benchmark.get_worst_performance()

    if benchmark_name == 'lcbench':
        # convert the incumbent iteration performance to the normalized regret for every iteration
        baseline_incumbent_curve = [incumbent_best_performance - incumbent_it_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]
    elif benchmark_name == 'taskset':
        if method_name != 'power_law' and method_name != 'smac':
            baseline_incumbent_curve = [worst_performance - curve_element for curve_element in baseline_incumbent_curve]

        baseline_incumbent_curve = [(incumbent_it_performance - incumbent_best_performance)  # / gap_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]
    elif benchmark_name == 'pd1':
        if method_name == 'dragonfly' or method_name == 'ahb':
            baseline_incumbent_curve = [1.0 - curve_element for curve_element in baseline_incumbent_curve]
            # convert the incumbent iteration performance to the normalized regret for every iteration
        baseline_incumbent_curve = [(incumbent_it_performance - incumbent_best_performance) / gap_performance
                                    for incumbent_it_performance in baseline_incumbent_curve]

    return iteration_cost, baseline_incumbent_curve, baseline_incumbent_std


def get_method_dataset_number_configs(
        benchmark: BaseBenchmark,
        dataset_name: str,
        method_name: str,
        benchmark_surrogate_results: str,
        benchmark_name: str = 'lcbench',
) -> float:
    """Calculate the number of unique configurations
    explored.

    Calculates the number of unique configurations that
    were explored during the HPO phase.

    Args:
        benchmark: BaseBenchmark
            The benchmark object.
        dataset_name: str
            The dataset for which the number of configurations will
            be calculated.
        method_name: str
            The method name.
        benchmark_surrogate_results: str
            The path where the results are located.
        benchmark_name: str
            The name of the benchmark

    Returns: float
        The number of configurations explored averaged over the repetitions.
    """
    repeat_nrs = [nr for nr in range(0, 10)]

    number_configs_repeats = []
    for repeat_nr in repeat_nrs:
        dataset_result_file = os.path.join(benchmark_surrogate_results, method_name, f'{dataset_name}_{repeat_nr}.json')
        try:
            with open(dataset_result_file, 'r') as fp:
                result_info = json.load(fp)
        except FileNotFoundError:
            # no results for this repeat/seed, move on to the next
            continue

        if benchmark_name == 'pd1':
            if benchmark.max_budget < 50:
                max_epochs = int(benchmark.max_budget * 20)
                result_info = result_info['hp']
                result_info = result_info[0:max_epochs]
            else:
                return -1
        else:
            result_info = result_info['hp']

        unique_configs = len(set(result_info))
        number_configs_repeats.append(unique_configs)

    return np.mean(number_configs_repeats)


def get_method_dataset_number_max_configs(
        benchmark: BaseBenchmark,
        dataset_name: str,
        method_name: str,
        benchmark_surrogate_results: str,
        benchmark_name: str = 'lcbench',
) -> float:
    """Return the number of unique configurations that
    were explored at the max budget.

    Return the number of unique configurations explored during
    the HPO phase until the end/max budget.

    Args:
        benchmark: BaseBenchmark
            The benchmark object.
        dataset_name: str
            The name of the dataset.
        method_name: str
            The method name.
        benchmark_surrogate_results: str
            The path where the results are located.
        benchmark_name: str
            The benchmark name.

    Returns: float
        The number of configurations explored maximally.
    """
    repeat_nrs = [nr for nr in range(0, 10)]

    number_configs_repeats = []
    for repeat_nr in repeat_nrs:
        dataset_result_file = os.path.join(benchmark_surrogate_results, method_name, f'{dataset_name}_{repeat_nr}.json')
        try:
            with open(dataset_result_file, 'r') as fp:
                result_info = json.load(fp)
        except FileNotFoundError:
            # no results for this repeat/seed, move on to the next
            continue

        if benchmark_name == 'pd1':
            if benchmark.max_budget < 60:
                max_epochs = int(benchmark.max_budget * 18)
                result_info = result_info['hp']
                result_budgets = result_info['epochs']
                result_info = result_info[0:max_epochs]
                result_budgets = result_budgets[0:max_epochs]
            else:
                return -1
        else:
            result_info = result_info['hp']
            result_budgets = result_info['epochs']

        config_ids = []
        for budget_index, budget in enumerate(result_budgets):
            if budget == benchmark.max_budget:
                config_ids.append(result_info[budget_index])

        unique_configs = len(set(result_info))
        number_configs_repeats.append(unique_configs)

    return np.mean(number_configs_repeats)


def generate_walltime_data(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
):
    benchmark_class = {
        'pd1': PD1,
    }

    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit_g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    method_dataset_times = {}
    method_dataset_performances = {}
    updated_dataset_names = []
    success_datasets = dict()

    for method_name in method_names:
        dataset_iteration_overhead = []
        dataset_normalized_regret = []
        for dataset_name in dataset_names:
            if benchmark_name == 'lcbench':
                benchmark.set_dataset_name(dataset_name)
            elif benchmark_name == 'taskset':
                benchmark = TaskSet(benchmark_data_path, dataset_name)
            else:
                benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

            iterations_overhead, baseline_incumbent_performance = get_method_dataset_regret_time_performance(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name=benchmark_name
            )
            if benchmark_name == 'pd1':
                if benchmark.max_budget < 50:
                    max_epochs = int(benchmark.max_budget * 20)
                    iterations_overhead = iterations_overhead[0: max_epochs]
                    baseline_incumbent_performance = baseline_incumbent_performance[0:max_epochs]
                    if method_name == 'random':
                        updated_dataset_names.append(dataset_name)
                else:
                    continue
            # if len(baseline_incumbent_performance) < 1000:
            #    continue

            if method_name in success_datasets:
                success_datasets[method_name].append(dataset_name)
            else:
                success_datasets[method_name] = [dataset_name]
            dataset_iteration_overhead.append(iterations_overhead)
            dataset_normalized_regret.append(baseline_incumbent_performance)

        method_dataset_times[method_name] = dataset_iteration_overhead
        method_dataset_performances[method_name] = dataset_normalized_regret

    random_times = method_dataset_times['random']

    for method_name in method_names:
        method_times = method_dataset_times[method_name]
        reconstructed_method_times = []
        dataset_names = success_datasets[method_name]
        for dataset_index, _ in enumerate(dataset_names):
            total_random_time = random_times[dataset_index][-1]
            dataset_times = method_times[dataset_index]

            dataset_times = np.array(dataset_times) / total_random_time
            reconstructed_method_times.append(dataset_times)
        method_dataset_times[method_name] = reconstructed_method_times

    datasets_mean_performances = []
    datasets_mean_times = []
    for method_name in method_names:
        method_times = method_dataset_times[method_name]
        method_performances = method_dataset_performances[method_name]

        if benchmark_name == 'pd1':
            dataset_index = 0
            for dataset_method_performance, dataset_method_time in zip(method_performances, method_times):
                dataset_time_curve = []
                dataset_performance_curve = []

                for dataset_index_point, dataset_time_point in enumerate(dataset_method_time):
                    if dataset_time_point <= 1.0:
                        dataset_performance_curve.append(dataset_method_performance[dataset_index_point])
                        dataset_time_curve.append(dataset_method_time[dataset_index_point])
                    elif dataset_time_point > 1:
                        method_performances[dataset_index] = dataset_performance_curve
                        method_times[dataset_index] = dataset_time_curve

                dataset_index += 1

        if benchmark_name == 'pd1':
            min_curve_length = 1000
            min_curve_index = -1

            for i in range(0, len(method_times)):
                curve_length = len(method_times[i])
                if curve_length < min_curve_length:
                    min_curve_length = curve_length
                    min_curve_index = i

            min_curve = method_times[min_curve_index]
            for i in range(0, len(method_times)):
                current_curve = method_times[i]
                current_performance_curve = method_performances[i]
                min_curve_point_index = 0
                transformed_curve = []
                transformed_performances = []
                for point_index, point in enumerate(current_curve):
                    curve_time = point
                    min_curve_time = min_curve[min_curve_point_index]
                    if curve_time >= min_curve_time:
                        min_curve_point_index += 1
                        if min_curve_point_index == len(min_curve):
                            break
                        transformed_curve.append(curve_time)
                        transformed_performances.append(current_performance_curve[point_index])

                method_times[i] = transformed_curve
                method_performances[i] = transformed_performances

            max_curve_length = -1
            max_curve_index = -1
            for i in range(0, len(method_times)):
                curve_length = len(method_times[i])
                if curve_length > max_curve_length:
                    max_curve_length = curve_length
                    max_curve_index = i

            for i in range(0, len(method_times)):
                if i == max_curve_index:
                    continue
                else:
                    method_dataset_performance = method_performances[i]
                    method_dataset_time = method_times[i]

                    current_curve_length = len(method_dataset_time)
                    difference = max_curve_length - current_curve_length
                    method_dataset_performance.extend([method_dataset_performance[-1]] * difference)
                    method_dataset_time.extend([method_dataset_time[-1]] * difference)
                    method_times[i] = method_dataset_time
                    method_performances[i] = method_dataset_performance

        dataset_iteration_overhead = np.mean(method_times, axis=0)
        dataset_normalized_regret = np.mean(method_performances, axis=0)
        cut_dataset_iteration_overhead = []
        cut_dataset_normalized_regret = []
        for time, regret in zip(dataset_iteration_overhead, dataset_normalized_regret):
            if time <= 1:
                cut_dataset_iteration_overhead.append(time)
                cut_dataset_normalized_regret.append(regret)

        datasets_mean_times.append(cut_dataset_iteration_overhead)
        datasets_mean_performances.append(cut_dataset_normalized_regret)

    return datasets_mean_times, datasets_mean_performances


def generate_epoch_performance_data(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
):
    benchmark_class = {
        'pd1': PD1,
    }

    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit_g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    method_mean_regret_performance = []
    method_mean_epochs = []
    for method_name in method_names:
        method_epochs_taken = []
        dataset_normalized_regret = []
        dataset_normalized_std = []
        for dataset_name in dataset_names:
            if benchmark_name == 'lcbench':
                benchmark.set_dataset_name(dataset_name)
            elif benchmark_name == 'taskset':
                benchmark = TaskSet(benchmark_data_path, dataset_name)
            else:
                benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

            dataset_epochs_taken, baseline_incumbent_performance, baseline_incumbent_std = get_method_dataset_regret_epoch_performance(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name,
            )

            if len(dataset_epochs_taken) < 1000:
                continue

            if benchmark_name == 'pd1':
                if benchmark.max_budget <= 50:
                    max_epochs = int(benchmark.max_budget * 20)
                    dataset_epochs_taken = dataset_epochs_taken[0: max_epochs]
                    baseline_incumbent_performance = baseline_incumbent_performance[0:max_epochs]
                    baseline_incumbent_std = baseline_incumbent_std[0:max_epochs]
                    dataset_epochs_taken = [point / max_epochs for point in dataset_epochs_taken]
                else:
                    continue

            method_epochs_taken.append(dataset_epochs_taken)
            dataset_normalized_regret.append(baseline_incumbent_performance)
            dataset_normalized_std.append(baseline_incumbent_std)

        if benchmark_name == 'pd1':
            min_curve_length = 1000
            min_curve_index = -1
            for i in range(0, len(method_epochs_taken)):
                curve_length = len(method_epochs_taken[i])
                if curve_length < min_curve_length:
                    min_curve_length = curve_length
                    min_curve_index = i

            min_curve = method_epochs_taken[min_curve_index]
            for i in range(0, len(method_epochs_taken)):
                if i == min_curve_index:
                    continue
                current_curve = method_epochs_taken[i]
                current_performance_curve = dataset_normalized_regret[i]
                current_std_curve = dataset_normalized_std[i]
                curve_epochs = 0
                min_curve_index = 0
                transformed_curve = []
                transformed_performances = []
                tranformed_stds = []
                for point_index, point in enumerate(current_curve):
                    curve_epochs = point
                    if curve_epochs > min_curve[min_curve_index]:
                        min_curve_index += 1
                        if min_curve_index == len(min_curve):
                            break
                        transformed_curve.append(curve_epochs)
                        transformed_performances.append(current_performance_curve[point_index])
                        tranformed_stds.append(current_std_curve[point_index])
                dataset_normalized_regret[i] = transformed_performances
                method_epochs_taken[i] = transformed_curve
                dataset_normalized_std[i] = tranformed_stds

        mean_epochs_taken_method = np.mean(method_epochs_taken, axis=0)
        dataset_normalized_regret = np.mean(dataset_normalized_regret, axis=0)
        dataset_normalized_std = np.mean(dataset_normalized_std, axis=0)

        method_mean_regret_performance.append(dataset_normalized_regret)
        method_mean_epochs.append(mean_epochs_taken_method)

    return method_mean_epochs, method_mean_regret_performance, dataset_normalized_std

def plot_rank_performance(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
):
    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    benchmark_class = {
        'pd1': PD1,
        'nasbench201': NASBench201,
    }
    # It loads all datasets in one go, so we cannot load it for every dataset like the
    # other ones.
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit-g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    plotting_method_ranks = dict()
    plotting_method_epochs = dict()

    for dataset_index, dataset_name in enumerate(dataset_names):
        if benchmark_name == 'lcbench':
            benchmark.set_dataset_name(dataset_name)
        elif benchmark_name == 'taskset':
            benchmark = TaskSet(benchmark_data_path, dataset_name)
        else:
            benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

        method_performances = []
        for method_name in method_names:
            dataset_epochs_taken, baseline_incumbent_performance, _ = get_method_dataset_regret_epoch_performance(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name,
            )
            if len(baseline_incumbent_performance) < 1000:
                continue

            if method_name in plotting_method_epochs:
                plotting_method_epochs[method_name].append(dataset_epochs_taken)
            else:
                plotting_method_epochs[method_name] = [dataset_epochs_taken]

            method_performances.append(baseline_incumbent_performance)

        dataset_method_ranks = dict()
        for iteration_point in range(0, 1000):
            method_point_performances = []
            for method_index in range(0, len(method_names)):
                method_curve = method_performances[method_index]
                method_point_curve = method_curve[iteration_point]
                method_point_performances.append(method_point_curve)

            method_ranks = rankdata(method_point_performances, method='min')
            for method_name, method_rank in zip(method_names, method_ranks):
                if method_name in dataset_method_ranks:
                    dataset_method_ranks[method_name].append(method_rank)
                else:
                    dataset_method_ranks[method_name] = [method_rank]

        for method_name in method_names:
            if method_name in plotting_method_ranks:
                plotting_method_ranks[method_name].append(dataset_method_ranks[method_name])
            else:
                plotting_method_ranks[method_name] = [dataset_method_ranks[method_name]]

    all_data = []
    for method_name in method_names:
        plotting_method_ranks[method_name] = np.mean(plotting_method_ranks[method_name], axis=0)
        all_data.append(plotting_method_ranks[method_name])
        plotting_method_epochs[method_name] = np.mean(plotting_method_epochs[method_name], axis=0)

        plt.plot(plotting_method_epochs[method_name], plotting_method_ranks[method_name], label=method_name)

    # plt.violinplot(all_data)

    plt.ylabel('Rank')
    # set style for the axes
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.xticks([i for i in range(1, len(method_names) + 1)], method_names)
    plt.savefig(f'comparison_ranks_{benchmark_name}.pdf', bbox_inches="tight")


def build_cd_diagram(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
        half: bool,
) -> pd.DataFrame:
    table_results = {
        'Method': [],
        'Dataset Name': [],
        'Regret': [],
    }

    benchmark_class = {
        'pd1': PD1,
        'nasbench201': NASBench201,
    }

    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit_g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    for method_name in method_names:
        for dataset_name in dataset_names:
            if benchmark_name == 'lcbench':
                benchmark.set_dataset_name(dataset_name)
            elif benchmark_name == 'taskset':
                benchmark = TaskSet(benchmark_data_path, dataset_name)
            else:
                benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

            _, baseline_incumbent_performance, _ = get_method_dataset_regret_epoch_performance(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name,
                half,
            )

            if benchmark_name == 'pd1':
                if benchmark.max_budget * 20 > 1000:
                    continue
                baseline_incumbent_performance = baseline_incumbent_performance[0:int(benchmark.max_budget * 20)]
            if half:
                baseline_incumbent_performance = baseline_incumbent_performance[
                                                 0:int(len(baseline_incumbent_performance) / 2)]
            final_performance = baseline_incumbent_performance[-1]
            table_results['Method'].append(pretty_names[method_name])
            table_results['Dataset Name'].append(dataset_name)
            table_results['Regret'].append(final_performance)

    result_df = pd.DataFrame(data=table_results)
    if half:
        file_extension = 'half'
    else:
        file_extension = 'full'

    result_df.to_csv(f'{benchmark_name}_cdinfo_{file_extension}.csv', index=False)

    return result_df


def plot_config_distribution(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
):
    benchmark_class = {
        'pd1': PD1,
        'nasbench201': NASBench201,
    }

    config_info_dict = {
        'Method': [],
        'Dataset Name': [],
        'Number Of Configurations': [],
    }
    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit_g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    # TODO remove hack
    method_names = ['power_law', 'ahb', 'dehb', 'dragonfly', 'random', 'bohb', 'smac', 'hyperband']  # 'nonlc_power_law'
    for method_name in method_names:
        for dataset_name in dataset_names:
            if benchmark_name == 'lcbench':
                benchmark.set_dataset_name(dataset_name)
            elif benchmark_name == 'taskset':
                benchmark = TaskSet(benchmark_data_path, dataset_name)
            else:
                benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

            dataset_nr_configs = get_method_dataset_number_configs(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name,
            )
            if dataset_nr_configs == -1:
                continue

            config_info_dict['Method'].append(pretty_names[method_name])
            config_info_dict['Dataset Name'].append(dataset_name)
            config_info_dict['Number Of Configurations'].append(dataset_nr_configs)

    ax = sns.violinplot(y='Method', x='Number Of Configurations', data=config_info_dict, kind='violin')

    if benchmark_name == 'lcbench':
        plt.title('LCBench')
        # plt.ylabel('Average Normalized Regret')
    elif benchmark_name == 'taskset':
        plt.title('TaskSet')
    elif benchmark_name == 'pd1':
        plt.title('PD1')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.savefig(f'config_distribution_{benchmark_name}.pdf', bbox_inches="tight")


def plot_dataset_epoch_performance(
        benchmark_data_path: str,
        surrogate_results_path: str,
        benchmark_name: str,
):
    benchmark_class = {
        'pd1': PD1,
        'nasbench201': NASBench201,
    }

    benchmark_surrogate_results = os.path.join(surrogate_results_path, benchmark_name)
    if benchmark_name == 'lcbench':
        benchmark = LCBench(benchmark_data_path, 'credit_g')
    elif benchmark_name == 'taskset':
        benchmark = TaskSet(benchmark_data_path, 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128')

    if benchmark_name == 'lcbench' or benchmark_name == 'taskset':
        dataset_names = benchmark.load_dataset_names()
    else:
        dataset_names = benchmark_class[benchmark_name].load_dataset_names()

    for dataset_index, dataset_name in enumerate(dataset_names):
        if benchmark_name == 'lcbench':
            benchmark.set_dataset_name(dataset_name)
        elif benchmark_name == 'taskset':
            benchmark = TaskSet(benchmark_data_path, dataset_name)
        else:
            benchmark = benchmark_class[benchmark_name](benchmark_data_path, dataset_name)

        fig = plt.figure()
        if benchmark_name == 'pd1':
            if benchmark.max_budget > 50:
                continue

        for method_name in method_names:

            dataset_epochs_taken, baseline_incumbent_mean, baseline_incumbent_std = get_method_dataset_regret_epoch_performance(
                benchmark,
                dataset_name,
                method_name,
                benchmark_surrogate_results,
                benchmark_name,
            )

            if len(dataset_epochs_taken) < 1000:
                continue

            if benchmark_name == 'pd1':
                if benchmark.max_budget <= 50:
                    max_epochs = int(benchmark.max_budget * 20)
                    dataset_epochs_taken = dataset_epochs_taken[0: max_epochs]
                    baseline_incumbent_mean = baseline_incumbent_mean[0:max_epochs]
                    baseline_incumbent_std = baseline_incumbent_std[0:max_epochs]
                    dataset_epochs_taken = [point / max_epochs for point in dataset_epochs_taken]
                else:
                    continue
            hpo_budget = dataset_epochs_taken / dataset_epochs_taken[-1]
            plt.plot(hpo_budget, baseline_incumbent_mean, label=pretty_names[method_name], linewidth=4)

        plt.ylabel('Average Regret')
        if benchmark_name == 'lcbench':
            plt.xlabel('HPO Budget')
            plt.title(f'{dataset_name}')
        elif benchmark_name == 'taskset':
            plt.xlabel('HPO Budget')
            plt.title(f'{dataset_name[0:26]}\n{dataset_name[27:-1]}')
        else:
            plt.xlabel('HPO Budget')
            plt.title(f'{dataset_name}')
        plt.tick_params(left=True, bottom=True)
        plt.legend(bbox_to_anchor=(0.5, -0.52), loc='lower center', ncol=3)
        plt.savefig(f'./{benchmark_name}/{dataset_name}_regret_epochs.pdf', bbox_inches='tight')
        plt.close(fig)


def plot_all_baselines_epoch_performance(project_folder, result_path):
    fig, ax = plt.subplots(3, 3, sharey=True)
    benchmark_names = ['lcbench', 'taskset', 'pd1']

    for axes in ax[1]:
        axes.set_visible(False)

    for axes in ax[2]:
        axes.set_visible(False)

    for axes, benchmark_name in zip(ax[0], benchmark_names):

        if benchmark_name == 'lcbench':
            benchmark_extension = os.path.join(
                'lc_bench',
                'results',
                'data_2k.json',
            )
        elif benchmark_name == 'taskset':
            benchmark_extension = os.path.join(
                'data',
                'taskset',
            )
        else:
            benchmark_extension = ''

        benchmark_data_path = os.path.join(
            project_folder,
            benchmark_extension,
        )

        methods_mean_epochs, methods_mean_regret, methods_mean_std = \
            generate_epoch_performance_data(
                benchmark_data_path,
                result_path,
                benchmark_name
            )
        for method_index, method_name in enumerate(method_names):
            axes.plot(
                methods_mean_epochs[method_index],
                methods_mean_regret[method_index],
                label=pretty_names[method_name],
                linewidth=4,
            )
            axes.fill_between(methods_mean_regret[method_index], np.add(methods_mean_regret[method_index], methods_mean_std[method_index]), np.subtract(methods_mean_regret[method_index], methods_mean_std[method_index]), alpha=0.1)

        if benchmark_name == 'lcbench':
            axes.set_title('LCBench')
            axes.set_xlabel('Epochs')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(250))
            axes.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(10))
            axes.set_ylabel('Average Normalized Regret')
        elif benchmark_name == 'taskset':
            axes.set_title('TaskSet')
            axes.set_xlabel('Steps')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(250))
            axes.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(100))
        elif benchmark_name == 'pd1':
            axes.set_title('PD1')
            axes.set_xlabel('Fraction of Iterations')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(0.25))
            axes.xaxis.set_major_formatter('{x:.2f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(0.1))

        axes.set_yscale('log')
        handles, labels = axes.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.52), loc='lower center', ncol=8)
    fig.savefig(f'comparison_epochs.pdf', bbox_inches="tight")


def plot_all_baselines_time_performance(project_folder, result_path):
    fig, ax = plt.subplots(2, 2, sharey=True)
    benchmark_names = ['lcbench', 'pd1']

    for axes in ax[1]:
        axes.set_visible(False)

    for axes, benchmark_name in zip(ax[0], benchmark_names):

        if benchmark_name == 'lcbench':
            benchmark_extension = os.path.join(
                'lc_bench',
                'results',
                'data_2k.json',
            )
        elif benchmark_name == 'taskset':
            benchmark_extension = os.path.join(
                'data',
                'taskset',
            )
        else:
            benchmark_extension = ''

        benchmark_data_path = os.path.join(
            project_folder,
            benchmark_extension,
        )

        methods_mean_epochs, methods_mean_regret = \
            generate_walltime_data(
                benchmark_data_path,
                result_path,
                benchmark_name,
            )

        for method_index, method_name in enumerate(method_names):
            axes.plot(
                methods_mean_epochs[method_index],
                methods_mean_regret[method_index],
                label=pretty_names[method_name],
                linewidth=4,
            )

        if benchmark_name == 'lcbench':
            axes.set_title('LCBench')
            axes.set_xlabel('Normalized Walltime')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(0.2))
            axes.xaxis.set_major_formatter('{x:.2f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(0.1))
            axes.set_ylabel('Average Normalized Regret')
        elif benchmark_name == 'taskset':
            axes.set_title('TaskSet')
            axes.set_xlabel('Steps')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(250))
            axes.xaxis.set_major_formatter('{x:.0f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(100))
        elif benchmark_name == 'pd1':
            axes.set_title('PD1')
            axes.set_xlabel('Normalized Walltime')
            axes.tick_params(left=True, bottom=True)
            axes.xaxis.set_major_locator(MultipleLocator(0.2))
            axes.xaxis.set_major_formatter('{x:.2f}')
            # For the minor ticks, use no labels; default NullFormatter.
            axes.xaxis.set_minor_locator(MultipleLocator(0.1))

        axes.set_yscale('log')
        handles, labels = axes.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.4), loc='lower center', ncol=8)
    fig.savefig(f'comparison_time.pdf', bbox_inches="tight")
