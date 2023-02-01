from collections import OrderedDict
import math

import numpy as np
import pandas as pd
from syne_tune.blackbox_repository import load_blackbox
from syne_tune.config_space import is_log_space

from benchmarks.benchmark import BaseBenchmark


class PD1(BaseBenchmark):

    def __init__(self, path_to_json_files: str, dataset_name: str, eta=3, number_of_brackets=3):

        super().__init__(path_to_json_files)
        self.dataset_name = dataset_name
        self.blackbox = self._load_benchmark()

        self.hp_names = None
        self.hp_candidates = self.blackbox[dataset_name].hyperparameters.to_numpy()

        # hyperparameter candidates, seed, fidelity, metrics
        # 0 is the validation train error
        self.validation_error_rate = self.blackbox[dataset_name].objectives_evaluations[:, :, :, 0]
        self.validation_error_rate = np.mean(self.validation_error_rate, axis=1)

        self.max_value = 0.0
        self.min_value = 1.0

        self.eta = eta
        self.number_of_brackets = number_of_brackets

        filtered_indices = self.filter_curves()
        self.max_budget = self.blackbox[dataset_name].num_fidelities
        # considering an eta of 3
        self.min_budget = int(self.max_budget / math.pow(self.eta, self.number_of_brackets))
        self.min_budget = self.min_budget if self.min_budget > 0 else 1
        self.log_indicator = self.get_log_indicator()
        self.categorical_indicator = [False] * self.hp_candidates[1]
        self.validation_error_rate = self.validation_error_rate[filtered_indices]
        self.hp_candidates = self.hp_candidates[filtered_indices]
        self.nr_hyperparameters = self.validation_error_rate.shape[0]

        hp_names = list(self.blackbox[dataset_name].hyperparameters.columns)

        self.param_space = OrderedDict(
            [
                (hp_names[i], [self.blackbox[dataset_name].hyperparameters[hp_names[i]].min(), self.blackbox[dataset_name].hyperparameters[hp_names[i]].max(), float, self.log_indicator[i]]) for i in range(len(hp_names))
            ]
        )

    def get_worst_performance(self):

        return np.amax(self.validation_error_rate)

    def _load_benchmark(self):

        return load_blackbox('pd1')

    @staticmethod
    def load_dataset_names():

        dataset_names = []
        enough_lc_points = 10

        blackbox = load_blackbox('pd1')
        for dataset_name in blackbox:
            if blackbox[dataset_name].num_fidelities > enough_lc_points:
                dataset_names.append(dataset_name)

        return dataset_names

    def get_log_indicator(self):

        log_indicator = [is_log_space(v) for v in self.blackbox[self.dataset_name].configuration_space.values()]

        return log_indicator

    def get_hyperparameter_candidates(self) -> np.ndarray:

        return self.hp_candidates

    def get_performance(self, hp_index: int, budget: int) -> float:

        budget = int(budget)
        val_performance = self.validation_error_rate[hp_index, budget - 1]

        return float(val_performance)

    def get_curve(self, hp_index: int, budget: int) -> float:

        budget = int(budget)
        val_curve = self.validation_error_rate[hp_index, 0:budget]
        return val_curve.tolist()

    def get_incumbent_curve(self):

        best_value = np.inf
        best_index = -1
        for index in range(0, self.validation_error_rate.shape[0]):
            val_error_curve = self.validation_error_rate[index, :]
            best_performance = min(val_error_curve)

            if best_performance < best_value:
                best_value = best_performance
                best_index = index

        return self.validation_error_rate[best_index]

    def get_gap_performance(self):

        incumbent_curve = self.get_incumbent_curve()
        best_value = min(incumbent_curve)
        worst_value = self.get_worst_performance()

        return worst_value - best_value

    def get_incumbent_config_index(self):

        best_value = np.inf
        best_index = -1
        for index in range(0, self.validation_error_rate.shape[0]):
            val_error_curve = self.validation_error_rate[index, :]
            best_performance = min(val_error_curve)

            if best_performance < best_value:
                best_value = best_performance
                best_index = index

        return best_index

    def filter_curves(self):

        validation_curves = pd.DataFrame(self.validation_error_rate)
        # TODO do a query for both values instead of going through the df twice
        non_nan_idx = validation_curves.notnull().all(axis=1)
        non_diverging_idx = (validation_curves < validation_curves.quantile(0.95).min()).all(axis=1)

        idx = non_nan_idx & non_diverging_idx

        return idx
