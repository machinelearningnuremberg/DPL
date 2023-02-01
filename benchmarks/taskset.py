import json
import os
from typing import List

import numpy as np
import pandas as pd

from benchmarks.benchmark import BaseBenchmark


class TaskSet(BaseBenchmark):

    nr_hyperparameters = 1000
    max_budget = 51

    hp_names = [
        'learning_rate',
        'beta1',
        'beta2',
        'epsilon',
        'l1',
        'l2',
        'linear_decay',
        'exponential_decay',
    ]

    log_indicator = [True, False, False, True, True, True, True, True]

    def __init__(self, path_to_json_files: str, dataset_name: str):

        super().__init__(path_to_json_files)
        self.dataset_name = dataset_name
        self.hp_candidates = []
        self.training_curves = []
        self.validation_curves = []
        self.test_curves = []

        self._load_benchmark()

        filtered_indices = self.filter_curves()
        self.validation_curves = np.array(self.validation_curves)
        self.validation_curves = self.validation_curves[filtered_indices]
        self.hp_candidates = np.array(self.hp_candidates)
        self.hp_candidates = self.hp_candidates[filtered_indices]

        self.categorical_indicator = [False] * self.hp_candidates[1]
        self.min_value = self.get_worst_performance()
        self.max_value = self.get_best_performance()

    def get_worst_performance(self):
        # for taskset we have loss, so the worst value possible value
        # is infinity
        worst_value = 0
        for hp_index in range(0, self.validation_curves.shape[0]):
            val_curve = self.validation_curves[hp_index]
            worst_performance_hp_curve = max(val_curve)
            if worst_performance_hp_curve > worst_value:
                worst_value = worst_performance_hp_curve

        return worst_value

    def get_best_performance(self):

        incumbent_curve = self.get_incumbent_curve()
        best_value = min(incumbent_curve)

        return best_value

    def _load_benchmark(self):

        dataset_file = os.path.join(self.path_to_json_file, f'{self.dataset_name}.json')

        with open(dataset_file, 'r') as fp:
            dataset_info = json.load(fp)

        for optimization_iteration in dataset_info:
            hp_configuration = optimization_iteration['hp']
            train_curve = optimization_iteration['train']['loss']
            validation_curve = optimization_iteration['valid']['loss']
            test_curve = optimization_iteration['test']['loss']

            # keep a fixed order for the hps and their values,
            # just in case
            new_hp_configuration = []
            for hp_name in self.hp_names:
                new_hp_configuration.append(hp_configuration[hp_name])

            self.hp_candidates.append(new_hp_configuration)
            self.training_curves.append(train_curve)
            self.validation_curves.append(validation_curve)
            self.test_curves.append(test_curve)

    def load_dataset_names(self) -> List[str]:

        dataset_file_names = [
            dataset_file_name[:-5] for dataset_file_name in os.listdir(self.path_to_json_file)
            if os.path.isfile(os.path.join(self.path_to_json_file, dataset_file_name))
        ]

        return dataset_file_names

    def get_hyperparameter_candidates(self) -> np.ndarray:

        return np.array(self.hp_candidates)

    def get_performance(self, hp_index: int, budget: int) -> float:

        val_curve = self.validation_curves[hp_index]

        budget = int(budget)

        return val_curve[budget - 1]

    def get_curve(self, hp_index: int, budget: int) -> float:

        val_curve = self.validation_curves[hp_index]

        budget = int(budget)

        return val_curve[0:budget].tolist()

    def get_incumbent_curve(self):

        best_value = np.inf
        best_index = -1
        for index in range(0, self.validation_curves.shape[0]):
            val_curve = self.validation_curves[index]
            min_loss = min(val_curve)

            if min_loss < best_value:
                best_value = min_loss
                best_index = index

        return self.validation_curves[best_index]

    def get_gap_performance(self):

        incumbent_curve = self.get_incumbent_curve()
        best_value = min(incumbent_curve)
        worst_value = self.get_worst_performance()

        return worst_value - best_value

    def get_incumbent_config_index(self):

        best_value = np.inf
        best_index = -1
        for index in range(0, self.validation_curves.shape[0]):
            val_curve = self.validation_curves[index]
            min_loss = min(val_curve)

            if min_loss < best_value:
                best_value = min_loss
                best_index = index

        return best_index

    def log_transform_labels(self):

        validation_curves = np.array(self.validation_curves).flatten()
        max_value = np.amax(validation_curves)
        min_value = np.amin(validation_curves)
        self.max_value = max_value
        self.min_value = min_value

        f = lambda x: (np.log(x) - np.log(min_value)) / (np.log(max_value) - np.log(min_value))

        log_transformed_values = f(self.validation_curves)

        return log_transformed_values.tolist()

    def filter_curves(self):

        validation_curves = np.array(self.validation_curves)
        validation_curves = pd.DataFrame(validation_curves)
        # TODO do a query for both values instead of going through the df twice
        non_nan_idx = validation_curves.notnull().all(axis=1)
        non_diverging_idx = (validation_curves < validation_curves.quantile(0.95).min()).all(axis=1)

        idx = non_nan_idx & non_diverging_idx

        return idx
