from collections import OrderedDict
from typing import List

import numpy as np

from benchmarks.benchmark import BaseBenchmark
from lc_bench.api import Benchmark


class LCBench(BaseBenchmark):

    nr_hyperparameters = 2000
    # Declaring the search space for LCBench
    param_space = OrderedDict([
        ('batch_size', [16, 512, int, True]),
        ('learning_rate', [0.0001, 0.1, float, True]),
        ('momentum', [0.1, 0.99, float, False]),
        ('weight_decay', [0.00001, 0.1, float, False]),
        ('num_layers', [1, 5, int, False]),
        ('max_units', [64, 1024, int, True]),
        ('max_dropout', [0.0, 1.0, float, False]),
    ])
    max_budget = 51
    min_budget = 1

    hp_names = list(param_space.keys())

    log_indicator = [True, True, False, False, False, True, False]

    # if the best value corresponds to a lower value
    minimization_metric = False

    def __init__(self, path_to_json_file: str, dataset_name: str):

        super().__init__(path_to_json_file)
        self.benchmark = self._load_benchmark()
        self.dataset_name = dataset_name
        self.dataset_names = self.load_dataset_names()
        self.categorical_indicator = [False] * len(self.param_space)
        self.max_value = 1.0
        self.min_value = 0.0

    def get_worst_performance(self):

        # since it is accuracy for LCBench
        min_value = 100
        for hp_index in range(0, LCBench.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=hp_index,
                tag='Train/val_balanced_accuracy',
            )
            val_curve = val_curve[1:]
            worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def _load_benchmark(self):

        bench = Benchmark(
            data_dir=self.path_to_json_file,
        )

        return bench

    def load_dataset_names(self) -> List[str]:

        return self.benchmark.get_dataset_names()

    def get_hyperparameter_candidates(self) -> np.ndarray:

        hp_names = list(LCBench.param_space.keys())
        hp_configs = []
        for i in range(LCBench.nr_hyperparameters):
            hp_config = []
            config = self.benchmark.query(
                dataset_name=self.dataset_name,
                tag='config',
                config_id=i,
            )
            for hp_name in hp_names:
                hp_config.append(config[hp_name])
            hp_configs.append(hp_config)

        hp_configs = np.array(hp_configs)

        return hp_configs

    def get_performance(self, hp_index: int, budget: int) -> float:

        val_curve = self.benchmark.query(
            dataset_name=self.dataset_name,
            config_id=hp_index,
            tag='Train/val_balanced_accuracy',
        )
        val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[budget - 1]

    def get_curve(self, hp_index: int, budget: int) -> List[float]:

        val_curve = self.benchmark.query(
            dataset_name=self.dataset_name,
            config_id=hp_index,
            tag='Train/val_balanced_accuracy',
        )
        val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[0:budget]

    def get_incumbent_curve(self):

        inc_curve = self.benchmark.query_best(
            self.dataset_name,
            "Train/val_balanced_accuracy",
            "Train/val_balanced_accuracy",
            0,
        )
        inc_curve = inc_curve[1:]

        return inc_curve

    def get_max_value(self):

        return max(self.get_incumbent_curve())

    def get_incumbent_config_id(self):

        best_value = 0
        best_index = -1
        for index in range(0, LCBench.nr_hyperparameters):
            val_curve = self.benchmark.query(
                dataset_name=self.dataset_name,
                config_id=index,
                tag='Train/val_balanced_accuracy',
            )
            val_curve = val_curve[1:]
            max_value = max(val_curve)

            if max_value > best_value:
                best_value = max_value
                best_index = index

        return best_index

    def get_gap_performance(self):

        incumbent_curve = self.get_incumbent_curve()
        best_value = max(incumbent_curve)
        worst_value = self.get_worst_performance()

        return best_value - worst_value

    def get_step_cost(self, hp_index: int, budget: int):

        time_cost_curve = self.benchmark.query(
            dataset_name=self.dataset_name,
            config_id=hp_index,
            tag='time',
        )
        time_cost_curve = time_cost_curve[1:]
        budget = int(budget)
        if budget > 1:
            step_cost = time_cost_curve[budget - 1] - time_cost_curve[budget - 2]
        else:
            step_cost = time_cost_curve[budget - 1]

        return step_cost

    def set_dataset_name(self, dataset_name: str):

        self.dataset_name = dataset_name
