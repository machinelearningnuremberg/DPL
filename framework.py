import argparse
import json
import os
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from benchmarks.lcbench import LCBench
from benchmarks.taskset import TaskSet
from benchmarks.hyperbo import PD1
from surrogate_models.power_law_surrogate import PowerLawSurrogate
from surrogate_models.asha import AHBOptimizer
from surrogate_models.dehb.interface import DEHBOptimizer
from surrogate_models.dragonfly import DragonFlyOptimizer
from surrogate_models.random_search import RandomOptimizer


class Framework:

    def __init__(
        self,
        args: argparse.Namespace,
        seed: int,
    ):
        """
        Args:
            args: Namespace
                Includes all the arguments given as variables to the main_experiment
                script.
            seed: int
                The seed for the experiment.
        """

        if args.benchmark_name == 'lcbench':
            benchmark_extension = os.path.join(
                'lc_bench',
                'results',
                'data_2k.json',
            )
        elif args.benchmark_name == 'taskset':
            benchmark_extension = os.path.join(
                'data',
                'taskset',
            )
        elif args.benchmark_name == 'pd1':
            benchmark_extension = 'pd1'
        else:
            raise ValueError(f'Benchmark {args.benchmark_name} not supported')

        benchmark_data_path = os.path.join(
            args.project_dir,
            benchmark_extension,
        )

        benchmark_types = {
            'lcbench': LCBench,
            'taskset': TaskSet,
            'pd1': PD1,
        }

        surrogate_types = {
            'power_law': PowerLawSurrogate,
            'asha': AHBOptimizer,
            'dehb': DEHBOptimizer,
            'dragonfly': DragonFlyOptimizer,
            'random': RandomOptimizer,
        }

        disable_preprocessing = {
            'dehb',
        }

        self.benchmark = benchmark_types[args.benchmark_name](benchmark_data_path, args.dataset_name)
        self.dataset_name = args.dataset_name
        self.seed = seed
        self.max_value = self.benchmark.max_value
        self.min_value = self.benchmark.min_value
        self.total_budget = args.budget_limit
        self.fantasize_step = args.fantasize_step

        self.categorical_indicator = self.benchmark.categorical_indicator
        self.log_indicator = self.benchmark.log_indicator
        self.hp_names = self.benchmark.hp_names
        self.minimization_metric = self.benchmark.minimization_metric
        self.info_dict = dict()
        self.result_dir = os.path.join(
            args.output_dir,
            args.benchmark_name,
            args.surrogate_name,
        )
        os.makedirs(self.result_dir, exist_ok=True)

        self.result_file = os.path.join(
            self.result_dir,
            f'{self.dataset_name}_{self.seed}.json',
        )

        if args.surrogate_name not in disable_preprocessing:
            self.hp_candidates = self.preprocess(self.benchmark.get_hyperparameter_candidates())
        else:
            self.hp_candidates = self.benchmark.get_hyperparameter_candidates()

        if args.surrogate_name == 'power_law':
            self.surrogate = surrogate_types[args.surrogate_name](
                self.hp_candidates,
                seed=seed,
                max_benchmark_epochs=self.benchmark.max_budget,
                ensemble_size=args.ensemble_size,
                nr_epochs=args.nr_epochs,
                fantasize_step=self.fantasize_step,
                minimization=self.minimization_metric,
                total_budget=args.budget_limit,
                device='cpu',
                dataset_name=args.dataset_name,
                output_path=self.result_dir,
                max_value=self.max_value,
                min_value=self.min_value,
            )
        else:
            self.surrogate = surrogate_types[args.surrogate_name](
                hyperparameter_candidates=self.hp_candidates,
                param_space=self.benchmark.param_space,
                min_budget=self.benchmark.min_budget,
                max_budget=self.benchmark.max_budget,
                eta=3,
                seed=seed,
                max_nr_trials=args.budget_limit,
                maximization=not self.benchmark.minimization_metric,
            )

    def run(self):

        evaluated_configs = dict()
        surrogate_budget = 0

        if self.benchmark.minimization_metric:
            best_value = np.inf
        else:
            best_value = 0

        while surrogate_budget < self.total_budget:

            start_time = time.time()
            hp_index, budget = self.surrogate.suggest()
            hp_curve = self.benchmark.get_curve(hp_index, budget)

            self.surrogate.observe(hp_index, budget, hp_curve)
            time_duration = time.time() - start_time

            if hp_index in evaluated_configs:
                previous_budget = evaluated_configs[hp_index]
            else:
                previous_budget = 0

            budget_cost = budget - previous_budget
            evaluated_configs[hp_index] = budget

            step_time_duration = time_duration / budget_cost

            for epoch in range(previous_budget + 1, budget + 1):
                epoch_performance = float(hp_curve[epoch - 1])
                if self.benchmark.minimization_metric:
                    if best_value > epoch_performance:
                        best_value = epoch_performance
                else:
                    if best_value < epoch_performance:
                        best_value = epoch_performance

                surrogate_budget += 1

                if surrogate_budget > self.total_budget:
                    exit(0)

                self.log_info(
                    int(hp_index),
                    epoch_performance,
                    epoch,
                    best_value,
                    step_time_duration,
                )

        exit(0)

    def preprocess(self, hp_candidates: np.ndarray) -> np.ndarray:
        """Preprocess the hyperparameter candidates.

        Performs min-max standardization for the numerical attributes and
        additionally one-hot encoding for the categorical attributes.

        Args:
            hp_candidates: np.ndarray
                The hyperparameter candidates in their raw form as taken
                from the benchmark.

        Returns:
            preprocessed_candidates: np.ndarray
                The transformed hyperparameter candidates after being
                preprocessed.
        """
        column_transformers = []
        numerical_columns = [
            col_index for col_index, category_indicator in enumerate(self.categorical_indicator)
            if not category_indicator
        ]
        categorical_columns = [
            col_index for col_index, category_indicator in enumerate(self.categorical_indicator)
            if category_indicator
        ]

        general_transformers = []

        if len(numerical_columns) > 0:

            if self.log_indicator is not None and any(self.log_indicator):
                log_columns = [col_index for col_index, log_indicator in enumerate(self.log_indicator) if log_indicator]
                log_transformer = FunctionTransformer(np.log)
                column_transformers.append(
                    (
                        'log_pre',
                        ColumnTransformer(
                            [('log', log_transformer, log_columns)],
                            remainder='passthrough'
                        )
                    )
                )

            general_transformers.append(('num', MinMaxScaler(), numerical_columns))

        if len(categorical_columns) > 0:

            general_transformers.append(
                (
                    'cat',
                    OneHotEncoder(
                        categories=[self.hp_names] * hp_candidates.shape[1],
                        sparse=False,
                    ),
                    categorical_columns,
                )
            )
        column_transformers.append(('feature_types_pre', ColumnTransformer(general_transformers)))

        preprocessor = Pipeline(
            column_transformers
        )
        # TODO log preprocessing will push numerical columns to the right
        # so a mapping has to happen for the feature_types_pre
        preprocessed_candidates = preprocessor.fit_transform(hp_candidates)

        return preprocessed_candidates

    def log_info(
            self,
            hp_index: int,
            performance: float,
            budget: int,
            best_value_observed: float,
            time_duration: float,
    ):
        """Log information after every HPO iteration.

        Args:
            hp_index: int
                The index of the suggested hyperparameter candidate.
            performance: float
                The performance of the hyperparameter candidate.
            budget: int
                The budget at which the hyperpararameter candidate has been evaluated so far.
            best_value_observed: float
                The incumbent value observed so far during the optimization.
            time_duration: float
                The time taken for the HPO iteration.
        """
        if 'hp' in self.info_dict:
            self.info_dict['hp'].append(hp_index)
        else:
            self.info_dict['hp'] = [hp_index]

        accuracy_performance = performance

        if 'scores' in self.info_dict:
            self.info_dict['scores'].append(accuracy_performance)
        else:
            self.info_dict['scores'] = [accuracy_performance]

        incumbent_acc_performance = best_value_observed

        if 'curve' in self.info_dict:
            self.info_dict['curve'].append(incumbent_acc_performance)
        else:
            self.info_dict['curve'] = [incumbent_acc_performance]

        if 'epochs' in self.info_dict:
            self.info_dict['epochs'].append(budget)
        else:
            self.info_dict['epochs'] = [budget]

        if 'overhead' in self.info_dict:
            self.info_dict['overhead'].append(time_duration)
        else:
            self.info_dict['overhead'] = [time_duration]

        with open(self.result_file, 'w') as fp:
            json.dump(self.info_dict, fp)
