from typing import Dict, List, OrderedDict, Tuple

import numpy as np

import optuna


class AHBOptimizer:

    def __init__(
        self,
        hyperparameter_candidates: np.ndarray,
        param_space: OrderedDict,
        min_budget: int,
        max_budget: int,
        eta: int,
        seed: int = 11,
        max_nr_trials: int = 1000,
        maximization: bool = True,
        **kwargs,
    ):
        """
        Wrapper for the Async Hyperband algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        param_space: OrderedDict
            The hyperparameter search-space, indicating the type and range of every
            hyperparameter.
        min_budget: int
            Minimum number of epochs available.
        max_budget: int
            Maximum number of epochs available.
        eta: int
            Halving factor
        seed: int
            Seed used to reproduce the experiments.
        max_nr_trials: int
            Maximum number of HPO trials.
        maximization: bool
            If the inner objective is to maximize or minimize.
        """
        self.maximization = maximization
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.max_nr_trials = max_nr_trials
        self.extra_arguments = kwargs

        self.param_space = param_space
        self.hyperparameter_candidates = hyperparameter_candidates
        self.hyperparameter_mapping = self.create_configuration_to_indices(
            hyperparameter_candidates,
        )
        self.transformed_hp_candidates = self.from_hp_value_to_unit_cube_values(
            hyperparameter_candidates,
        )

        self.distribution = self.get_optuna_search_space()

        # empty configuration, empty budget, empty information for config
        self.next_conf = None
        self.trial = None
        self.conf_budget = None
        self.conf_info = None
        self.fidelity_index = None
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)

        self.evaluated_configurations = dict()
        self.evaluated_hp_curves = dict()

        # define study with hyperband pruner.
        sampler = optuna.samplers.RandomSampler(seed=seed)
        self.study = optuna.create_study(
            sampler=sampler,
            direction='maximize' if self.maximization else 'minimize',
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self.min_budget,
                max_resource=self.max_budget,
                reduction_factor=self.eta,
            ),
        )

    def suggest(self) -> Tuple[int, int]:
        """
        Get information about the next configuration.

        Returns:
        ________
        next_conf, conf_budget: tuple
            A tuple that contains information about the next
            configuration (index in the hyperparameter_candidates it was
            given) and the budget for the hyperparameter to be evaluated
            on.
        """
        if self.next_conf is None:

            self.trial = self.study.ask(self.distribution)
            self.next_conf = self.get_hp_config_from_trial(self.conf_budget)
            self.conf_budget = 1

            # if the hyperparameter has been evaluated before
            while self.next_conf in self.evaluated_hp_curves:

                val_curve = self.evaluated_hp_curves[self.next_conf]
                # it was not evaluated as far as now, it can go to the framework
                if self.conf_budget > len(val_curve):
                    break
                else:
                    pruned_trial = False

                    score = val_curve[self.conf_budget - 1]
                    self.trial.report(score, self.conf_budget)

                    if self.trial.should_prune():
                        pruned_trial = True

                    if pruned_trial:
                        self.study.tell(self.trial, state=optuna.trial.TrialState.PRUNED)
                        # hyperparameter config was pruned, sample another one
                        self.trial = self.study.ask(self.distribution)
                        if self.conf_budget in self.evaluated_configurations:
                            self.evaluated_configurations[self.conf_budget].add(self.next_conf)
                        else:
                            self.evaluated_configurations[self.conf_budget] = set([self.next_conf])
                        self.conf_budget = 1
                        self.next_conf = self.get_hp_config_from_trial(self.conf_budget)

                    else:
                        if self.conf_budget == self.max_budget:
                            self.study.tell(self.trial, val_curve[-1])
                            self.trial = self.study.ask(self.distribution)
                            if self.conf_budget in self.evaluated_configurations:
                                self.evaluated_configurations[self.conf_budget].add(self.next_conf)
                            else:
                                self.evaluated_configurations[self.conf_budget] = set([self.next_conf])
                            self.conf_budget = 1
                            self.next_conf = self.get_hp_config_from_trial(self.conf_budget)
                        else:
                            # Increase the budget
                            self.conf_budget += 1

        return self.next_conf, self.conf_budget

    def observe(
        self,
        hp_index: int,
        budget: int,
        learning_curve: List[float],
    ):
        """
        Respond regarding the performance of a
        hyperparameter configuration. get_next should
        be called first to retrieve the configuration.

        Args:
        -----
        hp_index: int
            The index of the evaluated hyperparameter configuration.
        budget: int
            The budget for which the hyperparameter configuration was evaluated.
        learning curve: np.ndarray, list
            validation accuracy curve. The last value is the same as the score.
        """
        assert self.next_conf is not None, 'Call get_next first.'
        pruned_trial = False

        score = learning_curve[-1]
        self.trial.report(score, self.conf_budget)

        if self.trial.should_prune():
            pruned_trial = True

        if pruned_trial:
            self.study.tell(self.trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
            self.evaluated_hp_curves[self.next_conf] = learning_curve
            if self.conf_budget in self.evaluated_configurations:
                self.evaluated_configurations[self.conf_budget].add(self.next_conf)
            else:
                self.evaluated_configurations[self.conf_budget] = set([self.next_conf])
            self.next_conf = None

        if self.conf_budget == self.max_budget:
            self.study.tell(self.trial, score, state=optuna.trial.TrialState.COMPLETE)
            self.evaluated_hp_curves[self.next_conf] = learning_curve
            if self.conf_budget in self.evaluated_configurations:
                self.evaluated_configurations[self.conf_budget].add(self.next_conf)
            else:
                self.evaluated_configurations[self.conf_budget] = set([self.next_conf])
            self.next_conf = None
        else:
            self.conf_budget += 1

    def create_configuration_to_indices(
        self,
        hyperparameter_candidates: np.ndarray,
    ) -> Dict[tuple, int]:
        """
        Maps every configuration to its index as specified
        in hyperparameter_candidates.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            All the possible hyperparameter candidates given
            by the calling framework.

        Returns:
        ________
        hyperparameter_mapping: dict
            A dictionary where the keys are tuples representing
            hyperparameter configurations and the values are indices
            representing their placement in hyperparameter_candidates.
        """
        hyperparameter_mapping = dict()
        for i in range(0, hyperparameter_candidates.shape[0]):
            hyperparameter_mapping[tuple(hyperparameter_candidates[i])] = i

        return hyperparameter_mapping

    def map_configuration_to_index(
        self,
        hyperparameter_candidate: np.ndarray,
    ) -> int:
        """
        Return the index of the hyperparameter_candidate from
        the given initial array of possible hyperparameters.

        Args:
        -----
        hyperparameter_candidate: np.ndarray
            Hyperparameter configuration.

        Returns:
        ________
        index of the hyperparameter_candidate.
        """
        hyperparameter_candidate = tuple(hyperparameter_candidate)

        return self.hyperparameter_mapping[hyperparameter_candidate]

    def get_optuna_search_space(self):
        """
        Get the optuna hyperparameter search space distribution.

        Returns:
        --------
        distribution: dict
            The hyperparameter search space distribution for optuna.
        """
        distribution = {}
        for i, (k, v) in enumerate(self.param_space.items()):
            hp_type = v[2]
            is_log = v[3]
            if hp_type == str:
                distribution[k] = optuna.distributions.UniformDistribution(0, 1)
            else:
                if is_log:
                    distribution[k] = optuna.distributions.LogUniformDistribution(0.00001, 1)
                else:
                    distribution[k] = optuna.distributions.UniformDistribution(0, 1)

        return distribution

    def get_hp_config_from_trial(self, budget: int):
        """
        Get the hyperparameter config index from the
        optuna trial.

        Args:
        -----
        budget: int
            The budget to run the hyperparameter configuration for.

        Returns:
        --------
        conf_index: int
            The hyperparameter config index.
        """
        hp_config = []
        for hp_name in self.param_space.keys():
            hp_config.append(self.trial.params[hp_name])

        conf_index = self.map_closest_evaluated(hp_config, budget)

        return conf_index

    def map_closest_evaluated(
        self,
        config: List,
        budget: int,
    ) -> int:
        """
        Maps the hyperparameter configuration to the closest
        available hyperparameter configuration.

        Args:
        -----
        config: List
            The hyperparameter configuration suggested by the baseline.
        budget: int
            The budget of the hyperparameter configuration.

        Returns:
        --------
        closest_configuration_index: int
            An index of the closest matching configuration.
        """
        closest_configuration_index = None
        smallest_distance = np.inf

        for i in range(0, self.transformed_hp_candidates.shape[0]):
            current_distance = 0
            possible_config = self.transformed_hp_candidates[i, :]
            for hyperparameter_index in range(0, len(config)):
                main_config_hyperparameter_value = config[hyperparameter_index]
                candidate_config_hyperparameter_value = possible_config[hyperparameter_index]
                current_distance += abs(main_config_hyperparameter_value - candidate_config_hyperparameter_value)
            if current_distance < smallest_distance:
                if len(self.evaluated_configurations) != 0:
                    # if a hyperparameter has already been evaluated for a certain
                    # budget, we do not consider it anymore.
                    if budget in self.evaluated_configurations and i in self.evaluated_configurations[budget]:
                        continue
                smallest_distance = current_distance
                closest_configuration_index = i

        return closest_configuration_index

    def from_hp_value_to_unit_cube_values(
        self,
        hp_candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Maps the hyperparameter configurations from the original
        space to the unit cube space.

        Args:
        -----
        hp_candidates: np.ndarray
            The hyperparameter configuration suggested by the baseline.

        Returns:
        --------
        new_configs: np.ndarray
            An array representing the hyperparameter configurations
            in unit cube space.
        """
        assert len(hp_candidates[0]) == len(self.param_space)

        new_configs = []

        for i in range(0, hp_candidates.shape[0]):
            new_config = []
            configuration = hp_candidates[i]
            for hp_index, (k, v) in enumerate(self.param_space.items()):
                hp_type = v[2]
                value = configuration[hp_index]
                lower, upper = v[0], v[1]
                is_log = v[3]
                if hp_type == str:
                    unique_values = v[0]
                    ranges = np.arange(start=0, stop=1, step=1 / len(unique_values))
                    for range_index, unique_value in enumerate(unique_values):
                        if unique_value == value:
                            step_size = (1 / len(unique_values))
                            # set the value at the middle of the hyperparameter
                            # allocated range
                            value = ranges[range_index] + step_size / 0.5
                        else:
                            # do nothing
                            pass
                else:
                    if is_log:
                        log_range = np.log(upper) - np.log(lower)
                        value = (np.log(value) - np.log(lower)) / log_range
                    else:
                        value = (value - lower) / (upper - lower)
                    new_config.append(value)
            new_configs.append(new_config)

        return np.array(new_configs)
