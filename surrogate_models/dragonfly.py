from argparse import Namespace
import threading

from typing import Dict, List, OrderedDict, Tuple

from dragonfly import load_config, maximize_multifidelity_function, minimize_multifidelity_function

import numpy as np


class DragonFlyOptimizer:
    def __init__(
        self,
        hyperparameter_candidates: np.ndarray,
        param_space: OrderedDict,
        seed: int = 0,
        max_budget: int = 52,
        max_nr_trials: int = 2000,
        maximization: bool = True,
        **kwargs,
    ):
        """
        Wrapper for the BOCA algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        param_space: OrderedDict
            The hyperparameter search-space, indicating the type and range of every
            hyperparameter.
        seed: int
            Seed used to reproduce the experiments.
        max_budget: int
            The number of maximal steps for a hyperparameter configuration.
        max_nr_trials: int
            The total runtime budget, given as the number of epochs spent during HPO.
        maximization: bool
            If the inner objective is to maximize or minimize.
        """
        self.maximization = maximization
        self.hyperparameter_candidates = hyperparameter_candidates
        self.param_space = param_space
        self.extra_arguments = kwargs

        self.hyperparameter_mapping = self.create_configuration_to_indices()

        # empty configuration, empty budget, empty information for config
        self.next_conf = None
        self.conf_budget = None
        self.conf_info = None
        self.fidelity_index = None
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)

        self.evaluated_configurations = dict()
        self.evaluated_hp_curves = dict()
        # Basically the same as evaluated_hp_curves. However, this will
        # be used to estimate the evaluation cost for a certain fidelity.
        # If we used evaluated_hp_curves, the cost would always be zero
        # since the configuration index is added there as evaluated already
        # before.
        self.fidelity_hp_curves = dict()
        domain_vars = [
            {'type': 'discrete_euclidean', 'items': list(self.hyperparameter_candidates)},
        ]
        fidel_vars = [
            {'type': 'int', 'min': 1, 'max': max_budget},
        ]

        fidel_to_opt = [int(max_budget)]

        config = {
            'domain': domain_vars,
            'fidel_space': fidel_vars,
            'fidel_to_opt': fidel_to_opt,
        }
        # How frequently to build a new (GP) model
        # --build_new_model_every 17
        options_namespace = Namespace(
            gpb_hp_tune_criterion='ml',
        )
        config = load_config(config)

        self.dragonfly_run = threading.Thread(
            target=maximize_multifidelity_function if self.maximization else minimize_multifidelity_function,
            kwargs={
                'func': self.target_function,
                'max_capital': max_nr_trials,
                'config': config,
                'domain': config.domain,
                'fidel_space': config.fidel_space,
                'fidel_to_opt': config.fidel_to_opt,
                'options': options_namespace,
                'fidel_cost_func': self.fidel_cost_function,
            },
            daemon=True,
        )
        self.dragonfly_run.start()

    def fidel_cost_function(self, fidelity):

        fidelity_value = fidelity[0]
        while True:
            if self.fidelity_index is not None:
                config_index = self.fidelity_index
                if config_index in self.fidelity_hp_curves:
                    budget_evaluated = self.fidelity_hp_curves[config_index]
                    # the hyperparameter configuration has been evaluated before
                    # and it was evaluated for a higher\same budget
                    if budget_evaluated >= fidelity_value:
                        # there was a curve which was evaluated for longer
                        fidelity_opt_cost = 0
                    else:
                        # will only resume training for the extra query
                        fidelity_opt_cost = fidelity_value - budget_evaluated
                        self.fidelity_hp_curves[config_index] = fidelity_value
                else:
                    # first evaluation
                    fidelity_opt_cost = fidelity_value
                    self.fidelity_hp_curves[config_index] = fidelity_value
                self.fidelity_index = None
                break

        return fidelity_opt_cost

    def target_function(
        self,
        budget: List[int],
        config: List[np.ndarray],
    ) -> float:
        """
        Function to evaluate for a given configuration.

        Args:
        -----
        budget: list
            The budget for which the configuration will be run.
        config: list
            Configuration suggested by DragonFly.

        Returns:
        ________
        score: float
            A score which indicates the validation performance
            of the configuration.
        """
        # the budget is a list initially with only one value
        budget = budget[0]
        if budget is not None:
            budget = int(budget)

        # initially the config is a list consisting of a single np.ndarray
        config = list(config[0])

        config_index = self.map_configuration_to_index(config)

        # not the first hyperparameter to be evaluated for the selected
        # budget
        if budget in self.evaluated_configurations:
            self.evaluated_configurations[budget].add(config_index)
        else:
            self.evaluated_configurations[budget] = set([config_index])

        self.conf_budget = budget

        need_to_query_framework = True
        if config_index in self.evaluated_hp_curves:
            config_curve = self.evaluated_hp_curves[config_index]
            # the hyperparameter configuration has been evaluated before
            # and it was evaluated for a higher\same budget
            if len(config_curve) >= budget:
                need_to_query_framework = False

        # Save the config index in fidelity index, since sometimes this config
        # if evaluated before it is not passed to the framework, but it would be
        # still needed for the cost estimation.
        self.fidelity_index = config_index

        if need_to_query_framework:
            # update the field so the framework can take the index and
            # reply
            self.next_conf = config_index
            while True:
                if self.conf_info is not None:
                    score = self.conf_info['score']
                    val_curve = self.conf_info['val_curve']
                    # save the curve for the evaluated hyperparameter
                    # configuration
                    self.evaluated_hp_curves[config_index] = val_curve
                    break
                else:
                    # The framework has not yet responded with a value,
                    # keep checking
                    # TODO add a delay
                    pass
        else:
            score = config_curve[budget - 1]

        # need to make the previous response None since DragonFly
        # continues running in the background
        self.conf_info = None

        return score

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
        while self.next_conf is None:
            # DragonFly has not generated the config yet
            pass
        self.conf_info = None

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
        self.next_conf = None

        self.conf_info = {
            'score': learning_curve[-1],
            'val_curve': learning_curve,
        }

    def create_configuration_to_indices(
        self,
    ) -> Dict[Tuple, int]:
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
        for i in range(0, self.hyperparameter_candidates.shape[0]):
            hyperparameter_mapping[tuple(self.hyperparameter_candidates[i])] = i

        return hyperparameter_mapping

    def map_configuration_to_index(
        self,
        hyperparameter_candidate: List,
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