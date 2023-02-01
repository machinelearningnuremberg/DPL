from typing import List, Tuple

import numpy as np


class RandomOptimizer:
    def __init__(
        self,
        hyperparameter_candidates: np.ndarray,
        max_budget: int = 52,
        seed: int = 0,
        max_nr_trials=1000,
        **kwargs,
    ):
        """
        Wrapper for the Random search algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        max_budget: int
            The number of max epochs used during the HPO optimization.
        seed: int
            Seed used to reproduce the experiments.
        max_nr_trials: int
            The total runtime budget, given as the number of epochs spent during HPO.
        """
        self.hyperparameter_candidates = hyperparameter_candidates
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)
        self.evaluated_configurations = set()
        self.max_budget = max_budget
        self.max_trials = max_nr_trials
        self.extra_args = kwargs

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
        possible_candidates = {i for i in range(self.hyperparameter_candidates.shape[0])}
        not_evaluated_candidates = possible_candidates - self.evaluated_configurations
        config_index = np.random.choice(list(not_evaluated_candidates))
        self.evaluated_configurations.add(config_index)

        # if not enough budget to give max fidelity, give max budget
        max_budget = min(self.max_budget, self.max_trials)

        return config_index, max_budget

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
        pass
