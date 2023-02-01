import threading
import time
from typing import Dict, List, OrderedDict, Tuple, Union

import ConfigSpace
import numpy as np

from surrogate_models.dehb.dehb import DEHB


class DEHBOptimizer:
    def __init__(
        self,
        hyperparameter_candidates: np.ndarray,
        min_budget: int,
        max_budget: int,
        param_space: OrderedDict,
        seed: int = 0,
        exhaustive_search_space: bool = False,
        total_cost=np.inf,
        maximization: bool = 'True',
        **kwargs,
    ):
        """
        Wrapper for the DEHB algorithm.

        Args:
        -----
        hyperparameter_candidates: np.ndarray
            2d array which contains all possible configurations which can be queried.
        min_budget: int
            Minimum number of epochs available.
        max_budget: int
            Maximum number of epochs available.
        param_space: OrderedDict
            The hyperparameter search-space, indicating the type and range of every
            hyperparameter.
        seed: int
            Seed used to reproduce the experiments.
        **kwargs: dict
            DEHB configuration if desired.
        """
        search_space_dimensions = hyperparameter_candidates.shape[1]
        self.maximization = maximization

        if len(kwargs) != 6:
            # default DEHB configuration
            configuration = {
                'strategy': 'rand1_bin',
                'mutation_factor': 0.5,
                'crossover_prob': 0.5,
                'eta': 3,
                'boundary_fix_type': 'random',
                'gens': 1,
                'nr_workers': 1,
            }
        else:
            configuration = kwargs

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.hyperparameter_mapping = self.create_configuration_to_indices(
            hyperparameter_candidates,
        )
        self.hyperparameter_candidates = hyperparameter_candidates
        self.param_space = param_space
        self.exhaustive_search_space = exhaustive_search_space
        self.transformed_hp_candidates = self.from_hp_value_to_dehb_values(
            self.hyperparameter_candidates,
        )

        # empty configuration, empty budget, empty information for config
        self.next_conf = None
        self.conf_budget = None
        self.conf_info = None
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)

        configuration_space = None
        if self.exhaustive_search_space:
            configuration_space = self.create_config_space()

        self.evaluated_configurations = dict()
        self.evaluated_hp_curves = dict()

        # Initializing DEHB object
        self.dehb = \
            DEHB(
                dimensions=search_space_dimensions,
                f=self.target_function,
                cs=configuration_space,
                strategy=configuration['strategy'],
                mutation_factor=configuration['mutation_factor'],
                crossover_prob=configuration['crossover_prob'],
                eta=configuration['eta'],
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                generations=configuration['gens'],
                boundary_fix_type=configuration['boundary_fix_type'],
                n_workers=configuration['nr_workers'],
                hyperparameter_canditates=self.hyperparameter_candidates if not self.exhaustive_search_space else None,
                param_space=self.param_space if not self.exhaustive_search_space else None,
                random_state=self.rng,
            )

        self.debh_run = threading.Thread(
            target=self.dehb.run,
            kwargs={
                'total_cost': total_cost,
                'verbose': False,

            },
            daemon=True,
        )
        self.debh_run.start()

    def target_function(
        self,
        config: np.ndarray,
        budget: int = 100,
    ) -> Dict[str, Union[float, Dict]]:
        """
        Function to evaluate for a given configuration.

        Args:
        -----
        config: np.ndarray
            Configuration suggested by DEHB.
        budget: int
            The budget for which the configuration will be run.

        Returns:
        ________
        result: dict
            A dictionary which contains information about the performance
            of the configuration, the cost as well as additional information.
        """
        if budget is not None:
            budget = int(budget)
        # if the search space is not exhaustive the configuration sampled may
        # not be present in the available hyperparameter configurations. Map it
        # to the closest available hyperparameter configuration.
        if not self.exhaustive_search_space:
            config_index = self.map_closest_evaluated(
                config,
                budget,
            )
        else:
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

        score = None
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
            val_curve = config_curve[0:budget]

        if self.maximization:
            # DE/DEHB minimizes
            fitness = - score
        else:
            fitness = score

        cost = None
        # if cost is given (from a benchmark) use the value given
        # otherwise, just calculate the duration.
        if self.conf_info is not None:
            if 'cost' in self.conf_info:
                cost = self.conf_info['cost']
            else:
                start_time = time.time()
        else:
            start_time = time.time()

        if cost is None:
            end_time = time.time()
            cost = end_time - start_time

        try:
            val_curve = val_curve.tolist()
        except Exception:
            # val_curve is not a numpy array
            # but instead it is a list, do nothing.
            pass

        result = {
            "fitness": fitness,
            "cost": cost,
            "info": {
                "val_curve": val_curve,
                "val_score": score,
                "budget": budget

            }
        }

        # need to make the previous response None since DEHB
        # continues running in the background
        self.conf_info = None

        return result

    def suggest(self) -> Tuple[int, float]:
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
            # DEHB has not generated the config yet
            pass
        self.conf_info = None

        return self.next_conf, self.conf_budget

    def observe(self,
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
        score: float
            validation accuracy, the higher the better.
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

    def transform_space(
        self,
        param_space: Dict[str, List],
        configuration: Tuple,
    ) -> Tuple:
        """
        Scales the [0, 1] - ranged parameter linearly to [lower, upper].

        Args:
        -----
        param_space: dict
            A dictionary containing the parameters and their meta-info.
        configuration: tuple
            A vector with each dimension in [0, 1] (from DEHB).

        Returns:
        --------
        config: tuple
            A tuple of the ordered hyperparameter values.
        """
        assert len(configuration) == len(param_space)

        config = []
        for i, (k, v) in enumerate(param_space.items()):
            value = configuration[i]
            if len(v) > 2:
                hp_type = v[2]
            else:
                hp_type = v[1]
            # str can be passed for categorical variables
            if hp_type == str:
                # Unique values
                unique_values = v[0]
                ranges = np.arange(start=0, stop=1, step=1 / len(unique_values))
                value = unique_values[np.where((value < ranges) == False)[0][-1]]
            # float and integer hyperparameters
            else:
                lower, upper = v[0], v[1]
                is_log = v[3]
                if is_log:
                    # performs linear scaling in the log-space
                    log_range = np.log(upper) - np.log(lower)
                    value = np.exp(np.log(lower) + log_range * value)
                else:
                    # linear scaling within the range of the parameter
                    value = lower + (upper - lower) * value
                if hp_type == int:
                    value = np.round(value).astype(int)
            config.append(value)

        config = tuple(config)

        return config

    def map_closest_evaluated(
        self,
        config: Tuple,
        budget: int,
    ) -> np.ndarray:
        """
        Maps the hyperparameter configuration to the closest
        available hyperparameter configuration.

        Args:
        -----
        config: tuple
            The hyperparameter configuration suggested by DEHB.
        budget: int
            The budget of the hyperparameter configuration.

        Returns:
        --------
        config: np.ndarray
            An array representing the nearest available
            hyperparameter configuration.
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

    def create_config_space(self) -> ConfigSpace.ConfigurationSpace:
        """
        Return a configuration space based on the specifications given
        at the param_space object.

        Returns:
        ________
        cs - ConfigurationSpace
            A configuration space from which the hyperparameters will
            be sampled.
        """
        cs = ConfigSpace.ConfigurationSpace()
        for i, (k, v) in enumerate(self.param_space.items()):
            hp_type = v[2]
            lower, upper = v[0], v[1]
            is_log = v[3]
            if hp_type == str:
                unique_values = v[0]
                cs.add_hyperparameter(
                    ConfigSpace.CategoricalHyperparameter(
                        k,
                        choices=unique_values,
                    )
                )
            else:
                if hp_type == int:
                    numerical_hp = ConfigSpace.UniformIntegerHyperparameter
                elif hp_type == float:
                    numerical_hp = ConfigSpace.UniformFloatHyperparameter
                else:
                    raise ValueError('Illegal hyperparameter type given')

                cs.add_hyperparameter(
                    numerical_hp(
                        k,
                        lower=lower,
                        upper=upper,
                        log=is_log,
                    )
                )

        return cs

    def from_hp_value_to_dehb_values(
        self,
        hp_candidates: np.ndarray,
    ) -> np.ndarray:
        """
        Maps the hyperparameter configurations from the original
        space to the DEHB unit cube space.

        Args:
        -----
        hp_candidates: np.ndarray
            The hyperparameter configuration suggested by DEHB.

        Returns:
        --------
        new_configs: np.ndarray
            An array representing the hyperparameter configurations
            represented in the DEHB search space.
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
                        value = (value - lower) /  (upper - lower)
                    new_config.append(value)
            new_configs.append(new_config)

        return np.array(new_configs)
