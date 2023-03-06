from copy import deepcopy
import logging
import os
import time
from typing import List, Tuple

import numpy as np

from scipy.stats import norm
import torch
from torch.utils.data import DataLoader

from data_loader.tabular_data_loader import WrappedDataLoader
from dataset.tabular_dataset import TabularDataset

from models.conditioned_power_law import ConditionedPowerLaw


class PowerLawSurrogate:

    def __init__(
        self,
        hp_candidates: np.ndarray,
        surrogate_configs: dict = None,
        seed: int = 11,
        max_benchmark_epochs: int = 52,
        ensemble_size: int = 5,
        nr_epochs: int = 250,
        fantasize_step: int = 1,
        minimization: bool = True,
        total_budget: int = 1000,
        device: str = None,
        output_path: str = '.',
        dataset_name: str = 'unknown',
        pretrain: bool = False,
        backbone: str = 'power_law',
        max_value: float = 100,
        min_value: float = 0,
        fill_value: str = 'zero',
    ):
        """
        Args:
            hp_candidates: np.ndarray
                The full list of hyperparameter candidates for a given dataset.
            surrogate_configs: dict
                The model configurations for the surrogate.
            seed: int
                The seed that will be used for the surrogate.
            max_benchmark_epochs: int
                The maximal budget that a hyperparameter configuration
                has been evaluated in the benchmark for.
            ensemble_size: int
                The number of members in the ensemble.
            nr_epochs: int
                Number of epochs for which the surrogate should be
                trained.
            fantasize_step: int
                The number of steps for which we are looking ahead to
                evaluate the performance of a hpc.
            minimization: bool
                If for the evaluation metric, the lower the value the better.
            total_budget: int
                The total budget given. Used to calculate the initialization
                percentage.
            device: str
                The device where the experiment will be run on.
            output_path: str
                The path where all the output will be stored.
            dataset_name: str
                The name of the dataset that the experiment will be run on.
            pretrain: bool
                If the surrogate will be pretrained before with a synthetic
                curve.
            backbone: str
                The backbone, which can either be 'power_law' or 'nn'.
            max_value: float
                The maximal value for the dataset.
            min_value: float
                The minimal value for the dataset.
            fill_value: str = 'zero',
                The filling strategy for when learning curves are used.
                Either 'zero' or 'last' where last represents the last value.
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.total_budget = total_budget
        self.fill_value = fill_value
        self.max_value = max_value
        self.min_value = min_value
        self.backbone = backbone

        self.pretrained_path = os.path.join(
            output_path,
            'power_law',
            f'checkpoint_{seed}.pth',
        )

        self.model_instances = [
            ConditionedPowerLaw,
            ConditionedPowerLaw,
            ConditionedPowerLaw,
            ConditionedPowerLaw,
            ConditionedPowerLaw,
        ]

        if device is None:
            self.dev = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.learning_rate = 0.001
        self.batch_size = 64
        self.refine_batch_size = 64

        self.criterion = torch.nn.L1Loss()
        self.hp_candidates = hp_candidates

        self.minimization = minimization
        self.seed = seed

        self.logger = logging.getLogger('power_law')
        logging.basicConfig(
            filename=f'power_law_surrogate_{dataset_name}_{seed}.log',
            level=logging.INFO,
            force=True,
        )

        # with what percentage configurations will be taken randomly instead of being sampled from the model
        self.fraction_random_configs = 0.1
        self.iteration_probabilities = np.random.rand(self.total_budget)

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.examples = dict()
        self.performances = dict()

        # set a seed already, so that it is deterministic when
        # generating the seeds of the ensemble
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.seeds = np.random.choice(100, ensemble_size, replace=False)
        self.max_benchmark_epochs = max_benchmark_epochs
        self.ensemble_size = ensemble_size
        self.nr_epochs = nr_epochs
        self.refine_nr_epochs = 20
        self.fantasize_step = fantasize_step

        self.pretrain = pretrain

        initial_configurations_nr = 1
        conf_individual_budget = 1
        init_conf_indices = np.random.choice(self.hp_candidates.shape[0], initial_configurations_nr, replace=False)
        init_budgets = [i for i in range(1, conf_individual_budget + 1)]

        self.rand_init_conf_indices = []
        self.rand_init_budgets = []

        # basically add every config index up to a certain budget threshold for the initialization
        # we will go through both lists during the initialization
        for config_index in init_conf_indices:
            for config_budget in init_budgets:
                self.rand_init_conf_indices.append(config_index)
                self.rand_init_budgets.append(config_budget)

        self.initial_random_index = 0

        if surrogate_configs is None:

            self.surrogate_configs = []
            for i in range(0, self.ensemble_size):
                self.surrogate_configs.append(
                    {
                        'nr_units': 128,
                        'nr_layers': 2,
                        'kernel_size': 3,
                        'nr_filters': 4,
                        'nr_cnn_layers': 2,
                        'use_learning_curve': False,
                    }
                )
        else:
            self.surrogate_configs = surrogate_configs

        self.nr_features = self.hp_candidates.shape[1]
        self.best_value_observed = np.inf

        self.diverged_configs = set()

        # Where the models of the ensemble will be stored
        self.models = []
        # A tuple which will have the last evaluated point
        # It will be used in the refining process
        # Tuple(config_index, budget, performance, curve)
        self.last_point = None

        self.initial_full_training_trials = 10

        # a flag if the surrogate should be trained
        self.train = True

        # the times it was refined
        self.refine_counter = 0
        # the surrogate iteration counter
        self.iterations_counter = 0
        # info dict to drop every surrogate iteration
        self.info_dict = dict()

        # the start time for the overhead of every surrogate iteration
        # will be recorded here
        self.suggest_time_duration = 0

        self.output_path = output_path
        self.dataset_name = dataset_name

        self.no_improvement_threshold = int(self.max_benchmark_epochs + 0.2 * self.max_benchmark_epochs)
        self.no_improvement_patience = 0

    def _prepare_dataset(self) -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        train_examples, train_labels, train_budgets, train_curves = self.history_configurations()

        train_curves = self.prepare_training_curves(train_budgets, train_curves)
        train_examples = np.array(train_examples, dtype=np.single)
        train_labels = np.array(train_labels, dtype=np.single)
        train_budgets = np.array(train_budgets, dtype=np.single)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.max_benchmark_epochs

        train_dataset = TabularDataset(
            train_examples,
            train_labels,
            train_budgets,
            train_curves,
        )

        return train_dataset

    def _refine_surrogate(self):
        """Refine the surrogate model.
        """
        for model_index, model_seed in enumerate(self.seeds):

            train_dataset = self._prepare_dataset()
            self.logger.info(f'Started refining model with index: {model_index}')
            refined_model = self.train_pipeline(
                model_index,
                train_dataset,
                nr_epochs=self.refine_nr_epochs,
                refine=True,
                weight_new_example=True,
                batch_size=self.refine_batch_size,
            )

            self.models[model_index] = refined_model

    def _train_surrogate(self, pretrain: bool = False):
        """Train the surrogate model.

        Trains all the models of the ensemble
        with different initializations and different
        data orders.

        Args:
            pretrain: bool
                If we have pretrained weights and we will just
                refine the models.
        """
        for model_index, model_seed in enumerate(self.seeds):
            train_dataset = self._prepare_dataset()
            self.logger.info(f'Started training model with index: {model_index}')

            if pretrain:
                # refine the models that were already pretrained
                trained_model = self.train_pipeline(
                    model_index,
                    train_dataset,
                    nr_epochs=self.refine_nr_epochs,
                    refine=True,
                    weight_new_example=False,
                    batch_size=self.batch_size,
                    early_stopping_it=self.refine_nr_epochs,  # basically no early stopping
                )
                self.models[model_index] = trained_model
            else:
                # train the models for the first time
                trained_model = self.train_pipeline(
                    model_index,
                    train_dataset,
                    nr_epochs=self.nr_epochs,
                    refine=False,
                    weight_new_example=False,
                    batch_size=self.batch_size,
                    early_stopping_it=self.nr_epochs,  # basically no early stopping
                )
                self.models.append(trained_model)

    def train_pipeline(
        self,
        model_index: int,
        train_dataset: TabularDataset,
        nr_epochs: int,
        refine: bool = False,
        weight_new_example: bool = True,
        batch_size: int = 64,
        early_stopping_it: int = 10,
        activate_early_stopping: bool = False,
    ) -> torch.nn.Module:
        """Train an algorithm to predict the performance
        of the hyperparameter configuration based on the budget.

        Args:
            model_index: int
                The index of the model.
            train_dataset: TabularDataset
                The tabular dataset featuring the examples, labels,
                budgets and curves.
            nr_epochs: int
                The number of epochs to train the model for.
            refine: bool
                If an existing model will be refined or if the training
                will start from scratch.
            weight_new_example: bool
                If the last example that was added should be weighted more
                by being included in every batch. This is only applicable
                when refine is True.
            batch_size: int
                The batch size to be used for training.
            early_stopping_it: int
                The early stopping iteration patience.
            activate_early_stopping: bool
                Flag controlling the activation.

        Returns:
            model: torch.nn.Module
                A trained model.
        """
        if model_index == 0:
            self.iterations_counter += 1
            self.logger.info(f'Iteration number: {self.iterations_counter}')

        surrogate_config = self.surrogate_configs[model_index]
        seed = self.seeds[model_index]
        torch.manual_seed(seed)
        np.random.seed(seed)

        if refine:
            model = self.models[model_index]
        else:
            model = self.model_instances[model_index](
                nr_initial_features=self.nr_features + 1 if self.backbone == 'nn' else self.nr_features,
                nr_units=surrogate_config['nr_units'],
                nr_layers=surrogate_config['nr_layers'],
                use_learning_curve=surrogate_config['use_learning_curve'],
                kernel_size=surrogate_config['kernel_size'],
                nr_filters=surrogate_config['nr_filters'],
                nr_cnn_layers=surrogate_config['nr_cnn_layers'],
            )
            model.to(self.dev)

        # make the training dataset here
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        train_dataloader = WrappedDataLoader(train_dataloader, self.dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        patience_rounds = 0
        best_loss = np.inf
        best_state = deepcopy(model.state_dict())

        for epoch in range(0, nr_epochs):
            running_loss = 0
            model.train()

            for batch_examples, batch_labels, batch_budgets, batch_curves in train_dataloader:

                nr_examples_batch = batch_examples.shape[0]
                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnormalization.
                if nr_examples_batch == 1:
                    continue

                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # in case we are refining, we add the new example to every
                # batch to give it more importance.
                if refine and weight_new_example:
                    newp_index, newp_budget, newp_performance, newp_curve = self.last_point
                    new_example = np.array([self.hp_candidates[newp_index]], dtype=np.single)
                    newp_missing_values = self.prepare_missing_values_channel([newp_budget])
                    newp_budget = np.array([newp_budget], dtype=np.single) / self.max_benchmark_epochs
                    newp_performance = np.array([newp_performance], dtype=np.single)
                    modified_curve = deepcopy(newp_curve)

                    difference = self.max_benchmark_epochs - len(modified_curve) - 1
                    if difference > 0:
                        modified_curve.extend([modified_curve[-1] if self.fill_value == 'last' else 0] * difference)

                    modified_curve = np.array([modified_curve], dtype=np.single)
                    newp_missing_values = np.array(newp_missing_values, dtype=np.single)

                    # add depth dimension to the train_curves array and missing_value_matrix
                    modified_curve = np.expand_dims(modified_curve, 1)
                    newp_missing_values = np.expand_dims(newp_missing_values, 1)
                    modified_curve = np.concatenate((modified_curve, newp_missing_values), axis=1)

                    new_example = torch.tensor(new_example, device=self.dev)
                    newp_budget = torch.tensor(newp_budget, device=self.dev)
                    newp_performance = torch.tensor(newp_performance, device=self.dev)
                    modified_curve = torch.tensor(modified_curve, device=self.dev)

                    batch_examples = torch.cat((batch_examples, new_example))
                    batch_budgets = torch.cat((batch_budgets, newp_budget))
                    batch_labels = torch.cat((batch_labels, newp_performance))
                    batch_curves = torch.cat((batch_curves, modified_curve))

                outputs = model(batch_examples, batch_budgets, batch_budgets, batch_curves)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

            running_loss = running_loss / len(train_dataloader)
            self.logger.info(f'Epoch {epoch +1}, Loss:{running_loss}')

            if activate_early_stopping:
                if running_loss < best_loss:
                    best_state = deepcopy(model.state_dict())
                    best_loss = running_loss
                    patience_rounds = 0
                elif running_loss > best_loss:
                    patience_rounds += 1
                    if patience_rounds == early_stopping_it:
                        model.load_state_dict(best_state)
                        self.logger.info(f'Stopping training since validation loss is not improving')
                        break

        if activate_early_stopping:
            model.load_state_dict(best_state)

        return model

    def _predict(self) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray]:
        """
        Predict the performances of the hyperparameter configurations
        as well as the standard deviations based on the ensemble.

        Returns:
            mean_predictions, std_predictions, hp_indices, real_budgets:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                The mean predictions and the standard deviations over
                all model predictions for the given hyperparameter
                configurations with their associated indices and budgets.

        """
        configurations, hp_indices, budgets, real_budgets, hp_curves = self.generate_candidate_configurations()
        # scale budgets to [0, 1]
        budgets = np.array(budgets, dtype=np.single)
        hp_curves = self.prepare_training_curves(real_budgets, hp_curves)
        budgets = budgets / self.max_benchmark_epochs
        real_budgets = np.array(real_budgets, dtype=np.single)
        configurations = np.array(configurations, dtype=np.single)

        configurations = torch.tensor(configurations)
        configurations = configurations.to(device=self.dev)
        budgets = torch.tensor(budgets)
        budgets = budgets.to(device=self.dev)
        hp_curves = torch.tensor(hp_curves)
        hp_curves = hp_curves.to(device=self.dev)
        network_real_budgets = torch.tensor(real_budgets / self.max_benchmark_epochs)
        network_real_budgets.to(device=self.dev)
        all_predictions = []

        for model in self.models:
            model = model.eval()
            predictions = model(configurations, budgets, network_real_budgets, hp_curves)
            all_predictions.append(predictions.detach().cpu().numpy())

        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions, hp_indices, real_budgets

    def suggest(self) -> Tuple[int, int]:
        """Suggest a hyperparameter configuration and a budget
        to evaluate.

        Returns:
            suggested_hp_index, budget: Tuple[int, int]
                The index of the hyperparamter configuration to be evaluated
                and the budget for what it is going to be evaluated for.
        """
        suggest_time_start = time.time()

        if self.initial_random_index < len(self.rand_init_conf_indices):
            self.logger.info(
                'Not enough configurations to build a model. \n'
                'Returning randomly sampled configuration'
            )
            suggested_hp_index = self.rand_init_conf_indices[self.initial_random_index]
            budget = self.rand_init_budgets[self.initial_random_index]
            self.initial_random_index += 1
        else:
            mean_predictions, std_predictions, hp_indices, real_budgets = self._predict()
            best_prediction_index = self.find_suggested_config(
                mean_predictions,
                std_predictions,
            )
            # actually do the mapping between the configuration indices and the best prediction
            # index
            suggested_hp_index = hp_indices[best_prediction_index]

            if suggested_hp_index in self.examples:
                evaluated_budgets = self.examples[suggested_hp_index]
                max_budget = max(evaluated_budgets)
                budget = max_budget + self.fantasize_step
                if budget > self.max_benchmark_epochs:
                    budget = self.max_benchmark_epochs
            else:
                budget = self.fantasize_step

        suggest_time_end = time.time()
        self.suggest_time_duration = suggest_time_end - suggest_time_start

        return suggested_hp_index, budget

    def observe(
        self,
        hp_index: int,
        b: int,
        hp_curve: List[float],
    ):
        """Receive information regarding the performance of a hyperparameter
        configuration that was suggested.

        Args:
            hp_index: int
                The index of the evaluated hyperparameter configuration.
            b: int
                The budget for which the hyperparameter configuration was evaluated.
            hp_curve: List
                The performance of the hyperparameter configuration.
        """
        for index, curve_element in enumerate(hp_curve):
            if np.isnan(curve_element):
                self.diverged_configs.add(hp_index)
                # only use the non-nan part of the curve and the corresponding
                # budget to still have the information in the network
                hp_curve = hp_curve[0:index + 1]
                b = index
                break

        if not self.minimization:
            hp_curve = np.subtract([self.max_value] * len(hp_curve), hp_curve)
            hp_curve = hp_curve.tolist()

        best_curve_value = min(hp_curve)

        self.examples[hp_index] = np.arange(1, b + 1)
        self.performances[hp_index] = hp_curve

        if self.best_value_observed > best_curve_value:
            self.best_value_observed = best_curve_value
            self.no_improvement_patience = 0
            self.logger.info(f'New Incumbent value found '
                             f'{1 - best_curve_value if not self.minimization else best_curve_value}')
        else:
            self.no_improvement_patience += 1
            if self.no_improvement_patience == self.no_improvement_threshold:
                self.train = True
                self.no_improvement_patience = 0
                self.logger.info(
                    'No improvement in the incumbent value threshold reached, '
                    'restarting training from scratch'
                )

        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0
        if self.initial_random_index >= len(self.rand_init_conf_indices):
            performance = self.performances[hp_index]
            self.last_point = (hp_index, b, performance[b-1], performance[0:b-1] if b > 1 else [initial_empty_value])

            if self.train:
                # delete the previously stored models
                self.models = []
                if self.pretrain:
                    # TODO Load the pregiven weights.
                    pass

                self._train_surrogate(pretrain=self.pretrain)

                if self.iterations_counter <= self.initial_full_training_trials:
                    self.train = True
                else:
                    self.train = False
            else:
                self.refine_counter += 1
                self._refine_surrogate()

    def prepare_examples(self, hp_indices: List) -> List:
        """
        Prepare the examples to be given to the surrogate model.

        Args:
            hp_indices: List
                The list of hp indices that are already evaluated.

        Returns:
            examples: List
                A list of the hyperparameter configurations.
        """
        examples = []
        for hp_index in hp_indices:
            examples.append(self.hp_candidates[hp_index])

        return examples

    def generate_candidate_configurations(self) -> Tuple[List, List, List, List, List]:
        """Generate candidate configurations that will be
        fantasized upon.

        Returns:
            (configurations, hp_indices, hp_budgets, real_budgets, hp_curves): Tuple
                A tuple of configurations, their indices in the hp list,
                the budgets that they should be fantasized upon, the maximal
                budgets they have been evaluated and their corresponding performance
                curves.
        """
        hp_indices = []
        hp_budgets = []
        hp_curves = []
        real_budgets = []
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0

        for hp_index in range(0, self.hp_candidates.shape[0]):

            if hp_index in self.examples:
                budgets = self.examples[hp_index]
                # Take the max budget evaluated for a certain hpc
                max_budget = budgets[-1]
                if max_budget == self.max_benchmark_epochs:
                    continue
                real_budgets.append(max_budget)
                learning_curve = self.performances[hp_index]

                hp_curve = learning_curve[0:max_budget-1] if max_budget > 1 else [initial_empty_value]
            else:
                real_budgets.append(1)
                hp_curve = [initial_empty_value]

            hp_indices.append(hp_index)
            hp_budgets.append(self.max_benchmark_epochs)
            hp_curves.append(hp_curve)

        configurations = self.prepare_examples(hp_indices)

        return configurations, hp_indices, hp_budgets, real_budgets, hp_curves

    def history_configurations(self) -> Tuple[List, List, List, List]:
        """
        Generate the configurations, labels, budgets and curves
        based on the history of evaluated configurations.

        Returns:
            (train_examples, train_labels, train_budgets, train_curves): Tuple
                A tuple of examples, labels and budgets for the
                configurations evaluated so far.
        """
        train_examples = []
        train_labels = []
        train_budgets = []
        train_curves = []
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.hp_candidates[hp_index]

            for budget in budgets:
                example_curve = performances[0:budget-1]
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                train_curves.append(example_curve if len(example_curve) > 0 else [initial_empty_value])

        return train_examples, train_labels, train_budgets, train_curves

    @staticmethod
    def acq(
        best_values: np.ndarray,
        mean_predictions: np.ndarray,
        std_predictions: np.ndarray,
        explore_factor: float = 0.25,
        acq_choice: str = 'ei',
    ) -> np.ndarray:
        """
        Calculate the acquisition function based on the network predictions.

        Args:
        -----
        best_values: np.ndarray
            An array with the best value for every configuration.
            Depending on the implementation it can be different for every
            configuration.
        mean_predictions: np.ndarray
            The mean values of the model predictions.
        std_predictions: np.ndarray
            The standard deviation values of the model predictions.
        explore_factor: float
            The explore factor, when ucb is used as an acquisition
            function.
        acq_choice: str
            The choice for the acquisition function to use.

        Returns
        -------
        acq_values: np.ndarray
            The values of the acquisition function for every configuration.
        """
        if acq_choice == 'ei':
            z = (np.subtract(best_values, mean_predictions))
            difference = deepcopy(z)
            not_zero_std_indicator = [False if example_std == 0.0 else True for example_std in std_predictions]
            zero_std_indicator = np.invert(not_zero_std_indicator)
            z = np.divide(z, std_predictions, where=not_zero_std_indicator)
            np.place(z, zero_std_indicator, 0)
            acq_values = np.add(np.multiply(difference, norm.cdf(z)), np.multiply(std_predictions, norm.pdf(z)))
        elif acq_choice == 'ucb':
            # we are working with error rates so we multiply the mean with -1
            acq_values = np.add(-1 * mean_predictions, explore_factor * std_predictions)
        elif acq_choice == 'thompson':
            acq_values = np.random.normal(mean_predictions, std_predictions)
        else:
            acq_values = mean_predictions

        return acq_values

    def find_suggested_config(
            self,
            mean_predictions: np.ndarray,
            mean_stds: np.ndarray,
    ) -> int:
        """Return the hyperparameter with the highest acq function value.

        Given the mean predictions and mean standard deviations from the DPL
        ensemble for every hyperparameter configuraiton, return the hyperparameter
        configuration that has the highest acquisition function value.

        Args:
            mean_predictions: np.ndarray
                The mean predictions of the ensemble for every hyperparameter
                configuration.
            mean_stds: np.ndarray
                The standard deviation predictions of the ensemble for every
                hyperparameter configuration.

        Returns:
            max_value_index: int
                the index of the maximal value.

        """
        best_values = np.array([self.best_value_observed] * mean_predictions.shape[0])
        acq_func_values = self.acq(
            best_values,
            mean_predictions,
            mean_stds,
            acq_choice='ei',
        )

        max_value_index = np.argmax(acq_func_values)

        return max_value_index

    def calculate_fidelity_ymax(self, fidelity: int) -> float:
        """Calculate the incumbent for a certain fidelity level.

        Args:
            fidelity: int
                The given budget fidelity.

        Returns:
            best_value: float
                The incumbent value for a certain fidelity level.
        """
        config_values = []
        for example_index in self.examples.keys():
            try:
                performance = self.performances[example_index][fidelity - 1]
            except IndexError:
                performance = self.performances[example_index][-1]
            config_values.append(performance)

        # lowest error corresponds to best value
        best_value = min(config_values)

        return best_value

    def patch_curves_to_same_length(self, curves: List):
        """
        Patch the given curves to the same length.

        Finds the maximum curve length and patches all
        other curves that are shorter with zeroes.

        Args:
            curves: List
                The hyperparameter curves.
        """
        for curve in curves:
            difference = self.max_benchmark_epochs - len(curve) - 1
            if difference > 0:
                fill_value = [curve[-1]] if self.fill_value == 'last' else [0]
                curve.extend(fill_value * difference)

    def prepare_missing_values_channel(self, budgets: List) -> List:
        """Prepare an additional channel for learning curves.

        The additional channel will represent an existing learning
        curve value with a 1 and a missing learning curve value with
        a 0.

        Args:
            budgets: List
                A list of budgets for every training point.

        Returns:
            missing_value_curves: List
                A list of curves representing existing or missing
                values for the training curves of the training points.
        """
        missing_value_curves = []

        for i in range(len(budgets)):
            budget = budgets[i]
            budget = budget - 1
            budget = int(budget)

            if budget > 0:
                example_curve = [1] * budget
            else:
                example_curve = []

            difference_in_curve = self.max_benchmark_epochs - len(example_curve) - 1
            if difference_in_curve > 0:
                example_curve.extend([0] * difference_in_curve)
            missing_value_curves.append(example_curve)

        return missing_value_curves

    def get_mean_initial_value(self):
        """Returns the mean initial value
        for all hyperparameter configurations in the history so far.

        Returns:
            mean_initial_value: float
                Mean initial value for all hyperparameter configurations
                observed.
        """
        first_values = []
        for performance_curve in self.performances.values():
            first_values.append(performance_curve[0])

        mean_initial_value = np.mean(first_values)

        return mean_initial_value

    def prepare_training_curves(
            self,
            train_budgets: List[int],
            train_curves: List[float]
    ) -> np.ndarray:
        """Prepare the configuration performance curves for training.

        For every configuration training curve, add an extra dimension
        regarding the missing values, as well as extend the curve to have
        a fixed uniform length for all.

        Args:
            train_budgets: List
                A list of the budgets for all training points.
            train_curves: List
                A list of curves that pertain to every training point.

        Returns:
            train_curves: np.ndarray
                The transformed training curves.
        """
        missing_value_matrix = self.prepare_missing_values_channel(train_budgets)
        self.patch_curves_to_same_length(train_curves)
        train_curves = np.array(train_curves, dtype=np.single)
        missing_value_matrix = np.array(missing_value_matrix, dtype=np.single)

        # add depth dimension to the train_curves array and missing_value_matrix
        train_curves = np.expand_dims(train_curves, 1)
        missing_value_matrix = np.expand_dims(missing_value_matrix, 1)
        train_curves = np.concatenate((train_curves, missing_value_matrix), axis=1)

        return train_curves
