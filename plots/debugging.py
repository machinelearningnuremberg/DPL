import json
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')  # no need for tk
import numpy as np
import seaborn as sns
import scipy
from scipy import stats

sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 45,
        'axes.titlesize': 45,
        'axes.labelsize': 45,
        'xtick.labelsize': 45,
        'ytick.labelsize': 45,
        'legend.fontsize': 39,
    },
    style="white"
)


def gradients_and_parameters(
    parameters: List[List],
    parameter_gradients: List[List],
    parameter_names: List,
    final_predicted_curve: List,
    final_true_curve: List,
    loss_curve,
    hp_index: int,
    max_budget: int = 1000,
    curve_length: int = 25,
):
    # Create four subplots and unpack the output array immediately
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    for parameter_name, parameter_values in zip(parameter_names, parameters):
        ax1.plot(np.arange(1, max_budget + 1), parameter_values, label=f'{parameter_name} value')
    #ax1.set_aspect('equal', 'box')
    ax1.legend()
    for parameter_name, parameter_values in zip(parameter_names, parameter_gradients):
        ax2.plot(np.arange(1, max_budget + 1), parameter_values, label=f'{parameter_name} gradients')
        ax2.set_ylim(-0.5, 0.5)
    #ax2.set_aspect('equal', 'box')
    ax2.legend()


    ax3.plot(np.arange(1, curve_length + 1), final_true_curve, label=f'True validation curve')
    ax3.plot(np.arange(1, curve_length + 1), final_predicted_curve, label=f'Predicted validation curve')
    #ax3.set_aspect('equal', 'box')
    ax3.legend()

    ax4.plot(np.arange(1, max_budget + 1), loss_curve, label=f'{parameter_name} gradients')


    f.tight_layout()
    plt.savefig(f'training_info_{hp_index}.pdf')


def plot_grad_flow(named_parameters, i):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'gradients_epoch{i}.pdf')


def plot_conditioned_surrogates(result_dir: str):

    models = [
        'conditioned_power_law',
        'conditioned_nn',
    ]

    method_names_to_pretty = {
        'conditioned_power_law': 'DPL',
        'conditioned_nn': 'Cond NN',
        'power_law': 'PL',
        'nn': 'NN',
        'gp': 'GP',
    }
    seed = 11
    val_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    dataset_names = ['APSFailure', 'Amazon_employee_access', 'Australian', 'Fashion-MNIST', 'KDDCup09_appetency',
                     'MiniBooNE', 'adult', 'airlines', 'albert', 'bank-marketing', 'blood-transfusion-service-center',
                     'car', 'christine', 'cnae-9', 'connect-4', 'covertype', 'credit-g', 'dionis', 'fabert', 'helena',
                     'higgs', 'jannis', 'jasmine', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kr-vs-kp',
                     'mfeat-factors', 'nomao', 'numerai28.6', 'phoneme', 'segment', 'shuttle', 'sylvine', 'vehicle',
                     'volkert']

    for model in models:
        model_correlation_means = []
        model_correlation_stds = []

        for val_fraction in val_fractions:

            run_information_folder = os.path.join(
                result_dir,
                f'{model}',
                f'{seed}',
                f'{val_fraction}',
            )

            dataset_correlations = []
            for dataset_name in dataset_names:
                # for dataset_name in dataset_names:
                information_file = os.path.join(run_information_folder, f'{dataset_name}.json')
                with open(information_file, 'r') as fp:
                    information = json.load(fp)
                dataset_correlation = information['correlation']
                if not np.isnan(dataset_correlation):
                    dataset_correlations.append(dataset_correlation)

            if model == 'conditioned_power_law':
                print(f'Validation fraction: {val_fraction}')
                print(f'Dataset correlations: {dataset_correlations}')

            model_correlation_means.append(np.mean(dataset_correlations))
            model_correlation_stds.append(np.std(dataset_correlations))

        plt.plot(val_fractions, model_correlation_means, label=method_names_to_pretty[model], marker='o', linestyle='--', linewidth=7)

    for unconditioned_model_name in ['power_law', 'nn', 'gp']:
        model_correlation_means = []
        model_correlation_stds = []

        for val_fraction in val_fractions:
            run_information_folder = os.path.join(
                result_dir,
                f'{unconditioned_model_name}',
                f'{seed}',
                f'{val_fraction}',
                'config_6' if unconditioned_model_name == 'nn' else 'config_1',
            )
            dataset_correlations = []
            for dataset_name in dataset_names:
                # for dataset_name in dataset_names:
                information_file = os.path.join(run_information_folder, f'{dataset_name}.json')
                with open(information_file, 'r') as fp:
                    information = json.load(fp)

                hp_true_performances = []
                hp_predicted_performances = []
                for hp_information in information:
                    hp_predicted_performances.append(hp_information['hp_predicted_performance'])
                    hp_true_performances.append(hp_information['hp_true_performance'])

                dataset_correlation, _ = scipy.stats.pearsonr(hp_predicted_performances, hp_true_performances)
                if not np.isnan(dataset_correlation):
                    dataset_correlations.append(dataset_correlation)

            model_correlation_means.append(np.mean(dataset_correlations))
            model_correlation_stds.append(np.std(dataset_correlations))

        plt.plot(val_fractions, model_correlation_means, label=method_names_to_pretty[unconditioned_model_name], marker='o', linestyle='--', linewidth=7)

    plt.xlabel('LC Length Fraction')
    plt.xticks(val_fractions, [f'{val_fraction}' for val_fraction in val_fractions])
    plt.ylabel('Correlation: Est. vs. True')
    plt.legend(bbox_to_anchor=(0.5, -0.42), loc='lower center', ncol=5)
    plt.savefig('conditioned_model_correlations.pdf', bbox_inches="tight")


def plot_uncertainty_estimation(mean_values, std_values, evaluated_configs, point_to_be_evaluated, hp_indices, counter):

    plt.figure()
    point_indices = np.arange(0, 2000, 20)
    point_indices = np.append(point_indices, list(evaluated_configs.keys()))
    point_indices = np.append(point_indices, point_to_be_evaluated)
    point_indices = np.sort(point_indices)
    hp_indices = np.array(hp_indices)
    mean_point_x = hp_indices[point_indices]
    mean_point_y = mean_values[point_indices]
    mean_point_std =  std_values[point_indices]
    plt.plot(mean_point_x, mean_point_y, color='red', label='Mean Surrogate Predictions')

    plt.fill_between(mean_point_x, np.add(mean_point_y, mean_point_std), np.subtract(mean_point_y, mean_point_std), color='red', alpha=0.2)
    plt.plot([hp_indices[point_to_be_evaluated]], mean_values[point_to_be_evaluated], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green", label='Chosen Point')

    already_evaluated_x = []
    already_evaluated_y = []
    for hp_index, mean_performance in zip(hp_indices, mean_values):
        if hp_index in evaluated_configs:
            already_evaluated_x.append(hp_index)
            already_evaluated_y.append(mean_performance)

    plt.scatter(already_evaluated_x, already_evaluated_y, color='black', label='Evaluated Points')
    plt.xlabel('Hyperparameter indices')
    plt.ylabel('Surrogate Prediction')
    plt.legend(loc=8, ncol=5)
    counter = int(counter)
    plt.savefig(f'surrogate_uncertainty_{counter}.pdf', bbox_inches="tight")


def plot_top_conditioned_surrogates(result_dir: str):

    models = [
        'conditioned_power_law',
    ]

    seed = 11
    val_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    dataset_names = ['APSFailure', 'Amazon_employee_access', 'Australian', 'Fashion-MNIST', 'KDDCup09_appetency',
                     'MiniBooNE', 'adult', 'airlines', 'albert', 'bank-marketing', 'blood-transfusion-service-center',
                     'car', 'christine', 'cnae-9', 'connect-4', 'covertype', 'credit-g', 'dionis', 'fabert', 'helena',
                     'higgs', 'jannis', 'jasmine', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kr-vs-kp',
                     'mfeat-factors', 'nomao', 'numerai28.6', 'phoneme', 'segment', 'shuttle', 'sylvine', 'vehicle',
                     'volkert']

    model_correlations = []
    for model in models:
        model_correlation_means = []
        model_correlation_stds = []
        model_mae = []

        collection_mae = []
        for val_fraction in val_fractions:

            run_information_folder = os.path.join(
                result_dir,
                f'{model}',
                f'{seed}',
                f'{val_fraction}',
            )

            dataset_correlations = []
            mean_absolute_relative_errors = []

            dataset_errors = []
            for dataset_name in dataset_names:
                # for dataset_name in dataset_names:
                information_file = os.path.join(run_information_folder, f'{dataset_name}.json')
                with open(information_file, 'r') as fp:
                    information = json.load(fp)
                real_labels = information['real_labels']
                predicted_labels = information['predicted_labels']
                info_dict = dict()
                for real_label, predicted_label in zip(real_labels, predicted_labels):
                    info_dict[real_label] = predicted_label
                real_labels.sort(reverse=True)
                real_top_labels = []
                predicted_top_labels = []
                config_errors = []
                for i in range(0, len(real_labels)):
                    example_label = real_labels[i]
                    real_top_labels.append(example_label)
                    predicted_label = info_dict[example_label]
                    predicted_top_labels.append(info_dict[example_label])
                    mae = abs((example_label - predicted_label)) / example_label
                    mean_absolute_relative_errors.append(mae)
                    config_errors.append(mae)

                dataset_correlation, _ = stats.pearsonr(real_top_labels, predicted_top_labels)
                dataset_errors.append(np.mean(config_errors))
                if not np.isnan(dataset_correlation):
                    dataset_correlations.append(dataset_correlation)

            model_correlations.append(dataset_correlations)
            model_correlation_means.append(np.mean(dataset_correlations))
            model_mae.append(dataset_errors)
            model_correlation_stds.append(np.std(dataset_correlations))
            collection_mae.append(mean_absolute_relative_errors)

    meanlineprops = dict(linewidth=4)
    whiskersprops = dict(linewidth=3)
    plt.boxplot(model_mae, positions=val_fractions, widths=0.02, showfliers=False, whis=0.5, medianprops=meanlineprops, capprops=whiskersprops, boxprops=whiskersprops, whiskerprops=whiskersprops)
    plt.xlabel('LC Length Fraction')
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.4)
    plt.xticks(val_fractions, [f'{val_fraction}' for val_fraction in val_fractions])
    plt.ylabel('Absolute Relative Error')
    plt.savefig('pl_mae_distribution.pdf', bbox_inches="tight")
