import argparse

import numpy as np

from framework import Framework

parser = argparse.ArgumentParser(
    description='DPL publication experiments.',
)
parser.add_argument(
    '--index',
    type=int,
    default=1,
    help='The worker index. Every worker runs the same experiment, however, with a different seed.',
)
parser.add_argument(
    '--fantasize_step',
    type=int,
    default=1,
    help='The step used in fantasizing the next learning curve value from the last'
         'observed one for a certain hyperparameter configuration.',
)
parser.add_argument(
    '--budget_limit',
    type=int,
    default=1000,
    help='The maximal number of HPO iterations.',
)
parser.add_argument(
    '--ensemble_size',
    type=int,
    default=5,
    help='The ensemble size for the DPL surrogate.',
)
parser.add_argument(
    '--nr_epochs',
    type=int,
    default=250,
    help='The number of epochs used to train (not refine) the HPO surrogate.',
)
parser.add_argument(
    '--dataset_name',
    type=str,
    default='credit-g',
    help='The name of the dataset used in the experiment.'
         'The dataset names must be matched with the benchmark they belong to.',
)
parser.add_argument(
    '--benchmark_name',
    type=str,
    default='lcbench',
    help='The name of the benchmark used in the experiment. '
         'Every benchmark offers its own distinctive datasets. Available options are lcbench, taskset and pd1.',
)
parser.add_argument(
    '--surrogate_name',
    type=str,
    default='power_law',
    help='The method that will be run.',
)
parser.add_argument(
    '--project_dir',
    type=str,
    default='.',
    help='The directory where the project files are located.',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output',
    help='The directory where the project output files will be stored.',
)

args = parser.parse_args()
seeds = np.arange(10)
seed = seeds[args.index - 1]

framework = Framework(args, seed)
framework.run()
