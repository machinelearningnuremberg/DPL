# Scaling Laws for Hyperparameter Optimization

Hyperparameter optimization is an important subfield of machine learning that focuses on tuning the hyperparameters of a chosen algorithm to achieve peak performance. Recently, there has been a stream of methods that tackle the issue of hyperparameter optimization, however, most of the methods do not exploit the scaling law property of learning curves. In this work, we propose Deep Power Law (DPL), a neural network model conditioned to yield predictions that follow a power-law scaling pattern. Our model dynamically decides which configurations to pause and train incrementally by making use of multi-fidelity estimation. We compare our method against 7 state-of-the-art competitors on 3 benchmarks related to tabular, image, and NLP datasets covering 57 diverse search spaces. Our method achieves the best results across all benchmarks by obtaining the best any-time results compared to all competitors.

Authors: Arlind Kadra, Maciej Janowski, Martin Wistuba, Josif Grabocka


## Setting up the virtual environment

```
# The following commands assume the user is in the cloned directory
conda create -n dpl python=3.8
conda activate dpl
cat requirements.txt | xargs -n 1 -L 1 pip install
```

## Add the LCBench code & data


Copy the contents of `https://github.com/automl/LCBench` into a folder `lc_bench` in the root DPL repo.

From `https://figshare.com/projects/LCBench/74151` download `data_2k.zip` and extract the json file into `DPL/lc_bench/results/data_2k.json`.

## Running the Deep Power Laws (DPL) code

The entry script to running the experiment is `main_experiment.py`. The module can be used to start a full HPO search.

The main arguments for `main_experiment.py` are:

- `--index`: The worker index. Every worker runs the same experiment, however, with a different seed.
- `--fantasize_step`: The step used in fantasizing the next learning curve value from the last observed one for a certain hyperparameter configuration. 
- `--budget_limit`: The maximal number of HPO iterations.
- `--ensemble_size`: The ensemble size for the DPL surrogate.
- `--nr_epochs`: The number of epochs used to train (not refine) the HPO surrogate.
- `--dataset_name`: The name of the dataset used in the experiment. The dataset names must be matched with the benchmark they belong to.
- `--benchmark_name`: The name of the benchmark used in the experiment. Every benchmark offers its own distinctive datasets. Available options are lcbench, taskset and pd1.
- `--surrogate_name`: The method that will be run. 
- `--project_dir`: The directory where the project files are located.
- `--output_dir`: The directory where the project output files will be stored.

**A minimal example of running DPL**:

```
python main_experiment.py --index 1 --fantasize_step 1 --budget_limit 1000 --ensemble_size 5 --nr_epochs 250 --dataset_name "credit-g" --benchmark_name "lcbench" --surrogate_name "power_law" --project_dir "." --output_dir "."

```

The example above will run the first repetition (pertaining to the first seed) for a HPO budget of 1000 trials. It will use dataset credit-g from the lcbench benchmark.
The experiment will run the power law surrogate with an ensemble size of 5 members, where we will run each selected hyperparameter configuration by the acquisition function with 1 more step.
In the beginning and everytime that the training procedure is restarted, the models will be trained for 250 epochs. The script will consider the current folder as the project folder and it
will save the output files at the current folder.

## Plots

The plots that are included in our paper were generated from the functions in the module `plots/normalized_regret.py`.
The plots expect the following result folder structure:

```
├── results_folder
│   ├── benchmark_name
│   │   ├── method_name
│   │   │   ├── dataset_name_repetitionid.json

```
## Citation
```
@inproceedings{
kadra2023scaling,
title={Scaling Laws for Hyperparameter Optimization},
author={Arlind Kadra and Maciej Janowski and Martin Wistuba and Josif Grabocka},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=ghzEUGfRMD}
}
```

