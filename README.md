# Lumberjack

This repo contains the python prototype of Lumberjack used in the paper [Lumberjack: Better Differentially Private Random Forests through Heavy Hitter Detection in Trees](https://arxiv.org/abs/2605.22756).

Lumberjack is a Differentially Private Random Forest algorithm for classification tasks. 
The project uses Hydra configs to choose a dataset, choose a model, and run the experiment from `experiments/run.py`.

NOTE: The implementation ignores potential issues with floating points. The code should not be used in production environments as-is. A secure implementation should use trusted DP primitives such as those avilable in the OpenDP library.  

Contact Christian Janos Lebeda or David Erb for questions.

## Setup

Create and activate a Python environment with Python 3.11 and then install
the project in editable mode from the repository root:

```bash
pip install -e .
```

This installs the local `dp_random_forest` package and the experiment
dependencies declared in `pyproject.toml`.

## Download Datasets

Before running the cached datasets, download the network-backed data:

```bash
python scripts/download_all_datasets.py
```

By default this caches the Adult dataset and Folktables data under `data/`.
Folktables downloads can be customized, for example:

```bash
python scripts/download_all_datasets.py --folktables-states CA NY TX
```

## Configure An Experiment

The main experiment config is:

```text
experiments/conf/random_forest.yaml
```

It selects the dataset and model through Hydra defaults:

```yaml
defaults:
  - _self_
  - dataset: adult
  - model: dp_sparse_random_forest
```

Change `dataset` to any file stem in `experiments/conf/dataset/`, such as:

- `adult`
- `folktables`
- `moons`

Change `model` to any file stem in `experiments/conf/model/`, such as:

- `dp_sparse_random_forest` (Our technique, DP Lumberjack)
- `DiPriMeFlip_Forest`
- `diffprivlib_random_forest`
- `smooth_random_forest`
- `SNR_DP_forest`
- `notebook_dprf`
- `sklearn_decision_tree`
- `sklearn_extra_trees`

Then tune the selected config files. For example:

- Model hyperparameters live in `experiments/conf/model/<model>.yaml`.
  Typical values include `epsilon`, `delta`, `n_estimators`, `max_depth`,
  `random_state`, and model-specific options.
- Dataset hyperparameters live in `experiments/conf/dataset/<dataset>.yaml`.
  For `adult`, you can set `one_hot_encode`. For `folktables`, you can set
  `task`, `states`, `survey_year`, `one_hot_encode`, and categorical encoding
  options.

Some baseline models require dense ordinal categorical inputs, so the dataset
loaders may disable one-hot encoding automatically for those model configs.

## Run An Experiment

Run the active Hydra configuration with:

```bash
python experiments/run.py
```

The script loads the selected dataset, fits the selected forest, prints train and
test accuracy, and stores the fitted model as `forest.pkl` in a Hydra output
directory:

```text
outputs/<dataset-name>-<timestamp>/
```

## Example Workflow

1. Install the project:

   ```bash
   pip install -e .
   ```

2. Download datasets:

   ```bash
   python scripts/download_all_datasets.py
   ```

3. Edit `experiments/conf/random_forest.yaml` to select a dataset and model.

4. Edit the corresponding files in `experiments/conf/dataset/` and
   `experiments/conf/model/` to tune preprocessing and hyperparameters.

5. Run:

   ```bash
   python experiments/run.py
   ```
