# Application of the Digital Annealer Unit in Optimizing Chemical Reaction Conditions for Enhanced Production Yields
<p align="center"> [<b><a href="https://arxiv.org/abs/2407.17485">Paper</a></b>]

The repository contains all of the code and instructions needed to reproduce the experiments and results of **[Application of the Digital Annealer Unit in Optimizing Chemical Reaction Conditions for Enhanced Production Yields](https://arxiv.org/abs/2407.17485)**.

# Installation
```
git clone https://github.com/shihchengli/qubo_ml.git
cd qubo_ml
conda env create -f environment.yml
conda activate qubo_ml
```
**Tip 1:** mamba might be faster: try running `conda install -c conda-forge mamba` and then replacing `conda` with `mamba` in each of the steps above.

**Tip 2:** if you plan to use GPU, make sure you have the correct pytorch version with GPU support

# How to train a ML-based QUBO model?
We only support using Python scripts to train models. More examples can be found in the [scripts/ML](https://github.com/shihchengli/qubo_ml/tree/main/scripts/ML) folder.
```python
from qubo_ml.cross_validate import cross_validate

cross_validate(
    path="/path/to/data.csv",  # Path to the data csv file.
    features_path=["/path/to/data_features.csv"],  # Path to the features csv file(s).
    seed=0,  # Random seed to use when splitting data into train/val/test sets.
    pytorch_seed=0,  # Seed for PyTorch randomness (e.g., random initial weights).
    split_type="random",  # Method of splitting the data into train/val/test. Currently, 'cv', 'random', and 'scaffold_balanced' are supported.
    split_sizes=(0.8, 0.1, 0.1),  # Split proportions for train/validation/test sets.
    num_folds=1,  # Number of folds when performing cross-validation.
    ensemble_size=5,  # Number of models in ensemble.
    batch_size=5,  # Batch size.
    epochs=200,  # Number of epochs.
    lr=1e-3,  # Learning rate.
    smiles_columns=["substrate_smiles"],  # SMILES column for feature generation.
    task_names=["yield"],  # The task name.
    features_generator=None,  # Method(s) of generating additional features. Currently, 'MACCS', 'Avalon', and 'ECFP4' are supported.
    save_splits=True,  # Whether to save the split used during training.
    num_workers=8,  # Number of workers for the parallel data loading (0 means sequential).
    no_cuda=False,  # Turn off CUDA (i.e., use CPU instead of GPU).
    gpu=None,  # Which GPU to use.
    main_metric="rmse",  # Metric to use during evaluation. It is also used with the validation set for early stopping.
    metrics=["rmse", "mae", "r2"],  # Metrics to use to evaluate the model. Not used for early stopping.
    save_dir="/path/to/save_dir",  # Directory to save the results.
    save_preds=True,  # Whether to save test split predictions during training.
    reaction_mode=None,  # Choices for construction of features for reactions (e.g., reac_prod, reac_only, prod_only, diff_only, reac_diff, prod_diff).
    quiet=False  # Skip non-essential print statements.
)
```
# How to train a DAU-based QUBO model?

The script we used for this work is located in the [scripts/DAU](https://github.com/shihchengli/qubo_ml/tree/main/scripts/DAU) folder. The Python code specifies several requirements, including preparing files in a specified format and changing the current directory to the dataset's location.


# Relationship to Chemprop
I ([@shihchengli](https://github.com/shihchengli)) am also a developer of Chemprop, so I adopted most of the code from Chemprop. It allows users to benchmark the performance of the DAU-based model against other baselines (e.g., FFN, SVM, random forest) and the D-MPNN implemented in Chemprop.
