from collections import defaultdict
import csv
import json
import os
import sys
from typing import List, Tuple, Optional
from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch.nn as nn

from .train import run_training
from .data.utils import get_data
from .utils import create_logger, makedirs, multitask_mean

Metric = Literal['rmse', 'mae', 'mse']
TRAIN_LOGGER_NAME = 'train'
TEST_SCORES_FILE_NAME = 'test_scores.csv'
def cross_validate(path: str,
                   separate_val_path: str = None,
                   separate_test_path: str = None,
                   seed: int = 0,
                   pytorch_seed: int = 0,
                   split_type: str = 'random',
                   split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   split_key_molecule: int = 0,
                   num_folds: int = 1,
                   ensemble_size: int = 1,
                   batch_size: int = 50,
                   epochs: int = 30,
                   lr: float = 1e-4,
                   task_names: List[str] = None,
                   num_tasks: int = 0,
                   features_path: List[str] = None,
                   separate_val_features_path: List[str] = None,
                   separate_test_features_path: List[str] = None,
                   elements_to_ignore_interaction_between: Optional[List[List[List[int]]]] = None,
                   elements_to_ignore_internal_interaction: Optional[List[List[int]]] = None,
                   smiles_columns: List[str] = None,
                   features_generator: List[str] = None,
                   save_splits: bool = True,
                   num_workers: int = 8,
                   no_cuda: bool = False,
                   gpu: int = None,
                   loss_function = nn.MSELoss(reduction="none"),
                   main_metric: Metric = None,
                   metrics: List[Metric] = [],
                   save_dir: str = None,
                   save_preds: bool = False,
                   save_separate_preds: bool = True,
                   save_ensemble_variance: bool = True,
                   is_reaction: bool = False,
                   reaction_mode: str = None,
                   quiet: bool = False,
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=save_dir, quiet=quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = seed

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Get data
    debug('Loading data')
    data = get_data(
        path=path,
        smiles_columns=smiles_columns,
        target_columns=task_names,
        features_path=features_path,
        features_generator=features_generator,
        is_reaction=is_reaction,
        reaction_mode=reaction_mode,
        logger=logger,
    )

    # Run training on different random seeds for each fold
    all_scores = defaultdict(list)
    for fold_num in range(num_folds):
        info(f'Fold {fold_num}')
        seed = init_seed + fold_num
        tmp_save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(tmp_save_dir)
        data.reset_features_targets_and_limits()

        # If resuming experiment, load results from trained models
        test_scores_path = os.path.join(tmp_save_dir, 'test_scores.json')
        model_scores = run_training(data_path = path,
                                    data = data,
                                    separate_val_path = separate_val_path,
                                    separate_test_path = separate_test_path,
                                    features_generator = features_generator,
                                    seed = seed,
                                    pytorch_seed = pytorch_seed,
                                    split_type = split_type,
                                    split_sizes = split_sizes,
                                    split_key_molecule = split_key_molecule,
                                    num_folds = num_folds,
                                    ensemble_size = ensemble_size,
                                    batch_size = batch_size,
                                    epochs = epochs,
                                    lr = lr,
                                    task_names = task_names,
                                    num_tasks = num_tasks,
                                    features_path = features_path,
                                    separate_val_features_path = separate_val_features_path,
                                    separate_test_features_path = separate_test_features_path,
                                    elements_to_ignore_interaction_between = elements_to_ignore_interaction_between,
                                    elements_to_ignore_internal_interaction = elements_to_ignore_internal_interaction,
                                    smiles_columns = smiles_columns,
                                    save_splits = save_splits,
                                    num_workers = num_workers,
                                    no_cuda = no_cuda,
                                    gpu = gpu,
                                    loss_function = loss_function,
                                    main_metric = main_metric,
                                    metrics = metrics,
                                    save_dir = tmp_save_dir,
                                    save_preds = save_preds,
                                    save_separate_preds = save_separate_preds,
                                    save_ensemble_variance = save_ensemble_variance,
                                    is_reaction = is_reaction,
                                    reaction_mode = reaction_mode,
                                    logger = logger)

        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)
    all_scores = dict(all_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    # Report results
    info(f'{num_folds}-fold cross validation')

    # Report scores for each fold
    contains_nan_scores = False
    for fold_num in range(num_folds):
        for metric, scores in all_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {multitask_mean(scores[fold_num], metric):.6f}')

    # Report scores across folds
    for metric, scores in all_scores.items():
        avg_scores = multitask_mean(scores, axis=1, metric=metric)  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(num_folds)]
        writer.writerow(header)

        for task_num, task_name in enumerate(task_names):
            row = [task_name]
            for metric, scores in all_scores.items():
                task_scores = scores[:, task_num]
                mean, std = np.mean(task_scores), np.std(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)

    # Determine mean and std score of main metric
    avg_scores = multitask_mean(all_scores[main_metric], metric=main_metric, axis=1)
    mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)

    # Optionally merge and save test preds
    if save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                                  for fold_num in range(num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score
