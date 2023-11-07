import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
import logging
from typing import Callable
import pandas as pd
import json
from logging import Logger
import os
from typing import Dict, List, Tuple
from typing_extensions import Literal

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm, trange

from .data.data import MoleculeDataLoader, MoleculeDataset
from .model import MoleculeModel
from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .data.utils import get_data, split_data
from .utils import param_count_all, build_optimizer, load_checkpoint, makedirs, save_checkpoint, save_smiles_splits, multitask_mean

MODEL_FILE_NAME = 'model.pt'

def train(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    n_iter: int = 0,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
    torch_device = None,
    batch_size = 50,
) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        features_batch, target_batch = batch.features(), batch.targets()
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # shape(batch, tasks)

        # Run model
        model.zero_grad()
        preds = model(features_batch, torch_device)

        # Move tensors to correct device
        targets = targets.to(torch_device)

        # Calculate losses
        loss = loss_func(preds, targets)
        loss = loss.sum()
        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        model.zero_lower_triangle_gradients()
        #nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        n_iter += len(batch)

        # Log and/or add to tensorboard
        log_frequency = 10
        if (n_iter // batch_size) % log_frequency == 0:
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0
            debug(f"Loss = {loss_avg:.4e}")

            if writer is not None:
                writer.add_scalar("train_loss", loss_avg, n_iter)

    return n_iter

Metric = Literal['rmse', 'mae', 'mse']
def run_training(data_path: str,
                 data: MoleculeDataset,
                 separate_val_path: str = None,
                 separate_test_path: str = None,
                 features_generator: List[str] = None,
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
                 num_elements_per_additional_feature: List[int] = None,
                 smiles_columns: List[str] = None,
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
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {seed}')

    if separate_val_path:
        val_data = get_data(
            path=separate_val_path,
            smiles_columns=smiles_columns,
            target_columns=task_names,
            features_path=separate_val_features_path,
            features_generator=features_generator,
            is_reaction=is_reaction,
            reaction_mode=reaction_mode,
            logger=logger,
        )

    if separate_test_path:
        test_data = get_data(
            path=separate_test_path,
            smiles_columns=smiles_columns,
            target_columns=task_names,
            features_path=separate_test_features_path,
            features_generator=features_generator,
            is_reaction=is_reaction,
            reaction_mode=reaction_mode,
            logger=logger,
        )

    if separate_val_path and separate_test_path:
        train_data = data
    elif separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=split_type,
                                              sizes=split_sizes,
                                              key_molecule_index=split_key_molecule,
                                              seed=seed,
                                              num_folds=num_folds,
                                              logger=logger)
    elif separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                            split_type=split_type,
                                            sizes=split_sizes,
                                            key_molecule_index=split_key_molecule,
                                            seed=seed,
                                            num_folds=num_folds,
                                            logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                    split_type=split_type,
                                                    sizes=split_sizes,
                                                    key_molecule_index=split_key_molecule,
                                                    seed=seed,
                                                    num_folds=num_folds,
                                                    logger=logger)

    if num_elements_per_additional_feature is not None:
        for data in [train_data, val_data, test_data]:
            assert len(data[0].extra_features) == sum(num_elements_per_additional_feature), f"The sum of `num_elements_per_additional_feature`, {sum(num_elements_per_additional_feature)}, should equla to the number of extra features, {len(data[0].extra_features)}."

    if save_splits:
        save_smiles_splits(
            data_path=data_path,
            save_dir=save_dir,
            task_names=task_names,
            features_path=features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            logger=logger,
            smiles_columns=smiles_columns,
        )

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if len(val_data) == 0:
        raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
            a validation set is required to conduct early stopping according to the selected evaluation metric. This \
            may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

    if len(test_data) == 0:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')
        empty_test_set = True
    else:
        empty_test_set = False

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    debug('Fitting scaler')
    scaler = train_data.normalize_targets()

    # Get loss function
    loss_func = nn.MSELoss(reduction="none")

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    each_test_preds = np.zeros((len(test_smiles), num_tasks, ensemble_size))
    sum_test_preds = np.zeros((len(test_smiles), num_tasks))
    sum_test_preds_squared = np.zeros((len(test_smiles), num_tasks))

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Train ensemble of models
    for model_idx in range(ensemble_size):
        # Tensorboard writer
        model_save_dir = os.path.join(save_dir, f'model_{model_idx}')
        makedirs(model_save_dir)
        try:
            writer = SummaryWriter(log_dir=model_save_dir)
        except:
            writer = SummaryWriter(logdir=model_save_dir)

        # Load/build model
        debug(f'Building model {model_idx}')
        model = MoleculeModel(num_features=len(data[0].features),
                              num_elements_per_additional_feature=num_elements_per_additional_feature,
                              loss_function=loss_function)
        debug(model)
        debug(f'Number of parameters = {param_count_all(model):,}')


        cuda = not no_cuda and torch.cuda.is_available()
        if cuda:
            debug('Moving model to cuda')
            device = torch.device('cuda', gpu)
        else:
          device = torch.device('cpu')
        model = model.to(device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(model_save_dir, MODEL_FILE_NAME), model, scaler)

        # Optimizers
        optimizer = build_optimizer(model, lr=lr)

        # Run training
        best_score = float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(epochs):
            debug(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                torch_device = device,
                batch_size = batch_size,
            )

            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=num_tasks,
                metrics=metrics,
                scaler=scaler,
                torch_device = device,
                logger=logger
            )


            for metric, scores in val_scores.items():
                # Average validation score\
                mean_val_score = multitask_mean(scores, metric=metric)
                debug(f'Validation {metric} = {mean_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', mean_val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(val_scores[main_metric], metric=main_metric)
            if mean_val_score < best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(model_save_dir, MODEL_FILE_NAME), model, scaler)

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {main_metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(model_save_dir, MODEL_FILE_NAME),
                                num_features=len(data[0].features),
                                num_elements_per_additional_feature=num_elements_per_additional_feature,
                                loss_function=loss_function,
                                device=device,
                                logger=logger)

        if empty_test_set:
            info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        else:
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler,
                torch_device = device,
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=num_tasks,
                metrics=metrics,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)
                each_test_preds[:, :, model_idx] = np.array(test_preds)
                sum_test_preds_squared += np.square(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)

        writer.close()

    # Evaluate ensemble on test set
    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for task in task_names] for metric in metrics
        }
    else:
        avg_test_preds = (sum_test_preds / ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=num_tasks,
            metrics=metrics,
            logger=logger
        )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
        info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')

    # Save scores
    with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if save_preds and not empty_test_set:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

            if save_separate_preds:
                for model_idx in range(ensemble_size):
                    test_preds_dataframe[f"{task_name}_model_{model_idx}"] = each_test_preds[:,:,model_idx]

            if save_ensemble_variance:
                ensemble_vars = (
                    sum_test_preds_squared / ensemble_size
                    - np.square(sum_test_preds) / ensemble_size**2
                )
                test_preds_dataframe[f"{task_name}_var"] = [var[i] for var in ensemble_vars]

        test_preds_dataframe.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores
