import csv
import logging
import os
import pickle
import re
from typing import Callable, List, Union

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from scipy.stats.mstats import gmean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data.scaler import StandardScaler
from .data.data import MoleculeDataset
from .model import MoleculeModel

MODEL_FILE_NAME = 'model.pt'

def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    logger: logging.Logger = None,
    smiles_columns: List[str] = None
    )-> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

    with open(data_path) as f:
        f = open(data_path)
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    features_header = []
    if features_path is not None:
        extension_sets = set([os.path.splitext(feat_path)[1] for feat_path in features_path])
        if extension_sets == {'.csv'}:
            for feat_path in features_path:
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    feat_header = next(reader)
                    features_header.extend(feat_header)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                targets = [x.tolist() if isinstance(x, np.ndarray) else x for x in dataset_targets[i]]
                writer.writerow(smiles + targets)

        if features_path is not None:
            dataset_features = dataset.extra_features()
            if extension_sets == {'.csv'}:
                with open(os.path.join(save_dir, f"{name}_features.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features_header)
                    writer.writerows(dataset_features)
            else:
                np.save(os.path.join(save_dir, f"{name}_features.npy"), dataset_features)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

    if save_split_indices:
        with open(os.path.join(save_dir, "split_indices.pckl"), "wb") as f:
            pickle.dump(all_split_indices, f)

def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)

def save_checkpoint(
    path: str,
    model: MoleculeModel,
    scaler: StandardScaler = None,
) -> None:
    """
    Saves a model checkpoint.

    :param path: Path where checkpoint will be saved.
    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    """
    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None

    state = {
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
    }
    torch.save(state, path)

def build_optimizer(model: nn.Module, lr: float) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param lr: The learninig rate.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(),
               "lr": lr,
               "weight_decay": 0}]

    return Adam(params)

def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def param_count_all(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return mean_squared_error(targets, preds, squared=False)

def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    raise ValueError(f'Metric "{metric}" not supported.')

def multitask_mean(
    scores: np.ndarray,
    metric: str,
    axis: int = None,
) -> float:
    """
    A function for combining the metric scores across different
    model tasks into a single score. When the metric being used
    is one that varies with the magnitude of the task (such as RMSE),
    a geometric mean is used, otherwise a more typical arithmetic mean
    is used. This prevents a task with a larger magnitude from dominating
    over one with a smaller magnitude (e.g., temperature and pressure).

    :param scores: The scores from different tasks for a single metric.
    :param metric: The metric used to generate the scores.
    :axis: The axis along which to take the mean.
    :return: The combined score across the tasks.
    """
    scale_dependent_metrics = ["rmse", "mae", "mse", "bounded_rmse", "bounded_mae", "bounded_mse"]
    nonscale_dependent_metrics = [
        "auc", "prc-auc", "r2", "accuracy", "cross_entropy",
        "binary_cross_entropy", "sid", "wasserstein", "f1", "mcc",
    ]

    if metric in scale_dependent_metrics:
        return gmean(scores, axis=axis)
    elif metric in nonscale_dependent_metrics:
        return np.mean(scores, axis=axis)
    else:
        raise NotImplementedError(
            f"The metric used, {metric}, has not been added to the list of\
                metrics that are scale-dependent or not scale-dependent.\
                This metric must be added to the appropriate list in the multitask_mean\
                function in `chemprop/utils.py` in order to be used."
        )

def load_checkpoint(
    path: str,
    num_features: int,
    num_elements_per_additional_feature: List[int] = None,
    loss_function=nn.MSELoss(reduction="none"),
    device: torch.device = None,
    logger: logging.Logger = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = state["state_dict"]

    # Build model
    model = MoleculeModel(num_features, num_elements_per_additional_feature, loss_function)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    return model

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger
