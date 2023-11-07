from typing import List

import torch
from tqdm import tqdm

from .data.data import MoleculeDataLoader, MoleculeDataset
from .data.scaler import StandardScaler
from .model import MoleculeModel

def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    torch_device = None,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()
    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        features_batch = batch.features()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(features_batch, torch_device)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
