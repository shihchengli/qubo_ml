from typing import Callable

import torch
import torch.nn as nn
import numpy as np

def bounded_mse_loss(
    predictions: torch.tensor,
    targets: torch.tensor,
    limits: torch.tensor,
) -> torch.tensor:
    """
    Loss function for use with regression when some targets are presented as inequalities.

    :param predictions: Model predictions with shape(batch_size, tasks).
    :param targets: Target values with shape(batch_size, tasks).
    :param limits: A tensor of shape(batch_size, 2) containing the lower and upper bounds for each target.
    :return: A tensor containing loss values of shape(batch_size, tasks).
    """
    lower_limits = limits[:, 0].view(-1, 1)
    upper_limits = limits[:, 1].view(-1, 1)

    less_than_limits = predictions < lower_limits
    greater_than_limits = predictions > upper_limits

    predictions = torch.where(torch.logical_and(predictions < targets, less_than_limits), lower_limits, predictions)

    predictions = torch.where(torch.logical_and(predictions > targets, greater_than_limits), upper_limits, predictions)

    return nn.functional.mse_loss(predictions, targets, reduction="none")
