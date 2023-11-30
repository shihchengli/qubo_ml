from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self,
                 num_features: int,
                 elements_to_ignore_interaction_between: Optional[List[List[List[int]]]] = None,
                 elements_to_ignore_internal_interaction: Optional[List[List[int]]] = None,
                 loss_function: nn.MSELoss = nn.MSELoss(reduction="none")):
        """
        :param elements_to_ignore_interaction_between: A list of indices indicating the start and end of two subgroup of elements.
                                                       The interaction of the elements between the two subgroups will be ignored.
        :param elements_to_ignore_internal_interaction: A list of indices indicating the start and end of a subgroup of elements.
                                                        The interaction of the elements within the subgroup will be ignored.
        They are used to avoid interactions between the elements within the same feature set during the learning process.
        """
        super(MoleculeModel, self).__init__()
        self.num_features = num_features
        self.Q = nn.Parameter(torch.zeros(num_features, num_features))  # Initialize Q as a learnable parameter
        # self.bias = nn.Parameter(torch.zeros(1))
        self.loss_function = loss_function
        initialize_weights(self)
        mask = torch.tril(torch.ones(num_features, num_features), diagonal=-1).bool()
        # mask = torch.zeros(num_features, num_features).bool() <- slightly less helpful

        if elements_to_ignore_interaction_between is not None:
            for (range1_start, range1_end), (range2_start, range2_end) in elements_to_ignore_interaction_between:
                mask[range1_start-1:range1_end, range2_start-1:range2_end] = True
                mask[range2_start-1:range2_end, range1_start-1:range1_end] = True
        
        if elements_to_ignore_internal_interaction is not None:
            for start, end in elements_to_ignore_internal_interaction:
                mask[start-1:end, ::] = True
                mask[::, start-1:end] = True

        diagonal_mask = np.eye(num_features, num_features, dtype=bool)
        diagonal_mask = torch.tensor(diagonal_mask)
        mask[diagonal_mask] = False
        self.mask = mask
        self.Q.data.masked_fill_(mask, 0)

    def zero_lower_triangle_gradients(self):
        with torch.no_grad():
            self.Q.grad[self.mask] = 0

    def forward(
        self,
        features_batch: List[np.ndarray] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param features_batch: A list of numpy arrays containing additional features.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """
        # x^T * Q * x + b
        x = torch.from_numpy(np.stack(features_batch)).float().to(device)
        xQx = torch.matmul(torch.matmul(x, self.Q), x.t())
        output = xQx.diag().view(-1, 1)
        # return output + self.bias
        return output
