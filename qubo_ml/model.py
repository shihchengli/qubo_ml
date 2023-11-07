from typing import List

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
                 num_elements_per_additional_feature: List[int],# = None,
                 loss_function: nn.MSELoss = nn.MSELoss(reduction="none")):
        """
        :param num_elements_per_additional_feature: A list of the number of elements in each additional feature set.
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

        if num_elements_per_additional_feature is not None:
            print("The interaction for the elements in each of the following subgrop is ignored:")
            start = end = num_features - sum(num_elements_per_additional_feature)
            print(f"{start}/{num_features} of features are not frozen...")

            for i in num_elements_per_additional_feature:
                end = start + i - 1
                print(f"{start} to {end}")
                mask[start:end+1, start:end+1] = True
                start = end + 1

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
