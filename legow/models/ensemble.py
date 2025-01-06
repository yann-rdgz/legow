"""Module for ensembling multiple PyTorch modules."""

from typing import Optional, Union

import torch
from torch import DeviceObjType, dtype, nn
from typing_extensions import Self


class EnsembleModule(nn.Module):
    """Module for ensembling multiple PyTorch modules."""

    def __init__(self, list_modules: list[nn.Module]):
        """
        Initialize the ensemble module.
        Args:
        - config_modules (ConfigDict): Configuration for the individual modules.
        - n_rep (int): Number of modules to ensemble.
        """
        super().__init__()
        self.modules = list_modules

    def forward(self, x, mask):
        """
        Forward pass through all the ensemble modules and aggregate their predictions.
        Args:
        - x (torch.Tensor): Input tensor.
        Returns:
        - ensemble_output (torch.Tensor): Aggregated output of all ensemble modules.
        """
        list_output = []
        list_scores = []
        for module in self.modules:
            logits, scores = module(x, mask)
            list_output.append(logits)
            list_scores.append(scores)
        ensemble_output = torch.stack(list_output).mean(dim=0)
        ensemble_scores = torch.stack(list_scores).mean(dim=0)
        return ensemble_output, ensemble_scores

    def to(
        self,
        device: Optional[DeviceObjType] = None,
        dtype: Optional[Union[dtype, str]] = None,
        non_blocking: bool = False,
    ) -> Self:
        """
        Moves the ensemble model to the specified device.
        Parameters
        ----------
        device : Optional[DeviceObjType]
            The device to which the ensemble model should be moved.
        dtype : Optional[Union[dtype, str]]
            The desired data type of the ensemble model.
        non_blocking : bool
            If True and the source is in pinned memory, the copy will be asynchronous
            with respect to the host.
        Returns
        -------
        Self
            The ensemble model after being moved to the specified device.
        """
        for module in self.modules:
            module.to(device, dtype, non_blocking)
        return self
