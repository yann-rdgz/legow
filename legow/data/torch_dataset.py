from typing import Optional

import torch
from torch.utils.data import Dataset


class MultiSizeTensorDataset(Dataset):
    """Dataset for multi-size tensors.

    This class is used to create a dataset for multi-size tensors.

    Parameters
    ----------
    features : list
        List of features.
    labels : list, optional
        List of labels (default is None).
    """

    def __init__(self, features, labels: Optional[list] = None):
        super().__init__()
        self.features = features

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32).view((-1, 1))
        else:
            self.labels = torch.empty(0)

    def __getitem__(self, index):
        features = torch.tensor(self.features[index], dtype=torch.float32)
        if self.labels.nelement() != 0:
            return features, self.labels[index]
        return (features,)

    def __len__(self):
        return len(self.features)
