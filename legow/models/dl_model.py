from typing import Callable, Optional

import numpy as np
import torch
import torch.optim.optimizer
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from tqdm import tqdm

from legow.data.torch_dataset import MultiSizeTensorDataset


class DeepLearningEstimator(BaseEstimator):
    """
    A scikit-learn compatible estimator for deep learning models.

    Parameters:
    - model: PyTorch model
    - optimizer: PyTorch optimizer
    - criterion: PyTorch loss function
    - device: Device to run the model on (default is "cpu")
    - lr: Learning rate for the optimizer (default is 0.001)
    - batch_size: Batch size for training (default is 32)
    - epochs: Number of epochs to train for (default is 10)
    - collate_fn: Collate function for DataLoader (default is None)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        criterion: torch.nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.model = self.model.to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.collate_fn = collate_fn

    def fit(self, X, y):
        train_loader = self._create_loader(X, y)

        for epoch in tqdm(range(self.epochs), desc="Training", total=self.epochs):
            self.model.train()
            running_loss = 0.0
            # TODO: adapt to API different from MIL from Classic Algo
            for inputs, masks, labels in train_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)  # Not necessary for all models
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.model.zero_grad(set_to_none=True)

                # Forward + backward + optimize
                logits, _ = self.model(inputs, masks)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.cpu().detach().numpy()

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {avg_train_loss:.4f}")

    def predict(self, X):
        """
        Generates class predictions for the provided features.

        Parameters:
        - X: Features for data to predict on
        Returns:
        - List of predicted class labels
        """
        predictions = self.predict_proba(X)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def predict_proba(self, X):
        data_loader = self._create_dataloader(X, shuffle=False)
        self.model.eval()
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, masks in data_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                logits, _ = self.model(inputs, masks)
                scores = torch.sigmoid(logits)
                predictions.extend(scores.cpu().numpy())
        return np.array(predictions)

    def _create_dataloader(self, X, y=None, shuffle: bool = True):
        """
        Creates a DataLoader from features and labels.

        Parameters:
        - features: Input features
        - labels: Labels (optional)
        - shuffle: Whether to shuffle the data
        Returns:
        - DataLoader object
        """
        dataset = MultiSizeTensorDataset(X, y)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )
