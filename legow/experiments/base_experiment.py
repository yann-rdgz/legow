from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from legow.metrics import HistoryMetrics

from .utils import InstantiateDict, instantiate


class BaseExperiment:
    """Base class for experiments.

    This class is used to define the structure of an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    logging_dir : str, optional
        Directory to save the logs (default is None).
    seed : int, optional
        Seed for the experiment (default is None).

    Attributes
    ----------
    is_random : bool
        Whether the experiment is random.
    """

    def __init__(
        self,
        experiment_name,
        list_config_dict_seed_func: Optional[list[dict]] = None,
        logging_dir=None,
    ):
        self._experiment_name: str = experiment_name
        self._logging_dir = Path(logging_dir or "")
        self.is_random: Optional[bool] = None
        self._experiment_uri: Optional[str] = None
        if list_config_dict_seed_func is not None:
            self.seed_experiment([InstantiateDict(inst_dict) for inst_dict in list_config_dict_seed_func])

    @abstractmethod
    def launch_experiment(self): ...

    def seed_experiment(self, instance_dict_seed_functions: list[InstantiateDict]):
        """Seed the experiment.

        This method seeds the experiment.

        Parameters
        ----------
        instance_dict_seed_functions : list[InstantiateDict]
            List of dictionaries with the seed functions to instantiate.

        """
        self.is_random = False
        for instantiate_dict in instance_dict_seed_functions:
            instantiate(instantiate_dict)
            # TODO: find a way to check that the seed is set


class ModelingExperiment(BaseExperiment):
    """Base class for modeling experiments.

    This class is used to define the structure of a modeling experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    config_dict_model : dict
        Configuration dictionary for the model.
    config_dict_data : dict
        Configuration dictionary for the data.
    config_dict_seed_func : list[dict], optional
        List of dictionaries with the seed functions to instantiate (default is None).

    Attributes
    ----------
    config_dict_model : dict
        Configuration dictionary for the model.
    models_collections : dict
        Collection of models.
    history_metrics_train : HistoryMetrics
        History of training metrics.
    history_metrics_val : HistoryMetrics
        History of validation metrics.
    history_metrics_test : HistoryMetrics
        History of test metrics.
    func_metrics_binary : tuple
        Tuple of binary metrics functions.
    func_metrics_continuous : tuple
        Tuple of continuous metrics functions.

    """

    def __init__(
        self,
        experiment_name,
        config_dict_model: dict,
        config_dict_data: dict,
        list_config_dict_seed_func: Optional[list[dict]] = None,
        list_config_dict_metrics: Optional[list[dict]] = None,
        list_config_dict_metrics_proba: Optional[list[dict]] = None,
    ):
        super().__init__(experiment_name, list_config_dict_seed_func)
        self.config_dict_model = InstantiateDict(config_dict_model)
        self.config_dict_data = InstantiateDict(config_dict_data)
        self.models_collections: dict[str, BaseEstimator] = {}

        self._dataset_train, self._labels_train, self._dataset_test, self._labels_test = instantiate(
            self.config_dict_data
        )

        self._train_index = None
        self._val_index = None
        self.history_metrics_train = HistoryMetrics()
        self.history_metrics_val = HistoryMetrics()
        self.history_metrics_test = HistoryMetrics()

        self.func_metrics_binary: list[Callable] = []
        if list_config_dict_metrics is not None:
            self.func_metrics_binary = [
                instantiate(InstantiateDict(inst_dict)) for inst_dict in list_config_dict_metrics
            ]
        self.func_metrics_continuous: list[Callable] = []
        if list_config_dict_metrics_proba is not None:
            self.func_metrics_continuous = [
                instantiate(InstantiateDict(inst_dict)) for inst_dict in list_config_dict_metrics_proba
            ]

    @abstractmethod
    def launch_experiment(self):
        pass

    @abstractmethod
    def training(self):
        pass

    def evaluation(self):
        if self.X_test is not None:
            test_predict = self.model.predict(self.X_test)
            test_metrics_bin = self.measure_predict(test_predict, self.y_test)
            self.history_metrics_test.update(test_metrics_bin)

            if hasattr(self.model, "predict_proba"):
                test_val_predict_proba = self.model.predict_proba(self.X_test)
                metrics_cont = self.measure_predict_proba(test_val_predict_proba, self.y_test)
                self.history_metrics_test.update(metrics_cont)
        else:
            warn("No Test set provided", stacklevel=1)

    @property
    def X_train(self):
        return self._dataset_train[self._train_index]

    @property
    def y_train(self):
        return self._labels_train[self._train_index]

    @property
    def X_val(self):
        return self._dataset_train[self._val_index]

    @property
    def y_val(self):
        return self._labels_train[self._val_index]

    @property
    def X_test(self):
        return self._dataset_test

    @property
    def y_test(self):
        return self._labels_test

    @abstractmethod
    def measure_predict(self, y_true: np.ndarray, y_pred: np.ndarray, key_suffix="") -> dict:
        results: dict[str, float] = {}
        for metric in self.func_metrics_binary:
            results[f"{metric.__name__}{key_suffix}"] = metric(y_true, y_pred)

        return results

    @abstractmethod
    def measure_predict_proba(self, y_true: np.ndarray, y_pred: np.ndarray, key_suffix="") -> dict:
        results: dict[str, float] = {}
        for metric in self.func_metrics_continuous:
            results[f"{metric.__name__}{key_suffix}"] = metric(y_true, y_pred)

        return results

    def set_train_val_index(self, train_index, val_index):
        self._train_index = train_index
        self._val_index = val_index

    def instantiate_new_model(self):
        return instantiate(self.config_dict_model)

    @property
    def model(self):
        # TODO: add regression and other kind of experiments
        return VotingClassifier(estimators=self.models_collections.items(), voting="soft")
