from typing import Optional

from sklearn.model_selection import RepeatedStratifiedKFold

from legow.metrics import HistoryMetrics

from .base_experiment import ModelingExperiment


class CrossValExperiment(ModelingExperiment):
    """Cross Validation Experiment.

    This class is used to perform a cross-validation experiment.

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
    list_config_dict_metrics : list[dict], optional
        List of dictionaries with the metrics functions to instantiate (default is None).
    list_config_dict_metrics_proba : list[dict], optional
        List of dictionaries with the metrics functions for probabilities to instantiate (default is None).

    Attributes
    ----------
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
        experiment_name: str,
        config_dict_model: dict,
        config_dict_data: dict,
        list_config_dict_seed_func: Optional[list[dict]] = None,
        list_config_dict_metrics: Optional[list[dict]] = None,
        list_config_dict_metrics_proba: Optional[list[dict]] = None,
    ):
        super().__init__(
            experiment_name,
            config_dict_model,
            config_dict_data,
            list_config_dict_seed_func,
            list_config_dict_metrics,
            list_config_dict_metrics_proba,
        )
        self.history_metrics_train = HistoryMetrics(cv=True)
        self.history_metrics_val = HistoryMetrics(cv=True)

    def training(self, n_splits=5, n_repeats=1):
        split_kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

        indices = [(i, j) for i in range(n_repeats) for j in range(n_splits)]

        for (i_rep, i_fold), (train_index, val_index) in zip(
            indices, split_kfold.split(self.X_train, self.y_train), strict=True
        ):
            self._train_index = train_index
            self._val_index = val_index
            model = self.instantiate_new_model()
            model.fit(self.X_train, self.y_train)

            val_pred = model.predict(self.X_val)
            val_pred_proba = model.predict_proba(self.X_val)

            metrics_predict = self.measure_predict(self.y_val, val_pred)
            metrics_predic_proba = self.measure_predict_proba(self.y_val, val_pred_proba)
            self.history_metrics_val.update(
                {"repeat": i_rep, "fold": i_fold, **metrics_predict, **metrics_predic_proba}
            )
            self.models_collections[f"rep_{i_rep}.fold_{i_fold}"] = model
