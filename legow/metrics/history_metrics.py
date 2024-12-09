import pandas as pd


class HistoryMetrics:
    """HistoryMetrics is a class for tracking and updating metrics over multiple epochs.
    Attributes:
        metrics (pd.DataFrame): A DataFrame to store metrics with epochs as the index.
    Methods:
        __init__():
            Initializes an empty DataFrame to store metrics with 'epoch' as the index name.
        update(metrics: dict):
            Updates the metrics DataFrame with a new row of metrics.
        __str__():
            Returns a string representation of the metrics DataFrame.
        mean():
            Returns the mean of the metrics DataFrame.
        std():
            Returns the standard deviation of the metrics DataFrame.
    """

    def __init__(self, cv: bool = False):
        # Initialize an empty DataFrame to store metrics
        self.metrics = pd.DataFrame()
        self._cv = cv
        if self._cv:
            self.metrics.index = pd.MultiIndex.from_tuples([], names=["repeat", "fold"])
        else:
            self.metrics.index.name = "epoch"
        self.columns: set[str] = set()

    def update(self, metrics: dict):
        """
        Update the metrics DataFrame with a new row of metrics.

        Args:
        - metrics (dict): A dictionary where keys are metric names and values are metric values.
        """
        self._validate_incoming_metrics(metrics)
        df_metrics = pd.DataFrame([metrics])
        if self._cv:
            df_metrics = df_metrics.set_index(["rep", "fold"])
        self.metrics = pd.concat([self.metrics, pd.DataFrame([metrics])])

    def __str__(self):
        return self.metrics.__str__()

    def mean(self):
        return self.metrics.mean()

    def std(self):
        return self.metrics.std()

    def _validate_incoming_metrics(self, dict_metrics: dict):
        col_dict_metrics = set(dict_metrics.keys())

        if self.metrics.empty:
            self.columns = col_dict_metrics
        else:
            if self._cv:
                if "rep" or "fold" not in col_dict_metrics:
                    raise ValueError("Missing 'rep' or 'fold' in the incoming metrics.")
                col_dict_metrics - {"rep", "fold"}

            if self.columns != col_dict_metrics:
                raise ValueError(f"New metrics {col_dict_metrics} do not match existing metrics {self.columns}.")
