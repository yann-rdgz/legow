def build_tabular_dataset(list_dataframes, list_indexes):
    """
    Builds a TabularDataset from a list of dataframes and a list of indexes.

    :param list_dataframes: List of dataframes.
    :param list_indexes: List of indexes.
    :return: TabularDataset.
    """
    pass


class TabularDataset:
    def __init__(self):
        """
        Initializes the TabularDataset with data and validators.

        :param data: List of dictionaries representing rows of the dataset.
        :param validators: Dictionary where keys are column names and values are
        functions to validate the column values.
        """
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
