from importlib import import_module
from warnings import warn


INSTANTIATE_DICT_KEYS = ["callable", "args", "kwargs", "partial"]


class InstantiateDict(dict):
    """Dictionary to instantiate a class from a dictionary.
    Args:
        input_dict (dict): Dictionary with keys "callable", "args", "kwargs", and "partial".

    Attributes:
        callable (str): Name of the class to instantiate.
        args (tuple): Arguments to pass to the class constructor.
        kwargs (dict): Keyword arguments to pass to the class constructor
        partial (bool): Whether to use functools.partial to instantiate the class.
    """

    def __init__(self, input_dict):
        super().__init__(input_dict)
        if "callable" not in self.keys():
            return KeyError("No callable key in the dictionary")
        if "args" not in self.keys():
            self["args"] = ()
        if "kwargs" not in self.keys():
            self["kwargs"] = {}
        if "partial" in self.keys():
            self["callable"] = f"functools.partial({self['callable']})"
            del self["partial"]

        for key in self.keys():
            if key not in INSTANTIATE_DICT_KEYS:
                warn(f"Key {key} not recognized in InstantiateDict object.", stacklevel=0)


def instantiate(instanciate_dict: InstantiateDict):
    """Instantiate a class or call a function from a dictionary."""
    # Import the module and extract the callable
    module_path, callable_name = instanciate_dict["callable"].rsplit(".", 1)
    module = import_module(module_path)
    _callable = getattr(module, callable_name)

    # Call the callable with the provided arguments
    instance = _callable(*instanciate_dict["args"], **instanciate_dict["kwargs"])
    return instance
