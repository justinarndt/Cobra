# cobra/stdlib/frame.py
#
# This file implements the CobraFrame, a JIT-compilable DataFrame object
# designed to provide a familiar, pandas-like API while enabling massive
# performance gains through its lazy evaluation and kernel fusion engine.

from .array import CobraArray

class CobraFrame:
    """
    A JIT-compilable DataFrame object that enables automatic kernel fusion.
    """
    def __init__(self, data: dict):
        """
        Constructs a CobraFrame from a dictionary of column names to data.
        The data is immediately converted to CobraArray objects.
        
        Args:
            data (dict): A dictionary where keys are column names (str) and
                         values are list-like or NumPy arrays.
        """
        if not isinstance(data, dict):
            raise TypeError("CobraFrame must be initialized with a dictionary.")

        # The internal storage is a dictionary of column names to CobraArrays.
        self._data = {name: CobraArray(values) for name, values in data.items()}
        self._columns = list(data.keys())

    def __repr__(self):
        # A simple representation for the frame.
        # A real implementation would be more sophisticated, like pandas.
        header = " | ".join(f"{col:<10}" for col in self._columns)
        divider = "-" * len(header)
        return f"{header}\n{divider}"