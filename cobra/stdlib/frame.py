# MODIFIED: cobra/stdlib/frame.py

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

        self._data = {name: CobraArray(values) for name, values in data.items()}
        self._columns = list(data.keys())

    def __repr__(self):
        header = " | ".join(f"{col:<10}" for col in self._columns)
        divider = "-" * len(header)
        # A real implementation would show some data rows.
        return f"{header}\n{divider}"

    def __getitem__(self, key):
        """
        Retrieves a column by its name.
        
        Args:
            key (str): The name of the column.
            
        Returns:
            CobraArray: The array object for the requested column.
        """
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Assigns a new column or overwrites an existing one. The value can
        be a literal, a CobraArray, or crucially, an Expression Tree node
        which represents a pending computation.
        
        Args:
            key (str): The name of the new or existing column.
            value: The data or computation to assign.
        """
        # When an expression tree is assigned, it's stored directly. The JIT
        # compiler will handle its evaluation later.
        self._data[key] = value
        if key not in self._columns:
            self._columns.append(key)