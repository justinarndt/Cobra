import numpy as np
from cobra.runtime import manager as memory_manager

class CobraArray:
    """
    A multi-dimensional array object that manages memory on a specified device (CPU or GPU)
    through the Cobra runtime.
    """
    def __init__(self, data, device=memory_manager.DeviceType.CPU):
        """
        Initializes a CobraArray.

        For this phase, the actual data is NOT copied to the device. We are only
        allocating the required memory and storing the array's metadata.

        Args:
            data: An object that can be converted to a NumPy array, like a list or tuple.
            device: The target device for memory allocation (e.g., DeviceType.CPU).
        """
        # Use NumPy to easily handle data conversion, shape, and type.
        _np_array = np.asarray(data)

        self.shape = _np_array.shape
        self.dtype = _np_array.dtype
        self.device = device
        self.nbytes = _np_array.nbytes

        # Call our Python runtime wrapper to allocate memory on the C++ backend.
        # The returned value is a placeholder handle to this memory.
        self._handle = memory_manager.allocate(self.nbytes, self.device)

        print(
            f"INFO [CobraArray.__init__]: Allocated {self.nbytes} bytes for array with "
            f"shape {self.shape} on device {self.device}. Handle: {self._handle}"
        )

    def __del__(self):
        """
        The Python garbage collector calls this method when a CobraArray object
        is no longer referenced. This is our chance to clean up the C++ memory.
        """
        if hasattr(self, '_handle') and self._handle is not None:
            print(
                f"INFO [CobraArray.__del__]: Freeing memory for array with "
                f"shape {self.shape} on device {self.device}. Handle: {self._handle}"
            )
            # Call our Python runtime wrapper to free the memory on the C++ backend.
            memory_manager.free(self._handle)

    def __repr__(self):
        """
        Provides a clean, descriptive string representation of the object.
        """
        return (
            f"<CobraArray shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device.name}>"
        )