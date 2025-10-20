import numpy as np
from cobra.runtime import manager as memory_manager
import ctypes

class CobraArray:
    """A dense, one-dimensional array object managed by the Cobra runtime."""

    def __init__(self, data, device=memory_manager.DeviceType.CPU):
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError("Input data must be a list or NumPy array.")
        
        self.array = np.array(data, dtype=np.float64, order='C')
        self.shape = self.array.shape
        self.dtype = self.array.dtype
        self.nbytes = self.array.nbytes
        self.size = self.array.size
        self.device = device

        self._handle = memory_manager.allocate(self.nbytes, self.device)
        
        # Copy data from the source numpy array to the newly allocated memory
        ctypes.memmove(self._data_ptr, self.array.ctypes.data, self.nbytes)

        print(f"INFO [CobraArray.__init__]: Allocated {self.nbytes} bytes for array with shape {self.shape} on device {self.device}. Handle: {self._handle}")

    @property
    def _data_ptr(self):
        """
        A property that returns the raw memory address of the allocated data.
        This is the crucial bridge that allows the JIT compiler to pass this
        array's data to a native C function. It uses ctypes to interpret the
        py::capsule handle as a raw pointer address (an integer).
        """
        return ctypes.cast(self._handle, ctypes.c_void_p).value

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            print(f"INFO [CobraArray.__del__]: Freeing memory for array with shape {self.shape} on device {self.device}. Handle: {self._handle}")
            memory_manager.free(self._handle)

    def to_numpy(self):
        """Returns a NumPy array that is a copy of the data in this CobraArray."""
        # Create a ctypes pointer to the raw data
        ptr_type = ctypes.POINTER(ctypes.c_double * self.size)
        ptr = ctypes.cast(self._data_ptr, ptr_type)
        # Create a NumPy array that views this memory (copy to be safe)
        return np.copy(np.frombuffer(ptr.contents, dtype=np.float64))

    def __repr__(self):
        return f"<CobraArray shape={self.shape}, dtype={self.dtype.name}, device={self.device.name}>"

    def __str__(self):
        return str(self.to_numpy())

