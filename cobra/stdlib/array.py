import numpy as np
from cobra.runtime import manager as memory_manager
import ctypes

# --- Capsule Handling Helper ---
# This is the definitive fix. We use ctypes to access Python's own C-API
# to correctly unwrap the py::capsule object and get the raw pointer address.
PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

def get_pointer_from_capsule(capsule):
    """Extracts the raw memory address from a py::capsule object."""
    return PyCapsule_GetPointer(capsule, None)
# -----------------------------

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
        It uses our helper function to correctly unwrap the C++ pointer from
        the py::capsule handle.
        """
        return get_pointer_from_capsule(self._handle)

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

