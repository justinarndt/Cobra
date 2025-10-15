"""
This module provides a user-friendly Python interface to the core C++ MemoryManager.
"""
import os
import shutil

# --- Module Loading ---
# This logic handles finding and importing the compiled C++ extension module.
try:
    # First, try a direct import. This works if the module is already in the path
    # or has been installed properly.
    from cobra_core import DeviceType, MemoryManager as _CppMemoryManager
except ImportError:
    # If the direct import fails, it likely means we are running from the source
    # directory without a proper installation. We'll try to find the compiled
    # .pyd file in the build directory and copy it locally.
    
    # Define the new name for our local module copy
    _module_local_name = 'cobra_core.pyd'
    
    # Path to the build directory
    _build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'Debug')

    _found_module_path = None
    if os.path.exists(_build_dir):
        for f in os.listdir(_build_dir):
            if f.startswith('cobra_core') and f.endswith('.pyd'):
                _found_module_path = os.path.join(_build_dir, f)
                break

    if _found_module_path:
        # Copy the found module to the root of the cobra package directory
        _destination_path = os.path.join(os.path.dirname(__file__), '..', _module_local_name)
        shutil.copy(_found_module_path, _destination_path)
        
        # Now, perform the import again. This should succeed.
        from cobra_core import DeviceType, MemoryManager as _CppMemoryManager
    else:
        raise ImportError(
            "Cobra C++ core module not found. "
            "Please build the project first by running 'cmake --build .' in the 'build' directory."
        )

# --- Public API ---

# Re-export the DeviceType enum so users can access it as 'cobra.runtime.DeviceType'
DeviceType = DeviceType

# Get the single, global instance of the C++ MemoryManager.
# We keep it as a "private" variable within this module.
_manager_instance = _CppMemoryManager.get_instance()

# Define our public Python functions that users will call.
# These functions simply delegate to the underlying C++ instance.

def allocate(size: int, device: DeviceType):
    """
    Allocates a block of memory on the specified device.

    Args:
        size (int): The number of bytes to allocate.
        device (DeviceType): The device (CPU or GPU) to allocate on.

    Returns:
        A handle to the allocated memory.
    """
    return _manager_instance.allocate(size, device)

def free(ptr):
    """
    Frees a previously allocated block of memory.

    Args:
        ptr: The memory handle to free.
    """
    _manager_instance.free(ptr)