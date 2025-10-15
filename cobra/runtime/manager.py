"""
This module provides a user-friendly Python interface to the core C++ MemoryManager.
It is responsible for explicitly finding and loading the compiled C++ extension module.
"""
import os
import sys
import importlib.util

# --- Explicit Module Loading ---

# Determine the name of the compiled module file.
# On Windows it's .pyd, on Linux/macOS it's .so
MODULE_EXTENSION = '.pyd' if sys.platform == 'win32' else '.so'

# Construct the absolute path to the project's root directory.
# __file__ is the path to this file (manager.py)
# os.path.dirname() gets the directory of the file (c:\Cobra\cobra\runtime)
# We go up two levels to get to the root (c:\Cobra)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the absolute path to the build directory.
_build_dir = os.path.join(_project_root, 'build', 'Debug')

# Search for the compiled module file within the build directory.
_module_path = None
if os.path.exists(_build_dir):
    for f in os.listdir(_build_dir):
        if f.startswith('cobra_core') and f.endswith(MODULE_EXTENSION):
            _module_path = os.path.join(_build_dir, f)
            break

# If the module file was not found, raise a clear error.
if not _module_path:
    raise ImportError(
        f"Cobra C++ core module not found in '{_build_dir}'. "
        "Please build the project first by running 'cmake --build .' in the 'build' directory."
    )

# Use the importlib library to load the module directly from its file path.
# This is the most robust method and avoids all sys.path ambiguity.
spec = importlib.util.spec_from_file_location('cobra_core', _module_path)
_cobra_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_cobra_core)


# --- Public API ---

# Now that we have loaded the module into the _cobra_core variable,
# we can safely access its contents.
DeviceType = _cobra_core.DeviceType

_manager_instance = _cobra_core.MemoryManager.get_instance()


def allocate(size: int, device: DeviceType):
    """Allocates memory via the C++ backend."""
    return _manager_instance.allocate(size, device)


def free(ptr):
    """Frees memory via the C++ backend."""
    _manager_instance.free(ptr)