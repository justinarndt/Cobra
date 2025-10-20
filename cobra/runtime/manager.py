# cobra/runtime/manager.py

import importlib.util
import sys
import os

# Determine the name of the compiled module file (.pyd on Windows)
MODULE_EXTENSION = '.pyd' if sys.platform == 'win32' else '.so'

# Construct the absolute path to the project's root directory
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the absolute path to the build directory (using Release config)
_build_dir = os.path.join(_project_root, 'build', 'Release')

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
        "Please build the project first."
    )

# Use importlib to load the module directly from its file path.
spec = importlib.util.spec_from_file_location('cobra_core', _module_path)
cobra_core = importlib.util.module_from_spec(spec)
sys.modules['cobra_core'] = cobra_core  # Add to sys.modules to make it globally importable
spec.loader.exec_module(cobra_core)


# --- Public API ---

# Import the C++ objects from the loaded module
from cobra_core import MemoryManager, DeviceType

# Get the singleton instance from the C++ extension.
manager = MemoryManager.get_instance()

# --- THE FIX ---
# The C++ bindings expose DeviceType at the module level, but the Python
# code expects it to be an attribute of the manager. We graft it onto the
# manager instance and the class here to correct the API mismatch.
# This is now possible because 'dynamic_attr' was enabled in C++.
manager.DeviceType = DeviceType
MemoryManager.DeviceType = DeviceType

