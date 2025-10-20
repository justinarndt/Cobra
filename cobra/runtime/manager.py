# cobra/runtime/manager.py

import importlib.util
import sys
import os

# --- THE DEFINITIVE FIX ---
# This block programmatically solves the "DLL load failed" error. It finds the
# Intel oneAPI installation and adds its compiler 'bin' directory to the system's
# DLL search path. This must be done *before* any attempt to import cobra_core.
oneapi_root = os.getenv("ONEAPI_ROOT")
if oneapi_root and sys.platform == 'win32':
    compiler_bin_path = os.path.join(oneapi_root, 'compiler', 'latest', 'bin')
    if os.path.isdir(compiler_bin_path):
        os.add_dll_directory(compiler_bin_path)
# -----------------------------


# Determine the name of the compiled module file (.pyd on Windows)
MODULE_EXTENSION = '.pyd' if sys.platform == 'win32' else '.so'

# Construct the absolute path to the project's root directory
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# The Ninja generator places build artifacts directly in the build directory.
_build_dir = os.path.join(_project_root, 'build')

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
sys.modules['cobra_core'] = cobra_core
spec.loader.exec_module(cobra_core)


# --- Public API ---

# Import the C++ objects from the loaded module
from cobra_core import MemoryManager, DeviceType

# Get the singleton instance from the C++ extension.
manager = MemoryManager.get_instance()

# --- THE WARM-UP FIX ---
# Call the new warm_up function immediately after loading the module.
# This safely initializes the SYCL runtime and prevents deadlocks.
manager.warm_up()

# Graft the DeviceType enum onto the manager object to correct the API mismatch.
manager.DeviceType = DeviceType
MemoryManager.DeviceType = DeviceType

