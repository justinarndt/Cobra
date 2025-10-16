# cobra/runtime/manager.py

# With the build system corrected, Python can find cobra_core as a sub-module.
from cobra.cobra_core import MemoryManager, DeviceType

# Get the singleton instance from the C++ extension.
manager = MemoryManager.get_instance()

# --- THE FIX ---
# The C++ bindings expose DeviceType at the module level, but the Python
# code expects it to be an attribute of the manager. We graft it onto the
# manager instance and the class here to correct the API mismatch.
manager.DeviceType = DeviceType
MemoryManager.DeviceType = DeviceType