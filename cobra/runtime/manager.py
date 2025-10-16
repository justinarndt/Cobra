# cobra/runtime/manager.py

# With the project installed via 'pip install -e .', Python's import
# system can now find the compiled C++ module automatically. All the
# manual importlib logic is no longer needed and has been removed.

from cobra.cobra_core.memory import MemoryManager, DeviceType

# Create a single, global instance of the MemoryManager.
manager = MemoryManager.get_instance()

