import sys
import os

# This is a common Python pattern to make the 'cobra' package in the current
# project directory importable, even if it's not formally installed yet.
# It adds the project's root directory to the list of places Python looks for modules.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# Now we can perform a clean, high-level import
from cobra.runtime import manager as memory_manager
from cobra.runtime.manager import DeviceType


print("--- Starting Python Runtime Test ---")

# Access the DeviceType enum through the new runtime module
print(f"Accessed device types: {DeviceType.CPU}, {DeviceType.GPU}")

# Use the clean, Pythonic functions
print("\nCalling C++ functions via Python runtime...")
cpu_ptr = memory_manager.allocate(16384, DeviceType.CPU)
gpu_ptr = memory_manager.allocate(32768, DeviceType.GPU)

memory_manager.free(cpu_ptr)
memory_manager.free(gpu_ptr)

print("\n--- Python Runtime Test Finished ---")