import sys
import os
import gc # Import the garbage collector module

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now we can do a clean, top-level import
from cobra import CobraArray
from cobra.runtime.manager import DeviceType

def run_test():
    print("--- Starting CobraArray Lifecycle Test ---")

    print("\nStep 1: Creating a CPU array.")
    cpu_arr = CobraArray([1.0, 2.0, 3.0, 4.0], device=DeviceType.CPU)
    print(f"Created array: {cpu_arr}")

    print("\nStep 2: Creating a GPU array.")
    gpu_arr = CobraArray([[1, 2], [3, 4], [5, 6]], device=DeviceType.GPU)
    print(f"Created array: {gpu_arr}")

    print("\nStep 3: Explicitly deleting references to the arrays.")
    # The 'del' keyword removes the variable name, dropping the reference count to zero.
    del cpu_arr
    del gpu_arr

    print("\nStep 4: Forcing garbage collection to run now.")
    # This tells the GC to find all objects with no references and call their __del__ methods.
    gc.collect()

    print("\n--- Test function finished ---")

if __name__ == "__main__":
    run_test()
    print("\nScript finished. All arrays should now be freed.")