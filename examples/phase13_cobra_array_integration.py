# examples/phase13_cobra_array_integration.py

from cobra import jit, CobraArray
import numpy as np

@jit
def scale_array(arr, scalar):
    """
    Multiplies every element in a CobraArray by a scalar value,
    operating directly on the array's memory. This function has no return value.
    """
    i = 0
    while i < arr.size:
        arr[i] = arr[i] * scalar
        i += 1

def main():
    """Runs the demonstration."""
    print("--- Running CobraArray JIT Integration Demonstration ---")

    initial_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 2.5
    
    # Create a CobraArray
    my_array = CobraArray(initial_data)
    
    print(f"\nInitial array: {my_array}")
    
    print(f"\nCalling JIT-compiled function to scale array by {scale_factor}...")
    scale_array(my_array, scale_factor)
    
    print(f"\nArray after JIT execution: {my_array}")

    # Verification step
    expected_data = np.array(initial_data) * scale_factor
    print(f"Expected result (from NumPy): {expected_data}")
    assert np.allclose(my_array.to_numpy(), expected_data), "JIT result does not match expected result!"
    print("\n[SUCCESS] JIT compiled result is correct.")

    print("\n--- CobraArray JIT Integration Demonstration Complete ---")


if __name__ == "__main__":
    main()

