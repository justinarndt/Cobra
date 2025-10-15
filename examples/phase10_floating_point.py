# examples/phase10_floating_point.py

from cobra.compiler.jit import jit
import math

@jit
def calculate_area(radius):
    """
    Calculates the area of a circle using a float constant
    and floating point multiplication.
    """
    return 3.14159 * radius * radius

def main():
    """Runs the demonstration."""
    print("--- Running Floating-Point Demonstration ---")
    
    radius = 5.5
    
    print(f"\nNow, calling the '@jit' decorated 'calculate_area' function with radius: {radius}")
    area = calculate_area(radius)
    
    print(f"\nResult of calculate_area({radius}): {area}")

    # Verification step
    # Use the same constant as the JIT'd function for a fair comparison.
    expected_area = 3.14159 * radius * radius
    print(f"Expected result (from Python): {expected_area}")
    assert math.isclose(area, expected_area), "JIT result does not match expected result!"
    print("\n[SUCCESS] JIT compiled result is correct.")

    print("\n--- Floating-Point Demonstration Complete ---")


if __name__ == "__main__":
    main()

