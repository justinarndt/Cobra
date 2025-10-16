# examples/phase11_conditional_logic.py

from cobra.compiler.jit import jit

@jit
def maximum(a, b):
    """
    Returns the greater of two numbers using an if/else statement.
    """
    if a > b:
        return a
    else:
        return b

def main():
    """Runs the demonstration."""
    print("--- Running Conditional Logic Demonstration ---")
    
    val1, val2 = 100, 50
    print(f"\nTesting maximum({val1}, {val2})...")
    result1 = maximum(val1, val2)
    print(f"Result: {result1}")
    assert result1 == val1, "Test case 1 failed!"
    print("[SUCCESS] Test case 1 passed.")

    val3, val4 = -20, 40
    print(f"\nTesting maximum({val3}, {val4})...")
    result2 = maximum(val3, val4)
    print(f"Result: {result2}")
    assert result2 == val4, "Test case 2 failed!"
    print("[SUCCESS] Test case 2 passed.")

    print("\n--- Conditional Logic Demonstration Complete ---")


if __name__ == "__main__":
    main()

