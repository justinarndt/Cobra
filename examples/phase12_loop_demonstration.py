# examples/phase12_loop_demonstration.py

from cobra.compiler.jit import jit

@jit
def summation(n):
    """
    Calculates the sum of numbers from 0 to n-1 using a while loop.
    """
    total = 0
    i = 0
    while i < n:
        total += i
        i += 1
    return total

def main():
    """Runs the demonstration."""
    print("--- Running Loop Demonstration ---")
    
    limit = 10
    
    print(f"\nNow, calling the '@jit' decorated 'summation' function with limit: {limit}")
    result = summation(limit)
    
    print(f"\nResult of summation({limit}): {result}")

    # Verification step
    expected_result = sum(range(limit))
    print(f"Expected result (from Python sum): {expected_result}")
    assert result == expected_result, "JIT result does not match expected result!"
    print("\n[SUCCESS] JIT compiled result is correct.")

    print("\n--- Loop Demonstration Complete ---")


if __name__ == "__main__":
    main()

