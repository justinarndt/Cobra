# examples/phase14_function_calls.py

from cobra import jit

@jit
def add_one(x):
    """A simple JIT-compiled function that adds one to its input."""
    return x + 1

@jit
def process_value(y):
    """
    A JIT-compiled function that calls another JIT-compiled function.
    """
    # This call will be compiled into a native function call.
    result = add_one(y)
    return result * 2

def main():
    """Runs the demonstration."""
    print("--- Running Function Call Demonstration ---")
    
    input_val = 10
    
    print(f"\nCalling the top-level JIT function 'process_value' with input: {input_val}")
    # This first call will trigger the compilation of both functions.
    result = process_value(input_val)
    
    print(f"\nResult of process_value({input_val}): {result}")

    # Verification step
    expected_result = (input_val + 1) * 2
    print(f"Expected result (from Python): {expected_result}")
    assert result == expected_result, "JIT result does not match expected result!"
    print("\n[SUCCESS] JIT compiled result is correct.")
    
    print("\n--- A second call to test the cache ---")
    # This second call should be faster as it will use the cached, already-compiled function.
    result2 = process_value(20)
    print(f"Result of process_value(20): {result2}")
    assert result2 == 42

    print("\n--- Function Call Demonstration Complete ---")


if __name__ == "__main__":
    main()

