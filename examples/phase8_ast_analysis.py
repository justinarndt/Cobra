# examples/phase8_ast_analysis.py

"""
Demonstrates the AST analysis capability of the @cobra.jit decorator.

This script shows that the decorator parses the function's source code
into an Abstract Syntax Tree (AST) at compile time (when the script is loaded)
and then executes the function logic at runtime.
"""

import sys
import os

# Add the project root to the Python path to allow importing 'cobra'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cobra.compiler.jit import jit

@jit
def calculate_sum(a, b):
    """A simple function with a variable and a return statement."""
    # This comment is ignored by the AST
    total = a + b
    return total

def main():
    """Main execution function."""
    print("\n--- Running AST Analysis Demonstration ---")
    
    # Note: The 'Compiling function' output from the decorator appears
    # before this main function even starts, as the decoration happens
    # when the Python interpreter first loads the script.
    
    print("\nNow, calling the '@jit' decorated 'calculate_sum' function:")
    result_sum = calculate_sum(30, 12)
    # The print statement from within the original function is gone because
    # we are now imagining this is executing "compiled" code. In our
    # current implementation, we just call the original function.
    print(f"Result of calculate_sum(30, 12): {result_sum}")
    
    print("\n--- AST Analysis Demonstration Complete ---")
    
    # Verify the results
    assert result_sum == 42

if __name__ == "__main__":
    main()

