# cobra/compiler/jit.py

import functools
import inspect
import ast
from collections import namedtuple

# Define the structure for a single instruction in our Intermediate Representation.
# opcode: The operation to perform (e.g., 'LOAD', 'ADD', 'STORE').
# arg: The argument for the operation (e.g., a variable name or a constant value).
Instruction = namedtuple('Instruction', ['opcode', 'arg'])


class CobraIrGenerator(ast.NodeVisitor):
    """
    An AST visitor that walks the tree and generates a linear
    Intermediate Representation (IR) as a list of Instructions.
    """
    def __init__(self):
        self.ir = []
        self._temp_count = 0

    def fresh_temp(self):
        """Returns a new temporary variable name."""
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def visit_FunctionDef(self, node):
        """Handles the function definition."""
        # Process the function body
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Handles assignment statements (e.g., 'total = a + b')."""
        # First, visit the right-hand side of the assignment to generate
        # the code that computes the value.
        value_source = self.visit(node.value)
        
        # The left-hand side is the target variable.
        target = node.targets[0].id
        
        # Generate the STORE instruction.
        self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_BinOp(self, node):
        """Handles binary operations (e.g., 'a + b')."""
        # Visit the left and right operands to get their sources (variable names).
        left_source = self.visit(node.left)
        right_source = self.visit(node.right)
        
        # Create a new temporary variable to hold the result of the operation.
        target_temp = self.fresh_temp()

        # Map the AST operator to our IR opcode.
        if isinstance(node.op, ast.Add):
            opcode = 'ADD'
        # Add other operators like Sub, Mult, etc. here in the future.
        else:
            raise NotImplementedError(f"Operator {type(node.op).__name__} not supported")
            
        self.ir.append(Instruction(opcode, (target_temp, left_source, right_source)))
        return target_temp # Return the name of the temp var holding the result.

    def visit_Name(self, node):
        """Handles variable names."""
        # When we see a variable name, it's being used as a value.
        # We just return its name to be used by the calling visitor.
        return node.id

    def visit_Return(self, node):
        """Handles return statements."""
        # Visit the expression being returned to get its source.
        value_source = self.visit(node.value)
        self.ir.append(Instruction('RETURN', value_source))


def jit(func):
    """
    A decorator that intercepts a function call, translates it to
    Cobra IR, and then executes the original function.
    """
    print(f"[COBRA JIT] Compiling function: '{func.__name__}'...")
    
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    
    # Generate the Intermediate Representation
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    
    print("[COBRA JIT] Generated Intermediate Representation (IR):")
    for instruction in cobra_ir:
        print(f"    {instruction}")
    print("[COBRA JIT] Compilation complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper function that executes the 'compiled' code."""
        print(f"\n[COBRA JIT] Executing compiled function: '{func.__name__}'")
        
        # In the future, an interpreter will execute the IR.
        # For now, we still execute the original Python function.
        result = func(*args, **kwargs)
        
        print(f"[COBRA JIT] Successfully executed function '{func.__name__}'.")
        
        return result
        
    return wrapper

