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


class CobraIrInterpreter:
    """
    Executes a list of Cobra IR instructions.
    """
    def __init__(self, cobra_ir, arg_names, arg_values):
        self.ir = cobra_ir
        self.memory = {}

        # Load the function's arguments into memory.
        for name, value in zip(arg_names, arg_values):
            self.memory[name] = value

    def run(self):
        """Executes the IR instructions sequentially."""
        print("[COBRA IR] --- Interpreter Start ---")
        for instruction in self.ir:
            opcode, arg = instruction.opcode, instruction.arg
            print(f"[COBRA IR] Executing: {opcode} {arg}")

            if opcode == 'ADD':
                target, left_src, right_src = arg
                left_val = self.memory[left_src]
                right_val = self.memory[right_src]
                self.memory[target] = left_val + right_val
            
            elif opcode == 'STORE':
                target, source = arg
                self.memory[target] = self.memory[source]

            elif opcode == 'RETURN':
                return_val = self.memory[arg]
                print("[COBRA IR] --- Interpreter End ---")
                return return_val
        
        # In case there is no return statement.
        print("[COBRA IR] --- Interpreter End (No Return) ---")
        return None


def jit(func):
    """
    A decorator that intercepts a function call, translates it to
    Cobra IR, and then executes it using an interpreter.
    """
    print(f"[COBRA JIT] Compiling function: '{func.__name__}'...")
    
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    
    # Generate the Intermediate Representation
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    
    # Get the names of the function's arguments
    arg_names = inspect.getfullargspec(func).args

    print("[COBRA JIT] Generated Intermediate Representation (IR):")
    for instruction in cobra_ir:
        print(f"    {instruction}")
    print("[COBRA JIT] Compilation complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper function that executes the 'compiled' code."""
        print(f"\n[COBRA JIT] Executing compiled function: '{func.__name__}'")
        
        # Create and run the interpreter instead of the original function.
        interpreter = CobraIrInterpreter(cobra_ir, arg_names, args)
        result = interpreter.run()
        
        print(f"[COBRA JIT] Successfully executed IR for function '{func.__name__}'.")
        
        return result
        
    return wrapper

