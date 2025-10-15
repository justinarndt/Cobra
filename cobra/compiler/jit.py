# cobra/compiler/jit.py

import functools
import inspect
import ast
from collections import namedtuple

# Define the structure for a single instruction in our Intermediate Representation.
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
        """
        Handles the function definition.
        This version correctly identifies and skips the docstring.
        """
        # ast.get_docstring checks if the first node in the body is a
        # string constant and returns it. We use this to decide if we
        # should skip the first node.
        docstring = ast.get_docstring(node)
        
        body_nodes = node.body
        if docstring:
            # If a docstring exists, it's the first node, so we skip it.
            body_nodes = node.body[1:]

        # Visit only the actual code statements.
        for statement in body_nodes:
            self.visit(statement)

    def visit_Assign(self, node):
        """Handles assignment statements (e.g., 'total = a + b')."""
        value_source = self.visit(node.value)
        target = node.targets[0].id
        self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_BinOp(self, node):
        """Handles binary operations (e.g., 'a + 5', 'b * c')."""
        left_source = self.visit(node.left)
        right_source = self.visit(node.right)
        target_temp = self.fresh_temp()

        # Map the AST operator to our IR opcode.
        if isinstance(node.op, ast.Add):
            opcode = 'ADD'
        elif isinstance(node.op, ast.Sub):
            opcode = 'SUB'
        elif isinstance(node.op, ast.Mult):
            opcode = 'MUL'
        else:
            raise NotImplementedError(f"Operator {type(node.op).__name__} not supported")
            
        self.ir.append(Instruction(opcode, (target_temp, left_source, right_source)))
        return target_temp

    def visit_Constant(self, node):
        """Handles literal constants (e.g., 5, 10.0)."""
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_CONST', (target_temp, node.value)))
        return target_temp

    def visit_Name(self, node):
        """Handles variable names."""
        return node.id

    def visit_Return(self, node):
        """Handles return statements."""
        value_source = self.visit(node.value)
        self.ir.append(Instruction('RETURN', value_source))


class CobraIrInterpreter:
    """
    Executes a list of Cobra IR instructions.
    """
    def __init__(self, cobra_ir, arg_names, arg_values):
        self.ir = cobra_ir
        self.memory = dict(zip(arg_names, arg_values))

    def run(self):
        """Executes the IR instructions sequentially."""
        print("[COBRA IR] --- Interpreter Start ---")
        for opcode, arg in self.ir:
            print(f"[COBRA IR] Executing: {opcode} {arg}")

            if opcode == 'LOAD_CONST':
                target, value = arg
                self.memory[target] = value
            
            elif opcode in ('ADD', 'SUB', 'MUL'):
                target, left_src, right_src = arg
                left_val = self.memory[left_src]
                right_val = self.memory[right_src]

                if opcode == 'ADD':
                    self.memory[target] = left_val + right_val
                elif opcode == 'SUB':
                    self.memory[target] = left_val - right_val
                elif opcode == 'MUL':
                    self.memory[target] = left_val * right_val
            
            elif opcode == 'STORE':
                target, source = arg
                self.memory[target] = self.memory[source]

            elif opcode == 'RETURN':
                return_val = self.memory[arg]
                print("[COBRA IR] --- Interpreter End ---")
                return return_val
        
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
    
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    
    arg_names = inspect.getfullargspec(func).args

    print("[COBRA JIT] Generated Intermediate Representation (IR):")
    for instruction in cobra_ir:
        print(f"    {instruction}")
    print("[COBRA JIT] Compilation complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper function that executes the 'compiled' code."""
        print(f"\n[COBRA JIT] Executing compiled function: '{func.__name__}'")
        
        interpreter = CobraIrInterpreter(cobra_ir, arg_names, args)
        result = interpreter.run()
        
        print(f"[COBRA JIT] Successfully executed IR for function '{func.__name__}'.")
        
        return result
        
    return wrapper

