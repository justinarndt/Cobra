# cobra/compiler/jit.py

import functools
import inspect
import ast
from collections import namedtuple

# New import for LLVM integration
from llvmlite import ir, binding

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
        docstring = ast.get_docstring(node)
        body_nodes = node.body
        if docstring:
            body_nodes = node.body[1:]
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


class CobraLlvmIrGenerator:
    """
    Translates our high-level Cobra IR into low-level LLVM IR.
    """
    def __init__(self, cobra_ir, arg_names):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        
        # For now, we assume all variables are 64-bit integers.
        self.int_type = ir.IntType(64)

        # Setup the LLVM module and function
        self.module = ir.Module(name="cobra_module")
        func_type = ir.FunctionType(self.int_type, [self.int_type] * len(arg_names))
        self.function = ir.Function(self.module, func_type, name="cobra_func")

        # Name the function arguments
        for i, arg_name in enumerate(self.arg_names):
            self.function.args[i].name = arg_name

        # Setup the IR builder
        self.block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
        
        # This will map our Cobra variable names to LLVM value pointers.
        self.symbol_table = {}

    def generate(self):
        """Generates the LLVM IR from the Cobra IR."""
        # First, store the initial function arguments in the symbol table
        for arg in self.function.args:
            self.symbol_table[arg.name] = arg

        # Process each Cobra instruction
        for opcode, arg in self.cobra_ir:
            if opcode == 'LOAD_CONST':
                target, value = arg
                self.symbol_table[target] = ir.Constant(self.int_type, value)
            
            elif opcode in ('ADD', 'SUB', 'MUL'):
                target, left_src, right_src = arg
                left_val = self.symbol_table[left_src]
                right_val = self.symbol_table[right_src]

                if opcode == 'ADD':
                    self.symbol_table[target] = self.builder.add(left_val, right_val, name=target)
                elif opcode == 'SUB':
                    self.symbol_table[target] = self.builder.sub(left_val, right_val, name=target)
                elif opcode == 'MUL':
                    self.symbol_table[target] = self.builder.mul(left_val, right_val, name=target)
            
            elif opcode == 'STORE':
                target, source = arg
                self.symbol_table[target] = self.symbol_table[source]

            elif opcode == 'RETURN':
                return_val = self.symbol_table[arg]
                self.builder.ret(return_val)

        return str(self.module)


def jit(func):
    """
    A decorator that performs JIT compilation via LLVM.
    """
    print(f"[COBRA JIT] Compiling function: '{func.__name__}'...")
    
    # 1. Python AST -> Cobra IR
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    arg_names = inspect.getfullargspec(func).args

    print("[COBRA JIT] Generated Intermediate Representation (IR):")
    for instruction in cobra_ir:
        print(f"    {instruction}")

    # 2. Cobra IR -> LLVM IR
    llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names)
    llvm_ir = llvm_ir_generator.generate()
    print("\n[COBRA JIT] Generated LLVM IR:")
    print(llvm_ir)
    
    print("[COBRA JIT] Compilation complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper function that executes the 'compiled' code."""
        print(f"\n[COBRA JIT] Executing function: '{func.__name__}'")
        
        # For now, we still use the Python interpreter to verify logic.
        # The next step will be to execute the LLVM-compiled code.
        interpreter = CobraIrInterpreter(cobra_ir, arg_names, args)
        result = interpreter.run()
        
        print(f"[COBRA JIT] Successfully executed function '{func.__name__}'.")
        
        return result
        
    return wrapper

