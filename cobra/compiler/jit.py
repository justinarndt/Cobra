# cobra/compiler/jit.py

import functools
import inspect
import ast
from collections import namedtuple
import ctypes

# LLVM imports
from llvmlite import ir, binding

# --- LLVM Initialization ---
# This must be done once per process.
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

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
        docstring = ast.get_docstring(node)
        body_nodes = node.body
        if docstring:
            body_nodes = node.body[1:]
        for statement in body_nodes:
            self.visit(statement)

    def visit_Assign(self, node):
        value_source = self.visit(node.value)
        target = node.targets[0].id
        self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_BinOp(self, node):
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
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_CONST', (target_temp, node.value)))
        return target_temp

    def visit_Name(self, node):
        return node.id

    def visit_Return(self, node):
        value_source = self.visit(node.value)
        self.ir.append(Instruction('RETURN', value_source))


class CobraLlvmIrGenerator:
    """
    Translates our high-level Cobra IR into low-level LLVM IR.
    """
    def __init__(self, cobra_ir, arg_names, func_name="cobra_func"):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        self.int_type = ir.IntType(64)
        self.module = ir.Module(name="cobra_module")
        func_type = ir.FunctionType(self.int_type, [self.int_type] * len(arg_names))
        self.function = ir.Function(self.module, func_type, name=func_name)
        for i, arg_name in enumerate(self.arg_names):
            self.function.args[i].name = arg_name
        self.block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
        self.symbol_table = {}

    def generate(self):
        """Generates the LLVM IR from the Cobra IR."""
        for arg in self.function.args:
            self.symbol_table[arg.name] = arg

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
        return self.module


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

    # 2. Cobra IR -> LLVM IR
    func_name = f"cobra_{func.__name__}"
    llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names, func_name)
    llvm_module = llvm_ir_generator.generate()
    llvm_ir = str(llvm_module)
    print("\n[COBRA JIT] Generated LLVM IR:")
    print(llvm_ir)

    # 3. LLVM IR -> Native Machine Code
    target_machine = binding.Target.from_default_triple().create_target_machine()
    llvm_module_ref = binding.parse_assembly(llvm_ir)
    engine = binding.create_mcjit_compiler(llvm_module_ref, target_machine)
    engine.finalize_object()
    
    # 4. Get a pointer to the compiled function
    func_ptr = engine.get_function_address(func_name)

    # 5. Define the Python-callable function signature using ctypes
    cfunc_type = ctypes.CFUNCTYPE(ctypes.c_int64, *([ctypes.c_int64] * len(arg_names)))
    cfunc = cfunc_type(func_ptr)
    
    print("[COBRA JIT] Compilation to native code complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper that calls the compiled native code."""
        print(f"\n[COBRA JIT] Executing native code for function: '{func.__name__}'")
        result = cfunc(*args)
        print(f"[COBRA JIT] Successfully executed native code.")
        return result
        
    return wrapper

