# cobra/compiler/jit.py

import functools
import inspect
import ast
from collections import namedtuple
import ctypes

# LLVM imports
from llvmlite import ir, binding

# --- LLVM Initialization ---
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Define the structure for a single instruction in our Intermediate Representation.
Instruction = namedtuple('Instruction', ['opcode', 'arg'])


class CobraIrGenerator(ast.NodeVisitor):
    """
    An AST visitor that walks the tree, generates a linear IR,
    and performs basic type inference.
    """
    def __init__(self):
        self.ir = []
        self._temp_count = 0
        self.types = {}  # Tracks the type ('int' or 'float') of variables

    def fresh_temp(self):
        """Returns a new temporary variable name."""
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def visit_FunctionDef(self, node):
        # Infer argument types (assuming they are passed in correctly)
        for arg in node.args.args:
            # We'll refine this later; for now, assume float if not specified
            self.types[arg.arg] = 'float' 
        
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
        self.types[target] = self.types.get(value_source, 'float')


    def visit_BinOp(self, node):
        left_source = self.visit(node.left)
        right_source = self.visit(node.right)
        target_temp = self.fresh_temp()

        # Type promotion: if either operand is a float, the result is a float.
        left_type = self.types.get(left_source, 'int')
        right_type = self.types.get(right_source, 'int')
        if left_type == 'float' or right_type == 'float':
            self.types[target_temp] = 'float'
        else:
            self.types[target_temp] = 'int'

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
        if isinstance(node.value, float):
            self.types[target_temp] = 'float'
        else:
            self.types[target_temp] = 'int'
        return target_temp

    def visit_Name(self, node):
        return node.id

    def visit_Return(self, node):
        value_source = self.visit(node.value)
        self.ir.append(Instruction('RETURN', value_source))
        self.types['return'] = self.types.get(value_source)


class CobraLlvmIrGenerator:
    """
    Translates our high-level Cobra IR into low-level LLVM IR,
    handling both integer and floating-point types.
    """
    def __init__(self, cobra_ir, arg_names, inferred_types, func_name="cobra_func"):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        self.types = inferred_types
        
        self.int_type = ir.IntType(64)
        self.double_type = ir.DoubleType()

        self.module = ir.Module(name="cobra_module")

        # Determine function signature from inferred types
        return_type = self._get_llvm_type(self.types.get('return', 'int'))
        arg_types = [self._get_llvm_type(self.types.get(name, 'int')) for name in arg_names]

        func_type = ir.FunctionType(return_type, arg_types)
        self.function = ir.Function(self.module, func_type, name=func_name)

        for i, arg_name in enumerate(self.arg_names):
            self.function.args[i].name = arg_name

        self.block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.block)
        self.symbol_table = {}

    def _get_llvm_type(self, type_str):
        if type_str == 'float':
            return self.double_type
        return self.int_type

    def generate(self):
        """Generates the LLVM IR from the Cobra IR."""
        for arg in self.function.args:
            self.symbol_table[arg.name] = arg

        for opcode, arg in self.cobra_ir:
            if opcode == 'LOAD_CONST':
                target, value = arg
                const_type = self._get_llvm_type(self.types.get(target))
                self.symbol_table[target] = ir.Constant(const_type, value)
            
            elif opcode in ('ADD', 'SUB', 'MUL'):
                target, left_src, right_src = arg
                target_type_str = self.types.get(target)
                
                left_val = self.symbol_table[left_src]
                right_val = self.symbol_table[right_src]

                # If it's a float operation, use floating point instructions
                if target_type_str == 'float':
                    if opcode == 'ADD':
                        self.symbol_table[target] = self.builder.fadd(left_val, right_val, name=target)
                    elif opcode == 'SUB':
                        self.symbol_table[target] = self.builder.fsub(left_val, right_val, name=target)
                    elif opcode == 'MUL':
                        self.symbol_table[target] = self.builder.fmul(left_val, right_val, name=target)
                else: # Otherwise, use integer instructions
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
    
    # 1. Python AST -> Cobra IR (with type inference)
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    inferred_types = ir_generator.types
    arg_names = inspect.getfullargspec(func).args

    # 2. Cobra IR -> LLVM IR
    func_name = f"cobra_{func.__name__}"
    llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names, inferred_types, func_name)
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
    def _get_ctype(type_str):
        if type_str == 'float':
            return ctypes.c_double
        return ctypes.c_int64

    cfunc_return_type = _get_ctype(inferred_types.get('return'))
    cfunc_arg_types = [_get_ctype(inferred_types.get(name)) for name in arg_names]
    cfunc_type = ctypes.CFUNCTYPE(cfunc_return_type, *cfunc_arg_types)
    cfunc = cfunc_type(func_ptr)
    
    print("[COBRA JIT] Compilation to native code complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """The wrapper that calls the compiled native code."""
        print(f"\n[COBRA JIT] Executing native code for function: '{func.__name__}'")
        result = cfunc(*args)
        print(f"[COBRA JIT] Successfully executed native code.")
        return result
        
    wrapper.engine = engine
    
    return wrapper

