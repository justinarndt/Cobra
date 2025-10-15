# cobra/compiler/jit.py

import functools
import inspect
import ast
import ctypes

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
        self._label_count = 0

    def fresh_temp(self):
        """Returns a new temporary variable name."""
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def fresh_label(self):
        """Returns a new label name."""
        name = f"L{self._label_count}"
        self._label_count += 1
        return name

    def visit_FunctionDef(self, node):
        """Handles the function definition."""
        docstring = ast.get_docstring(node)
        body_nodes = node.body[1:] if docstring else node.body
        for statement in body_nodes:
            self.visit(statement)

    def visit_Assign(self, node):
        """Handles assignment statements."""
        value_source = self.visit(node.value)
        target = node.targets[0].id
        self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_BinOp(self, node):
        """Handles binary operations."""
        left_source = self.visit(node.left)
        right_source = self.visit(node.right)
        target_temp = self.fresh_temp()
        op_map = {ast.Add: 'ADD', ast.Sub: 'SUB', ast.Mult: 'MUL'}
        opcode = op_map.get(type(node.op))
        if opcode:
            self.ir.append(Instruction(opcode, (target_temp, left_source, right_source)))
            return target_temp
        raise NotImplementedError(f"Operator {type(node.op).__name__} not supported")

    def visit_Compare(self, node):
        """Handles comparison operations (e.g., a > b)."""
        left_source = self.visit(node.left)
        right_source = self.visit(node.comparators[0])
        target_temp = self.fresh_temp()
        
        # For now, only support one comparison operator
        op_map = {ast.Gt: 'GT'} # Greater Than
        opcode = op_map.get(type(node.ops[0]))
        if opcode:
            self.ir.append(Instruction('COMPARE', (opcode, target_temp, left_source, right_source)))
            return target_temp
        raise NotImplementedError(f"Comparison {type(node.ops[0]).__name__} not supported")

    def visit_If(self, node):
        """Handles if statements."""
        # 1. Evaluate the test condition
        test_result_var = self.visit(node.test)
        
        # 2. Create labels for the 'then' and 'else' blocks
        then_label = self.fresh_label()
        else_label = self.fresh_label()
        end_label = self.fresh_label()

        # 3. Branch instruction
        self.ir.append(Instruction('JUMP_IF_TRUE', (test_result_var, then_label, else_label)))
        
        # 4. 'Then' block
        self.ir.append(Instruction('LABEL', then_label))
        for stmt in node.body:
            self.visit(stmt)
        self.ir.append(Instruction('JUMP', end_label))
        
        # 5. 'Else' block
        self.ir.append(Instruction('LABEL', else_label))
        for stmt in node.orelse:
            self.visit(stmt)
        
        # 6. End label
        self.ir.append(Instruction('LABEL', end_label))

    def visit_Constant(self, node):
        """Handles literal constants."""
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


class CobraLlvmIrGenerator:
    """Translates Cobra IR into LLVM IR."""
    def __init__(self, cobra_ir, arg_names):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        self.module = ir.Module(name="cobra_module")
        self.symbol_table = {}
        self.type_map = {}
        self._infer_types()

        func_type = ir.FunctionType(self.return_type, self.arg_types)
        self.function = ir.Function(self.module, func_type, name="cobra_func")
        for i, arg_name in enumerate(self.arg_names):
            self.function.args[i].name = arg_name

        self.entry_block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.entry_block)
        
        self.llvm_labels = {}

    def _infer_types(self):
        """A simple type inference pass over the Cobra IR."""
        # Default to integer
        var_types = {name: 'int' for name in self.arg_names}
        
        for opcode, arg in self.cobra_ir:
            if opcode == 'LOAD_CONST':
                target, value = arg
                if isinstance(value, float):
                    var_types[target] = 'float'
                else:
                    var_types[target] = 'int'
            elif opcode in ('ADD', 'SUB', 'MUL', 'COMPARE'):
                # Simple propagation: if any operand is float, result is float
                operands = arg[1:]
                is_float = any(var_types.get(op, 'int') == 'float' for op in operands)
                var_types[arg[0]] = 'float' if is_float else 'int'
            elif opcode == 'RETURN':
                self.return_type_str = var_types.get(arg, 'int')

        self.arg_types_str = [var_types.get(name, 'int') for name in self.arg_names]
        
        type_str_map = {'int': ir.IntType(64), 'float': ir.DoubleType(), 'bool': ir.IntType(1)}
        self.arg_types = [type_str_map[t] for t in self.arg_types_str]
        self.return_type = type_str_map[self.return_type_str]
        
        for name, type_str in var_types.items():
            self.type_map[name] = type_str_map.get(type_str)


    def generate(self):
        """Generates the LLVM IR."""
        for arg in self.function.args:
            self.symbol_table[arg.name] = arg

        for opcode, arg in self.cobra_ir:
            if opcode == 'LABEL':
                self.llvm_labels[arg] = self.function.append_basic_block(name=arg)
                self.builder.position_at_end(self.llvm_labels[arg])
                
            elif opcode == 'LOAD_CONST':
                target, value = arg
                var_type = self.type_map[target]
                self.symbol_table[target] = ir.Constant(var_type, value)
            
            elif opcode in ('ADD', 'SUB', 'MUL'):
                target, left_src, right_src = arg
                left_val = self.symbol_table[left_src]
                right_val = self.symbol_table[right_src]
                op_type = self.type_map.get(left_src.name if hasattr(left_src, 'name') else left_src, 'int')

                if isinstance(op_type, ir.DoubleType):
                    op_map = {'ADD': self.builder.fadd, 'SUB': self.builder.fsub, 'MUL': self.builder.fmul}
                else:
                    op_map = {'ADD': self.builder.add, 'SUB': self.builder.sub, 'MUL': self.builder.mul}
                self.symbol_table[target] = op_map[opcode](left_val, right_val, name=target)

            elif opcode == 'COMPARE':
                op, target, left_src, right_src = arg
                left_val = self.symbol_table[left_src]
                right_val = self.symbol_table[right_src]
                op_type = self.type_map.get(left_src.name if hasattr(left_src, 'name') else left_src, 'int')

                if isinstance(op_type, ir.DoubleType):
                    self.symbol_table[target] = self.builder.fcmp_ordered('>', left_val, right_val, name=target)
                else:
                    self.symbol_table[target] = self.builder.icmp_signed('>', left_val, right_val, name=target)

            elif opcode == 'JUMP_IF_TRUE':
                cond_var, then_label, else_label = arg
                cond_val = self.symbol_table[cond_var]
                self.builder.cbranch(cond_val, self.llvm_labels[then_label], self.llvm_labels[else_label])
                
            elif opcode == 'JUMP':
                self.builder.branch(self.llvm_labels[arg])

            elif opcode == 'STORE':
                target, source = arg
                self.symbol_table[target] = self.symbol_table[source]

            elif opcode == 'RETURN':
                return_val = self.symbol_table[arg]
                self.builder.ret(return_val)

        return str(self.module)


def jit(func):
    """A decorator that performs JIT compilation via LLVM."""
    print(f"[COBRA JIT] Compiling function: '{func.__name__}'...")
    
    # Python AST -> Cobra IR
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    arg_names = inspect.getfullargspec(func).args
    
    # Cobra IR -> LLVM IR
    llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names)
    llvm_ir_str = llvm_ir_generator.generate()
    
    print("\n[COBRA JIT] Generated LLVM IR:")
    print(llvm_ir_str)
    
    # LLVM JIT Compilation
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    llvm_module = binding.parse_assembly(llvm_ir_str)
    llvm_module.verify()

    target_machine = binding.Target.from_default_triple().create_target_machine()
    engine = binding.create_mcjit_compiler(llvm_module, target_machine)
    engine.finalize_object()
    
    func_ptr = engine.get_function_address("cobra_func")

    # Define the ctypes function signature
    cfunc_arg_types = []
    for t in llvm_ir_generator.arg_types:
        if isinstance(t, ir.IntType):
            cfunc_arg_types.append(ctypes.c_int64)
        elif isinstance(t, ir.DoubleType):
            cfunc_arg_types.append(ctypes.c_double)
    
    cfunc_return_type = None
    if isinstance(llvm_ir_generator.return_type, ir.IntType):
        cfunc_return_type = ctypes.c_int64
    elif isinstance(llvm_ir_generator.return_type, ir.DoubleType):
        cfunc_return_type = ctypes.c_double

    cfunc = ctypes.CFUNCTYPE(cfunc_return_type, *cfunc_arg_types)(func_ptr)
    
    print("[COBRA JIT] Compilation to native code complete.")


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n[COBRA JIT] Executing native code for function: '{func.__name__}'")
        result = cfunc(*args)
        print("[COBRA JIT] Successfully executed native code.")
        return result
    
    # Keep the engine alive
    wrapper.engine = engine
    return wrapper

