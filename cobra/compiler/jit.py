# cobra/compiler/jit.py

import functools
import inspect
import ast
import ctypes
from collections import namedtuple

from llvmlite import ir, binding

from cobra.stdlib import CobraArray

# --- Global JIT Registry ---
# This cache stores metadata about compiled functions, allowing them
# to find and call each other. The key is the original Python function object.
_JIT_REGISTRY = {}
# ---------------------------

Instruction = namedtuple('Instruction', ['opcode', 'arg'])


class CobraIrGenerator(ast.NodeVisitor):
    """
    AST visitor that generates a linear Intermediate Representation (IR).
    """
    def __init__(self):
        self.ir = []
        self._temp_count = 0
        self._label_count = 0

    def fresh_temp(self):
        name = f"t{self._temp_count}"
        self._temp_count += 1
        return name

    def fresh_label(self):
        name = f"L{self._label_count}"
        self._label_count += 1
        return name

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        body_nodes = node.body[1:] if docstring else node.body
        for statement in body_nodes:
            self.visit(statement)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Subscript):
            array_name = node.targets[0].value.id
            index_var = self.visit(node.targets[0].slice)
            value_source = self.visit(node.value)
            self.ir.append(Instruction('STORE_ELEMENT', (array_name, index_var, value_source)))
        else:
            value_source = self.visit(node.value)
            target = node.targets[0].id
            self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_AugAssign(self, node):
        target_name = node.target.id
        value_source = self.visit(node.value)
        temp_result = self.fresh_temp()
        op_map = {ast.Add: 'ADD', ast.Sub: 'SUB', ast.Mult: 'MUL'}
        opcode = op_map.get(type(node.op))
        if not opcode:
            raise NotImplementedError(f"Augmented assignment operator {type(node.op).__name__} not supported")
        self.ir.append(Instruction(opcode, (temp_result, target_name, value_source)))
        self.ir.append(Instruction('STORE', (target_name, temp_result)))

    def visit_BinOp(self, node):
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
        left_source = self.visit(node.left)
        right_source = self.visit(node.comparators[0])
        target_temp = self.fresh_temp()
        op_map = {ast.Gt: 'GT', ast.Lt: 'LT'}
        opcode = op_map.get(type(node.ops[0]))
        if opcode:
            self.ir.append(Instruction('COMPARE', (opcode, target_temp, left_source, right_source)))
            return target_temp
        raise NotImplementedError(f"Comparison {type(node.ops[0]).__name__} not supported")

    def visit_Call(self, node):
        """Handles function calls (e.g., add_one(x))."""
        func_name = node.func.id
        arg_sources = [self.visit(arg) for arg in node.args]
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('CALL', (target_temp, func_name, arg_sources)))
        return target_temp

    def visit_If(self, node):
        test_result_var = self.visit(node.test)
        then_label, else_label = self.fresh_label(), self.fresh_label()
        end_label = self.fresh_label() if node.orelse else else_label
        jump_else_label = else_label if node.orelse else end_label
        self.ir.append(Instruction('JUMP_IF_TRUE', (test_result_var, then_label, jump_else_label)))
        self.ir.append(Instruction('LABEL', then_label))
        for stmt in node.body: self.visit(stmt)
        if node.orelse: self.ir.append(Instruction('JUMP', end_label))
        if node.orelse:
            self.ir.append(Instruction('LABEL', else_label))
            for stmt in node.orelse: self.visit(stmt)
        self.ir.append(Instruction('LABEL', end_label))

    def visit_While(self, node):
        header_label, body_label, exit_label = self.fresh_label(), self.fresh_label(), self.fresh_label()
        self.ir.append(Instruction('JUMP', header_label))
        self.ir.append(Instruction('LABEL', header_label))
        test_result_var = self.visit(node.test)
        self.ir.append(Instruction('JUMP_IF_TRUE', (test_result_var, body_label, exit_label)))
        self.ir.append(Instruction('LABEL', body_label))
        for stmt in node.body: self.visit(stmt)
        self.ir.append(Instruction('JUMP', header_label))
        self.ir.append(Instruction('LABEL', exit_label))

    def visit_Attribute(self, node):
        source_var, attr_name = node.value.id, node.attr
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_ATTR', (target_temp, source_var, attr_name)))
        return target_temp

    def visit_Subscript(self, node):
        array_name, index_var = node.value.id, self.visit(node.slice)
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_ELEMENT', (target_temp, array_name, index_var)))
        return target_temp

    def visit_Constant(self, node):
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_CONST', (target_temp, node.value)))
        return target_temp

    def visit_Name(self, node):
        return node.id

    def visit_Return(self, node):
        value_source = self.visit(node.value) if node.value else None
        self.ir.append(Instruction('RETURN', value_source))


class CobraLlvmIrGenerator:
    """Translates Cobra IR into LLVM IR."""
    def __init__(self, cobra_ir, arg_names, arg_types_info, func_name, module):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        self.arg_types_info = arg_types_info
        self.func_name = func_name
        self.module = module  # Use the shared module
        self.symbol_table = {}
        self.type_map = {}
        self.double_ptr_type = ir.DoubleType().as_pointer()
        self._infer_types()

        func_type = ir.FunctionType(self.return_type, self.arg_types)
        # Check if the function already exists in the module before creating it
        if not any(f.name == f"cobra_{func_name}" for f in self.module.functions):
             self.function = ir.Function(self.module, func_type, name=f"cobra_{func_name}")
        else:
             self.function = self.module.get_global(f"cobra_{func_name}")

        i = 0
        for name, type_info in zip(self.arg_names, self.arg_types_info):
            if type_info == 'CobraArray':
                self.function.args[i].name, self.function.args[i+1].name = f"{name}_data", f"{name}_size"
                i += 2
            else:
                self.function.args[i].name = name
                i += 1
        self.entry_block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.entry_block)
        self.llvm_labels = {}

    def _infer_types(self):
        var_types = {name: type_info for name, type_info in zip(self.arg_names, self.arg_types_info)}
        return_var = next((arg for opcode, arg in reversed(self.cobra_ir) if opcode == 'RETURN'), None)
        
        for opcode, arg in self.cobra_ir:
            if opcode == 'STORE': var_types[arg[0]] = var_types.get(arg[1], 'int')
            elif opcode == 'LOAD_CONST': var_types[arg[0]] = 'float' if isinstance(arg[1], float) else 'int'
            elif opcode == 'COMPARE': var_types[arg[1]] = 'bool'
            elif opcode in ('ADD', 'SUB', 'MUL'):
                is_float = any(var_types.get(op, 'int') == 'float' for op in arg[1:])
                var_types[arg[0]] = 'float' if is_float else 'int'
            elif opcode == 'LOAD_ATTR': var_types[arg[0]] = 'int' # arr.size
            elif opcode == 'LOAD_ELEMENT': var_types[arg[0]] = 'float'
            elif opcode == 'CALL':
                 # This needs a proper signature lookup in a real compiler
                 var_types[arg[0]] = 'int'

        self.return_type_str = var_types.get(return_var, 'int') if return_var else 'void'
        self.arg_types = []
        for type_info in self.arg_types_info:
            if type_info == 'CobraArray': self.arg_types.extend([self.double_ptr_type, ir.IntType(64)])
            elif type_info == 'float': self.arg_types.append(ir.DoubleType())
            else: self.arg_types.append(ir.IntType(64))

        type_str_map = {'int': ir.IntType(64), 'float': ir.DoubleType(), 'bool': ir.IntType(1), 'void': ir.VoidType()}
        self.return_type = type_str_map.get(self.return_type_str, ir.IntType(64))
        for name, type_str in var_types.items():
            if type_str != 'CobraArray': self.type_map[name] = type_str_map.get(type_str)

    def _get_value(self, name):
        """
        Retrieves a value from the symbol table. If the symbol is a pointer
        to a scalar value (e.g. an alloca), it loads the value.
        Raw array data pointers (ending in .data) are returned as-is.
        """
        if name is None: return None
        val = self.symbol_table.get(name)
        if val is None: raise NameError(f"Variable '{name}' not found.")
        
        # If the symbol is a pointer AND not a special raw data pointer, load it.
        if isinstance(val.type, ir.PointerType) and not name.endswith('.data'):
            return self.builder.load(val, name=f"{name}.val")
            
        return val

    def generate(self):
        for opcode, arg in self.cobra_ir:
            if opcode == 'LABEL': self.llvm_labels[arg] = self.function.append_basic_block(name=arg)

        arg_idx = 0
        for name, type_info in zip(self.arg_names, self.arg_types_info):
            if type_info == 'CobraArray':
                self.symbol_table[f"{name}.data"] = self.function.args[arg_idx]
                self.symbol_table[f"{name}.size"] = self.function.args[arg_idx+1]
                arg_idx += 2
            else:
                 # Check if the arg needs to be mutable (has a STORE operation)
                is_mutable = any(opcode == 'STORE' and arg[0] == name for opcode, arg in self.cobra_ir)
                if is_mutable:
                    # Allocate space for mutable args and store the initial value
                    alloc = self.builder.alloca(self.type_map[name], name=name)
                    self.builder.store(self.function.args[arg_idx], alloc)
                    self.symbol_table[name] = alloc
                else:
                    self.symbol_table[name] = self.function.args[arg_idx]
                arg_idx += 1
        
        self.builder.position_at_end(self.entry_block)
        
        for opcode, arg in self.cobra_ir:
            if opcode == 'LABEL':
                if not self.builder.block.is_terminated: self.builder.branch(self.llvm_labels[arg])
                self.builder.position_at_end(self.llvm_labels[arg])
            elif opcode == 'LOAD_CONST':
                target, value = arg
                self.symbol_table[target] = ir.Constant(self.type_map[target], value)
            elif opcode == 'STORE':
                target, source = arg
                if target not in self.symbol_table:
                     self.symbol_table[target] = self.builder.alloca(self.type_map.get(target, ir.IntType(64)), name=target)
                self.builder.store(self._get_value(source), self.symbol_table[target])
            elif opcode == 'LOAD_ATTR':
                target, source_obj, attr = arg
                if attr == 'size': self.symbol_table[target] = self.symbol_table[f"{source_obj}.size"]
                else: raise NotImplementedError(f"Attribute '{attr}' not supported.")
            elif opcode == 'LOAD_ELEMENT':
                target, array_name, index_var = arg
                array_ptr = self.symbol_table[f"{array_name}.data"]
                index_val = self._get_value(index_var)
                element_ptr = self.builder.gep(array_ptr, [index_val], name=f"{array_name}.gep")
                self.symbol_table[target] = self.builder.load(element_ptr, name=target)
            elif opcode == 'STORE_ELEMENT':
                array_name, index_var, value_var = arg
                array_ptr = self.symbol_table[f"{array_name}.data"]
                index_val = self._get_value(index_var)
                value_to_store = self._get_value(value_var)
                element_ptr = self.builder.gep(array_ptr, [index_val], name=f"{array_name}.gep")
                self.builder.store(value_to_store, element_ptr)
            elif opcode in ('ADD', 'SUB', 'MUL'):
                target, left_src, right_src = arg
                left_val, right_val = self._get_value(left_src), self._get_value(right_src)
                op_type = self.type_map[target]
                op_map = {'ADD': self.builder.fadd if isinstance(op_type, ir.DoubleType) else self.builder.add,
                          'SUB': self.builder.fsub if isinstance(op_type, ir.DoubleType) else self.builder.sub,
                          'MUL': self.builder.fmul if isinstance(op_type, ir.DoubleType) else self.builder.mul}
                self.symbol_table[target] = op_map[opcode](left_val, right_val, name=target)
            elif opcode == 'COMPARE':
                op, target, left_src, right_src = arg
                left_val, right_val = self._get_value(left_src), self._get_value(right_src)
                op_map = {'GT': '>', 'LT': '<'}
                if isinstance(left_val.type, ir.DoubleType):
                    self.symbol_table[target] = self.builder.fcmp_ordered(op_map[op], left_val, right_val, name=target)
                else:
                    self.symbol_table[target] = self.builder.icmp_signed(op_map[op], left_val, right_val, name=target)
            elif opcode == 'CALL':
                target, func_name, arg_sources = arg
                target_func = self.module.get_global(f"cobra_{func_name}")
                if target_func is None:
                    raise NameError(f"JIT function '{func_name}' not found in current module.")
                call_args = [self._get_value(s) for s in arg_sources]
                self.symbol_table[target] = self.builder.call(target_func, call_args, name=target)
            elif opcode == 'JUMP_IF_TRUE':
                cond_var, then_label, else_label = arg
                self.builder.cbranch(self._get_value(cond_var), self.llvm_labels[then_label], self.llvm_labels[else_label])
            elif opcode == 'JUMP':
                if not self.builder.block.is_terminated: self.builder.branch(self.llvm_labels[arg])
            elif opcode == 'RETURN':
                if not self.builder.block.is_terminated:
                    return_val = self._get_value(arg)
                    if return_val: self.builder.ret(return_val)
                    else: self.builder.ret_void()

        if not self.function.blocks[-1].is_terminated:
            self.builder.position_at_end(self.function.blocks[-1])
            if isinstance(self.return_type, ir.VoidType): self.builder.ret_void()
            else: self.builder.unreachable()

        return self.function

def _compile_function(py_func, arg_types_info, module):
    """Helper to compile a single Python function into a shared LLVM module."""
    func_name = py_func.__name__

    # Check if this specific specialization is already in the module
    if any(f.name == f"cobra_{func_name}" for f in module.functions):
        return module.get_global(f"cobra_{func_name}")

    source_code = inspect.getsource(py_func)
    tree = ast.parse(source_code)
    arg_names = inspect.getfullargspec(py_func).args

    ir_generator = CobraIrGenerator()
    ir_generator.visit(tree)
    cobra_ir = ir_generator.ir
    
    llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names, arg_types_info, func_name, module)
    return llvm_ir_generator.generate()

def jit(func):
    """
    A decorator that performs type-specialized JIT compilation via LLVM.
    """
    _JIT_REGISTRY[func.__name__] = func
    func._jit_cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_types_info = tuple('CobraArray' if isinstance(arg, CobraArray) else 
                               'float' if isinstance(arg, float) else 
                               'int' for arg in args)

        if arg_types_info in func._jit_cache:
            cfunc, engine = func._jit_cache[arg_types_info]
        else:
            main_func_name = func.__name__
            print(f"[COBRA JIT] Compiling new specialization for '{main_func_name}' with types {arg_types_info}...")
            
            # --- Unified Compilation Session ---
            module = ir.Module(name=f"cobra_module_{main_func_name}")
            
            # 1. Discover dependency graph
            dependencies = set()
            to_process = [func]
            processed = set()
            while to_process:
                current_func = to_process.pop(0)
                if current_func in processed: continue
                processed.add(current_func)
                
                source_code = inspect.getsource(current_func)
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        callee_name = node.func.id
                        if callee_name in _JIT_REGISTRY:
                            callee_func = _JIT_REGISTRY[callee_name]
                            dependencies.add(callee_func)
                            to_process.append(callee_func)
            
            # 2. Compile dependencies into the shared module
            for dep_func in dependencies:
                # This is a simplification; assumes dependencies have same arg types
                _compile_function(dep_func, arg_types_info, module)

            # 3. Compile the main function into the shared module
            main_llvm_func = _compile_function(func, arg_types_info, module)
            
            print("\n[COBRA JIT] Generated LLVM IR:")
            print(str(module))

            # 4. Compile the entire module
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
            target_machine = binding.Target.from_default_triple().create_target_machine()
            
            llvm_module = binding.parse_assembly(str(module))
            llvm_module.verify()
            
            engine = binding.create_mcjit_compiler(llvm_module, target_machine)
            engine.finalize_object()
            
            func_ptr = engine.get_function_address(main_llvm_func.name)

            cfunc_arg_types = []
            for t_info in arg_types_info:
                if t_info == 'CobraArray': cfunc_arg_types.extend([ctypes.c_void_p, ctypes.c_int64])
                elif t_info == 'float': cfunc_arg_types.append(ctypes.c_double)
                else: cfunc_arg_types.append(ctypes.c_int64)

            cfunc_return_type = None
            if isinstance(main_llvm_func.return_value.type, ir.IntType): cfunc_return_type = ctypes.c_int64
            elif isinstance(main_llvm_func.return_value.type, ir.DoubleType): cfunc_return_type = ctypes.c_double

            cfunc = ctypes.CFUNCTYPE(cfunc_return_type, *cfunc_arg_types)(func_ptr)
            
            func._jit_cache[arg_types_info] = (cfunc, engine)
            print("[COBRA JIT] Compilation to native code complete.")

        native_args = []
        for arg in args:
            if isinstance(arg, CobraArray): native_args.extend([arg._data_ptr, arg.size])
            else: native_args.append(arg)

        print(f"\n[COBRA JIT] Executing native code for function: '{func.__name__}' with types {arg_types_info}")
        result = cfunc(*args)
        print("[COBRA JIT] Successfully executed native code.")
        return result
    
    return wrapper

