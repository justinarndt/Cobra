# cobra/compiler/jit.py

import functools
import inspect
import ast
import ctypes
from collections import namedtuple

from llvmlite import ir, binding

# We must now import CobraArray to check for its type
from cobra.runtime import CobraArray

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
        # Handle array assignment: arr[i] = value
        if isinstance(node.targets[0], ast.Subscript):
            array_name = node.targets[0].value.id
            index_var = self.visit(node.targets[0].slice)
            value_source = self.visit(node.value)
            self.ir.append(Instruction('STORE_ELEMENT', (array_name, index_var, value_source)))
        # Handle regular variable assignment: x = value
        else:
            value_source = self.visit(node.value)
            target = node.targets[0].id
            self.ir.append(Instruction('STORE', (target, value_source)))

    def visit_AugAssign(self, node):
        """Handles augmented assignments (e.g., total += i)."""
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
        """Handles comparison operations."""
        left_source = self.visit(node.left)
        right_source = self.visit(node.comparators[0])
        target_temp = self.fresh_temp()
        
        op_map = {ast.Gt: 'GT', ast.Lt: 'LT'}
        opcode = op_map.get(type(node.ops[0]))
        if opcode:
            self.ir.append(Instruction('COMPARE', (opcode, target_temp, left_source, right_source)))
            return target_temp
        raise NotImplementedError(f"Comparison {type(node.ops[0]).__name__} not supported")

    def visit_If(self, node):
        """Handles if statements."""
        test_result_var = self.visit(node.test)
        
        then_label = self.fresh_label()
        else_label = self.fresh_label()
        end_label = self.fresh_label() if node.orelse else else_label

        jump_else_label = else_label if node.orelse else end_label

        self.ir.append(Instruction('JUMP_IF_TRUE', (test_result_var, then_label, jump_else_label)))
        
        self.ir.append(Instruction('LABEL', then_label))
        for stmt in node.body:
            self.visit(stmt)
        
        if node.orelse:
            self.ir.append(Instruction('JUMP', end_label))
        
        if node.orelse:
            self.ir.append(Instruction('LABEL', else_label))
            for stmt in node.orelse:
                self.visit(stmt)
        
        self.ir.append(Instruction('LABEL', end_label))
        
    def visit_While(self, node):
        """Handles while loops."""
        header_label = self.fresh_label()
        body_label = self.fresh_label()
        exit_label = self.fresh_label()

        self.ir.append(Instruction('JUMP', header_label))
        self.ir.append(Instruction('LABEL', header_label))

        test_result_var = self.visit(node.test)
        self.ir.append(Instruction('JUMP_IF_TRUE', (test_result_var, body_label, exit_label)))

        self.ir.append(Instruction('LABEL', body_label))
        for stmt in node.body:
            self.visit(stmt)
        self.ir.append(Instruction('JUMP', header_label))

        self.ir.append(Instruction('LABEL', exit_label))

    def visit_Attribute(self, node):
        """Handles attribute access (e.g., arr.size)."""
        source_var = node.value.id
        attr_name = node.attr
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_ATTR', (target_temp, source_var, attr_name)))
        return target_temp

    def visit_Subscript(self, node):
        """Handles subscript access (e.g., arr[i])."""
        # This visitor handles loading an element, e.g., x = arr[i]
        array_name = node.value.id
        index_var = self.visit(node.slice)
        target_temp = self.fresh_temp()
        self.ir.append(Instruction('LOAD_ELEMENT', (target_temp, array_name, index_var)))
        return target_temp

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
        value_source = self.visit(node.value) if node.value else None
        self.ir.append(Instruction('RETURN', value_source))


class CobraLlvmIrGenerator:
    """Translates Cobra IR into LLVM IR."""
    def __init__(self, cobra_ir, arg_names, arg_types_info, func_name):
        self.cobra_ir = cobra_ir
        self.arg_names = arg_names
        self.arg_types_info = arg_types_info # Info about which args are CobraArrays
        self.func_name = func_name
        self.module = ir.Module(name=f"cobra_module_{func_name}")
        self.symbol_table = {}
        self.type_map = {}
        
        # Define CobraArray struct: { double* data, i64 size }
        self.double_ptr_type = ir.DoubleType().as_pointer()
        
        self._infer_types()

        func_type = ir.FunctionType(self.return_type, self.arg_types)
        self.function = ir.Function(self.module, func_type, name=f"cobra_{func_name}")
        
        # Unpack CobraArray args and name all arguments
        i = 0
        for name, type_info in zip(self.arg_names, self.arg_types_info):
            if type_info == 'CobraArray':
                self.function.args[i].name = f"{name}_data"
                self.function.args[i+1].name = f"{name}_size"
                i += 2
            else:
                self.function.args[i].name = name
                i += 1

        self.entry_block = self.function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(self.entry_block)
        
        self.llvm_labels = {}

    def _infer_types(self):
        """A simple type inference pass over the Cobra IR."""
        var_types = {}
        # Infer types from function arguments first
        for name, type_info in zip(self.arg_names, self.arg_types_info):
            var_types[name] = type_info

        return_var = None
        for opcode, arg in reversed(self.cobra_ir):
            if opcode == 'RETURN':
                return_var = arg
                break
        
        for opcode, arg in self.cobra_ir:
            if opcode == 'STORE':
                 var_types[arg[0]] = var_types.get(arg[1], 'int')
            elif opcode == 'LOAD_CONST':
                target, value = arg
                var_types[target] = 'float' if isinstance(value, float) else 'int'
            elif opcode == 'COMPARE':
                 var_types[arg[1]] = 'bool'
            elif opcode in ('ADD', 'SUB', 'MUL'):
                operands = arg[1:]
                is_float = any(var_types.get(op, 'int') == 'float' for op in operands)
                var_types[arg[0]] = 'float' if is_float else 'int'
            elif opcode == 'LOAD_ATTR':
                 # For now, only arr.size is supported, which is an int
                 var_types[arg[0]] = 'int'
            elif opcode == 'LOAD_ELEMENT':
                 # Array elements are floats (doubles)
                 var_types[arg[0]] = 'float'
        
        self.return_type_str = var_types.get(return_var, 'int') if return_var else 'void'

        # Build LLVM arg types based on type info (unpacking CobraArray)
        self.arg_types = []
        for type_info in self.arg_types_info:
            if type_info == 'CobraArray':
                self.arg_types.extend([self.double_ptr_type, ir.IntType(64)])
            elif type_info == 'float':
                self.arg_types.append(ir.DoubleType())
            else: # int
                self.arg_types.append(ir.IntType(64))

        type_str_map = {'int': ir.IntType(64), 'float': ir.DoubleType(), 'bool': ir.IntType(1), 'void': ir.VoidType()}
        self.return_type = type_str_map.get(self.return_type_str, ir.IntType(64))
        
        for name, type_str in var_types.items():
            if type_str not in ['CobraArray']:
                self.type_map[name] = type_str_map.get(type_str)
    
    def _get_value(self, name):
        """Loads a value from the symbol table, handling pointers."""
        if name is None: return None
        val = self.symbol_table.get(name)
        if val is None:
             raise NameError(f"Variable '{name}' not found in symbol table.")
        if isinstance(val.type, ir.PointerType) and val.type.pointee != self.double_ptr_type:
            return self.builder.load(val, name=f"{name}.val")
        return val

    def generate(self):
        """Generates the LLVM IR."""
        for opcode, arg in self.cobra_ir:
            if opcode == 'LABEL':
                self.llvm_labels[arg] = self.function.append_basic_block(name=arg)

        # Handle arguments, creating local symbols for CobraArray parts
        arg_idx = 0
        for name, type_info in zip(self.arg_names, self.arg_types_info):
            if type_info == 'CobraArray':
                self.symbol_table[f"{name}.data"] = self.function.args[arg_idx]
                self.symbol_table[f"{name}.size"] = self.function.args[arg_idx+1]
                arg_idx += 2
            else:
                self.symbol_table[name] = self.function.args[arg_idx]
                arg_idx += 1

        # Create allocas for all mutable variables in the entry block.
        mutable_vars = {arg[0] for opcode, arg in self.cobra_ir if opcode == 'STORE'}
        for var_name in mutable_vars:
            var_type = self.type_map.get(var_name, ir.IntType(64))
            self.symbol_table[var_name] = self.builder.alloca(var_type, name=var_name)
        
        self.builder.position_at_end(self.entry_block)

        for opcode, arg in self.cobra_ir:
            if opcode == 'LABEL':
                if not self.builder.block.is_terminated:
                    self.builder.branch(self.llvm_labels[arg])
                self.builder.position_at_end(self.llvm_labels[arg])
                
            elif opcode == 'LOAD_CONST':
                target, value = arg
                var_type = self.type_map[target]
                self.symbol_table[target] = ir.Constant(var_type, value)
            
            elif opcode == 'STORE':
                target, source = arg
                source_val = self._get_value(source)
                self.builder.store(source_val, self.symbol_table[target])

            elif opcode == 'LOAD_ATTR':
                target, source_obj, attr = arg
                if attr == 'size':
                    self.symbol_table[target] = self.symbol_table[f"{source_obj}.size"]
                else:
                    raise NotImplementedError(f"Attribute '{attr}' not supported.")

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
                left_val = self._get_value(left_src)
                right_val = self._get_value(right_src)
                op_type = self.type_map[target]

                op_map = {
                    'ADD': self.builder.fadd if isinstance(op_type, ir.DoubleType) else self.builder.add,
                    'SUB': self.builder.fsub if isinstance(op_type, ir.DoubleType) else self.builder.sub,
                    'MUL': self.builder.fmul if isinstance(op_type, ir.DoubleType) else self.builder.mul,
                }
                self.symbol_table[target] = op_map[opcode](left_val, right_val, name=target)

            elif opcode == 'COMPARE':
                op, target, left_src, right_src = arg
                left_val = self._get_value(left_src)
                right_val = self._get_value(right_src)
                op_map = {'GT': '>', 'LT': '<'}
                
                if isinstance(left_val.type, ir.DoubleType):
                    self.symbol_table[target] = self.builder.fcmp_ordered(op_map[op], left_val, right_val, name=target)
                else:
                    self.symbol_table[target] = self.builder.icmp_signed(op_map[op], left_val, right_val, name=target)

            elif opcode == 'JUMP_IF_TRUE':
                cond_var, then_label, else_label = arg
                cond_val = self._get_value(cond_var)
                self.builder.cbranch(cond_val, self.llvm_labels[then_label], self.llvm_labels[else_label])
                
            elif opcode == 'JUMP':
                if not self.builder.block.is_terminated:
                    self.builder.branch(self.llvm_labels[arg])

            elif opcode == 'RETURN':
                if not self.builder.block.is_terminated:
                    return_val = self._get_value(arg)
                    if return_val:
                        self.builder.ret(return_val)
                    else:
                        self.builder.ret_void()

        last_block = self.function.blocks[-1]
        if not last_block.is_terminated:
            self.builder.position_at_end(last_block)
            if isinstance(self.return_type, ir.VoidType):
                self.builder.ret_void()
            else:
                self.builder.unreachable()

        return str(self.module)


def jit(func):
    """
    A decorator that performs type-specialized JIT compilation via LLVM.
    """
    func_cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Determine argument types to create a cache key
        arg_types_info = []
        for arg in args:
            if isinstance(arg, CobraArray):
                arg_types_info.append('CobraArray')
            elif isinstance(arg, float):
                arg_types_info.append('float')
            elif isinstance(arg, int):
                arg_types_info.append('int')
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")
        
        cache_key = tuple(arg_types_info)

        # 2. Check if a compiled version for these types already exists
        if cache_key in func_cache:
            cfunc, engine = func_cache[cache_key]
        else:
            # 3. If not, compile a new version
            func_name = func.__name__
            print(f"[COBRA JIT] Compiling new specialization for '{func_name}' with types {cache_key}...")
            
            source_code = inspect.getsource(func)
            tree = ast.parse(source_code)
            arg_names = inspect.getfullargspec(func).args

            ir_generator = CobraIrGenerator()
            ir_generator.visit(tree)
            cobra_ir = ir_generator.ir
            
            llvm_ir_generator = CobraLlvmIrGenerator(cobra_ir, arg_names, arg_types_info, func_name)
            llvm_ir_str = llvm_ir_generator.generate()
            
            print("\n[COBRA JIT] Generated LLVM IR:")
            print(llvm_ir_str)
            
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()

            llvm_module = binding.parse_assembly(llvm_ir_str)
            llvm_module.verify()

            target_machine = binding.Target.from_default_triple().create_target_machine()
            engine = binding.create_mcjit_compiler(llvm_module, target_machine)
            engine.finalize_object()
            
            func_ptr = engine.get_function_address(f"cobra_{func_name}")

            # Define ctypes signature based on unpacked types
            cfunc_arg_types = []
            for t_info in arg_types_info:
                if t_info == 'CobraArray':
                    cfunc_arg_types.extend([ctypes.c_void_p, ctypes.c_int64])
                elif t_info == 'float':
                    cfunc_arg_types.append(ctypes.c_double)
                else: # int
                    cfunc_arg_types.append(ctypes.c_int64)

            ret_type = llvm_ir_generator.return_type
            cfunc_return_type = None
            if isinstance(ret_type, ir.IntType) and ret_type.width == 64:
                cfunc_return_type = ctypes.c_int64
            elif isinstance(ret_type, ir.DoubleType):
                cfunc_return_type = ctypes.c_double

            cfunc = ctypes.CFUNCTYPE(cfunc_return_type, *cfunc_arg_types)(func_ptr)
            
            # Cache the compiled function and its engine
            func_cache[cache_key] = (cfunc, engine)
            print("[COBRA JIT] Compilation to native code complete.")

        # 4. Prepare arguments for the native call
        native_args = []
        for arg in args:
            if isinstance(arg, CobraArray):
                native_args.extend([arg._data_ptr, arg.size])
            else:
                native_args.append(arg)

        # 5. Execute the compiled code
        print(f"\n[COBRA JIT] Executing native code for function: '{func.__name__}' with types {cache_key}")
        result = cfunc(*native_args)
        print("[COBRA JIT] Successfully executed native code.")
        return result
    
    return wrapper

