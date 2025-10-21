# cobra/compiler/jit.py
#
# This file implements the Python frontend of the JIT compiler. Its main
# responsibilities are:
# 1. The user-facing `@cobra.jit` decorator that intercepts function calls.
# 2. An AST Visitor (`JitAstVisitor`) that analyzes the decorated function's
#    source code to find CobraFrame assignments.
# 3. An expression tree traverser (`_traverse_expr_tree`) that converts a
#    CobraFrame computation into a fused C-like kernel body string.
# 4. A generator that emits the final MLIR text for the `cobra.fused_kernel`
#    operation, which is then sent to the C++ backend for compilation.

import inspect
import ast
from .expr import ExprNode, BinaryOpNode, UnaryOpNode, ColumnNode, LiteralNode

# A real implementation would get this from the C++ runtime binding.
# from ..runtime import manager

# ============================================================================
# Step 1: Expression Tree Traversal
# ============================================================================

def _traverse_expr_tree(node: ExprNode, input_map: dict) -> str:
    """
    Recursively traverses an expression tree and generates a C-like string
    for the fused kernel's body, replacing column names with generic
    input/output variable names like `in0`, `in1`, etc.

    Args:
        node: The current node of the expression tree to traverse.
        input_map: A dictionary mapping column names to generic names (e.g., {'a': 'in0'}).

    Returns:
        A string representing the computation (e.g., "((in0 + in1) * 2.0)").
    """
    if isinstance(node, BinaryOpNode):
        left_str = _traverse_expr_tree(node.left, input_map)
        right_str = _traverse_expr_tree(node.right, input_map)
        return f"({left_str} {node.op} {right_str})"

    elif isinstance(node, UnaryOpNode):
        operand_str = _traverse_expr_tree(node.operand, input_map)
        # A full implementation would map op to function names like 'exp', 'log'
        return f"{node.op}({operand_str})"

    elif isinstance(node, ColumnNode):
        # Look up the column's placeholder name (e.g., 'a') and return
        # its corresponding generic kernel argument name (e.g., 'in0').
        return input_map.get(node.name, "UNKNOWN_COLUMN")

    elif isinstance(node, LiteralNode):
        return str(node.value)

    else:
        raise TypeError(f"Unknown expression tree node type: {type(node)}")


# ============================================================================
# Step 2: Python AST Analysis
# ============================================================================

class JitAstVisitor(ast.NodeVisitor):
    """
    Visits the AST of a JIT-decorated function to find assignments to
    CobraFrame columns and extracts the associated expression tree.
    """
    def __init__(self, frame_name: str, frame_obj):
        self.frame_name = frame_name
        self.frame_obj = frame_obj
        self.mlir_ops = []

    def visit_Assign(self, node: ast.Assign):
        """
        Processes an assignment statement like `df['c'] = df['a'] + df['b']`.
        """
        # We only care about single-target assignments
        if len(node.targets) != 1:
            return

        target = node.targets[0]

        # Check if the assignment is to a column of our target frame,
        # e.g., `df['c']` where `target.value.id` is `df`.
        if (isinstance(target, ast.Subscript) and
            isinstance(target.value, ast.Name) and
            target.value.id == self.frame_name):

            # In a real compiler, we would need to evaluate the Python code
            # for the right-hand side `node.value` to get the actual
            # expression tree object. This is a complex process involving
            # inspecting the function's frame and locals.
            # Here, we will simulate it by retrieving the pre-built tree
            # from the frame object itself.

            output_col_name = target.slice.value
            expr_tree = self.frame_obj._data.get(output_col_name)

            if not isinstance(expr_tree, ExprNode):
                # This assignment was not a lazy expression, so we can't JIT it.
                return

            # We found a JIT-able expression. Now, generate MLIR for it.
            self._generate_fused_kernel(output_col_name, expr_tree)

    def _generate_fused_kernel(self, output_col_name: str, expr_tree: ExprNode):
        """
        Generates the MLIR for a single `cobra.fused_kernel` operation.
        """
        # Discover all unique input columns by finding all ColumnNodes in the tree.
        input_nodes = set(n for n in ast.walk(expr_tree) if isinstance(n, ColumnNode))
        input_names = sorted([n.name for n in input_nodes]) # Sort for deterministic order

        # Create the mapping from column names to generic kernel arg names.
        input_map = {name: f"in{i}" for i, name in enumerate(input_names)}

        # Traverse the tree to get the final kernel body string.
        kernel_body = _traverse_expr_tree(expr_tree, input_map)
        
        # In our kernel, the output is always `out0`.
        final_kernel_str = f"out0 = {kernel_body}"

        # Get the MLIR types for the function signature (e.g., tensor<*xf32>)
        # This would come from inspecting the actual CobraArray dtypes.
        mlir_types = ["tensor<*xf32>"] * (len(input_names) + 1) # +1 for the output
        
        # Build the MLIR region's argument list (e.g., %in0: tensor<*xf32>, ...)
        arg_defs = ", ".join([f"%in{i}: {t}" for i, t in enumerate(mlir_types[:-1])])
        arg_defs += f", %out0: {mlir_types[-1]}"
        
        # Build the argument list for the operation itself
        input_args = ", ".join([f"%{name}" for name in input_names])
        output_arg = f"%{output_col_name}"

        mlir_op = f"""
    %result = cobra.fused_kernel(%{input_args} -> %{output_arg}) {{
    ^bb0({arg_defs}):
      // Fused kernel body generated from Python expression:
      // out0 = {kernel_body.replace('"', '""')}
      cobra.yield
    }} : ({", ".join(mlir_types[:-1])}) -> ({mlir_types[-1]})
"""
        self.mlir_ops.append(mlir_op)

# ============================================================================
# Step 3: The JIT Decorator
# ============================================================================

def jit(func):
    """
    A decorator that JIT-compiles functions containing CobraFrame operations.
    """
    def wrapper(*args, **kwargs):
        # 1. Get the function's source code and parse it into an AST.
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
        except (TypeError, OSError):
            print("Cobra JIT Error: Could not get source code for function.")
            return func(*args, **kwargs) # Fallback to pure Python

        # 2. Find the CobraFrame object among the function's arguments.
        frame_obj = None
        frame_name = None
        func_sig = inspect.signature(func)
        for i, param in enumerate(func_sig.parameters.values()):
            if i < len(args) and hasattr(args[i], '_data'): # A simple check for CobraFrame
                frame_obj = args[i]
                frame_name = param.name
                break
        
        if not frame_obj:
            print("Cobra JIT Warning: No CobraFrame argument found. Running in Python mode.")
            return func(*args, **kwargs)

        # 3. First, execute the original function in Python. This has the
        #    side-effect of building the expression trees inside the CobraFrame.
        result = func(*args, **kwargs)

        # 4. Now, analyze the function's AST to find assignments and generate MLIR.
        visitor = JitAstVisitor(frame_name, frame_obj)
        visitor.visit(tree)

        # 5. If any compilable operations were found, send them to the backend.
        if visitor.mlir_ops:
            print("="*20 + " Cobra JIT Compilation " + "="*20)
            print(f"Found {len(visitor.mlir_ops)} JIT-able operation(s) in '{func.__name__}'.")
            
            full_mlir_module = "module {\n" + "\n".join(visitor.mlir_ops) + "\n}"
            
            print("\n--- Generated MLIR Module ---")
            print(full_mlir_module)
            print("="*64)
            
            # This is where we would send the MLIR string to the C++ backend.
            # manager.compile_and_run(full_mlir_module, frame_obj)
            print("\n[Cobra JIT]>>> Sending to C++ backend for compilation and execution (simulated).")
        else:
            print(f"Cobra JIT: No compilable operations found in '{func.__name__}'.")

        return result
    return wrapper