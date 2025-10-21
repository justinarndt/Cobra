# cobra/compiler/expr.py
#
# This file defines the node classes for building the lazy evaluation
# expression tree. When a user performs an operation like `df['a'] + df['b']`,
# instead of computing the result immediately, the system creates an instance
# of `BinaryOpNode`. The entire chain of operations thus forms a tree
# of these nodes, which can be analyzed and compiled by the JIT.

class ExprNode:
    """Base class for all expression tree nodes."""
    pass

class BinaryOpNode(ExprNode):
    """Represents a binary operation (e.g., +, *, /)."""
    def __init__(self, op: str, left: ExprNode, right: ExprNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

class UnaryOpNode(ExprNode):
    """Represents a unary operation (e.g., neg, log, exp)."""
    def __init__(self, op: str, operand: ExprNode):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"{self.op}({self.operand})"

class ColumnNode(ExprNode):
    """Represents a reference to a column in a CobraFrame."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Column({self.name})"

class LiteralNode(ExprNode):
    """Represents a literal value (e.g., 2.0, 5)."""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)