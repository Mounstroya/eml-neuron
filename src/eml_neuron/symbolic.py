"""
Symbolic verification of snapped EML trees.

Converts an EMLNode tree to a SymPy expression and checks whether
it is symbolically equivalent to a target expression.
"""

import sympy as sp
from .node import EMLNode

_x = sp.Symbol("x")
_EPS = sp.Float("1e-7")


def to_sympy(node: EMLNode) -> sp.Expr:
    """Recursively convert an EMLNode tree to a SymPy expression."""
    if node.kind == "one":
        return sp.Integer(1)
    if node.kind == "x":
        return _x
    left = to_sympy(node.left)
    right = to_sympy(node.right)
    return sp.exp(left) - sp.log(right)


def snap_and_verify(
    soft_node,
    target_expr: sp.Expr,
    simplify: bool = True,
) -> tuple[EMLNode, sp.Expr, bool]:
    """
    Snap a SoftEMLNode to its argmax topology, build its SymPy form,
    and check symbolic equivalence against target_expr.

    Returns (hard_node, sympy_expr, is_exact).
    """
    hard = soft_node.snap()
    expr = to_sympy(hard)

    if simplify:
        expr = sp.simplify(expr)

    diff = sp.simplify(expr - target_expr)
    is_exact = diff == sp.Integer(0)

    return hard, expr, is_exact
