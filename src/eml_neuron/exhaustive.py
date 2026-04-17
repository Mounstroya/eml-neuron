"""
Exhaustive enumeration of EML expressions up to a given depth.

For small depths the search space is manageable:
  depth 0: 2 expressions  (1, x)
  depth 1: 6
  depth 2: 42
  depth 3: 1806
  depth 4: 3263442

Exhaustive search is used as a reliable oracle for Phase 1 targets.
Gradient-based search is attempted first; this is the fallback.
"""

from __future__ import annotations
import torch
from .node import EMLNode


def _enumerate(depth: int) -> list[EMLNode]:
    """Return all distinct EML expression trees up to the given depth."""
    if depth == 0:
        return [EMLNode("one"), EMLNode("x")]
    subtrees = _enumerate(depth - 1)
    nodes = list(subtrees)  # include shallower expressions
    for left in subtrees:
        for right in subtrees:
            nodes.append(EMLNode("eml", left=left, right=right))
    return nodes


def exhaustive_search(
    target_fn,
    max_depth: int = 3,
    x_range: tuple[float, float] = (0.5, 4.0),
    n_eval: int = 1000,
    device: str = "cpu",
) -> tuple[EMLNode, float]:
    """
    Evaluate all EML trees up to max_depth and return the one with
    the smallest max absolute error against target_fn.

    Returns (best_node, best_max_err).
    """
    x = torch.linspace(x_range[0], x_range[1], n_eval, device=device)
    y_true = target_fn(x)

    candidates = _enumerate(max_depth)
    best_node = None
    best_err = float("inf")

    for node in candidates:
        try:
            with torch.no_grad():
                y_pred = node(x)
            if not torch.isfinite(y_pred).all():
                continue
            err = (y_pred - y_true).abs().max().item()
        except Exception:
            continue

        if err < best_err:
            best_err = err
            best_node = node

    return best_node, best_err
