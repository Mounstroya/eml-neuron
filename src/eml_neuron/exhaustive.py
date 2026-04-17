"""
Beam-pruned numerical search over EML expression trees.

Strategy
--------
At each depth level we maintain a "beam" of the best K candidate nodes
ranked by max absolute error against the target on a fixed evaluation grid.
New depth-d nodes are built as eml(left, right) where left and right are
drawn from the beam of depth d-1 candidates.  This prunes the combinatorial
explosion: instead of O(n²) full enumeration we keep only the most
promising subtrees.

Search space (without pruning):
  depth 0:         2 expressions
  depth 1:         6
  depth 2:        42
  depth 3:     1 806
  depth 4: 3 263 442   ← enumerating is feasible, beam makes it fast

With beam_size=500 each depth generates at most 500×500=250 000 new
candidates before pruning back to 500 — tractable up to depth 7+.

This approach is empirical, not a proof of impossibility.  Reporting
"not found at depth ≤ D with beam K" is a measured bound on difficulty,
not a completeness claim.
"""

from __future__ import annotations
import torch
from .node import EMLNode


# ── helpers ──────────────────────────────────────────────────────────────────

def _eval_safe(node: EMLNode, x: torch.Tensor) -> torch.Tensor | None:
    try:
        with torch.no_grad():
            y = node(x)
        if torch.isfinite(y).all():
            return y
    except Exception:
        pass
    return None


def _score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return (y_pred - y_true).abs().max().item()


# ── public API ────────────────────────────────────────────────────────────────

def beam_search(
    target_fn,
    max_depth: int = 6,
    beam_size: int = 500,
    x_range: tuple[float, float] = (0.5, 4.0),
    n_eval: int = 512,
    exact_threshold: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[EMLNode | None, float, int]:
    """
    Beam-pruned search for the EML expression tree that best approximates
    target_fn on a uniform grid over x_range.

    At each depth, candidate nodes are scored by max absolute error.
    Only the best beam_size survive to become subtrees at the next depth.
    The depth-0 beam always includes both terminals regardless of score.

    Args:
        target_fn: callable torch.Tensor -> torch.Tensor
        max_depth: deepest level to search
        beam_size: number of candidates to keep per depth level
        x_range: evaluation interval (must stay positive for ln)
        n_eval: number of grid points
        exact_threshold: error below which we declare an exact match
        device: torch device
        verbose: print per-depth summary

    Returns:
        (best_node, best_error, depth_found)
        best_node is None only if every candidate produced nan/inf.
    """
    x = torch.linspace(x_range[0], x_range[1], n_eval, device=device)
    y_true = target_fn(x)

    global_best_node: EMLNode | None = None
    global_best_err: float = float("inf")
    global_best_depth: int = -1

    # depth-0 beam: always keep both terminals
    terminals = [EMLNode("one"), EMLNode("x")]
    beam: list[tuple[float, EMLNode]] = []
    for node in terminals:
        y = _eval_safe(node, x)
        if y is not None:
            err = _score(y, y_true)
            beam.append((err, node))
            if err < global_best_err:
                global_best_err = err
                global_best_node = node
                global_best_depth = 0

    beam.sort(key=lambda t: t[0])

    if verbose:
        print(f"  depth 0 | beam={len(beam):>6} | best_err={beam[0][0]:.4e}")

    if global_best_err < exact_threshold:
        return global_best_node, global_best_err, 0

    for depth in range(1, max_depth + 1):
        # generate all eml(left, right) from current beam
        prev_nodes = [node for _, node in beam]
        candidates: list[tuple[float, EMLNode]] = []

        for left in prev_nodes:
            for right in prev_nodes:
                node = EMLNode("eml", left=left, right=right)
                y = _eval_safe(node, x)
                if y is None:
                    continue
                err = _score(y, y_true)
                candidates.append((err, node))

        # also carry forward previous beam (shallower expressions stay eligible)
        candidates.extend(beam)

        # prune to beam_size
        candidates.sort(key=lambda t: t[0])
        beam = candidates[:beam_size]

        if beam:
            best_err_d, best_node_d = beam[0]
            if best_err_d < global_best_err:
                global_best_err = best_err_d
                global_best_node = best_node_d
                global_best_depth = depth

            if verbose:
                print(
                    f"  depth {depth} | beam={len(beam):>6} | "
                    f"new_candidates={len(candidates):>8} | "
                    f"best_err={best_err_d:.4e}"
                )

            if global_best_err < exact_threshold:
                if verbose:
                    print(f"  → exact match at depth {depth}")
                return global_best_node, global_best_err, depth

    return global_best_node, global_best_err, global_best_depth


# ── legacy exact enumeration (kept for depth ≤ 3 oracle use) ─────────────────

def _enumerate(depth: int) -> list[EMLNode]:
    """All EML trees up to depth (exact, no pruning). Feasible for depth ≤ 3."""
    if depth == 0:
        return [EMLNode("one"), EMLNode("x")]
    subtrees = _enumerate(depth - 1)
    nodes = list(subtrees)
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
) -> tuple[EMLNode | None, float]:
    """Exact exhaustive search. Use only for max_depth ≤ 3."""
    x = torch.linspace(x_range[0], x_range[1], n_eval, device=device)
    y_true = target_fn(x)
    best_node, best_err = None, float("inf")
    for node in _enumerate(max_depth):
        y = _eval_safe(node, x)
        if y is None:
            continue
        err = _score(y, y_true)
        if err < best_err:
            best_err, best_node = err, node
    return best_node, best_err
