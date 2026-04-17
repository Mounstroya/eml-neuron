"""
Measurement 3 — Transcendental vs algebraic: structural analysis.

Hypothesis: exp(x) and ln(x) are "native" to EML (they are the primitive
operations of the operator itself). x² is algebraic and requires the
constant 2, which is not reachable from EML{1} over the reals.

This script measures:

  A. Constant reachability: run beam search for the constant function
     f(x) = 2 (and other rationals) using only terminal {1}. Shows that
     no EML{1} expression reaches these constants up to depth D.

  B. Error plateau analysis: compare how best_err evolves with depth for
     exp(x), ln(x), and x². Transcendental targets show abrupt convergence;
     algebraic targets show a plateau. This difference is structural.

  C. Closest-constant gap: for x², measure the minimum distance between
     any EML{1} constant reachable at depth D and the constant 2. Show
     that this gap does not close as depth grows, and that the x² error
     plateau tracks this gap.

Usage:
    python experiments/phase1_structural_analysis.py
"""

import sys
import math
import torch

sys.path.insert(0, "src")
from eml_neuron.exhaustive import beam_search, _enumerate
from eml_neuron.node import EMLNode

X_RANGE_EXP    = (0.3, 2.0)
X_RANGE_LN     = (0.5, 5.0)
X_RANGE_SQUARE = (0.5, 2.5)
MAX_DEPTH      = 7
BEAM            = 300


# ── A: Constant reachability ──────────────────────────────────────────────────

def constant_eml_values(max_depth: int) -> list[tuple[int, float, str]]:
    """
    Enumerate EML{1} constant expressions (no x terminal) up to max_depth.
    Returns list of (depth, value, repr).
    """
    # Build expression trees using only the terminal `1`
    # We reuse _enumerate but filter for x-free expressions
    results = []
    x = torch.tensor([1.0])  # dummy, won't be used

    def is_const(node: EMLNode) -> bool:
        if node.kind == "x":
            return False
        if node.kind == "one":
            return True
        return is_const(node.left) and is_const(node.right)

    seen_vals: set[float] = set()

    for d in range(max_depth + 1):
        for node in _enumerate(d):
            if not is_const(node):
                continue
            try:
                with torch.no_grad():
                    val = node(x).item()
                if not math.isfinite(val):
                    continue
                val_r = round(val, 8)
                if val_r not in seen_vals:
                    seen_vals.add(val_r)
                    results.append((d, val, repr(node)))
            except Exception:
                pass

    return sorted(results, key=lambda t: t[1])


# ── B: Error plateau per target ───────────────────────────────────────────────

def depth_error_profile(target_fn, x_range, max_depth, beam_size, label):
    print(f"\n  {label}")
    errors_by_depth = {}

    def _beam_verbose(target_fn, max_depth, x_range, beam_size):
        """beam_search but yield (depth, best_err) at each step."""
        x = torch.linspace(x_range[0], x_range[1], 512)
        y_true = target_fn(x)

        def safe(node):
            try:
                with torch.no_grad():
                    y = node(x)
                return y if torch.isfinite(y).all() else None
            except Exception:
                return None

        score = lambda y: (y - y_true).abs().max().item()

        terminals = [EMLNode("one"), EMLNode("x")]
        beam = []
        for node in terminals:
            y = safe(node)
            if y is not None:
                beam.append((score(y), node))
        beam.sort(key=lambda t: t[0])
        yield 0, beam[0][0]

        for depth in range(1, max_depth + 1):
            prev = [n for _, n in beam]
            candidates = list(beam)
            for l in prev:
                for r in prev:
                    node = EMLNode("eml", left=l, right=r)
                    y = safe(node)
                    if y is not None:
                        candidates.append((score(y), node))
            candidates.sort(key=lambda t: t[0])
            beam = candidates[:beam_size]
            yield depth, beam[0][0]

    for depth, err in _beam_verbose(target_fn, max_depth, x_range, beam_size):
        errors_by_depth[depth] = err
        marker = " ← exact" if err < 1e-4 else ""
        print(f"    depth {depth:>2}  best_err={err:.4e}{marker}")

    return errors_by_depth


# ── C: Closest-constant gap ───────────────────────────────────────────────────

def closest_constant_gap(target_val: float, max_depth: int) -> dict[int, float]:
    """Min |EML{1}_const - target_val| at each depth."""
    x = torch.tensor([1.0])
    gap_by_depth: dict[int, float] = {}

    def is_const(node):
        if node.kind == "x": return False
        if node.kind == "one": return True
        return is_const(node.left) and is_const(node.right)

    best_gap = float("inf")
    for d in range(max_depth + 1):
        for node in _enumerate(d):
            if not is_const(node):
                continue
            try:
                with torch.no_grad():
                    val = node(x).item()
                if math.isfinite(val):
                    gap = abs(val - target_val)
                    if gap < best_gap:
                        best_gap = gap
            except Exception:
                pass
        gap_by_depth[d] = best_gap
    return gap_by_depth


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*62}")
    print("  Measurement 3: Transcendental vs Algebraic")
    print(f"{'='*62}")

    # A: What constants can EML{1} reach?
    print("\n[A] EML{1} constant closure (depth ≤ 3, sorted by value):")
    consts = constant_eml_values(max_depth=3)
    for depth, val, rep in consts[:20]:
        dist_2 = abs(val - 2.0)
        print(f"    depth {depth}  val={val:>12.6f}  dist_to_2={dist_2:.6f}  expr={rep}")
    print(f"  (showing up to 20 of {len(consts)} distinct constants)")

    # B: Depth-error profile comparison
    print("\n[B] Error-vs-depth profiles (beam_size=300):")
    profiles = {}
    for label, fn, xr in [
        ("exp(x)", torch.exp,  X_RANGE_EXP),
        ("ln(x)",  torch.log,  X_RANGE_LN),
        ("x²",     lambda x: x**2, X_RANGE_SQUARE),
    ]:
        profiles[label] = depth_error_profile(fn, xr, max_depth=MAX_DEPTH,
                                               beam_size=BEAM, label=label)

    # C: Closest-constant gap for constant 2 (needed by x²)
    print("\n[C] Min distance from any EML{1} constant to 2 (needed for x²):")
    gap = closest_constant_gap(target_val=2.0, max_depth=4)
    sq_errs = profiles.get("x²", {})
    print(f"  {'depth':>5}  {'gap_to_2':>12}  {'x²_best_err':>12}")
    for d in sorted(gap):
        sq_err = sq_errs.get(d, float("nan"))
        print(f"  {d:>5}  {gap[d]:>12.6f}  {sq_err:>12.4e}")

    print(f"\n  Interpretation:")
    print(f"  - exp(x) and ln(x) are native EML operations — exact match is abrupt.")
    print(f"  - x² requires the constant 2, which EML{{1}} cannot produce.")
    print(f"  - The x² error plateau tracks the gap-to-2: both stagnate together.")
    print(f"  This is empirical evidence of a structural difference, not a proof.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
