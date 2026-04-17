"""
Phase 1 experiment: single EML node proof of concept.

Trains a SoftEMLNode to recover exp(x), ln(x), or x^2,
then snaps to the hard topology and reports symbolic match.

Usage:
    python experiments/phase1_single_node.py --target exp
    python experiments/phase1_single_node.py --target ln
    python experiments/phase1_single_node.py --target square
"""

import sys
import argparse
import math
import torch
import sympy as sp

sys.path.insert(0, "src")
from eml_neuron.train import train_single_node
from eml_neuron.symbolic import snap_and_verify, to_sympy

_x = sp.Symbol("x")

TARGETS = {
    # exp(x) = eml(x, 1)  — minimal depth is 1
    "exp": {
        "fn": torch.exp,
        "sympy": sp.exp(_x),
        "x_range": (0.3, 2.0),
        "depth": 1,
    },
    # ln(x) — exact EML depth unknown, start at 3 and let curriculum find it
    "ln": {
        "fn": torch.log,
        "sympy": sp.log(_x),
        "x_range": (0.5, 4.0),
        "depth": 3,
    },
    # x^2 = exp(2*ln(x)) — deeper composition required
    "square": {
        "fn": lambda x: x ** 2,
        "sympy": _x ** 2,
        "x_range": (0.5, 2.5),
        "depth": 4,
    },
}


def evaluate_snap(hard_node, target_fn, x_range, n=1000):
    """Compute max absolute error of the snapped hard node on fresh data."""
    x = torch.linspace(x_range[0], x_range[1], n)
    with torch.no_grad():
        y_pred = hard_node(x)
        y_true = target_fn(x)
        max_err = (y_pred - y_true).abs().max().item()
    return max_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), default="exp")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tau_start", type=float, default=3.0)
    parser.add_argument("--tau_end", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of random restarts to try")
    args = parser.parse_args()

    cfg = TARGETS[args.target]
    print(f"\n{'='*60}")
    print(f"  Target: {args.target}(x) = {cfg['sympy']}")
    print(f"  Depth:  {cfg['depth']}  |  Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    import torch.nn.functional as F

    best_result = None
    best_err = float("inf")

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        print(f"--- seed {seed} ---")

        soft_node = train_single_node(
            target_fn=cfg["fn"],
            depth=cfg["depth"],
            epochs=args.epochs,
            lr=args.lr,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            x_range=cfg["x_range"],
            verbose=(seed == 0),  # only print progress for first seed
        )

        root_logits = soft_node.logits.detach()
        root_probs = F.softmax(root_logits / soft_node.tau, dim=0)
        labels = ["1", "x", "eml"][:len(root_probs)]
        probs_str = "  ".join(f"{l}={p:.3f}" for l, p in zip(labels, root_probs.tolist()))
        print(f"  root: {probs_str}")

        hard_node, expr, is_exact = snap_and_verify(soft_node, cfg["sympy"])
        err = evaluate_snap(hard_node, cfg["fn"], cfg["x_range"])
        print(f"  snap: {expr}  |  max_err={err:.2e}  exact={is_exact}")

        if err < best_err:
            best_err = err
            best_result = (hard_node, expr, is_exact, seed)

        if best_err < 1e-5:
            print("\n  Early stop: exact match found.")
            break

    print(f"\n{'='*60}")
    print(f"  BEST RESULT (seed {best_result[3]})")
    print(f"  Expression:   {best_result[1]}")
    print(f"  Target:       {cfg['sympy']}")
    print(f"  Exact match:  {best_result[2]}")
    print(f"  Max |error|:  {best_err:.2e}")
    if best_err < 1e-5:
        print("  STATUS: machine-epsilon accuracy after snap ✓")
    else:
        print(f"  STATUS: snap error above threshold ✗")


if __name__ == "__main__":
    main()
