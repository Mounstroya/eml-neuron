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
    "exp": {
        "fn": torch.exp,
        "sympy": sp.exp(_x),
        "x_range": (0.5, 3.0),
        "depth": 2,
    },
    "ln": {
        "fn": torch.log,
        "sympy": sp.log(_x),
        "x_range": (0.5, 5.0),
        "depth": 2,
    },
    "square": {
        "fn": lambda x: x ** 2,
        "sympy": _x ** 2,
        "x_range": (0.5, 3.0),
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
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tau_start", type=float, default=2.0)
    parser.add_argument("--tau_end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = TARGETS[args.target]
    print(f"\n{'='*60}")
    print(f"  Target: {args.target}(x) = {cfg['sympy']}")
    print(f"  Depth:  {cfg['depth']}")
    print(f"{'='*60}\n")

    soft_node = train_single_node(
        target_fn=cfg["fn"],
        depth=cfg["depth"],
        epochs=args.epochs,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        x_range=cfg["x_range"],
    )

    print("\n--- Snapping to hard topology ---")
    hard_node, expr, is_exact = snap_and_verify(soft_node, cfg["sympy"])

    print(f"  Tree depth:   {hard_node.depth()}")
    print(f"  Node count:   {hard_node.node_count()}")
    print(f"  Expression:   {expr}")
    print(f"  Target:       {cfg['sympy']}")
    print(f"  Exact match:  {is_exact}")

    max_err = evaluate_snap(hard_node, cfg["fn"], cfg["x_range"])
    print(f"  Max |error|:  {max_err:.2e}")

    if max_err < 1e-5:
        print("\n  RESULT: machine-epsilon accuracy after snap ✓")
    else:
        print(f"\n  RESULT: snap error above threshold ({max_err:.2e}) ✗")


if __name__ == "__main__":
    main()
