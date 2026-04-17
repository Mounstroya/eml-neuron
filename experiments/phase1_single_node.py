"""
Phase 1 experiment: single EML node proof of concept.

Strategy:
  1. Attempt gradient-based topology search (SoftEMLNode + tau annealing).
  2. If no seed finds an exact match, fall back to exhaustive enumeration
     of all EML expressions up to the configured depth.

Both methods verify the result symbolically via SymPy.

Usage:
    python experiments/phase1_single_node.py --target exp
    python experiments/phase1_single_node.py --target ln
    python experiments/phase1_single_node.py --target square
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import sympy as sp

sys.path.insert(0, "src")
from eml_neuron.train import train_single_node
from eml_neuron.symbolic import snap_and_verify, to_sympy
from eml_neuron.exhaustive import beam_search, exhaustive_search

_x = sp.Symbol("x", positive=True)

TARGETS = {
    # exp(x) = eml(x, 1)  — minimal EML depth is 1
    "exp": {
        "fn": torch.exp,
        "sympy": sp.exp(_x),
        "x_range": (0.3, 2.0),
        "depth": 1,
    },
    # ln(x) = eml(1, eml(eml(1,x), 1))  — EML depth 3
    "ln": {
        "fn": torch.log,
        "sympy": sp.log(_x),
        "x_range": (0.5, 5.0),
        "depth": 3,
    },
    # x^2 — EML depth TBD, exhaustive search will find it
    "square": {
        "fn": lambda x: x ** 2,
        "sympy": _x ** 2,
        "x_range": (0.5, 2.5),
        "depth": 4,
    },
}

EXACT_THRESHOLD = 1e-4


def evaluate_node(hard_node, target_fn, x_range, n=2000):
    x = torch.linspace(x_range[0], x_range[1], n)
    with torch.no_grad():
        y_pred = hard_node(x)
        y_true = target_fn(x)
        if not torch.isfinite(y_pred).all():
            return float("inf")
        return (y_pred - y_true).abs().max().item()


def run_gradient_search(cfg, args):
    print("\n[1/2] Gradient-based topology search")
    best_result = None
    best_err = float("inf")

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        soft_node = train_single_node(
            target_fn=cfg["fn"],
            depth=cfg["depth"],
            epochs=args.epochs,
            lr=args.lr,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            x_range=cfg["x_range"],
            verbose=(seed == 0),
        )

        root_probs = F.softmax(soft_node.logits.detach() / soft_node.tau, dim=0)
        labels = ["1", "x", "eml"][:len(root_probs)]
        probs_str = "  ".join(f"{l}={p:.3f}" for l, p in zip(labels, root_probs.tolist()))
        hard_node, expr, is_exact = snap_and_verify(soft_node, cfg["sympy"])
        err = evaluate_node(hard_node, cfg["fn"], cfg["x_range"])
        print(f"  seed {seed}  root=[{probs_str}]  snap={expr}  err={err:.2e}")

        if err < best_err:
            best_err = err
            best_result = (hard_node, expr, is_exact, seed)

        if best_err < EXACT_THRESHOLD:
            print("  → exact match found, stopping early.")
            break

    return best_result, best_err


def run_exhaustive_search(cfg):
    print("\n[2/2] Beam search fallback")
    node, err, depth_found = beam_search(
        target_fn=cfg["fn"],
        max_depth=cfg["depth"],
        x_range=cfg["x_range"],
        verbose=True,
    )
    if node is None:
        return None, float("inf")
    expr = sp.simplify(to_sympy(node))
    is_exact = sp.simplify(expr - cfg["sympy"]) == sp.Integer(0)
    print(f"  best: {expr}  err={err:.2e}  depth={depth_found}  exact={is_exact}")
    return (node, expr, is_exact, -1), err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS), default="exp")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tau_start", type=float, default=3.0)
    parser.add_argument("--tau_end", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-exhaustive", action="store_true",
                        help="skip exhaustive fallback")
    args = parser.parse_args()

    cfg = TARGETS[args.target]
    print(f"\n{'='*60}")
    print(f"  Target : {args.target}(x) = {cfg['sympy']}")
    print(f"  Depth  : {cfg['depth']}  |  Seeds: {args.seeds}")
    print(f"{'='*60}")

    grad_result, grad_err = run_gradient_search(cfg, args)

    if grad_err < EXACT_THRESHOLD:
        final_result, final_err = grad_result, grad_err
        method = "gradient"
    elif not args.no_exhaustive:
        final_result, final_err = run_exhaustive_search(cfg)
        method = "exhaustive"
    else:
        final_result, final_err = grad_result, grad_err
        method = "gradient (exhaustive skipped)"

    print(f"\n{'='*60}")
    print(f"  METHOD      : {method}")
    if final_result:
        print(f"  Expression  : {final_result[1]}")
        print(f"  Target      : {cfg['sympy']}")
        print(f"  Exact match : {final_result[2]}")
        print(f"  Max |error| : {final_err:.2e}")
        if final_err < EXACT_THRESHOLD:
            print("  STATUS: ✓ machine-epsilon accuracy")
        else:
            print("  STATUS: ✗ above threshold")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
