"""
Measurement 1 — GD recovery rate.

Runs gradient-based topology search over N seeds for each target and reports:
  - Fraction of seeds that reach exact accuracy (error < threshold)
  - Distribution of final snap errors (min, median, max)
  - Median epoch at which the best checkpoint was saved (proxy for convergence speed)

Usage:
    python experiments/phase1_gd_recovery.py --target exp --n_seeds 50
    python experiments/phase1_gd_recovery.py --target ln  --n_seeds 50
    python experiments/phase1_gd_recovery.py --target all --n_seeds 50
"""

import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from eml_neuron.node import SoftEMLNode
from eml_neuron.symbolic import snap_and_verify, to_sympy
import sympy as sp

EXACT_THRESHOLD = 1e-4

_x = sp.Symbol("x", positive=True)

TARGETS = {
    "exp": {
        "fn": torch.exp,
        "sympy": sp.exp(_x),
        "x_range": (0.3, 2.0),
        "depth": 1,
    },
    "ln": {
        "fn": torch.log,
        "sympy": sp.log(_x),
        "x_range": (0.5, 5.0),
        "depth": 3,
    },
}


def train_and_snap(target_fn, sympy_target, depth, x_range, epochs, lr, tau_start, tau_end):
    """Train one SoftEMLNode and return (snap_error, best_epoch_fraction, root_eml_prob)."""
    import math

    node = SoftEMLNode(tau=tau_start, depth=depth)
    optimizer = torch.optim.Adam(node.parameters(), lr=lr)
    k = math.log(tau_start / tau_end) / epochs
    tau_fn = lambda e: tau_end + (tau_start - tau_end) * math.exp(-k * e)

    n_samples = 256
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        tau = tau_fn(epoch)
        node.set_tau(tau)
        x = torch.empty(n_samples).uniform_(*x_range)
        y_pred = node(x)
        if not torch.isfinite(y_pred).all():
            continue
        loss = torch.nn.functional.mse_loss(y_pred, target_fn(x))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(node.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            best_state = {k2: v.clone() for k2, v in node.state_dict().items()}

    if best_state:
        node.load_state_dict(best_state)
        node.set_tau(tau_end)

    # snap and evaluate
    hard, expr, _ = snap_and_verify(node, sympy_target)
    x_eval = torch.linspace(x_range[0], x_range[1], 1000)
    with torch.no_grad():
        y_pred = hard(x_eval)
        y_true = target_fn(x_eval)
    if torch.isfinite(y_pred).all():
        err = (y_pred - y_true).abs().max().item()
    else:
        err = float("inf")

    # root routing probability toward eml branch
    root_probs = F.softmax(node.logits.detach() / node.tau, dim=0)
    p_eml = root_probs[-1].item() if len(root_probs) == 3 else 0.0

    return err, best_epoch / epochs, p_eml, str(expr)


def run(name, cfg, n_seeds, epochs, lr, tau_start, tau_end):
    print(f"\n{'='*62}")
    print(f"  Target   : {name}(x) = {cfg['sympy']}")
    print(f"  Depth    : {cfg['depth']}  |  Seeds: {n_seeds}  |  Epochs: {epochs}")
    print(f"{'='*62}")

    errors = []
    conv_fracs = []
    p_emls = []
    exact_count = 0
    expr_counts: dict[str, int] = {}

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        err, conv_frac, p_eml, expr_str = train_and_snap(
            cfg["fn"], cfg["sympy"], cfg["depth"],
            cfg["x_range"], epochs, lr, tau_start, tau_end,
        )
        errors.append(err)
        conv_fracs.append(conv_frac)
        p_emls.append(p_eml)
        expr_counts[expr_str] = expr_counts.get(expr_str, 0) + 1
        if err < EXACT_THRESHOLD:
            exact_count += 1
        if (seed + 1) % 10 == 0:
            print(f"  ... {seed+1}/{n_seeds} seeds done, exact so far: {exact_count}")

    errors_finite = [e for e in errors if e < 1e9]
    errors_sorted = sorted(errors_finite)
    n = len(errors_sorted)

    print(f"\n  Recovery rate    : {exact_count}/{n_seeds} = {exact_count/n_seeds*100:.1f}%")
    print(f"  Error  min       : {min(errors_finite):.4e}")
    print(f"  Error  median    : {errors_sorted[n//2]:.4e}")
    print(f"  Error  max       : {max(errors_finite):.4e}")
    print(f"  Avg conv. frac   : {sum(conv_fracs)/len(conv_fracs):.3f}  (1.0 = best at last epoch)")
    print(f"  Avg p(eml) root  : {sum(p_emls)/len(p_emls):.4f}")
    print(f"\n  Top snapped expressions:")
    for expr_str, count in sorted(expr_counts.items(), key=lambda t: -t[1])[:5]:
        marker = " ← exact" if errors[list(expr_counts.keys()).index(expr_str)] < EXACT_THRESHOLD else ""
        print(f"    {count:>3}x  {expr_str}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS) + ["all"], default="exp")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tau_start", type=float, default=3.0)
    parser.add_argument("--tau_end", type=float, default=0.02)
    args = parser.parse_args()

    targets = list(TARGETS.items()) if args.target == "all" else [(args.target, TARGETS[args.target])]
    for name, cfg in targets:
        run(name, cfg, args.n_seeds, args.epochs, args.lr, args.tau_start, args.tau_end)


if __name__ == "__main__":
    main()
