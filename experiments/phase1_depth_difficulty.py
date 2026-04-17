"""
Phase 1 supplementary: empirical depth-difficulty measurement.

For each target function, runs beam search and reports the best max-absolute
error found at each depth level.  This gives a measured, honest picture of
how hard the target is under the EML grammar with terminals {1, x} — without
claiming impossibility beyond the depths actually searched.

Usage:
    python experiments/phase1_depth_difficulty.py --target square
    python experiments/phase1_depth_difficulty.py --target all
"""

import sys
import argparse
import torch

sys.path.insert(0, "src")
from eml_neuron.exhaustive import beam_search

TARGETS = {
    "exp": {
        "fn": torch.exp,
        "x_range": (0.3, 2.0),
        "note": "exp(x) = eml(x,1) — should be exact at depth 1",
    },
    "ln": {
        "fn": torch.log,
        "x_range": (0.5, 5.0),
        "note": "ln(x) = eml(1,eml(eml(1,x),1)) — should be exact at depth 3",
    },
    "square": {
        "fn": lambda x: x ** 2,
        "x_range": (0.5, 2.5),
        "note": "x^2 — measuring difficulty empirically up to max_depth",
    },
    "sqrt": {
        "fn": torch.sqrt,
        "x_range": (0.5, 4.0),
        "note": "sqrt(x) — measuring difficulty empirically",
    },
}


def run(name: str, cfg: dict, max_depth: int, beam_size: int):
    print(f"\n{'='*62}")
    print(f"  Target : {name}(x)")
    print(f"  Note   : {cfg['note']}")
    print(f"  Search : depth 0..{max_depth}, beam_size={beam_size}")
    print(f"{'='*62}")

    node, err, depth_found = beam_search(
        target_fn=cfg["fn"],
        max_depth=max_depth,
        beam_size=beam_size,
        x_range=cfg["x_range"],
        exact_threshold=1e-4,
        verbose=True,
    )

    print(f"\n  Summary: best_err={err:.4e}  found_at_depth={depth_found}")
    if err < 1e-4:
        print("  STATUS: ✓ exact match found")
    else:
        print(f"  STATUS: ✗ not found at depth ≤ {max_depth} (beam_size={beam_size})")
        print("  This is an empirical bound, not a proof of non-representability.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGETS) + ["all"], default="square")
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--beam_size", type=int, default=500)
    args = parser.parse_args()

    targets = list(TARGETS.items()) if args.target == "all" else [(args.target, TARGETS[args.target])]
    for name, cfg in targets:
        run(name, cfg, args.max_depth, args.beam_size)


if __name__ == "__main__":
    main()
