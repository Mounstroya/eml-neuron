"""
Measurement 2 — Routing quality as search signal.

Question: when GD fails to snap to the correct expression, do the learned
routing probabilities still carry useful directional information?

Method:
  1. Train GD on the target (e.g. ln(x)) and extract the per-node routing
     probability distribution from the trained SoftEMLNode tree.
  2. Score every expression in the beam using both:
       (a) pure numerical error (baseline beam rank)
       (b) a routing-weighted score: numerical_error / routing_affinity
     where routing_affinity measures how well the expression's topology
     matches the routing probabilities learned by GD.
  3. Report the rank of the known correct expression under each scoring
     method and across multiple GD seeds.

If GD routing improves the rank of the correct expression (compared to
pure numerical ranking), gradients provide useful search signal even when
they don't fully converge.

Usage:
    python experiments/phase1_routing_signal.py
"""

import sys
import argparse
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from eml_neuron.node import SoftEMLNode, EMLNode
from eml_neuron.exhaustive import _enumerate
from eml_neuron.symbolic import to_sympy
import sympy as sp

_x = sp.Symbol("x", positive=True)

# Known correct expression for ln(x): eml(1, eml(eml(1,x), 1))
def _make_ln_exact() -> EMLNode:
    inner = EMLNode("eml", left=EMLNode("one"), right=EMLNode("x"))
    mid   = EMLNode("eml", left=inner, right=EMLNode("one"))
    return EMLNode("eml", left=EMLNode("one"), right=mid)


def routing_affinity(node: EMLNode, soft: SoftEMLNode) -> float:
    """
    Log-probability of the hard topology under the trained soft routing.
    Higher = the GD-trained routing assigns more probability to this tree.
    Returns -inf if the topologies are structurally incompatible.
    """
    with torch.no_grad():
        probs = F.softmax(soft.logits.detach() / max(soft.tau, 1e-6), dim=0)

    n_choices = len(probs)

    if node.kind == "one":
        if n_choices < 1:
            return -math.inf
        return math.log(probs[0].item() + 1e-12)
    if node.kind == "x":
        if n_choices < 2:
            return -math.inf
        return math.log(probs[1].item() + 1e-12)
    # node.kind == "eml"
    if n_choices < 3:
        return -math.inf
    log_p = math.log(probs[2].item() + 1e-12)
    if soft.depth > 0:
        log_p += routing_affinity(node.left,  soft.left)
        log_p += routing_affinity(node.right, soft.right)
    return log_p


def train_gd(target_fn, depth, x_range, epochs=4000, lr=3e-3,
             tau_start=3.0, tau_end=0.02, seed=0) -> SoftEMLNode:
    torch.manual_seed(seed)
    node = SoftEMLNode(tau=tau_start, depth=depth)
    optimizer = torch.optim.Adam(node.parameters(), lr=lr)
    k = math.log(tau_start / tau_end) / epochs
    tau_fn = lambda e: tau_end + (tau_start - tau_end) * math.exp(-k * e)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        tau = tau_fn(epoch)
        node.set_tau(tau)
        x = torch.empty(256).uniform_(*x_range)
        y = node(x)
        if not torch.isfinite(y).all():
            continue
        loss = F.mse_loss(y, target_fn(x))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(node.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k2: v.clone() for k2, v in node.state_dict().items()}
    if best_state:
        node.load_state_dict(best_state)
        node.set_tau(tau_end)
    return node


def rank_of(candidates_sorted, correct_node_repr: str) -> int:
    for i, (_, node) in enumerate(candidates_sorted):
        expr = sp.simplify(to_sympy(node))
        if sp.simplify(expr - sp.log(_x)) == 0:
            return i + 1
    return -1  # not found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4000)
    args = parser.parse_args()

    x_range = (0.5, 5.0)
    depth   = 3
    target  = torch.log
    correct = _make_ln_exact()

    print(f"\n{'='*62}")
    print(f"  Target   : ln(x)")
    print(f"  Depth    : {depth}  |  GD seeds: {args.n_seeds}")
    print(f"  Correct  : {sp.simplify(to_sympy(correct))}")
    print(f"{'='*62}")

    # Build candidate pool once (all depth-3 EML trees)
    print("\nBuilding candidate pool (depth ≤ 3)…")
    pool = _enumerate(depth)
    print(f"  {len(pool)} candidates")

    # Score by numerical error
    x_eval = torch.linspace(x_range[0], x_range[1], 512)
    y_true  = target(x_eval)
    num_scores: list[tuple[float, EMLNode]] = []
    for node in pool:
        try:
            with torch.no_grad():
                y = node(x_eval)
            if not torch.isfinite(y).all():
                continue
            err = (y - y_true).abs().max().item()
            num_scores.append((err, node))
        except Exception:
            pass
    num_scores.sort(key=lambda t: t[0])

    correct_num_rank = rank_of(num_scores, "correct")
    print(f"\n  Rank of correct expression by numerical error alone: {correct_num_rank}")

    # For each GD seed: compute routing-weighted rank
    routing_ranks = []
    for seed in range(args.n_seeds):
        print(f"\n  --- GD seed {seed} ---")
        soft = train_gd(target, depth, x_range, epochs=args.epochs, seed=seed)

        # Score each candidate: combined = numerical_error - lambda * log_affinity
        # lambda trades off numerical fit vs routing match
        lam = 0.05
        combined: list[tuple[float, EMLNode]] = []
        for err, node in num_scores:
            aff = routing_affinity(node, soft)
            if not math.isfinite(aff):
                aff = -50.0
            score = err - lam * aff
            combined.append((score, node))
        combined.sort(key=lambda t: t[0])

        r_num    = correct_num_rank
        r_routing = rank_of(combined, "correct")

        # root routing probs for reporting
        root_probs = F.softmax(soft.logits.detach() / soft.tau, dim=0)
        p_eml = root_probs[2].item() if len(root_probs) == 3 else 0.0

        print(f"    p(eml) at root = {p_eml:.4f}")
        print(f"    Rank of correct expr — numerical only: {r_num}")
        print(f"    Rank of correct expr — routing-weighted: {r_routing}")
        routing_ranks.append((r_num, r_routing))

    print(f"\n{'='*62}")
    print(f"  Summary over {args.n_seeds} GD seeds")
    nums    = [r[0] for r in routing_ranks]
    routing = [r[1] for r in routing_ranks]
    print(f"  Numerical rank  : always {nums[0]} (deterministic)")
    improved = sum(1 for n, r in routing_ranks if r < n)
    same     = sum(1 for n, r in routing_ranks if r == n)
    worse    = sum(1 for n, r in routing_ranks if r > n and r > 0)
    print(f"  Routing rank    : min={min(r for r in routing if r>0)}  "
          f"median={sorted(r for r in routing if r>0)[len(routing)//2]}  "
          f"max={max(r for r in routing if r>0)}")
    print(f"  Improved vs numerical: {improved}/{args.n_seeds}")
    print(f"  Same            : {same}/{args.n_seeds}")
    print(f"  Worse           : {worse}/{args.n_seeds}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
