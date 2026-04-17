"""
Training loop for Phase 1: single SoftEMLNode.

Tau annealing schedule: tau starts at tau_start and decays
exponentially toward tau_end, pushing soft routing toward discrete choices.
Gradient clipping prevents exploding gradients from deep eml compositions.
"""

import math
import torch
import torch.nn as nn
from .node import SoftEMLNode


def make_schedule(tau_start: float, tau_end: float, epochs: int):
    """Exponential tau decay: tau(t) = tau_end + (tau_start-tau_end)*exp(-k*t)."""
    k = math.log(tau_start / tau_end) / epochs
    return lambda epoch: tau_end + (tau_start - tau_end) * math.exp(-k * epoch)


def train_single_node(
    target_fn,
    depth: int = 3,
    epochs: int = 4000,
    lr: float = 3e-3,
    tau_start: float = 3.0,
    tau_end: float = 0.02,
    n_samples: int = 256,
    x_range: tuple[float, float] = (0.5, 2.5),
    log_every: int = 500,
    grad_clip: float = 1.0,
    entropy_beta_end: float = 0.0,  # kept for API compat, no longer used
    device: str = "cpu",
    verbose: bool = True,
) -> SoftEMLNode:
    """
    Train a single SoftEMLNode to match target_fn.

    An entropy regularizer (weight annealed from 0 → entropy_beta_end)
    penalizes diffuse routing distributions, forcing the soft probabilities
    to become peaked before snap.  Without this the optimizer can achieve
    low MSE via a smooth mixture without committing to any topology.

    Args:
        target_fn: callable torch.Tensor -> torch.Tensor
        depth: max tree depth
        epochs: gradient steps
        lr: Adam learning rate
        tau_start / tau_end: softmax temperature annealing
        n_samples: batch size (fresh uniform x each step)
        x_range: sampling range (keep ln args > 0)
        log_every: print interval
        grad_clip: max gradient norm
        entropy_beta_end: final weight on the entropy penalty
        device: torch device
        verbose: print training progress

    Returns trained SoftEMLNode.
    """
    node = SoftEMLNode(tau=tau_start, depth=depth).to(device)
    optimizer = torch.optim.Adam(node.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    tau_schedule = make_schedule(tau_start, tau_end, epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        tau = tau_schedule(epoch)
        node.set_tau(tau)

        x = torch.empty(n_samples, device=device).uniform_(*x_range)
        y_target = target_fn(x)
        y_pred = node(x)

        if not torch.isfinite(y_pred).all():
            continue

        mse = nn.functional.mse_loss(y_pred, y_target)
        loss = mse

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(node.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if mse.item() < best_loss:
            best_loss = mse.item()
            best_state = {k: v.clone() for k, v in node.state_dict().items()}

        if verbose and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch+1:>5}  mse={mse.item():.4e}  "
                f"best={best_loss:.4e}  tau={tau:.4f}"
            )

    if best_state is not None:
        node.load_state_dict(best_state)
        node.set_tau(tau_end)

    return node
