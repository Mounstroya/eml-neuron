"""
Training loop for Phase 1: single SoftEMLNode.

Tau annealing schedule: tau starts at tau_start and decays
exponentially toward tau_end over the training run, pushing the
soft routing toward discrete choices.
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
    epochs: int = 2000,
    lr: float = 1e-2,
    tau_start: float = 2.0,
    tau_end: float = 0.05,
    n_samples: int = 512,
    x_range: tuple[float, float] = (0.5, 3.0),
    log_every: int = 200,
    device: str = "cpu",
) -> SoftEMLNode:
    """
    Train a single SoftEMLNode to match target_fn.

    Args:
        target_fn: callable, maps torch.Tensor -> torch.Tensor
        depth: max tree depth of the soft node
        epochs: number of gradient steps
        lr: learning rate for Adam
        tau_start / tau_end: Gumbel-softmax temperature schedule
        n_samples: number of random x points per batch
        x_range: uniform sampling range (must keep ln args > 0)
        log_every: print loss every N epochs
        device: torch device string

    Returns trained SoftEMLNode.
    """
    node = SoftEMLNode(tau=tau_start, depth=depth).to(device)
    optimizer = torch.optim.Adam(node.parameters(), lr=lr)
    tau_schedule = make_schedule(tau_start, tau_end, epochs)

    for epoch in range(epochs):
        tau = tau_schedule(epoch)
        node.set_tau(tau)

        x = torch.empty(n_samples, device=device).uniform_(*x_range)
        y_target = target_fn(x)

        y_pred = node(x)
        loss = nn.functional.mse_loss(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1:>5}  loss={loss.item():.6e}  tau={tau:.4f}")

    return node
