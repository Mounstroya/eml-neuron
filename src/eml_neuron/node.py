"""
EML node implementations.

The EML operator: eml(a, b) = exp(a) - ln(b)

A single node has three routing choices:
  0 -> terminal: 1
  1 -> terminal: x
  2 -> recursive: eml(left_child, right_child)

SoftEMLNode uses a Gumbel-softmax over these choices so topology
gradients flow during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7  # clamp floor for ln arguments


def _eml(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.exp(a) - torch.log(b.clamp(min=EPS))


class SoftEMLNode(nn.Module):
    """
    A differentiable EML node with soft routing.

    Logits over three choices are learned; during forward a soft
    convex combination is computed so gradients flow through topology.
    At evaluation time call .snap() to get the hard discrete choice.

    Args:
        tau: Gumbel-softmax temperature (annealed during training).
        depth: Tree depth below this node (0 = leaf forced to terminal).
    """

    def __init__(self, tau: float = 1.0, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.tau = tau

        # logits: [p_one, p_x, p_eml]
        # if depth==0 the eml branch is unavailable (would need children)
        n_choices = 2 if depth == 0 else 3
        self.logits = nn.Parameter(torch.zeros(n_choices))

        if depth > 0:
            self.left = SoftEMLNode(tau=tau, depth=depth - 1)
            self.right = SoftEMLNode(tau=tau, depth=depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.gumbel_softmax(self.logits, tau=self.tau, hard=False)

        ones = torch.ones_like(x)

        if self.depth == 0:
            # only terminals available
            p_one, p_x = probs[0], probs[1]
            return p_one * ones + p_x * x

        p_one, p_x, p_eml = probs[0], probs[1], probs[2]

        left_val = self.left(x)
        right_val = self.right(x)
        eml_val = _eml(left_val, right_val)

        return p_one * ones + p_x * x + p_eml * eml_val

    def set_tau(self, tau: float) -> None:
        self.tau = tau
        self.logits  # touch to keep reference
        if self.depth > 0:
            self.left.set_tau(tau)
            self.right.set_tau(tau)

    def snap(self) -> "EMLNode":
        """Return the hard discrete EMLNode corresponding to the argmax choice."""
        with torch.no_grad():
            n_choices = 2 if self.depth == 0 else 3
            choice = int(self.logits[:n_choices].argmax().item())

        if choice == 0:
            return EMLNode(kind="one")
        if choice == 1:
            return EMLNode(kind="x")
        # choice == 2: eml branch
        return EMLNode(kind="eml", left=self.left.snap(), right=self.right.snap())


class EMLNode:
    """
    Hard (non-differentiable) EML expression node.

    Used for symbolic evaluation and verification after snapping a
    trained SoftEMLNode to its argmax topology.
    """

    def __init__(self, kind: str, left: "EMLNode | None" = None, right: "EMLNode | None" = None):
        assert kind in ("one", "x", "eml")
        self.kind = kind
        self.left = left
        self.right = right

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "one":
            return torch.ones_like(x)
        if self.kind == "x":
            return x
        return _eml(self.left(x), self.right(x))

    def __repr__(self) -> str:
        if self.kind == "one":
            return "1"
        if self.kind == "x":
            return "x"
        return f"eml({self.left!r}, {self.right!r})"

    def depth(self) -> int:
        if self.kind in ("one", "x"):
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        if self.kind in ("one", "x"):
            return 1
        return 1 + self.left.node_count() + self.right.node_count()
