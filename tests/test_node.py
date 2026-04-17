"""Unit tests for EML node primitives."""

import sys
import math
import torch
import pytest

sys.path.insert(0, "src")
from eml_neuron.node import EMLNode, SoftEMLNode, _eml


def test_eml_operator_identity():
    # eml(1, 1) = exp(1) - ln(1) = e - 0 = e
    x = torch.tensor([1.0])
    result = _eml(x, x).item()
    assert math.isclose(result, math.e, rel_tol=1e-6)


def test_hard_node_one():
    node = EMLNode(kind="one")
    x = torch.tensor([2.0, 3.0])
    out = node(x)
    assert (out == 1.0).all()


def test_hard_node_x():
    node = EMLNode(kind="x")
    x = torch.tensor([2.0, 3.0])
    assert torch.allclose(node(x), x)


def test_hard_node_eml():
    # eml(x, x) = exp(x) - ln(x), compare numerically
    node = EMLNode(kind="eml", left=EMLNode(kind="x"), right=EMLNode(kind="x"))
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.exp(x) - torch.log(x)
    assert torch.allclose(node(x), expected)


def test_soft_node_forward_shape():
    node = SoftEMLNode(tau=1.0, depth=2)
    x = torch.linspace(0.5, 2.0, 32)
    out = node(x)
    assert out.shape == x.shape


def test_soft_node_snap_returns_hard():
    torch.manual_seed(0)
    node = SoftEMLNode(tau=0.1, depth=2)
    hard = node.snap()
    assert isinstance(hard, EMLNode)
    assert hard.kind in ("one", "x", "eml")


def test_depth_zero_node():
    node = SoftEMLNode(tau=1.0, depth=0)
    x = torch.tensor([1.5, 2.5])
    out = node(x)
    assert out.shape == x.shape
    hard = node.snap()
    assert hard.kind in ("one", "x")
