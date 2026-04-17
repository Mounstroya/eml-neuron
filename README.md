# EML-Neuron: Exact Neural Networks via the Exp-Minus-Log Operator

> *"With enough weights and neurons you can approximate any function — but you never know exactly which one. There is always an error margin epsilon. If the phenomenon you are modeling is elementarily mathematical, an EML network would find it exactly and verifiably, not statistically."*

---

## Motivation

Modern neural networks are universal approximators. This is powerful, but comes with a fundamental limitation: **the result is always an approximation**. When a model discovers a pattern in data, there is no way to know if it found the real underlying law or a statistical artifact that works within the range of training data.

The EML operator — `eml(x, y) = exp(x) − ln(y)` — was shown by Odrzywolek (2026) to generate every elementary mathematical function through composition, using only itself and the constant `1`. Its grammar is:

```
S → 1 | x | eml(S, S)
```

This raises a question nobody has answered yet: **what if each neuron in a network were an EML node, and instead of training weights, the network learned the tree topology itself?**

If the phenomenon being modeled has an elementary mathematical structure, such a network would not approximate it — it would construct it exactly, in a verifiable symbolic form.

---

## What This Project Is

An experimental framework to test whether a neural network whose nodes are EML operators can learn **topology** instead of weights, producing exact symbolic expressions rather than numerical approximations.

This is not a replacement for conventional deep learning. It is an attempt to answer a specific open question: *can gradient-based search over EML tree structures recover exact mathematical laws from data?*

---

## Objectives

### Phase 1 — Single EML Neuron (Proof of Concept)
- Implement a single differentiable EML node with soft routing between `1`, `x`, and `eml(left, right)`
- Train it to recover simple functions: `exp(x)`, `ln(x)`, `x²`
- Verify that snapping soft weights to hard choices produces machine-epsilon accuracy
- Compare against a standard neuron with a sigmoid activation on the same tasks

### Phase 2 — EML Tree Network
- Extend to a full binary tree of EML nodes with trainable topology
- Implement curriculum learning: grow the tree depth incrementally when current depth fails
- Test on elementary functions up to depth 5: `sin(x)`, `cos(x)`, `x³`, `sqrt(x)`
- Log recovered symbolic expressions and compare against ground truth

### Phase 3 — Physical Law Discovery
- Test on data from the AI Feynman dataset (Udrescu & Tegmark, 2020), which contains known physical equations
- Measure recovery rate: how often does the network find the exact symbolic law vs. an approximation
- Compare against existing symbolic regression baselines (PySR, eml-sr)

### Phase 4 — Analysis and Documentation
- Characterize which classes of functions are recoverable and which are not
- Document failure cases honestly (functions outside the EML-expressible set)
- Write a technical report on findings regardless of outcome

---

## Key Hypothesis

A network where every node computes `eml(left, right)` and the only trainable decision is *which subtree to route through* can recover exact elementary functions from data. The uniform grammar `S → 1 | x | eml(S, S)` means:

- Every node is identical — no heterogeneous operator zoo
- Two expressions are equivalent if and only if their trees produce the same output — verification is structural, not semantic
- The search space is complete for elementary functions (with known exceptions for polynomial roots)

---

## Why This Is Different From Prior Work

| Approach | What is learned | Result |
|---|---|---|
| Standard neural network | Weights | Numerical approximation |
| Neural Architecture Search (NAS) | Topology, then weights | Better substrate for weights |
| Weight Agnostic Neural Networks | Topology with shared weight | Task-specific structure, no symbolic output |
| Symbolic Regression (PySR, eml-sr) | Expression tree via search | Symbolic output, but search not gradient-based over a uniform grammar |
| **This project** | EML tree topology via gradient | Exact symbolic expression, no weights |

The closest prior work is OxiEML and eml-sr, both of which use EML for symbolic regression but through heuristic or evolutionary search, not through a differentiable network that learns topology end-to-end.

---

## Scope and Honest Limitations

This is an experiment, not a product. There are real reasons it might not work:

- Gradient-based topology search over discrete tree structures is a hard optimization problem
- EML requires complex arithmetic internally for some real-valued functions
- Functions involving polynomial roots are provably outside the EML-expressible set (Khovanskii, via stylewarning.com, 2026)
- At deeper tree depths, random initialization may fail to converge (reported as 0% recovery at depth 6 in Odrzywolek's paper)

These limitations are documented up front. A negative result that clearly maps the boundary of what EML networks can and cannot do is still a valid and useful contribution.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Phase 1: single EML node proof of concept
python experiments/phase1_single_node.py --target exp --epochs 2000

# Available targets: exp, ln, square
python experiments/phase1_single_node.py --target ln --epochs 2000
python experiments/phase1_single_node.py --target square --epochs 2000
```

---

## Related Work

- Odrzywolek, A. (2026). *All elementary functions from a single operator*. arXiv:2603.21852
- OxiEML — Pure Rust EML implementation with symbolic regression: [github.com/cool-japan/oxieml](https://github.com/cool-japan/oxieml)
- EMLVM — Stack machine with EML as sole instruction: [github.com/nullwiz/emlvm](https://github.com/nullwiz/emlvm)
- eml-sr — Sklearn-compatible symbolic regression via EML: [github.com/oaustegard/eml-sr](https://github.com/oaustegard/eml-sr)
- uninum — Python symbolic runtime with EML lowering: [github.com/Brumbelow/uninum](https://github.com/Brumbelow/uninum)
- Udrescu & Tegmark (2020). *AI Feynman: a Physics-Inspired Method for Symbolic Regression*

---

## Stack

- Python 3.11+
- PyTorch (differentiable tree nodes, tau-annealing for discretization)
- SymPy (symbolic verification of recovered expressions)
- mpmath (high-precision evaluation)
- NumPy

---

## Status

🟡 **Phase 1 — In progress**

---

## Contributing

This project is in early experimental stage. If you find a function class that EML networks recover reliably, or one where they consistently fail, open an issue with the data — both outcomes are useful.
