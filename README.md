# Normalizing Flows for Bayesian Inference: An Interactive Tutorial

An interactive tutorial on normalizing flows for physics graduate students, built with [marimo](https://marimo.io).

## Overview

This tutorial introduces normalizing flows as an alternative to MCMC for sampling from Bayesian posteriors. It is designed for students who already understand Bayesian inference and have experience running MCMC, but haven't encountered normalizing flows.

**Topics covered:**

1. Why sampling from posteriors is hard (and why MCMC hurts)
2. The core idea: learning an invertible map from a simple distribution to the posterior
3. Planar flows — the simplest architecture, with full math
4. Training via KL divergence minimization (no training data needed)
5. MCMC vs flows: head-to-head interactive comparison
6. Architecture comparison: planar vs radial vs affine coupling
7. Challenging distributions: multimodal, banana, funnel, ring
8. Connections to gravitational-wave inference (DINGO, nessai)

## Getting started

```bash
pip install -r requirements.txt
marimo edit tutorial.py
```

Or to view as a read-only app:

```bash
marimo run tutorial.py
```

## Export to HTML

To share a static version (no Python needed):

```bash
marimo export html tutorial.py -o tutorial.html
```

## Requirements

- Python ≥ 3.10
- marimo
- numpy
- matplotlib

## Author

Chiara Mingarelli — Yale University / Flatiron Institute CCA

## License

MIT
