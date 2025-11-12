# What is OMpy?

OMpy bundles reusable analysis pipelines for the Oslo Method, covering
unfolding, first-generation extraction, and level-density/γ-strength analysis.
The library provides:

- acceleration backends (NumPy, Numba, JAX) so you can balance setup effort and
  runtime performance;
- a command-line interface via the `ompy` entry point for common data
  management, verification, and dataset fetching tasks;
- an extensible architecture that encourages notebooks, scripts, and custom
  workflows tailored to local experiments.

```{note}
For a higher-level overview, skim the project `README.md`. It lists published
works that rely on OMpy and highlights the major features in active
development.
```
