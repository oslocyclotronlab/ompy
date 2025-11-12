# Unfolding

Detector unfolding removes response effects from first-generation spectra,
recovering the physical distribution that downstream steps consume.

- The Richardson–Lucy implementations in
  `src/ompy/unfolding/richardsonlucy/` provide both NumPy and JAX backends.
- Configuration helpers expose high-level controls for iteration count,
  regularization, and convergence checks.
- Example notebooks in `examples/` walk through typical parameter choices,
  diagnostics, and comparisons between backends.

Bring tuned response matrices and monitor convergence plots to ensure the
unfolded spectra remain physical and stable.
