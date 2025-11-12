# Decomposing

Decomposition modules factorize observables into level-density and γ-strength
components, enabling Oslo Method fits and uncertainty studies.

- The code under `src/ompy/decomposition/` supplies CPU and JAX-accelerated
  kernels that pair with the array abstractions.
- Configuration objects capture model assumptions, boundary conditions, and
  convergence criteria.
- Worked examples highlight how to inspect residuals, perform parameter scans,
  and export intermediate results for publication figures.
