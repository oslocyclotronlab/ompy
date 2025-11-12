# Arrays

OMpy’s array types extend NumPy arrays with physical metadata (axes, units,
uncertainty) so spectra carry enough information for the rest of the pipeline.
Most workflows start with a `Vector` (1D spectrum) or a `Matrix` (2D coincidence
data).

## Build a vector

Vectors represent spectra sampled on a single axis (energy bins, excitation
levels, …). Construct them by pairing values with an axis definition:

```{eval-rst}
.. jupyter-execute::

   import os
   os.environ.setdefault("JAX_PLATFORMS", "cpu")

   from ompy.array import Vector
   from ompy.units import Unit

   counts = [12, 18, 9, 3]
   energies = [1.0, 1.5, 2.0, 2.5]  # MeV bin edges

   spec = Vector(
       X=energies,
       values=counts,
       unit=Unit("counts"),
       copy=True,
       metadata={"misc": {"detector": "SiRi"}},
       edge="left",
   )
   spec
```

Essentials to highlight:

- `values`: the y-axis data. Accepts lists, NumPy arrays, or other array-likes.
- `X`: bin edges or centres. Provide the same length as `values` (or `+1` for
  edges) so the vector can build its internal index.
- `unit`: optional physical unit string (e.g. `"counts"`, `"MeV"`). It flows to
  plot labels and saved metadata.
- `metadata`: dictionary for experiment-specific tags (use the `misc` key for
  custom notes such as detector names or acquisition settings).
- `copy`, `dtype`, `order`: control memory layout if you need to work with JAX
  or Numba backends.

### Creating indices quickly

Use helper constructors when you only know the bin width:

```{eval-rst}
.. jupyter-execute::

   import numpy as np

   rng = np.random.default_rng(42)
   energies = np.linspace(0.125, 5.875, 240)  # 25 keV spacing (bin centres)
   counts = rng.poisson(10, size=energies.size)

   wide_spec = Vector(X=energies, values=counts, unit="counts", edge="left")
   wide_spec.values[:5]
```

If you need labelled bins (e.g. level indices) rather than numeric edges, pass
an `Index` instance directly.

### Metadata access

`Vector` exposes convenience properties:

```{eval-rst}
.. jupyter-execute::

   spec.unit, spec.metadata.misc, spec.X
```

These fields help your downstream code adjust for unit conversions or embed the
right labels in plots and tables.

## Build a matrix

Matrices hold 2D coincidence data (e.g. excitation vs γ-ray energy). They accept
separate axes for each dimension:

```{eval-rst}
.. jupyter-execute::

   from ompy.array import Matrix

   E_exc = [0.25, 0.75, 1.25]  # MeV centres
   E_gamma = [0.125, 0.375, 0.625, 0.875]
   coincidences = [
       [12, 18, 11, 4],
       [20, 42, 27, 9],
       [17, 31, 21, 7],
   ]

   matrix = Matrix(
       X=E_exc,
       Y=E_gamma,
       values=coincidences,
   )
   matrix
```

Highlights:

- `X` and `Y` describe the two axes independently. Each takes edges or centres,
  mirroring the `Vector` constructor.
- `MatrixMetadata` tracks per-axis labels and any symmetry flags (e.g. whether
  the matrix is symmetric or triangular). You can populate it via the `metadata`
  argument or convenience properties (`matrix.metadata.x.label = "E_exc"`).
- `matrix.values` returns the underlying 2D `ndarray`, so you can interoperate
  with NumPy/NumExpr when needed.

### Slicing and projections

Matrices support the usual NumPy slicing semantics:

```{eval-rst}
.. jupyter-execute::

   row0 = matrix.values[0]              # first excitation row
   sub = matrix.values[:, 1:3]          # gamma-energy window
   projection = matrix.values.sum(axis=1)  # collapse along gamma axis

   row0, sub.shape, projection
```

Projections often feed back into `Vector`—e.g. `projection` above returns a
`Vector` with metadata preserved.

## Arithmetic and NumPy interop

Adding or subtracting arrays produces new instances while keeping axis metadata:

```{eval-rst}
.. jupyter-execute::

   normalized = spec / spec.values.max()
   combined = matrix + matrix

   normalized.values, combined.values
```

Scalars and raw NumPy arrays broadcast as expected, but combining two OMpy
arrays requires matching axes. If the binning differs, rebin first (see below
for `Vector.rebin`). Converting with `np.asarray(vector)` hands you the backing
`ndarray`; once you step outside the OMpy classes you are responsible for
tracking units and metadata yourself.

## Covariance and uncertainty

Vectors and matrices can carry covariance information, though most workflows
fall back to Poisson errors if you do not supply explicit uncertainties. Attach
them through metadata or helper constructors when you have precomputed standard
deviations.

## Rebinning and resampling

Use `Vector.rebin` to change bin widths without losing physical meaning:

```{eval-rst}
.. jupyter-execute::

   coarser = wide_spec.rebin(binwidth=0.2, preserve="counts")
   coarser.X[:4]
```

Common applications include matching detector responses to simulation grids and
aligning experimental data before unfolding.

## I/O helpers

Both array types ship with load/save helpers:

```python
# Load a spectrum stored as NumPy NPZ
spectrum = Vector.load_npz_1D("runs/oslo_2024-spectrum.npz")

# Save a matrix to CSV files + metadata sidecar
matrix.save_csv_2D("outputs/dy160-coincidences")
```

Formats include CSV, NumPy NPZ/NPY, ROOT, TAR bundles, and the legacy MAMA text
representation. Metadata (units, labels, experiment tags) accompanies the data
when supported by the format.

## Plotting utilities

`Vector.plot` offers quick inspection options (line, step, bar, scatter,
Poisson). You can pass an existing Matplotlib `Axes` via `ax=...` to embed the
plot in multi-panel figures. Matrices can be visualised with `matrix.imshow()`
or manual plotting using `matrix.values`.

## Worked example

```python
from pathlib import Path

from ompy.array import Matrix, Vector
from ompy.array.rebin import Preserve

# Load coincidence data (Matrix) and response spectrum (Vector)
coinc = Matrix.load_npz_2D(Path("data/dy160_coincidences.npz"))
response = Vector.load_npz_1D(Path("data/response_siri.npz"))

# Rebin to match analysis settings
coinc_rebinned = coinc.rebin(bin_width=(0.25, 0.25), preserve=Preserve.COUNT)
response_rebinned = response.rebin(bin_width=0.25, preserve=Preserve.COUNT)

# Subtract a flat background from the response spectrum
response_clean = response_rebinned - 0.1

# Quick sanity plots
response_clean.plot(kind="step", color="black")
coinc_rebinned.values.sum(axis=1)  # project onto excitation axis

# Persist processed outputs
response_clean.save_npz_1D("outputs/response_clean.npz")
coinc_rebinned.save_npz_2D("outputs/dy160_coinc_rebinned.npz")
```

## Where to go next

- Consult the API reference for full method signatures (`reference/api` →
  `ompy.array.vector.Vector`, `ompy.array.matrix.Matrix`).
- Explore `guide/usage/unfolding` to see how arrays feed into iterative
  unfolding algorithms.
- Review `guide/usage/normalization` for guidance on keeping units and
  uncertainties consistent through the Oslo Method pipeline.
