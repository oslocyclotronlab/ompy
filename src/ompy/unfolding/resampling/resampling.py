from __future__ import annotations

import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
    TYPE_CHECKING,
    Sequence,
)

import numpy as np
from tqdm.autonotebook import tqdm

from .stubs import SaveFormat
from ...array import ArrayList, Matrix, Vector
from ...accel import h5py_available
from ...ensemble import EnsembleVector, EnsembleMatrix
from ...helpers import (
    bytes_to_readable,
    print_readable_time,
    readable_time,
)
from ...version import FULLVERSION
from ..result import RESULT_CLASSES, Result

LOG = logging.getLogger(__name__)


if h5py_available():
    import h5py

if TYPE_CHECKING:
    from .resample1d import Resampling1D
    from .resample2d import Resampling2D

"""
TODO
- [?] Measure bootstrap convergence
- [ ] Automatic coverage test
- [x] Vector bootstrap
- [ ] Covariance
- [ ] The bootstrap uses *a lot* of memory. Can we reduce it?
      Remove the _boxes and use custom methods to broadcast over the lists instead
      Sparse matrices?
- [ ] Use float16 or some other dtype Jax likes
- [x] Packed unfolding is not working. there is some scaling
      or cross row interaction that is not taken into account.
"""


T = TypeVar("T", bound=Matrix | Vector)


@dataclass(kw_only=True)
class Resampling(ABC, Generic[T]):
    base: Result[T]
    bootstraps: list[T]
    unfolded: list[T]
    initials: list[T]
    costs: np.ndarray | list[np.ndarray]
    backgrounds: list[Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)
    _ubox: np.ndarray | None = None
    _etabox: np.ndarray | None = None
    _nubox: np.ndarray | None = None
    _xi_eta_box: np.ndarray | None = None
    _xi_nu_box: np.ndarray | None = None
    elapsed_time: float | None = None
    aux: list[dict[str, np.ndarray]] | None = None
    contaminants: list[list[T]] = field(default_factory=lambda: [[]])

    def save(
        self,
        path: str | Path,
        exist_ok: bool = False,
        format: SaveFormat = "hdf5",
        **kwargs,
    ) -> None:
        format_ = format.lower()
        LOG.debug(f"Saving bootstrap to {path} in {format_} format(?)")
        start = time.time()
        if format_ == "hdf5":
            self.save_hdf5(path, exist_ok=exist_ok, **kwargs)
        elif format_ == "npz":
            self.save_npz(path, exist_ok=exist_ok)
        else:
            raise ValueError(f"Expected format {SaveFormat}, not {format}")
        LOG.debug(
            f"Saved bootstrap to {path} in {format_} format in {readable_time(time.time() - start)}"
        )

    def save_hdf5(
        self, path: str | Path, exist_ok: bool = False, compression="gzip", **kwargs
    ) -> None:
        path = Path(path)
        if not h5py_available():
            LOG.error("h5py is not available. Install it or use `npz` format instead.")
            raise ImportError(
                "h5py is not available. Install it or use `npz` format instead."
            )
        LOG.debug(f"Making directory {path}, exist_ok={exist_ok}")
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(
            version=FULLVERSION, base=self.base.__class__.__name__, ndim=self.ndim
        )
        LOG.debug(f"Saving metadata to {path / 'metadata.json'}")
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        LOG.debug(f"Saving base to {path / 'base'}")
        self.base.save(path / "base", exist_ok=exist_ok)

        unfolded = ArrayList.from_list(self.unfolded)
        bootstraps = ArrayList.from_list(self.bootstraps)
        initials = ArrayList.from_list(self.initials)
        background_payload = self._flatten_backgrounds_for_storage()
        if background_payload is not None:
            background_flat, background_counts = background_payload
            background_arraylist = (
                ArrayList.from_list(background_flat) if len(background_flat) > 0 else None
            )
        with h5py.File(path / "matrices.h5", "w") as f:
            LOG.debug(
                f"Saving `bootstraps` to {path / 'matrices.h5' / 'bootstraps'}"
                f" with compression {compression}" + kwargs.get("compression_opts", "")
            )
            subg = f.create_group("bootstraps")
            bootstraps.insert_into_tree(
                f, "bootstraps/", compression=compression, **kwargs
            )

            LOG.debug(f"Saving `unfolded` to {path / 'matrices.h5' / 'unfolded'}")
            f.create_group("unfolded")
            unfolded.insert_into_tree(f, "unfolded/", compression=compression, **kwargs)

            LOG.debug(f"Saving `initials` to {path / 'matrices.h5' / 'initials'}")
            f.create_group("initials")
            initials.insert_into_tree(f, "initials/", compression=compression, **kwargs)
            if background_payload is not None:
                LOG.debug(
                    f"Saving `backgrounds` to {path / 'matrices.h5' / 'backgrounds'}"
                )
                bg_group = f.create_group("backgrounds")
                bg_group.create_dataset("counts", data=background_counts)
                if len(background_flat) > 0 and background_arraylist is not None:
                    background_arraylist.insert_into_tree(
                        f, "backgrounds/matrices/", compression=compression, **kwargs
                    )
            betas = getattr(self, "betas", None)
            if isinstance(betas, list):
                LOG.debug(f"Saving `betas` to {path / 'matrices.h5' / 'betas'}")
                beta_group = f.create_group("betas")
                total = len(betas)
                beta_group.attrs["total"] = total
                mask = np.array([beta is not None for beta in betas], dtype=np.int8)
                beta_group.create_dataset("mask", data=mask)
                if total > 0:
                    prototype = next(
                        (beta for beta in betas if beta is not None),
                        self.base.raw,
                    )
                    beta_group.attrs["shape"] = prototype.values.shape
                    beta_group.attrs["dtype"] = np.dtype(prototype.values.dtype).str
                    values = np.zeros(
                        (total,) + prototype.values.shape,
                        dtype=prototype.values.dtype,
                    )
                    for i, beta in enumerate(betas):
                        if beta is not None:
                            values[i] = beta.values
                    beta_group.create_dataset(
                        "values", data=values, compression=compression, **kwargs
                    )
                else:
                    shape = self.base.raw.values.shape
                    dtype_str = np.dtype(self.base.raw.values.dtype).str
                    beta_group.attrs["shape"] = shape
                    beta_group.attrs["dtype"] = dtype_str
                    beta_group.create_dataset(
                        "values",
                        data=np.empty((0,) + shape, dtype=np.dtype(dtype_str)),
                        compression=compression,
                        **kwargs,
                    )
            if self.aux:
                LOG.debug(f"Saving `aux` to {path / 'matrices.h5' / 'auxiliary'}")
                aux_group = f.create_group("auxiliary")
                aux_group.attrs["count"] = len(self.aux)
                for i, mapping in enumerate(self.aux):
                    sub = aux_group.create_group(str(i))
                    for key, value in mapping.items():
                        sub.create_dataset(
                            key,
                            data=np.asarray(value),
                            compression=compression,
                            **kwargs,
                        )
            if self.contaminants:
                LOG.debug(
                    f"Saving `contaminants` to {path / 'matrices.h5' / 'contaminants'}"
                )
                cont_group = f.create_group("contaminants")
                counts = np.asarray(
                    [len(entry) for entry in self.contaminants], dtype=np.int32
                )
                cont_group.create_dataset("counts", data=counts)
                flat_contaminants = [
                    contaminant
                    for entry in self.contaminants
                    for contaminant in entry
                ]
                if flat_contaminants:
                    cont_arraylist = ArrayList.from_list(flat_contaminants)
                    cont_arraylist.insert_into_tree(
                        f, "contaminants/matrices/", compression=compression, **kwargs
                    )
            LOG.debug(f"Saving `costs` to {path / 'matrices.h5' / 'costs'}")
            f.create_dataset(
                "costs", data=self.costs, compression=compression, **kwargs
            )
        LOG.warn("Saving `kwargs` is not implemented yet")

    def _flatten_backgrounds_for_storage(
        self,
    ) -> tuple[list[Matrix | Vector], np.ndarray] | None:
        if not self.backgrounds:
            return None

        flat: list[Matrix | Vector] = []
        counts: list[int] = []
        for entry in self.backgrounds:
            if entry is None:
                counts.append(0)
                continue
            if isinstance(entry, (Matrix, Vector)):
                flat.append(entry)
                counts.append(1)
                continue
            if isinstance(entry, Sequence) and all(
                isinstance(item, (Matrix, Vector)) for item in entry
            ):
                flat.extend(entry)
                counts.append(len(entry))
                continue
            LOG.warning(
                "Skipping background serialization for unsupported type %s", type(entry)
            )
            return None
        return flat, np.asarray(counts, dtype=np.int64)

    def save_npz(
        self, path: str | Path, exist_ok: bool = False, disable_tqdm: bool = False
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=exist_ok)
        metadata = dict(
            version=FULLVERSION, base=self.base.__class__.__name__, ndim=self.ndim
        )
        LOG.debug(f"Saving metadata to {path / 'metadata.json'}")
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        LOG.debug(f"Saving base to {path / 'base'}")
        self.base.save(path / "base", exist_ok=True)

        tqdm_ = tqdm if not disable_tqdm else lambda x: x
        LOG.debug(f"Saving {len(self.bootstraps)} matrices to {path}")
        for i in tqdm_(range(len(self.bootstraps))):
            self.bootstraps[i].save(path / f"boot_{i}.npz", exist_ok=True)
            self.unfolded[i].save(path / f"unfolded_{i}.npz", exist_ok=True)
            self.initials[i].save(path / f"initial_{i}.npz", exist_ok=True)
            if self.backgrounds is not None and len(self.backgrounds) > i:
                bg_entry = self.backgrounds[i]
                if isinstance(bg_entry, (Matrix, Vector)):
                    bg_entry.save(path / f"background_{i}.npz", exist_ok=True)
                elif isinstance(bg_entry, Sequence) and all(
                    isinstance(item, (Matrix, Vector)) for item in bg_entry
                ):
                    for j, item in enumerate(bg_entry):
                        item.save(path / f"background_{i}_{j}.npz", exist_ok=True)
                elif bg_entry is not None:
                    LOG.warning(
                        "Skipping background %d for NPZ serialization (unsupported type %s)",
                        i,
                        type(bg_entry),
                    )
            betas = getattr(self, "betas", None)
            if isinstance(betas, list) and len(betas) > i and betas[i] is not None:
                betas[i].save(path / f"beta_{i}.npz", exist_ok=True)  # type: ignore[call-arg]
            if self.aux and len(self.aux) > i:
                aux_mapping = {k: np.asarray(v) for k, v in self.aux[i].items()}
                np.savez(path / f"aux_{i}.npz", **aux_mapping)
            if self.contaminants and len(self.contaminants) > i:
                for j, contaminant in enumerate(self.contaminants[i]):
                    contaminant.save(path / f"contaminant_{i}_{j}.npz", exist_ok=True)

        LOG.debug(f"Saving {len(self.costs)} `costs` to {path}")
        costs = {f"cost_{i}": self.costs[i] for i in range(len(self.costs))}
        np.savez(path / "costs.npz", **costs)

        LOG.warn("Not saving kwargs")

    @overload
    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Matrix],
        basearray: type[Resampling2D],
        read_only: int | None = None,
    ) -> Resampling2D: ...

    @overload
    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Vector],
        basearray: type[Resampling1D],
        read_only: int | None = None,
    ) -> Resampling1D: ...

    @classmethod
    def _load(
        cls,
        path: Path,
        arraytype: type[Matrix] | type[Vector],
        basearray: type[Resampling2D] | type[Resampling1D],
        read_only: int | None = None,
    ) -> Resampling2D | Resampling1D:
        if (path / "matrices.h5").exists():
            return cls._load_h5(path, arraytype, basearray, read_only)
        return cls._load_npz(path, arraytype, basearray, read_only)

    @classmethod
    def _load_npz(cls, path, arraytype, basearray, read_only):
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata["ndim"] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore
        unfolded = []
        bootstraps = []
        costs = []
        initials = []
        if (path / "costs.npz").exists():
            costs = np.load(path / "costs.npz")
        backgrounds = []
        betas: list[T | None] = []
        aux_list: list[dict[str, np.ndarray]] = []
        contaminants: list[list[T]] = []
        for i in range(len(list(path.glob("boot_*.npz")))):
            unfolded.append(arraytype.from_path(path / f"unfolded_{i}.npz"))
            bootstraps.append(arraytype.from_path(path / f"boot_{i}.npz"))
            if (path / "background_i.npz").exists():
                backgrounds.append(arraytype.from_path(path / f"background_{i}.npz"))
            initials.append(arraytype.from_path(path / f"initial_{i}.npz"))
            beta_path = path / f"beta_{i}.npz"
            if beta_path.exists():
                betas.append(arraytype.from_path(beta_path))
            else:
                betas.append(None)
            aux_path = path / f"aux_{i}.npz"
            if aux_path.exists():
                data = np.load(aux_path)
                aux_list.append({k: data[k] for k in data.files})
            else:
                aux_list.append({})
            cont_group: list[T] = []
            j = 0
            while (path / f"contaminant_{i}_{j}.npz").exists():
                cont_group.append(arraytype.from_path(path / f"contaminant_{i}_{j}.npz"))
                j += 1
            contaminants.append(cont_group)
            if read_only is not None and i > read_only:
                break
        betas_use = betas if any(beta is not None for beta in betas) else None
        aux_use = aux_list if any(len(entry) > 0 for entry in aux_list) else None
        contaminants_use = (
            contaminants if any(len(entry) > 0 for entry in contaminants) else None
        )
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds if backgrounds else None,
            initials=initials,
            betas=betas_use,
            aux=aux_use,
            contaminants=contaminants_use,
        )

    @classmethod
    def _load_h5(cls, path, arraytype, basearray, read_only):
        if not h5py_available():
            raise ImportError("h5py is not available")
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata["ndim"] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore

        with h5py.File(path / "matrices.h5", "r") as f:
            bootstraps = list(
                ArrayList.from_tree(f, "bootstraps/", read_only=read_only).to_arrays()
            )
            unfolded = list(
                ArrayList.from_tree(f, "unfolded/", read_only=read_only).to_arrays()
            )
            initials = list(
                ArrayList.from_tree(f, "initials/", read_only=read_only).to_arrays()
            )
            costs = np.asarray(f["costs"])
            backgrounds = None
            if "backgrounds" in f:
                bg_group = f["backgrounds"]
                counts = (
                    np.asarray(bg_group["counts"])
                    if "counts" in bg_group
                    else None
                )
                flat_backgrounds: list[Matrix | Vector] = []
                if "matrices" in bg_group:
                    flat_backgrounds = list(
                        ArrayList.from_tree(
                            f, "backgrounds/matrices/", read_only=read_only
                        ).to_arrays()
                    )
                if counts is not None:
                    backgrounds = []
                    idx = 0
                    for count in counts:
                        count_int = int(count)
                        items = tuple(
                            flat_backgrounds[idx : idx + count_int]
                        )
                        idx += count_int
                        backgrounds.append(items)
                elif flat_backgrounds:
                    backgrounds = flat_backgrounds
            betas = None
            if "betas" in f:
                beta_group = f["betas"]
                total = int(beta_group.attrs.get("total", 0))
                mask = (
                    beta_group["mask"][...].astype(bool)
                    if "mask" in beta_group
                    else np.ones(total, dtype=bool)
                )
                values = (
                    beta_group["values"][...]
                    if "values" in beta_group
                    else np.empty(
                        (total,) + base.raw.values.shape, dtype=base.raw.values.dtype
                    )
                )
                betas = []
                for i in range(total):
                    if mask[i]:
                        betas.append(base.raw.clone(values=values[i]))
                    else:
                        betas.append(None)
            aux = None
            if "auxiliary" in f:
                aux_group = f["auxiliary"]
                count = int(aux_group.attrs.get("count", 0))
                aux = []
                for i in range(count):
                    sub = aux_group[str(i)]
                    mapping = {key: sub[key][...] for key in sub}
                    aux.append(mapping)
            contaminants = None
            if "contaminants" in f:
                cont_group = f["contaminants"]
                counts = cont_group["counts"][...] if "counts" in cont_group else np.empty(0, dtype=np.int32)
                flat = []
                if "matrices" in cont_group:
                    flat = list(
                        ArrayList.from_tree(
                            f, "contaminants/matrices/", read_only=read_only
                        ).to_arrays()
                    )
                contaminants = []
                idx = 0
                for count in counts:
                    inner = []
                    for _ in range(int(count)):
                        inner.append(flat[idx])
                        idx += 1
                    contaminants.append(inner)
        extra_kwargs: dict[str, Any] = {}
        if betas is not None:
            extra_kwargs["betas"] = betas
        if aux is not None:
            extra_kwargs["aux"] = aux
        if contaminants is not None:
            extra_kwargs["contaminants"] = contaminants
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds,
            initials=initials,
            **extra_kwargs,
        )

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path | str, n: int | None = None) -> Resampling: ...

    @property
    def G_ex(self) -> Matrix:
        return self.base.G_ex

    def has_G_ex(self) -> bool:
        return hasattr(self.base, "G_ex") and self.G_ex is not None

    @property
    def G_eg(self) -> Matrix:
        return self.base.G_eg

    @property
    def GegD(self) -> Matrix:
        return self.base.GegD

    @property
    def D(self) -> Matrix:
        return self.base.D_eg

    @property
    def raw(self) -> T:
        return self.base.raw

    @property
    def background(self) -> T | None:
        return self.base.background

    @property
    @abstractmethod
    def ubox(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def etabox(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def nubox(self) -> np.ndarray: ...

    def __len__(self) -> int:
        return len(self.unfolded)

    def memory_usage_rapport(self) -> None:
        g = bytes_to_readable
        memory_usage = {
            "bootstraps": sum([b.nbytes for b in self.bootstraps]),
            "unfolded": sum([b.nbytes for b in self.unfolded]),
            "costs": (
                self.costs.nbytes
                if isinstance(self.costs, np.ndarray)
                else sum([c.nbytes for c in self.costs])
            ),
            "backgrounds": (
                sum([b.nbytes for b in self.backgrounds])
                if self.backgrounds is not None
                else 0
            ),
            # 'base': self.base.nbytes,
            "_ubox": self._ubox.nbytes if self._ubox is not None else 0,
            "_etabox": self._etabox.nbytes if self._etabox is not None else 0,
            "_nubox": self._nubox.nbytes if self._nubox is not None else 0,
            "initial": sum([b.nbytes for b in self.initials]),
        }
        rapport = "MEMORY USAGE RAPPORT\n"
        for attr, mem in memory_usage.items():
            rapport += f"{attr:<15} {g(mem)}\n"
        rapport += "=============================\n"
        rapport += f"{'Total':<15} {g(sum(memory_usage.values()))}"
        print(rapport)

    def time(self) -> None:
        print_readable_time(self.elapsed_time)

    def to_ensemble(self, which: str = "eta") -> EnsembleVector | EnsembleMatrix:
        """Convert resampling results to Ensemble for statistical analysis.
        
        This method packages the bootstrap samples into an Ensemble object,
        which provides convenient statistical operations (mean, std, quantiles)
        and supports automatic lifting of operations via .each.
        
        Parameters
        ----------
        which : str, default: "eta"
            Which space to return as ensemble:
            
            - ``"unfolded"`` or ``"mu"`` : The unfolded bootstrap samples
            - ``"eta"`` : The eta space (G @ mu) bootstrap samples (default)
            - ``"nu"`` : The nu space (D @ G @ mu) bootstrap samples
            
        Returns
        -------
        EnsembleVector | EnsembleMatrix
            Ensemble containing N bootstrap samples, where N is the number
            of bootstrap iterations. The ensemble can be used for:
            
            - Statistical summaries: ``.mean()``, ``.std()``, ``.quantile()``
            - Uncertainty propagation: ``.each.rebin()``, ``.each.normalize()``
            - Error vector construction: ``.to_error_vector()``
            
        Examples
        --------
        Basic usage with statistics:
        
        >>> result = unfolder.unfold(data, background=bg)
        >>> resampling = result.resample(N=100)
        >>> ensemble = resampling.to_ensemble()
        >>> mean_spectrum = ensemble.mean()
        >>> uncertainty = ensemble.std()
        
        Get confidence intervals:
        
        >>> q16, q84 = ensemble.quantile([0.16, 0.84])
        >>> # 68% confidence interval
        
        Apply operations to all members:
        
        >>> rebinned = ensemble.each.rebin(factor=2.0)
        >>> normalized = ensemble.each.normalize()
        
        Get different spaces:
        
        >>> ensemble_mu = resampling.to_ensemble("mu")     # Unfolded
        >>> ensemble_eta = resampling.to_ensemble("eta")   # Smoothed
        >>> ensemble_nu = resampling.to_ensemble("nu")     # Forward folded
        
        See Also
        --------
        ensemble_unfolded : Convenience method for ``.to_ensemble("unfolded")``
        ensemble_eta : Convenience method for ``.to_ensemble("eta")``
        ensemble_nu : Convenience method for ``.to_ensemble("nu")``
        
        Notes
        -----
        The ensemble uses the appropriate bootstrap box (ubox, etabox, nubox)
        which are computed lazily and cached by the Resampling subclasses.
        """
        # Delegate to the appropriate box property
        match which:
            case "unfolded" | "mu":
                box = self.ubox
            case "eta":
                box = self.etabox
            case "nu":
                box = self.nubox
            case _:
                raise ValueError(
                    f"Unknown which={which!r}. "
                    f"Expected 'unfolded', 'mu', 'eta', or 'nu'"
                )
        
        # Use the best result from base as template
        template = self.base.best()
        
        # Create appropriate ensemble type based on dimensionality
        if self.ndim == 1:
            ensemble = EnsembleVector(box, template=template)
        elif self.ndim == 2:
            ensemble = EnsembleMatrix(box, template=template)
        else:
            raise ValueError(f"Unsupported ndim={self.ndim}")
        ensemble.stage = 'unfolded'
        return ensemble

    def ensemble_unfolded(self) -> EnsembleVector | EnsembleMatrix:
        """Return ensemble of unfolded bootstrap spectra.
        
        Convenience method equivalent to ``.to_ensemble("unfolded")``.
        
        Returns
        -------
        EnsembleVector | EnsembleMatrix
            Ensemble of unfolded spectra from bootstrap samples.
            
        Examples
        --------
        >>> resampling = result.resample(N=100)
        >>> ensemble = resampling.ensemble_unfolded()
        >>> mean_unfolded = ensemble.mean()
        """
        return self.to_ensemble("unfolded")
    
    def ensemble_eta(self) -> EnsembleVector | EnsembleMatrix:
        """Return ensemble of eta space bootstrap spectra.
        
        Convenience method equivalent to ``.to_ensemble("eta")``.
        Eta space is the smoothed unfolded space: eta = G @ mu
        
        Returns
        -------
        EnsembleVector | EnsembleMatrix
            Ensemble of eta space spectra.
            
        Examples
        --------
        >>> resampling = result.resample(N=100)
        >>> ensemble_eta = resampling.ensemble_eta()
        >>> mean_eta = ensemble_eta.mean()
        """
        return self.to_ensemble("eta")
    
    def ensemble_nu(self) -> EnsembleVector | EnsembleMatrix:
        """Return ensemble of nu space bootstrap spectra.
        
        Convenience method equivalent to ``.to_ensemble("nu")``.
        Nu space is the forward-folded space: nu = D @ G @ mu
        
        Returns
        -------
        EnsembleVector | EnsembleMatrix
            Ensemble of nu space spectra (forward-folded predictions).
            
        Examples
        --------
        >>> resampling = result.resample(N=100)
        >>> ensemble_nu = resampling.ensemble_nu()
        >>> mean_prediction = ensemble_nu.mean()
        >>> # Compare to raw data to assess fit quality
        """
        return self.to_ensemble("nu")
