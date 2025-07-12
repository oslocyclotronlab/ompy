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
)

import numpy as np
from tqdm.autonotebook import tqdm

from .stubs import SaveFormat
from ... import H5PY_AVAILABLE, ArrayList, Matrix, Vector
from ...helpers import (
    bytes_to_readable,
    print_readable_time,
    readable_time,
)
from ...version import FULLVERSION
from ..result import RESULT_CLASSES, Result

LOG = logging.getLogger(__name__)

if H5PY_AVAILABLE:
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
- [ ] Packed unfolding is not working. there is some scaling
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
    backgrounds: list[T] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)
    _ubox: np.ndarray | None = None
    _etabox: np.ndarray | None = None
    _nubox: np.ndarray | None = None
    _xi_eta_box: np.ndarray | None = None
    _xi_nu_box: np.ndarray | None = None
    elapsed_time: float | None = None
    aux: dict[str, Any] | None = None
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
        if not H5PY_AVAILABLE:
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
        if self.backgrounds is not None and len(self.backgrounds) > 0:
            backgrounds = ArrayList.from_list(self.backgrounds)
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
            if self.backgrounds:
                LOG.debug(
                    f"Saving `backgrounds` to {path / 'matrices.h5' / 'backgrounds'}"
                )
                f.create_group("backgrounds")
                backgrounds.insert_into_tree(
                    f, "backgrounds/", compression=compression, **kwargs
                )  # type: ignore
            LOG.debug(f"Saving `costs` to {path / 'matrices.h5' / 'costs'}")
            f.create_dataset(
                "costs", data=self.costs, compression=compression, **kwargs
            )
        LOG.warn("Saving `kwargs` is not implemented yet")

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
            if self.backgrounds is not None and len(self.backgrounds) > 0:
                self.backgrounds[i].save(
                    path / f"background_{i}.npz", exist_ok=True
                )  # np.save(path / f"cost_{i}.npy", self.costs[i])

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
        for i in range(len(list(path.glob("boot_*.npz")))):
            unfolded.append(arraytype.from_path(path / f"unfolded_{i}.npz"))
            bootstraps.append(arraytype.from_path(path / f"boot_{i}.npz"))
            if (path / "background_i.npz").exists():
                backgrounds.append(arraytype.from_path(path / f"background_{i}.npz"))
            initials.append(arraytype.from_path(path / f"initial_{i}.npz"))
            if read_only is not None and i > read_only:
                break
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds if backgrounds else None,
            initials=initials,
        )

    @classmethod
    def _load_h5(cls, path, arraytype, basearray, read_only):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not available")
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        if metadata["version"] != FULLVERSION:
            warnings.warn(f"Version mismatch: {metadata['version']} != {FULLVERSION}")
        if metadata["ndim"] != basearray.ndim:
            raise ValueError(f"Wrong ndim: {metadata['ndim']} != {cls.ndim}")
        result_cls: type[Result] = RESULT_CLASSES[metadata["base"]]
        base = result_cls.from_path(path / "base")  # type: ignore

        backgrounds = None
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
            if "backgrounds" in f:
                backgrounds = list(
                    ArrayList.from_tree(
                        f, "backgrounds/", read_only=read_only
                    ).to_arrays()
                )
        return basearray(
            base=base,
            bootstraps=bootstraps,
            unfolded=unfolded,
            costs=costs,
            backgrounds=backgrounds,
            initials=initials,
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
