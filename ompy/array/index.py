from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import groupby
from typing import TypeAlias, Literal, Type, overload, Any
from warnings import warn

import numpy as np

from .index_fn import _index_left, _index_mid_uniform, is_monotone, is_uniform, is_length_congruent
from .index_fn import _index_mid_nonuniform
from .rebin import _rebin_uniform_left_left, Preserve
from .. import float64, Unit, Quantity, DimensionalityError, njit
from ..library import only_one_not_none
from ..stubs import Unitlike

"""
TODO Mixin architecture is not quite right. Read up.
-[ ] Implement binary search
-[x] Implement conversion methods
-[ ] Batch index, batch index sorted
-[x] Rebinning
-[x] Slicing

How is rebinning handled?
The index has N+1 "edges" with respect to the N values of the
values it indexes. The 1 more edge is used to properly define the
step size between bins. 
Each Edge interprets the raw array differently. 
For Left the *last* value is the *rightmost* edge of the last bin.
For Mid the *first* value is the *leftmost* edge of the first bin.

The Layout determines the default handling of N bins. 
A Uniform layout extrapolates the step size into the right/leftmost edge.
A Nonuniform layout does not extrapolate, instead raising an exception.

For extrapolation one has the options: left, right, and auto.
`left` extrapolates the bins to the left, `right` extrapolates to the right,
using their respective last two bins. `auto` depends on the Edge.

There is another "extrapolation" extrapolate_boundary that adds the required boundary. 
Assumes the bins are the N bins with the +1 boundary lacking. 
For left, the boundary is the rightmost bin + the step size.
For mid, the boundary is the leftmost bin - the step size/2, as it
represents the leftmost edge, not leftmost bin.
"""

Direction: TypeAlias = Literal['left', 'right', 'auto']
Edges: TypeAlias = Literal['left', 'mid']


@dataclass(frozen=True, slots=True)
class Calibration:
    coefficients: np.ndarray

    def __post_init__(self):
        if len(self.coefficients) < 2:
            raise ValueError("Calibration must have exactly 3 coefficients.")
        if len(self.coefficients) > 2:
            warn("Non-linear calibration poorly supported")

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def shift(self) -> float:
        if self.order < 0:
            return 0.0
        return self.coefficients[0]

    @property
    def gain(self) -> float:
        if self.order < 1:
            return 0.0
        return self.coefficients[1]

    @property
    def quadratic(self) -> float:
        if self.order < 2:
            return 0.0
        return self.coefficients[2]

    @property
    def a0(self) -> float:
        if self.order < 0:
            return 0.0
        return self.coefficients[0]

    @property
    def a1(self) -> float:
        if self.order < 1:
            return 0.0
        return self.coefficients[1]

    @property
    def a2(self) -> float:
        if self.order < 2:
            return 0.0
        return self.coefficients[2]

    @property
    def start(self) -> float:
        if self.order < 0:
            return 0.0
        return self.coefficients[0]

    @property
    def step(self) -> float:
        if self.order < 1:
            return 0.0
        return self.coefficients[1]

    def from_index(self, index: Index) -> Calibration:
        if not index.is_uniform():
            raise ValueError("Index must be uniform.")
        return Calibration(np.array([index[0], index.step]))

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        if 'start' in d:
            return Calibration(np.array([d['start'], d['step']]))
        if 'a0' in d:
            vals = []
            for key, val in d.items():
                if key.startswith('a') and key[1:].isdigit() and val is not None:
                    vals.append((key, val))
            _, vals = zip(*sorted(vals, key=lambda x: x[0]))
            return Calibration(np.array(vals))
        raise ValueError("Invalid dictionary.")

    def linspace(self, n: int, **kwargs) -> np.ndarray:
        if self.order != 1:
            raise ValueError("Only linear polynomials supported.")
        return np.linspace(self.start, self.start + self.step * (n - 1), n, **kwargs)

    def __len__(self) -> int:
        return len(self.coefficients)

    def __getitem__(self, key: int | slice) -> float:
        return self.coefficients[key]


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    alias: str = ''
    label: str = ''
    unit: Unit = Unit('keV')

    def __post_init__(self):
        object.__setattr__(self, 'unit', Unit(self.unit))

    def clone(self, alias: str | None = None, label: str | None = None,
              unit: Unit | None = None) -> IndexMetadata:
        alias = alias if alias is not None else self.alias
        label = label if label is not None else self.label
        unit = unit if unit is not None else self.unit
        return IndexMetadata(alias, label, unit)

    def update(self, **kwargs) -> IndexMetadata:
        return self.clone(**kwargs)

    def to_dict(self) -> dict[str, str | Unit]:
        return dict(alias=self.alias, label=self.label, unit=self.unit)


class Index(ABC):
    """ Abstract base class for indexers """

    def __init__(self, bins: np.ndarray, boundary: float,
                 metadata: IndexMetadata = IndexMetadata(),
                 dtype: np.dtype = np.float64, **kwargs):
        if len(bins) <= 0:
            raise ValueError("Bins must have at least one element.")
        if not is_monotone(bins):
            raise ValueError("Bins must be monotonically increasing.")
        self.bins = np.asarray(bins, dtype=dtype)
        self.boundary = dtype(boundary)
        self.meta = metadata.update(**kwargs)
        self.__content_hash = None

    def rebin(self, other: Index | np.ndarray, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[
        Index, np.ndarray]:
        if not isinstance(other, Index):
            other = self.from_array(values)
        if self.leftmost > other.rightmost:
            raise ValueError(f"Bins do not overlap. {self.leftmost} < {other.rightmost}")
        if self.rightmost < other.leftmost:
            raise ValueError(f"Bins do not overlap. {self.rightmost} > {other.leftmost}")
        return self._rebin(other, values, preserve=preserve)

    @abstractmethod
    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        pass

    def to_same(self, other: Index) -> Index:
        return self.to_unit(other).to_same_edge(other)

    def to_same_edge(self, other: Index) -> Index:
        if self.is_left() and other.is_left():
            return self
        if self.is_mid() and other.is_mid():
            return self
        if self.is_left():
            return self.to_mid()
        return self.to_left()

    def to_same_unit(self, x: Unitlike) -> float:
        x = Quantity(x)
        if not x.dimensionless:
            x = x.to(self.meta.unit)
        return x.magnitude

    def index(self, x: Unitlike) -> int:
        """ Index of the bin containing x """
        return self._index(self.to_same_unit(x))

    @abstractmethod
    def _index(self, x: float) -> int:
        ...

    @classmethod
    def from_extrapolate(cls, bins: np.ndarray, n: int = 1, direction: Direction = 'auto',
                         extrapolate_boundary: bool = True, **kwargs) -> Index:
        bins = cls.extrapolate(bins, n=n, direction=direction)
        boundary = cls.extrapolate_boundary(bins, extrapolate_boundary=extrapolate_boundary)
        return cls(bins, boundary=boundary, **kwargs)

    @classmethod
    def from_array(cls, bins: np.ndarray, extrapolate_boundary: bool = True, **kwargs) -> Index:
        if extrapolate_boundary:
            boundary = cls.extrapolate_boundary(bins)
        else:
            boundary, bins = cls.split_bins_boundary(bins)
        return cls(bins, boundary=boundary, **kwargs)

    @overload
    def __getitem__(self, key: int) -> float:
        ...

    @overload
    def __getitem__(self, key: slice) -> Index:
        ...

    def __getitem__(self, key: int | slice) -> float | Index:
        match key:
            case slice():
                return self.from_array(self.bins[key], metadata=self.meta)
            case _:
                return self.bins[key]

    def __len__(self) -> int:
        return len(self.bins)

    def summary(self) -> str:
        e = 'left' if self.is_left() else 'mid'
        l = 'uniform' if self.is_uniform() else 'nonuniform'
        if self.is_uniform():
            summary = f": X₀ = {self.leftmost:.2f}, ΔX = {self.step(0):.2f}, Xₙ = {self.rightmost:.2f}"
        else:
            summary = f": X₀ = {self.leftmost:.2f}, Xₙ = {self.rightmost:.2f}"
            de = self.steps()
            de = ', '.join(f'{d:.3g}\u00D7{i}' for d, i in compress(de))
            summary += f"\nSteps: {de}"
        s = f"Index {e} {l} with {len(self)} bins [{self.meta.unit:~}] {summary}"
        if self.meta.label:
            s += f"\nLabel: {self.meta.label}"
        if self.meta.alias:
            s += f"\nAlias: {self.meta.alias}"
        return s

    def __str__(self) -> str:
        return self.summary() + f'\n{self.bins}'

    def __eq__(self, other) -> bool:
        return np.allclose(self.bins, other.bins) and np.isclose(self.boundary,
                                                                 other.boundary) and self.unit == other.unit

    def update_metadata(self, **kwargs) -> Index:
        return self.__class__(self.bins, self.boundary, metadata=self.meta.update(**kwargs))

    def to_unit(self, unit: Unitlike | Index) -> Index:
        if isinstance(unit, Index):
            unit = unit.unit
        unit = Unit(unit)
        if self.meta.unit == unit:
            return self
        if not self.meta.unit.is_compatible_with(unit):
            raise DimensionalityError(self.meta.unit, unit)
        factor = 1 / self.meta.unit.from_(Quantity(1, unit)).magnitude
        bins = self.bins * factor
        boundary = self.boundary * factor
        meta = self.meta.clone(unit=unit)
        return self.__class__(bins, boundary, meta)

    @property
    def unit(self) -> Unit:
        return self.meta.unit

    @overload
    def index_expression(self, expr: Unitlike | str, strict: bool) -> int:
        ...

    @overload
    def index_expression(self, expr: None, strict: bool) -> None:
        ...

    def index_expression(self, expr: Unitlike | str | None, strict: bool = True) -> int | None:
        if expr is None:
            return None
        lesser, greater, value = preparse(expr)
        if strict or not isinstance(value, int):
            index = self.index(value)
        else:
            index = value

        if lesser:
            index -= 1
        if greater:
            index += 1
        return index

    def index_slice(self, s: slice, strict: bool = True) -> slice:
        """ Convert a slice to a slice of indices

        Args:
            s: Slice to convert
            strict: Always treat ints as values (True) or indices (False).

        Returns: Slice of indices

        """
        start = self.index_expression(s.start, strict=strict)
        stop = self.index_expression(s.stop, strict=strict)

        if s.step is not None:
            raise ValueError("Step not supported for slices.")
        return slice(start, stop)

    @classmethod
    def from_calibration(cls, calibration: Calibration, n: int, **kwargs) -> Index:
        """ Create an index from a calibration

        Args:
            calibration: Calibration to convert
            n: Number of bins
            kwargs: Additional arguments to pass to the constructor

        Returns: Index

        """
        bins = calibration.linspace(n)
        return cls.from_array(bins, **kwargs)

    def to_dict(self) -> dict[str, any]:
        return dict(bins=self.bins, boundary=self.boundary, meta=self.meta.to_dict(),
                    class_name=self.__class__.__name__, length=len(self))

    @classmethod
    def from_dict(cls, d: dict[str, any]) -> Index:
        return _from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict[str, any]) -> Index:
        if cls.__name__ != d['class_name']:
            raise ValueError(f"Cannot deserialize {d['class_name']} to {cls.__name__}")
        meta = d['meta']
        return cls(d['bins'], boundary=d['boundary'], **meta)

    @property
    def label(self) -> str:
        return self.meta.label

    @property
    def alias(self) -> str:
        return self.meta.alias

    def is_compatible_with(self, other: Index, rtol: float = 1e-4, **kwargs) -> bool:
        if not self.unit.is_compatible_with(other.unit):
            return False
        if not len(self) == len(other):
            return False
        if self.is_uniform():
            if not other.is_uniform():
                return False
            factor = 1 / self.meta.unit.from_(Quantity(1, other.unit)).magnitude
            if not np.isclose(self.step(0) * factor, other.step(0), rtol=rtol, **kwargs):
                return False
            if not np.isclose(self.leftmost * factor, other.leftmost, rtol=rtol, **kwargs):
                return False
            return True
        else:
            other = other.to_same(self)
            if not np.allclose(self.bins, other.bins, rtol=rtol, **kwargs):
                return False
            return True

    def __clone(self, bins: np.ndarray | None = None,
                boundary: np.ndarray | None = None,
                meta: IndexMetadata | None = None, order: str | None = None,
                **kwargs) -> Index:
        if bins is None:
            bins = self.bins
        if order is not None:
            bins = np.asarray(bins, order=order)
        if boundary is None:
            boundary = self.boundary
        if meta is None:
            meta = self.meta
        meta = meta.clone(**kwargs)
        return self.__class__(bins, boundary, metadata=meta)

    def update(self, alias: str | None = None, label: str | None = None,
               unit: Unitlike | None = None) -> Index:
        meta = self.meta.clone(alias=alias, label=label, unit=unit)
        return self.__clone(meta=meta)

    def handle_rebin_arguments(self, *, bins: np.ndarray | Index | None = None,
                               factor: float | None = None,
                               binwidth: Unitlike | None = None,
                               numbins: int | None = None) -> Index:
        # TODO This is so ugly
        if not only_one_not_none(bins, factor, binwidth, numbins):
            raise ValueError("Either 'bins', 'factor', `numbins` or 'binwidth' must be"
                             " specified, but not more than one.")
        if binwidth is not None:
            binwidth = self.to_same_unit(binwidth)

        if factor is not None:
            if factor <= 0:
                raise ValueError("`factor` must be positive")
            numbins = int(len(self) / factor)

        if numbins is not None:
            if numbins <= 0:
                raise ValueError("`numbins` must be positive")
            bins, step = np.linspace(self.bins[0], self.bins[-1], num=numbins, retstep=True)
        if binwidth is not None:
            if binwidth <= 0:
                raise ValueError("`binwidth` must be positive")
            bins = np.arange(self.bins[0], self.bins[-1], binwidth, dtype=float)
        if isinstance(bins, Index):
            warn("Might be buggy")
            return bins.__clone(meta=self.meta)
        if len(bins) > len(self.bins):
            raise ValueError("Cannot rebin to a finer binning.")
        if not np.isclose(bins[-1], self.bins[-1]) or not np.isclose(bins[0], self.bins[0]):
            warn("The rebinning resizes the index.")
        if not self.is_uniform() and is_uniform(bins):
            warn("Rebinning non-uniform index to uniform index.")
            cls = self.uniform_cls()
            new = cls.from_array(bins=bins)
        else:
            new = self.from_array(bins=bins)
        return new.__clone(meta=self.meta)

    def is_content_close(self, other: Index) -> bool:
        if not np.allclose(self.bins, other.bins):
            return False
        if not np.isclose(self.boundary, other.boundary):
            return False
        return True

    def is_metadata_equal(self, other: Index) -> bool:
        return self.meta == other.meta


class Edge(ABC):
    @abstractmethod
    def left_edge(self, i: int) -> float64:
        """ Left edge of the bin """
        pass

    @abstractmethod
    def right_edge(self, i: int) -> float64:
        """ Right edge of the bin """
        pass

    @abstractmethod
    def mid(self, i: int) -> float64:
        """ Mid point of the bin """
        pass

    @property
    def leftmost(self) -> float64:
        """ Left edge of the leftmost bin """
        return self.left_edge(0)

    @property
    def rightmost(self) -> float64:
        """ Right edge of the rightmost bin """
        return self.right_edge(-1)

    def is_inbounds(self, x: float64) -> bool:
        """ Check if index is in bounds """
        return self.leftmost <= x <= self.rightmost

    @abstractmethod
    def is_left(self) -> bool:
        ...

    @abstractmethod
    def to_left(self) -> Edge:
        ...

    @abstractmethod
    def to_mid(self) -> Edge:
        ...

    def is_mid(self) -> bool:
        return not self.is_left()

    def to_edge(self, edge: Edges) -> Edge:
        match edge.lower():
            case 'left' | 'l':
                return self.to_left()
            case 'mid' | 'm':
                return self.to_mid()
            case _:
                raise ValueError(f"Unknown edge {edge}")

    @abstractmethod
    def other_edge_cls(self) -> Type[Index]:
        ...

    def assert_inbounds(self, x: float64):
        """ Assert that x is in bounds """
        if x < self.leftmost:
            raise IndexError(f"{x} is beyond the leftmost edge {self.leftmost}")
        if x >= self.rightmost:
            raise IndexError(f"{x} is beyond the rightmost edge {self.rightmost}")

    def is_inbounds(self, x) -> bool:
        return self.leftmost <= x < self.rightmost


    @staticmethod
    @abstractmethod
    def extrapolate_boundary(bins: np.ndarray) -> float:
        ...

    @staticmethod
    @abstractmethod
    def split_bins_boundary(bins: np.ndarray) -> tuple[float, np.ndarray]:
        ...

    @abstractmethod
    def ticks(self) -> np.ndarray:
        ...


class Left(Edge):
    def left_edge(self, i: int) -> float64:
        return self.bins[i]

    def right_edge(self, i: int) -> float64:
        return self.bins[i] + self.step(i)

    def mid(self, i: int) -> float64:
        return self.bins[i] + self.step(i) / 2

    def to_left(self) -> Index:
        return self

    def to_mid(self) -> Index:
        cls = self.other_edge_cls()
        bins = self.bins + self.steps() / 2
        return cls(bins, boundary=self.bins[0], metadata=self.meta)

    def _index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_left(self.bins, x)

    def is_left(self) -> bool:
        return True

    @staticmethod
    def extrapolate_boundary(bins: np.ndarray) -> float:
        return bins[-1] + (bins[-1] - bins[-2])

    @staticmethod
    def split_bins_boundary(bins: np.ndarray) -> tuple[float, np.ndarray]:
        return bins[-1], bins[:-1]

    def ticks(self) -> np.ndarray:
        return np.append(self.bins, self.rightmost)


class Mid(Edge):
    @property
    def leftmost(self) -> float64:
        return self.boundary

    def left_edge(self, i: int) -> float64:
        return self.bins[i] - self.step(i) / 2

    def right_edge(self, i: int) -> float64:
        return self.bins[i] + self.step(i) / 2

    def mid(self, i: int) -> float64:
        return self.bins[i]

    def to_mid(self) -> Index:
        return self

    def to_left(self) -> Index:
        cls = self.other_edge_cls()
        bins = self.bins - self.steps() / 2
        boundary = bins[-1] + self.step(-1)
        return cls(bins, boundary=boundary, metadata=self.meta)

    def _index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_mid_uniform(self.bins, x)

    def is_left(self) -> bool:
        return False

    @staticmethod
    def extrapolate_boundary(bins: np.ndarray) -> float:
        return bins[0] - (bins[1] - bins[0]) / 2

    @staticmethod
    def split_bins_boundary(bins: np.ndarray) -> tuple[float, np.ndarray]:
        # Mid boundary is the leftmost *edge*, not bin, hence the shift.
        return bins[0] + (bins[1] - bins[0]) / 2, bins[1:]

    def ticks(self) -> np.ndarray:
        return np.append(np.append(self.leftmost, self.bins), self.rightmost)


class Layout(ABC):
    @abstractmethod
    def steps(self) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, i: int) -> float64:
        ...

    @abstractmethod
    def is_uniform(self) -> bool:
        ...

    @classmethod
    def extrapolate(cls, bins: np.ndarray, direction: Literal['left', 'right'], n: int = 1) -> np.ndarray:
        if direction not in ('left', 'right'):
            raise ValueError(f"Direction must be 'left' or 'right', not '{direction}'.")
        if n < 1:
            if n == 0:
                return bins
            elif n >= -len(bins):
                if direction == 'left':
                    return bins[-n:]
                else:
                    return bins[:n]
            else:
                raise ValueError("`n` must either be > 1 or < len(bins).")

        match direction:
            case 'left':
                return cls.extrapolate_left(bins, n)
            case 'right':
                return cls.extrapolate_right(bins, n)

    @classmethod
    def extrapolate_left(cls, bins: np.ndarray, n: int = 1) -> np.ndarray:
        """ Linear extrapolation to the left.

        Args:
            bins: The bins to extrapolate
            n: The number of elements to extrapolate

        Returns:
            Extrapolated bins
        """
        dx = bins[1] - bins[0]
        extrapolated = np.linspace(bins[0] - n * dx, bins[0], n, endpoint=False)
        return np.concatenate((extrapolated, bins))

    @classmethod
    def extrapolate_right(cls, bins: np.ndarray, n: int = 1) -> np.ndarray:
        """ Linear extrapolation to the right.

        Args:
            bins: The bins to extrapolate
            n: The number of elements to extrapolate

        Returns:
            Extrapolated bins
        """
        dx = bins[-1] - bins[-2]
        extrapolated = np.linspace(bins[-1] + dx, bins[-1] + n * dx, n, endpoint=True)
        return np.concatenate((bins, extrapolated))

    @classmethod
    @abstractmethod
    def uniform_cls(cls) -> Type[Uniform]:
        ...


class Uniform(Layout):
    """ Index for equidistant binning """

    def __init__(self, bins: np.ndarray, *args, **kwargs):
        if len(bins) < 1:
            raise ValueError("Bins must be at least length 1.")
        if not is_uniform(bins):
            raise ValueError("Bins must be uniform.")
        self.dX = bins[1] - bins[0]
        self.__hash = None
        super().__init__(bins, *args, **kwargs)

    def step(self, i: int) -> float64:
        return self.dX

    def steps(self) -> np.ndarray:
        return np.repeat(self.dX, len(self))

    def is_uniform(self) -> bool:
        return True

    def is_congruent(self, other) -> bool:
        if not is_length_congruent(self.bins, other.X):
            return False
        if not other.uniform():
            return False
        dx = self.dX
        dy = other.dX
        if np.isclose(dx, dy) and np.isclose(self[0] % dx, other[0] % dy):
            return True
        return False

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash((self.bins[0], self.dX, len(self), self.boundary, self.unit))
        return self.__hash

    def to_calibration(self) -> Calibration:
        return Calibration([self.bins[0], self.dX])

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.pop('bins')
        d['calibration'] = self.to_calibration().coefficients
        return d

    @classmethod
    def from_dict(cls, d: dict[str, any]) -> Uniform:
        d = d.copy()
        calibration = d.pop('calibration')
        length = d['length']
        d['bins'] = Calibration(calibration).linspace(length)
        return cls._from_dict(d)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def uniform_cls(cls) -> Type[Uniform]:
        return cls


class LeftUniformIndex(Left, Uniform, Index):
    """ Index for left-binning """

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        if not other.is_uniform():
            raise NotImplementedError()
        if not other.is_left():
            other_ = other.to_left()
            new = _rebin_uniform_left_left(self.bins, other_.bins, values, preserve=preserve)
        else:
            new = _rebin_uniform_left_left(self.bins, other.bins, values, preserve=preserve)
        return other, new

    def other_edge_cls(self) -> Type[Index]:
        return MidUniformIndex

    @classmethod
    def extrapolate(cls, bins: np.ndarray, n: int = 1, direction: Direction = 'auto') -> np.ndarray:
        if direction == 'auto':
            direction = 'right'
        return super().extrapolate(bins, n=n, direction=direction)


class MidUniformIndex(Mid, Uniform, Index):
    """ Index for uniform mid-binning """

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        return self.to_left()._rebin(other, values, preserve=preserve)

    def other_edge_cls(self) -> Type[Index]:
        return LeftUniformIndex

    @classmethod
    def extrapolate(cls, bins: np.ndarray, n: int = 1, direction: Direction = 'auto') -> np.ndarray:
        if direction == 'auto':
            direction = 'left'
        return super().extrapolate(bins, n=n, direction=direction)


class NonUniform(Layout):
    """ Index for non-equidistant binning """

    def is_uniform(self) -> bool:
        return False

    def step(self, i: int) -> np.float64:
        return self.dX[i]

    def steps(self) -> np.ndarray:
        return self.dX

    def _rebin(self, other: Index, values: np.ndarray, preserve: Preserve = 'counts') -> tuple[Index, np.ndarray]:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, d: dict[str, any]) -> Uniform:
        return cls._from_dict(d)


class LeftNonUniformIndex(Left, NonUniform, Index):
    def __init__(self, bins: np.ndarray, boundary: float, *args, **kwargs):
        Index.__init__(self, bins, boundary, *args, **kwargs)
        self.dX = np.append(np.diff(bins), boundary - bins[-1])

    def _index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_left(self.bins, x)

    def other_edge_cls(self) -> Type[Index]:
        return MidNonUniformIndex

    @classmethod
    def uniform_cls(cls) -> Type[Uniform]:
        return LeftUniformIndex

    @classmethod
    def extrapolate(cls, bins: np.ndarray, n: int = 1, direction: Direction = 'auto') -> np.ndarray:
        if direction == 'auto':
            direction = 'right'
        return super().extrapolate(bins, n=n, direction=direction)


@njit
def widths_from_mid_left(mid, left):
    if left > mid[0]:
        raise ValueError("Unsolvable constraints: left boundary is smaller than first mid-point.")
    steps = np.empty(len(mid))
    steps[0] = abs(2 * (mid[0] - left))
    if steps[0] == 0:
        raise ValueError("Unsolvable constraints: first mid-point is equal to left boundary.")
    for i in range(1, len(mid)):
        delta = abs(mid[i] - left)
        left += steps[i - 1]
        steps[i] = abs(2 * (delta - steps[i - 1]))
        if steps[i] == 0:
            raise ValueError("Unsolvable constraints: mid-point equal to left edge.")
    return steps


@njit
def widths_from_mid_right(mid, right):
    """ Bin widths from mid-binned bins and the right edge of the final bin"""
    if right < mid[-1]:
        raise ValueError("Unsolvable constraints: right boundary is smaller than last mid-point.")
    steps = np.empty_like(mid)
    steps[-1] = 2 * (right - mid[-1])
    if steps[-1] == 0:
        raise ValueError("Unsolvable constraints: last mid-point is equal to right boundary.")
    i = len(mid) - 2
    while i >= 0:
        eta = mid[i + 1] - mid[i]
        steps[i] = 2 * eta - steps[i + 1]
        if steps[i] == 0:
            raise ValueError("Unsolvable constraints: mid-point equal to right edge.")
        i -= 1
    return steps


class MidNonUniformIndex(Mid, NonUniform, Index):
    def __init__(self, bins: np.ndarray, boundary: float,
                 direction: Literal['left', 'right'] = 'left',
                 *args, **kwargs):
        match direction:
            case 'left':
                self.dX = widths_from_mid_left(bins, boundary)
            case 'right':
                self.dX = widths_from_mid_right(bins, boundary)
                boundary = bins[0] - self.dX[0] / 2
            case d:
                raise ValueError(f"direction must be 'left' or 'right', not {d}")
        Index.__init__(self, bins, boundary, *args, **kwargs)

    def other_edge_cls(self) -> Type[Index]:
        return LeftNonUniformIndex

    @classmethod
    def uniform_cls(cls) -> Type[Uniform]:
        return MidUniformIndex

    # @classmethod
    # def extrapolate(cls, bins: np.ndarray, n: int = 1, direction: Direction = 'auto') -> np.ndarray:
    #    raise NotImplementedError("Extrapolation poorly defined for non-uniform mid-binning and is not supported.")

    @staticmethod
    def extrapolate_boundary(bins: np.ndarray, direction: Literal['left', 'right'] = 'left') -> float:
        match direction:
            case 'left':
                return bins[0] - (bins[1] - bins[0]) / 2
            case 'right':
                return bins[-1] + (bins[-1] - bins[-2]) / 2
            case d:
                raise ValueError(f"direction must be 'left' or 'right', not {d}")

    @staticmethod
    def split_bins_boundary(bins: np.ndarray) -> tuple[float, np.ndarray]:
        raise NotImplementedError("Extrapolation poorly defined for non-uniform mid-binning and is not supported.")

    @classmethod
    def from_array(cls, bins: np.ndarray, *,
                   extrapolate_boundary: bool = False,
                   boundary: float | None = None,
                   width: float | None = None,
                   direction: Literal['left', 'right'] = 'left', **kwargs) -> MidNonUniformIndex:
        """ Construct a MidNonUniformIndex from an array of bin edges.

        Mid bins lack a degree of freedom which must be specified. This can be done in three ways:
        Specify the left edge of either the first or the last bin, through `boundary` and `direction`.
        Specify the width of the first or last bin, through `width` and `direction`.
        Assume that the width of the first bin is equal to the second, or that the width of the last
        bin is equal to the second to last, through `extrapolate_boundary` and `direction`.
        The default behaviour is to let the user resolve this ambiguity. If in doubt, try a left extrapolation.
        """
        if extrapolate_boundary:
            if boundary is not None or width is not None:
                raise ValueError("Cannot specify `extrapolate_boundary` along with `boundary` or `width`.")
            boundary = cls.extrapolate_boundary(bins, direction=direction)
            return cls(bins, boundary, direction=direction, **kwargs)
        match boundary, width, direction:
            case None, None, _:
                raise ValueError("Either `boundary` or `width` must be specified")
            case [a, b, _] if a is not None and b is not None:
                raise ValueError("Only one of `boundary` or `width` can be specified")
            case [_, _, c] if c not in {'left', 'right'}:
                raise ValueError(f"Direction must be `left` or `right`, not `{c}`")
            case bound, _, dir if bound is not None:
                return cls(bins, bound, direction=dir, **kwargs)
            case _, width, 'left':
                boundary = bins[0] - width / 2
                return MidNonUniformIndex(bins, boundary, direction='left', **kwargs)
            case _, width, 'right':
                boundary = bins[-1] + width / 2
                return MidNonUniformIndex(bins, boundary, direction='right', **kwargs)
        raise RuntimeError("Unreachable D:")

    def __getitem__(self, key: int | slice) -> float | Index:
        match key:
            case slice():
                if key.start is None or key.start == 0:
                    boundary = self.boundary
                else:
                    boundary = self.bins[key.start - 1] + self.dX[key.start - 1] / 2
                return self.from_array(self.bins[key], metadata=self.meta,
                                       boundary=boundary)
            case _:
                return self.bins[key]

    def _index(self, x: float64) -> int:
        self.assert_inbounds(x)
        return _index_mid_nonuniform(self.bins, self.steps(), x)


def to_index(X: np.ndarray, edge: Edges = 'left',
             boundary: bool = False, **kwargs) -> Index:
    X = np.asarray(X)
    if edge not in {'left', 'mid'}:
        raise ValueError(f"`edge` must be on of {Edges} not {edge}")
    if not is_monotone(X):
        raise ValueError("Indices must be monotone")
    if is_uniform(X):
        if edge == 'left':
            return LeftUniformIndex.from_array(X, extrapolate_boundary=not boundary, **kwargs)
        else:
            return MidUniformIndex.from_array(X, extrapolate_boundary=not boundary, **kwargs)
    else:
        if edge == 'left':
            return LeftNonUniformIndex.from_array(X, extrapolate_boundary=not boundary, **kwargs)
        else:
            return MidNonUniformIndex.from_array(X, extrapolate_boundary=not boundary, **kwargs)


def make_or_update_index(X: Index | np.ndarray, unit: Unit, alias: str, label: str,
                         default_label: bool,
                         default_unit: bool,
                         edge: Edges = 'left',
                         boundary: bool = False, **kwargs) -> Index:
    if isinstance(X, Index):
        # Preserve the label if it exists, but overwrite if a new label is specified
        if X.meta.label and default_label:
            label = None
        if default_unit:
            unit = X.meta.unit
        if alias is None or alias == '':
            alias = X.meta.alias
        return X.update_metadata(unit=unit, alias=alias, label=label)
    else:
        return to_index(X, edge=edge, boundary=boundary, label=label,
                        unit=unit, alias=alias, **kwargs)


def preparse(s: Unitlike | None) -> (bool, bool, Unitlike | None):
    lesser = greater = False
    if isinstance(s, str):
        s = s.strip()
        if s[0] == '<':
            lesser = True
            s = s[1:]
        elif s[0] == '>':
            greater = True
            s = s[1:]

    return (lesser, greater, s)


def _from_dict(d: dict[str, any]) -> Index:
    match d['class_name']:
        case 'LeftUniformIndex':
            return LeftUniformIndex.from_dict(d)
        case 'MidUniformIndex':
            return MidUniformIndex.from_dict(d)
        case 'LeftNonUniformIndex':
            return LeftNonUniformIndex.from_dict(d)
        case 'MidNonUniformIndex':
            return MidNonUniformIndex.from_dict(d)
        case _:
            raise ValueError(f"Unknown class name {d['class_name']}")


def compress(x):
    return [(k, len(list(g))) for k, g in groupby(x)]
