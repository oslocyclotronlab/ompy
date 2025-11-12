from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Type, Dict, Any, Literal, TYPE_CHECKING

from ..result import Result

if TYPE_CHECKING:
    from ...array import Matrix, Vector


class Sampler(ABC):
    """Abstract sampler that knows how to draw components from a result."""

    ndim: ClassVar[int | None] = None
    _registry: ClassVar[Dict[int, type["Sampler"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.ndim is not None:
            Sampler._registry[int(cls.ndim)] = cls

    def __init__(self, result: Result):
        self._result = result

    @property
    def result(self) -> Result:
        return self._result

    @classmethod
    def from_result(cls, result: Result) -> Sampler:
        try:
            subclass = cls._registry[result.ndim]
        except KeyError as exc:
            raise ValueError(f"No sampler registered for ndim={result.ndim}") from exc
        return subclass(result)

    @abstractmethod
    def sample_data(
        self,
        count: int,
        base: Literal["raw", "folded", "nu"] | str = "folded",
    ) -> list[Matrix | Vector]:
        """Draw bootstrap replicas of the observed data."""

    @abstractmethod
    def sample_background(
        self,
        count: int,
        *,
        base: Literal["raw", "beta"] = "beta",
        bootstrap: bool = True,
    ) -> list[Any]:
        """Draw background replicas, returning the structure expected by the unfolder."""

    @abstractmethod
    def sample_total(
        self,
        count: int,
        *,
        base: Literal["folded", "raw"] = "folded",
    ) -> list[Matrix | Vector]:
        """Draw samples of the total predicted spectrum."""
