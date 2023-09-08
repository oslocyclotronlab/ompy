from __future__ import annotations
from dataclasses import dataclass
from ..array import Vector, Matrix, AbstractArray
from ..stubs import Axes, Plot1D, Plot2D, Lines
import matplotlib.pyplot as plt
from typing import TypeAlias
import numpy as np
from abc import ABC, abstractmethod
from .result import Result
from .result1d import UnfoldedResult1D
from .bootstrap import BootstrapVector
from ..helpers import make_combined_legend

@dataclass
class AStudy(ABC):
    name: str
    raw: AbstractArray | None = None
    unfolded: AbstractArray | None = None
    folded: AbstractArray | None = None
    eta: AbstractArray | None = None
    mu: AbstractArray | None = None

    @classmethod
    @abstractmethod
    def from_result(cls, result: Result) -> AStudy: ...


@dataclass
class Study1D(AStudy):
    raw: Vector | None = None
    unfolded: Vector | None = None
    folded: Vector | None = None
    eta: Vector | None = None
    mu: Vector | None = None

    @classmethod
    def from_result(cls, name: str, result: UnfoldedResult1D) -> Study1D:
        return cls(name, raw=result.raw, unfolded=result.best(), eta=result.best_eta(), folded=result.best_folded())

    @classmethod
    def from_bootstrap(cls, name: str, boot: BootstrapVector) -> Study1D:
        return cls(name, eta=boot.eta(), folded=boot.nu(), mu=boot.mu())


@dataclass
class Study2D(AStudy):
    raw: Matrix | None = None
    unfolded: Matrix | None = None
    folded: Matrix | None = None
    eta: Matrix | None = None
    mu: Matrix | None = None

Study: TypeAlias = Study1D | Study2D

@dataclass
class StudyGroup:
    studies: list[Study]

    def plot(self, ax: Axes | None = None,
             raw: bool = True,
             unfolded: bool = True,
             folded: bool = True, mu: bool = True,
             eta: bool = True) -> Plot1D | Plot2D:
        if ax is None:
            n = int(raw or folded) + int(unfolded or eta or mu)
            _, ax = plt.subplots(ncols=n, sharex=True, constrained_layout=True,
                                 figsize=(9, 5))
        ax = ax.ravel() if isinstance(ax, np.ndarray) else np.array([ax])
        assert isinstance(ax, np.ndarray)
        lines: list[Lines] = []
        i = 0
        # Raw and folded plot together
        if raw:
            for study in self.studies:
                if study.raw is not None:
                    _, line = study.raw.plot(ax=ax[i], label=study.name + ' raw')
                    lines.append(line)
        if folded:
            for study in self.studies:
                if study.folded is not None:
                    _, line = study.folded.plot(ax=ax[i], label=study.name + ' folded')
                    lines.append(line)
        make_combined_legend(ax[i], [], *lines)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        i = int(raw or folded)
        if unfolded:
            for study in self.studies:
                if study.unfolded is not None:
                    study.unfolded.plot(ax=ax[i], label=study.name)
        if eta:
            for study in self.studies:
                if study.eta is not None:
                    study.eta.plot(ax=ax[i], label=study.name)
        if mu:
            for study in self.studies:
                if study.mu is not None:
                    study.mu.plot(ax=ax[i], label=study.name)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        fig = ax[0].figure
        fig.supxlabel('Energy [keV]')
        return ax, lines

