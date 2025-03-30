from __future__ import annotations
from .detector import EgDetector, ExDetector, CompoundDetector, LambdaEgDetector
from .oscar import OSCAR
from .siri import SiRi
from ..response import Response, DiscreteInterpolation, ResponseMatrices
from ..array import Vector, Matrix
from typing import Self, overload
from ..stubs import Unitlike


class Oslo(CompoundDetector):
    def __init__(
        self,
        eg_detector: EgDetector | None = None,
        ex_detector: ExDetector | None = None,
    ):
        if eg_detector is None:
            eg_detector = OSCAR.from_default()
        if ex_detector is None:
            ex_detector = SiRi()
        super().__init__(eg_detector=eg_detector, ex_detector=ex_detector)

    @property
    def oscar(self) -> OSCAR:
        return self.eg_detector

    @property
    def siri(self) -> SiRi:
        return self.ex_detector

    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: bool = True
    ) -> tuple[Matrix, ResponseMatrices]:
        return self.siri.specialize_like(
            array, drop_eye=drop_eye
        ), self.oscar.specialize_like(array, drop_eye=drop_eye)

    @overload
    def normalize_oscar_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = True
    ) -> None: ...

    @overload
    def normalize_oscar_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self: ...

    def normalize_oscar_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self | None:
        if inplace:
            self.oscar.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            oscar = self.oscar.normalize_sigma(energy, sigma, inplace=inplace)
            return self.clone(eg_detector=oscar)

    @overload
    def normalize_siri_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = True
    ) -> None: ...

    @overload
    def normalize_siri_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self: ...

    def normalize_siri_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self | None:
        raise NotImplementedError("Not implemented")
        if inplace:
            self.siri.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            siri = self.siri.normalize_sigma(energy, sigma)
            return self.clone(ex_detector=siri)

    def __str__(self) -> str:
        return f"OSLO with eg detector {self.eg_detector} and ex detector {self.ex_detector}"
