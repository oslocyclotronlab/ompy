from __future__ import annotations
from .detector import EgDetector
from ..response import Response
from typing import Literal, Self, overload
from ..stubs import Unitlike
from ..stubs import array as Array
from ..array import Vector, Index, Matrix


class ResponseDetector(EgDetector):
    # From https://doi.org/10.1016/j.nima.2020.164678
    #a0 = 60.6473
    #a1 = 0.45802
    #a2 = 2.655517e-4

    def __init__(self, response: Response, title: str = ""):
        super().__init__(title=title)
        self.response = response

    def _FWHM(self, e: float) -> float:
        return self.response.interpolation.FWHM(e)

    @classmethod
    def from_db(cls, name: str) -> Self:
        response = Response.from_db(name)
        return cls(response)

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[False]) -> Self: ...

    @overload
    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: Literal[True]) -> None: ...

    def normalize_FWHM(self, energy: Unitlike, fwhm: Unitlike, inplace: bool = False) -> Self | None:
        if inplace:
            self.response.normalize_FWHM(energy, fwhm, inplace=inplace)
        else:
            return self.clone(response=self.response.normalize_FWHM(energy, fwhm, inplace=inplace))

    @overload
    def normalize_sigma(self, energy: Unitlike, sigma: Unitlike, inplace: Literal[False]) -> Self: ...

    @overload
    def normalize_sigma(self, energy: Unitlike, sigma: Unitlike, inplace: Literal[True]) -> None: ...

    def normalize_sigma(self, energy: Unitlike, sigma: Unitlike, inplace: bool = False) -> Self | None:
        if inplace:
            self.response.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            return self.clone(response=self.response.normalize_sigma(energy, sigma, inplace=inplace))

    @staticmethod
    def implements_discrete_response() -> bool:
        return True

    def _discrete_response(self, E: Index | Array, **kwargs) -> Matrix:
        return self.response.discrete_(E=E, **kwargs)

    def clone(self, *, response: Response | None = None, copy: bool = False, title: str | None = None) -> Self:
        if response is None:
            response = self.response

        if copy:
            response = response.copy()
        title = self.title if title is None else title
        return self.__class__(response, title=title)

    def efficiency_like(self, array: Vector | Matrix) -> Vector:
        return self.response.efficiency_like(array)