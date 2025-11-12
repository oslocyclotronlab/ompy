from __future__ import annotations
from .detector import EgDetector, ExDetector, CompoundDetector
from .oscar import OSCAR, ResponseDetector
from .cactus import CACTUS
from .siri import SiRi
from ..response import ResponseMatrices, Components
from ..array import Vector, Matrix
from typing import Self, overload
from ..stubs import Unitlike


class Oslo(CompoundDetector):
    """Base class for Oslo detector combinations."""
    
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

    @classmethod
    def from_db(cls, name: str) -> OsloOscar | OsloCactus:
        """Factory method that returns the appropriate Oslo subclass.
        
        Args:
            name: Detector name (e.g., 'OSCAR2020', 'CACTUS', etc.)
            
        Returns:
            OsloOscar if name matches OSCAR detector
            OsloCactus if name matches CACTUS detector
        """
        name_upper = name.upper()
        
        if 'OSCAR' in name_upper:
            eg_detector = OSCAR.from_str(name)
            return OsloOscar(eg_detector=eg_detector)
        elif 'CACTUS' in name_upper:
            eg_detector = CACTUS.from_str(name)
            return OsloCactus(eg_detector=eg_detector)
        else:
            # Generic fallback
            eg_detector = ResponseDetector.from_db(name)
            return cls(eg_detector=eg_detector)

    @property
    def siri(self) -> SiRi:
        return self.ex_detector

    def specialize_like(
        self, array: Vector | Matrix, *, drop_eye: bool = True,
        components: Components | None = None
    ) -> tuple[Matrix, ResponseMatrices]:
        return self.siri.specialize_like(
            array, drop_eye=drop_eye
        ), self.eg_detector.specialize_like(array, drop_eye=drop_eye, components=components)

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
        if inplace:
            self.siri.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            siri = self.siri.normalize_sigma(energy, sigma)
            return self.clone(ex_detector=siri)

    def __str__(self) -> str:
        return f"OSLO with eg detector {self.eg_detector} and ex detector {self.ex_detector}"


class OsloOscar(Oslo):
    """Oslo detector combination with OSCAR as the Eg detector."""
    
    def __init__(
        self,
        eg_detector: OSCAR | None = None,
        ex_detector: ExDetector | None = None,
    ):
        if eg_detector is None:
            eg_detector = OSCAR.from_default()
        if ex_detector is None:
            ex_detector = SiRi()
        super().__init__(eg_detector=eg_detector, ex_detector=ex_detector)

    @property
    def oscar(self) -> OSCAR:
        """Access the OSCAR detector."""
        return self.eg_detector

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
        """Normalize OSCAR detector sigma at a specific energy."""
        if inplace:
            self.oscar.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            oscar = self.oscar.normalize_sigma(energy, sigma, inplace=inplace)
            return self.clone(eg_detector=oscar)

    def efficiency_like(self, array: Vector | Matrix) -> Vector:
        return self.oscar.efficiency_like(array)


class OsloCactus(Oslo):
    """Oslo detector combination with CACTUS as the Eg detector."""
    
    def __init__(
        self,
        eg_detector: CACTUS | None = None,
        ex_detector: ExDetector | None = None,
    ):
        if eg_detector is None:
            eg_detector = CACTUS.from_default()
        if ex_detector is None:
            ex_detector = SiRi()
        super().__init__(eg_detector=eg_detector, ex_detector=ex_detector)

    @property
    def cactus(self) -> CACTUS:
        """Access the CACTUS detector."""
        return self.eg_detector

    @overload
    def normalize_cactus_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = True
    ) -> None: ...

    @overload
    def normalize_cactus_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self: ...

    def normalize_cactus_sigma(
        self, energy: Unitlike, sigma: Unitlike, inplace: bool = False
    ) -> Self | None:
        """Normalize CACTUS detector sigma at a specific energy."""
        if inplace:
            self.cactus.normalize_sigma(energy, sigma, inplace=inplace)
        else:
            cactus = self.cactus.normalize_sigma(energy, sigma, inplace=inplace)
            return self.clone(eg_detector=cactus)

    def efficiency_like(self, array: Vector | Matrix) -> Vector:
        return self.cactus.efficiency_like(array)
