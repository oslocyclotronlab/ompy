from __future__ import annotations
from .detector import ExDetector
from typing import Self

class SiRi(ExDetector):
    # From https://doi.org/10.1016/j.nima.2011.05.055
    # says ~= 100 keV
    # depends on experiment
    def __init__(self, a0: float = 100.0, title=''):
        if not title:
            title = f"SiRi response with FWHM {a0}"
        super().__init__(title)
        self.a0 = a0

    def _FWHM(self, e: float) -> float:
        return self.a0
    
    def normalize_sigma(self, energy: float, sigma: float, inplace: bool = False) -> Self | None:
        """Normalize the resolution (FWHM) to match a given sigma at a specific energy.

        Args:
            energy: The energy at which to normalize
            sigma: The target sigma value
            inplace: If True, modify this instance. If False, return a new instance.

        Returns:
            If inplace=False, returns a new SiRi instance with normalized sigma.
            If inplace=True, returns None.
        """
        # Calculate normalization factor
        if inplace:
            self.a0 = sigma
            return None
        else:
            new = self.__class__(a0=sigma)
            return new

    def __str__(self) -> str:
        return f"SiRi with FWHM a0={self.a0}"
