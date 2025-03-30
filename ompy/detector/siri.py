from __future__ import annotations
from .detector import ExDetector

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

    def __str__(self) -> str:
        return f"SiRi with FWHM a0={self.a0}"
