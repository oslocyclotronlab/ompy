import ROOT
import numpy as np
from typing import Sequence

class ROOTWriter:
    def write_number(self, label: str, number: int | float) -> None:
        ROOT.TNamed(label, str(number)).Write()

    def write_array_1d(self, label: str, arr: Sequence[np.number]) -> None:
        vec = ROOT.TVectorD(len(arr), label)
        for i, x in enumerate(arr):
            vec[i] = x
        vec.Write()

    def write_string(self, label: str, string: str) -> None:
        ROOT.TNamed(label, string).Write()


