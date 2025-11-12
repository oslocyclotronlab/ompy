from ..detector import CompoundDetector
from ..array import Vector, Index, Matrix
import numpy as np
from ..external.ripl3 import RIPL3Data, RIPL3Record, LevelEntry
import warnings

def _to_array(x) -> np.ndarray:
    match x:
        case Vector():
            return x.to_unit("MeV").to_mid().X
        case Index():
            return x.to_unit("MeV").to_mid()
        case _:
            return x


def _to_levels(x) -> np.ndarray:
    match x:
        case RIPL3Data(levels=levels) | RIPL3Record(levels=levels):
            return _to_levels(levels)
        case [*entries] if entries and isinstance(entries[0], LevelEntry):
            return np.asarray([entry.level.Elv for entry in entries])
        case _:
            return np.asarray(x)


def bin_levels_like(
    levels: np.ndarray | RIPL3Data | RIPL3Record,
    *,
    like: Vector | Index | np.ndarray | None = None,
    G: Matrix | None = None,
    eg: np.ndarray | Vector | Index | None = None,
    detector: CompoundDetector | None = None,
    num_bins: int = 1000,
    smooth: bool = True,
    cut: bool = False,
):
    """
    Bin discrete nuclear levels into a histogram (level density spectrum), with optional smoothing
    using a detector response matrix.

    The function supports binning levels into a fine grid to avoid edge effects, optional smoothing
    (folding) by a detector response matrix, then rebinning/aligning to a target "like" grid.

    Args:
        levels (np.ndarray | RIPL3Data | RIPL3Record): Discrete level energies to be binned (in MeV or convertible).
        like (Vector | Index | np.ndarray | None, optional): Target energy vector or bin midpoints/edges. 
            If provided, the output will be rebinned to match this vector.
        G (Matrix | None, optional): Precomputed response matrix for smoothing (Optional if detector is provided).
        eg (np.ndarray | Vector | Index | None, optional): Gamma energy vector, recommended if building response matrix.
        detector (CompoundDetector | None, optional): Detector used to generate response matrix if G not given.
        num_bins (int, optional): Number of bins for initial fine binning (default 1000).
        smooth (bool, optional): If True, apply smoothing using the response matrix/detector (default True).
        cut (bool, optional): If True, output density is rebinned with sharp cut to match the like vector (default False).

    Returns:
        Vector: Binned and optionally smoothed level density, in units MeV^-1,
            rebinned/aligned to the "like" vector if provided.
    """
    levels = _to_levels(levels)

    # If a like vector is provided, the levels must span the range of the like vector.
    if like is not None:
        arr = _to_array(like)
        # The maximum energy must span both the levels and the vector to bin to.
        ef_max = max(max(levels), arr[-1])
        ef_min = min(min(levels), arr[0])
    else:
        ef_max = max(levels)
        ef_min = min(levels)
    ef_min -= 1
    ef_max += 1

    ef = np.linspace(ef_min, ef_max, num_bins)
    dEf = ef[1] - ef[0]
    bins = np.concatenate([ef, [ef[-1] + dEf]])
    hist, _ = np.histogram(levels, bins=bins)
    
    if smooth:
        if G is None:
            if detector is None:
                raise ValueError("A response matrix was not provided, and a detector was not provided."
                                " You must provide a response matrix, or a detector and a gamma energy vector.")
            if eg is None:
                warnings.warn("Without a gamma energy vector, the response matrix will be calculated with a default vector."
                            "THIS IS PROBABLY NOT WHAT YOU WANT")
                eg = np.linspace(0, ef_max, num_bins)
            # TODO The detector assumes keV, which is a bug. Hence the factor 1e3.
            G = detector.mixture_response(Ef=ef*1e3, Eg=_to_array(eg)*1e3)
        else:
            if eg is not None:
                raise ValueError("A response matrix was provided, but a gamma energy vector was also provided.")
        hist = G @ hist
    else:
        if G is not None:
            raise ValueError("A response matrix was provided, but smoothing was disabled.")
        if eg is not None:
            raise ValueError("A gamma energy vector was provided, but smoothing was disabled.")
        if detector is not None:
            raise ValueError("A detector was provided, but smoothing was disabled.")
        
    levels = Vector(
        E=bins[:-1],
        values=hist,
        name="Smoothed discrete levels" if smooth else "Discrete levels",
        vlabel="density",
        edge="mid",
        unit="MeV",
        vunit="MeV^-1",
    )
    density = levels / dEf
    if like is not None:
        if cut:
            density = density.rebin_like(like, preserve="area")
        else:
            # We use arr instead of like because like may be in another unit.
            width = arr[1] - arr[0]
            density = density.rebin(binwidth=width, preserve="area")
    return density
