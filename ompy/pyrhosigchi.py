import numpy as np
from typing import Tuple, Sequence
from .matrix import Matrix
from .vector import Vector
from .rhosigchi import iterate


def find_limits(mat: Matrix,
                gain: float,
                shift: float,
                Ex_min: float,
                Ex_max: float,
                Eg_min: float) -> Tuple[int, int, int, Sequence[int], int]:
    """ This function will find the indices of the limits.
    Args:
        mat: Matrix to find the indices limits on (ndarray).
        gain: Energies are given by gain*i + shift
        shift: Energies are given by gain*i + shift
        Ex_min: Minimum energy to include on excitation axis
        Ex_max: Maximum energy to include on excitation axis
        Eg_min: Minimum energy to include on gamma axis
    Returns: A tuple with min Ex index, max Ex index (+1),
        min Eg index, a list of Eg max indices and the final energy offset.
    """
    imin = int(np.ceil((Ex_min - shift)/gain))
    imax = int(np.floor((Ex_max - shift)/gain)) + 1
    igmin = int(np.ceil((Eg_min - shift)/gain))
    igmax = np.ones(mat.shape[0], dtype=int) * igmin
    for i in range(imin, imax):
        for j in range(igmin, mat.shape[1]):
            if mat[i, j] > 0:
                igmax[i] = j + 1
    u0 = 0
    for i in range(imin, imax):
        for j in range(igmin, igmax[i]):
            if j - i > u0:
                u0 = j - i
    return imin, imax, igmin, igmax, u0


def normalize(mat: Matrix, std: Matrix, imin: int, imax: int,
              igmin: int, igmax: Sequence[int]) -> Tuple[Matrix, Matrix]:
    """ Normalize a matrix to 1 for each excitation bin within the limits.
    Args:
        mat: Matrix with data points.
        std: Uncertanty matrix.
        imin: Lowest bin on excitation axis to normalize.
        imax: Highest bin (+ 1) on excitation axis to normalize.
        igmin: Lowest gamma bin to include in normalization.
        igmax: List of highest gamma bin to include in normalization
            for each excitation bin.
    Returns: Normalized matrix and uncertanties.
    """
    res, err = np.zeros(mat.shape), np.zeros(mat.shape)
    for i in range(imin, imax):
        norm = np.sum(mat[i, igmin:igmax[i]], keepdims=True)
        res[i, igmin:igmax[i]] = mat[i, igmin:igmax[i]]/norm
        err[i, igmin:igmax[i]] = (std[i, igmin:igmax[i]] *
                                  (1 - res[i, igmin:igmax[i]])/norm)**2
        for j in range(igmin, igmax[i]):
            for k in range(igmin, igmax[i]):
                if k == j:
                    continue
                err[i, j] = (std[i, k]*res[i, j]/norm)**2
    return res, std


def normalize_array(mat: np.ndarray, imin: int, imax: int, igmin: int,
                    igmax: int) -> np.ndarray:
    """ Normalize a numpy array within the limits.
    Args:
        mat: Numpy array to normalize
        imin: Index of minimum bin on first axis to normalize.
        imax: Index (+1) of last bin on first axis to normalize.
        igmin: Index of minimum bin on second axis to include
            in normalization.
        igmax: List of indices of maximum bin (+1) on second axis to include
            in normalization.
    Returns: Normalized array
    """
    res = np.zeros(mat.shape)
    for i in range(imin, imax):
        norm = np.sum(mat[i, igmin:igmax[i]], keepdims=True)
        res[i, igmin:igmax[i]] = mat[i, igmin:igmax[i]]/norm
    return res


def pad_matrix(matrix: Matrix, N: int) -> Matrix:
    """ Takes an OMpy matrix and pads top and left hand side
    with N zeros. It then extends the Ex and Eg to these
    energies.
    Args:
        matrix: Matrix to pad
        N: Number of bins to pad with
    Returns: The padded matrix
    """
    if not np.allclose(matrix.Ex, matrix.Eg):
        raise ValueError("The matrix has different binning on Ex and Eg axis")
    gain, shift = matrix.Ex[1] - matrix.Ex[0], matrix.Ex[0]
    values = np.pad(matrix.values, ((N, 0), (N, 0)), "constant")
    E = np.concatenate((-np.flip(np.arange(1, N+1))*gain + shift, matrix.Ex))
    return Matrix(values=values, Ex=E, Eg=E)


def pyrhosigchi(first_gen: Matrix, first_gen_std: Matrix, Ex_min: float,
                Ex_max: float, Eg_min: float, nit: int=101, seed: int=4848):
    """ Reimplementation of rhosigchi in python/C++.
    Args:
        first_gen: First generation matrix
        first_gen_std: Standard diviation of the first generation matrix
        Ex_min: Minimum excitation energy
        Ex_max: Maximum excitation energy
        Eg_min: Minimum gamma energy
        nit: Number of iterations
    Returns: Extracted NLD and transmission coefficients
    """

    # First, we check that the two matrices have the same shape
    if first_gen.shape != first_gen_std.shape:
        raise ValueError("first_gen and first_gen_std have different shapes.")

    # Ensure equal binning in each axis
    if not np.allclose(first_gen.Ex, first_gen.Eg):
        raise ValueError("first_gen have different binning on Ex and Eg axis.")

    # Next ensure the same binning
    if not np.allclose(first_gen.Ex, first_gen_std.Ex):
        raise ValueError("first_gen and first_gen_std have different binning.")

    # Extract gain and shift
    gain, shift = (first_gen.Ex[1] - first_gen.Ex[0]), first_gen.Ex[0]

    # And the initial limits
    imin, imax, igmin, igmax, u0 = find_limits(first_gen, gain, shift,
                                               Ex_min, Ex_max, Eg_min)

    # Pad the matrices
    first_gen = pad_matrix(first_gen, u0)
    first_gen_std = pad_matrix(first_gen_std, u0)

    # Find the new limits
    imin, imax, igmin, igmax, u0 = find_limits(first_gen, gain, shift,
                                               Ex_min, Ex_max, Eg_min)

    # Next we will normalize
    FgN, sFgN = normalize(first_gen.values, first_gen_std.values,
                          imin, imax, igmin, igmax)

    # Setup data and the initial guess
    nld = np.zeros((nit, imax - igmin - u0))
    gsf = np.zeros((nit, max(igmax)))
    nld[0, :] = 1
    for i in range(imin, imax):
        for j in range(igmin, igmax[i]):
            gsf[0, j] += FgN[i, j]

    # Now we can apply the iteration method
    iterate(FgN, sFgN, nld, gsf, imin, imax, igmin, igmax, u0, nit)

    # Next we will do the error analysis
    # First we will perturb the FG and normalize
    rng = np.random.default_rng(seed)
    first_gen_perturbed = first_gen.copy()
    first_gen_perturbed.values += \
        rng.normal(size=first_gen.shape)*first_gen_std.values
    Fgv, _ = normalize(first_gen_perturbed.values, first_gen_std.values,
                       imin, imax, igmin, igmax)

    # Next we will now set our initial guess to the last NLD and GSF
    nldv = np.zeros((nit, imax - igmin - u0))
    gsfv = np.zeros((nit, max(igmax)))
    nldv[0, :] = nld[0, :]
    for i in range(imin, imax):
        for j in range(igmin, igmax[i]):
            gsfv[0, j] += Fgv[i, j]

    # Next, perform the iteration once more
    iterate(Fgv, sFgN, nldv, gsfv, imin, imax, igmin, igmax, u0, nit)

    # From this we can now get uncertanties
    nlderr = np.sqrt(np.sum((nldv - nld[-1, :][np.newaxis])**2, axis=0))/nit
    gsferr = np.sqrt(np.sum((gsfv - gsf[-1, :][np.newaxis])**2, axis=0))/nit

    nld = Vector(E=np.arange(imax-igmin-u0)*gain+shift,
                 values=nld[-1, :], std=nlderr)
    gsf = Vector(E=np.arange(max(igmax))*gain+shift,
                 values=gsf[-1, :], std=gsferr)
    return nld, gsf
