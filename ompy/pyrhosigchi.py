import numpy as np
from typing import Tuple, Sequence, Union
from .matrix import Matrix
from .vector import Vector
from .rhosigchi import iterate, Finv_vec


def find_limits(mat: Matrix, Ex_min: float, Ex_max: float, Eg_min: float
                ) -> Tuple[int, int, int, Sequence[int], int]:
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
    gain, shift = mat.Ex[1]-mat.Ex[0], mat.Ex[0]
    imin = int(np.ceil((Ex_min - shift)/gain))
    imax = int(np.floor((Ex_max - shift)/gain)) + 1
    igmin = int(np.ceil((Eg_min - shift)/gain))
    igmax = np.ones(mat.shape[0], dtype=int) * igmin
    for i in range(imin, imax):
        for j in range(igmin, mat.shape[1]):
            if mat[i, j] > 0:
                igmax[i] = j + 1
    u0 = int(abs(shift/gain) + 0.5)
    return imin, imax, igmin, igmax, u0


def normalize(mat: np.ndarray, std: np.ndarray, imin: int, imax: int,
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
    res, err = np.zeros(mat.shape), np.zeros(std.shape)
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
                    igmax: Sequence[int]) -> np.ndarray:
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


def iterate_python(FgN: np.ndarray, sFgN: np.ndarray,
                   Rho0: np.ndarray, Sig0: np.ndarray,
                   jmin: int, jmax: int, igmin: int,
                   igmax: Sequence[int], iu0: int, nit: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """ Perform the iterative ChiSq procedure to obtain the level density
    and transmission coefficients.
    Args:
        FgN: Numpy array with normalized first generation spectra. Normalized
            to 1 in each excitation bin, between gamma limits
        sFgN: Numpy array with normalized std of the first generation spectra.
        Rho0: A jmax-igmin+iu0 array containing initial guess for the nuclear
            level density.
        Sig0: A jmax+iu0 array containing the initial guess for the
            transmission coefficients.
        jmin: Lower bin number on the excitation axis to include in the fit
        jmax: Upper bin number + 1 on the excitation axis to include in the fit
        igmin: Lower bin number on the gamma axis to include in the fit
        igmax: Array with the upper gamma bin to include for each
            excitation bin
        iu0: Excitation offset (should be int(abs(gain/shift) + 0.5))
        nit: Number of iterations.
    Returns:
        rho: A matrix with nuclear level density on the second axis
            and the result of the ith iteration on the first.
        sig: A matrix with transmission coefficients on the second axis
            and the result of the ith iteration on the first.
    """
    fun1 = np.zeros(jmax)
    fun2 = np.zeros(jmax)
    fun3 = np.zeros(jmax)
    max_idx = max([max(igmax), jmax+iu0])
    nom = np.zeros((max_idx, max_idx))
    denom = np.zeros((max_idx, max_idx))
    Rho, Sig = np.zeros((nit, jmax-igmin+iu0)), np.zeros((nit, max_idx))
    Rho[0, :] = Rho0
    Sig[0, :] = Sig0
    var = 1.2
    for it in range(1, nit):
        if it <= 5:
            var = 1.2
        elif it <= 12:
            var = 1.1
        elif it <= 21:
            var = 1.05
        elif it <= 30:
            var = 1.025
        else:
            var = 1.01
        for ix in range(jmin, jmax):
            fun1[ix] = 0
            fun2[ix] = 0
            fun3[ix] = 0
            for ig in range(igmin, igmax[ix]):
                iu = ix - ig + iu0
                fun1[ix] += Sig[it-1, ig]*Rho[it-1, iu]
                if sFgN[ix, ig] > 0:
                    sr = Sig[it-1, ig]*Rho[it-1, iu]
                    fun2[ix] += (sr/sFgN[ix, ig])**2
                    fun3[ix] += sr*FgN[ix, ig]/sFgN[ix, ig]**2
            if fun1[ix] > 0:
                fun2[ix] /= fun1[ix]**3
                fun3[ix] /= fun1[ix]**2
            else:
                fun2[ix] = 0
                fun3[ix] = 0
            for ig in range(igmin, igmax[ix]):
                if fun1[ix]*sFgN[ix, ig] > 0:
                    nom[ix, ig] = fun2[ix] - fun3[ix]
                    nom[ix, ig] += FgN[ix, ig]/(fun1[ix]*sFgN[ix, ig]**2)
                    denom[ix, ig] = 1/(fun1[ix]*sFgN[ix, ig])**2
                else:
                    nom[ix, ig] = 0
                    denom[ix, ig] = 0

        for ig in range(igmin, max(igmax)):
            up = 0
            down = 0
            ii = max([jmin, ig])
            for ix in range(ii, jmax):
                iu = ix - ig + iu0
                up += Rho[it-1, iu]*nom[ix, ig]
                down += Rho[it-1, iu]*Rho[it-1, iu]*denom[ix, ig]
            if down > 0:
                if up/down > var*Sig[it-1, ig]:
                    Sig[it, ig] = var*Sig[it-1, ig]
                elif up/down < Sig[it-1, ig]/var:
                    Sig[it, ig] = Sig[it-1, ig]/var
                else:
                    Sig[it, ig] = up/down
            else:
                Sig[it, ig] = 0
        for iu in range(0, jmax-igmin+iu0):
            up = 0
            down = 0
            ii = max([jmin, iu])
            for ix in range(ii, jmax):
                ig = ix - iu + iu0
                up += Sig[it-1, ig]*nom[ix, ig]
                down += Sig[it-1, ig]*Sig[it-1, ig]*denom[ix, ig]
            if down > 0:
                if up/down > var*Rho[it-1, iu]:
                    Rho[it, iu] = var*Rho[it-1, iu]
                elif up/down < Rho[it-1, iu]/var:
                    Rho[it, iu] = Rho[it-1, iu]/var
                else:
                    Rho[it, iu] = up/down
            else:
                Rho[it, iu] = 0
    return Rho, Sig


def chisquare(FgN: np.ndarray, sFgN: np.ndarray,
              Rho: np.ndarray, Sig: np.ndarray,
              sRho: np.ndarray, sSig: np.ndarray,
              imin: int, imax: int,
              igmin: int, igmax: Sequence[int], iu0: int) -> np.ndarray:
    """ Find the ChiSq based on the errors in first generation matrix
    and the extracted NLD and GSF.
    Args:
        FgN: Normalized first generation matrix
        sFgN: Errors, normalized first generation matrix
        rho: All iterations of the NLD
        sig: All iterations of the GSF
        imin: Minimum Ex index
        imax: Maximum Ex index + 1
        igmin: Minimum Eg index
        igmax: Maximum Eg index for each Ex bin
        iu0: NLD offset
    Returns: The chi square value for each iteration.
    """

    degrees = 0
    for ix in range(imin, imax):
        degrees += igmax[ix] - igmin
    degrees -= (imax - imin + 1) + (max(igmax) - igmin + 1)
    chiSq = np.zeros(Rho.shape[0])
    for it in range(Rho.shape[0]):
        deg = degrees
        fg_teo = np.zeros(FgN.shape)
        sumFG = np.zeros(imax)
        for ix in range(imin, imax):
            for ig in range(igmin, igmax[ix]):
                iu = ix - ig + iu0
                sumFG[ix] += Sig[it, ig]*Rho[it, iu]
            for ig in range(igmin, igmax[ix]):
                iu = ix - ig + iu0
                if sumFG[ix] > 0:
                    fg_teo[ix, ig] = Sig[it, ig]*Rho[it, iu]/sumFG[ix]
                else:
                    fg_teo[ix, ig] = 0
                sRhoT = 0
                sAll = 0
                if fg_teo[ix, ig] > 0:
                    sRhoT = fg_teo[ix, ig]*np.sqrt((sRho[iu]/Rho[it, iu])**2 +
                                                   (sSig[ig]/Sig[it, ig])**2)
                if sFgN[ix, ig] > 0 and sRhoT > 0:
                    sAll = np.sqrt(sFgN[ix, ig]**2 + sRhoT**2)
                    chiSq[it] += ((fg_teo[ix, ig] - FgN[ix, ig])/sAll)**2
                else:
                    deg -= 1
            if deg > 0:
                chiSq[it] /= deg
    return chiSq


def pyrhosigchi(first_gen: Matrix, first_gen_std: Matrix, Ex_min: float,
                Ex_max: float, Eg_min: float, nit: int = 51, nunc: int = 100,
                extend_diagnoal: float = 800,
                rng: np.random.Generator = np.random.default_rng(4821),
                diagnostics: bool = False
                ) -> Union[Tuple[Vector, Vector], Tuple[Vector, Vector, dict]]:
    """ Reimplementation of rhosigchi in python.
    Args:
        first_gen: First generation matrix
        first_gen_std: Standard diviation of the first generation matrix
        Ex_min: Minimum excitation energy
        Ex_max: Maximum excitation energy
        Eg_min: Minimum gamma energy
        nit: Number of iterations. Default value is 51 (same as rhosigchi)
        extend_diagnoal: Energy to extend the diagonal to ensure all counts
            are included (as the detector has a resolution)
        rng: Random number generator
        diagnostics: If additional diagnostics should be returned.
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

    # Don't really know why this works, but this is how we get the
    # correct offsets...
    a1, aEx0 = first_gen.Ex[1]-first_gen.Ex[0], first_gen.Ex[0]
    a0 = (aEx0/a1 - int(aEx0/a1))*a1
    a0 = a0 - int(extend_diagnoal/a1 + 0.5)*a1

    # We will rebin with the new binning we found
    first_gen.rebin(axis="both",
                    mids=np.arange(a0, first_gen.Ex[-1]+a1, a1),
                    inplace=True)
    first_gen_std.rebin(axis="both",
                        mids=np.arange(a0, first_gen.Ex[-1]+a1, a1),
                        inplace=True)

    igmin = int((Eg_min - a0)/a1 + 0.5)
    imin = int((Ex_min - a0)/a1 + 0.5)
    imax = int((Ex_max - a0)/a1 + 0.5) + 1
    igmax = np.zeros(imax, dtype=int)
    for ix in range(imin, imax):
        for ig in range(igmin, len(first_gen.Eg)):
            if first_gen.values[ix, ig] > 0 and igmax[ix] < ig+1:
                igmax[ix] = ig+1
    u0 = int(abs(a0/a1) + 0.5)

    # Next we will normalize
    FgN, sFgN = normalize(first_gen.values, first_gen_std.values,
                          imin, imax, igmin, igmax)

    # Setup data and the initial guess
    nld = np.zeros((nit, imax - igmin + u0))
    gsf = np.zeros((nit, imax + u0))
    nld[0, :] = 1
    for ix in range(imin, imax):
        for ig in range(igmin, igmax[ix]):
            gsf[0, ig] += FgN[ix, ig]

    # Now we can apply the iteration method
    iterate(FgN, sFgN, nld, gsf, imin, imax, igmin, igmax, u0, nit)

    # Next we will do the error analysis
    nlderr = np.zeros(imax - igmin + u0)
    gsferr = np.zeros(imax + u0)
    nldv = np.zeros((nit, imax-igmin+u0))
    gsfv = np.zeros((nit, imax+u0))
    nldv[0, :] = nld[0, :]

    for i in range(nunc):
        first_gen_perturbed = first_gen.values + \
            Finv_vec(rng.random(size=first_gen.shape)) \
            * first_gen_std.values
        first_gen_perturbed[first_gen_perturbed < 0] = 0
        Fgv = normalize_array(first_gen_perturbed, imin, imax,
                              igmin, igmax)
        gsfv[0, :] = 0
        for ix in range(imin, imax):
            for ig in range(igmin, igmax[ix]):
                gsfv[0, ig] += Fgv[ix, ig]

        # Next, we will calculate the uncertanties
        iterate(Fgv, sFgN, nldv, gsfv,
                imin, imax, igmin, igmax, u0, nit)

        # From this we can now get uncertanties
        nlderr += (nldv[-1] - nld[-1])**2
        gsferr += (gsfv[-1] - gsf[-1])**2

    # Final errors
    nlderr = np.sqrt(nlderr/nunc)
    gsferr = np.sqrt(gsferr/nunc)

    chiSq = chisquare(FgN, sFgN, nld, gsf, nlderr, gsferr,
                      imin, imax, igmin, igmax, u0)

    diag = {'ChiSq': chiSq.copy(), 'rho': nld.copy(), 'sig': gsf.copy(),
            'rhoerr': nlderr.copy(), 'gsferr': gsferr.copy()}

    nld = Vector(E=np.arange(imax-igmin+u0)*a1+a0,
                 values=nld[-1, :], std=nlderr)
    gsf = Vector(E=np.arange(np.max(igmax))*a1+a0,
                 values=gsf[-1, :np.max(igmax)],
                 std=gsferr[:np.max(igmax)])

    if diagnostics:
        return nld, gsf, diag
    else:
        return nld, gsf
