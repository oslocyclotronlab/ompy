# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import logging
from typing import Optional, Tuple, List, Iterable, Callable, Any, Union
from . import zeros_like, empty_like, Matrix, Vector
from .library import log_interp1d
from .extractor import Extractor

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

Point2D = Tuple[float, float]
Points2D = Tuple[Point2D, Point2D]
Interval = Tuple[float, float]
array = np.ndarray


class Diagonal:
    def __init__(self, diagonal: Matrix, spin: float, parity: int):
        self.matrix = diagonal.copy()
        self.matrix.Eg.to('MeV', inplace=True)
        self.matrix.Ex.to('MeV', inplace=True)
        self.spin = spin
        self.parity = parity

    def compute_gsf(self, spinmodel) -> (Vector):
        values = np.nan_to_num(self.values)
        # Sum along Eg
        summed = values.sum(axis=0)
        # Compute the centroid along Eg for each Ex row
        centroid = self.Eg@values
        centroid /= summed
        # Correct for Eγ³ and the population difference due to spin
        gsf = summed / (centroid**3 * self.spinfactor(spinmodel))
        gsf = Vector(E=self.Ex, values=gsf)
        # Need to keep track of the central Eg values used for the sewing
        Eg = Vector(E=self.Ex, values=centroid)
        return gsf, Eg

    def spinfactor(self, spinmodel) -> array:
        #if self.spin == 0:
        #    return 1

        spinmodel.Ex = self.Ex
        spinmodel.J = self.spin + 1
        S = spinmodel.distribution()
        if self.spin == 0:
            return S

        spinmodel.J = self.spin
        S += spinmodel.distribution()
        if self.spin == 1/2:
            return S

        spinmodel.J = self.spin - 1
        S += spinmodel.distribution()

        #spinmodel.J = 1
        #S /= spinmodel.distribution()
        return S


    @property
    def values(self) -> array:
        return self.matrix.values

    @property
    def Eg(self) -> array:
        return self.matrix.Eg.magnitude

    @property
    def Ex(self) -> array:
        return self.matrix.Ex.magnitude


class Shape:
    """Implements the shape method

    """
    def __init__(self, matrix: Matrix):
        self.diagonals: List[Diagonal] = []
        self.matrix: Matrix = matrix
        self.spinmodel = None
        self.gsf: Vector = Vector([0], [0])
        self.gsfs: Tuple[Vector, Vector] = Vector([0], [0]), Vector([0], [0])
        self.reg: Any | None = None
        self.fit_region: array | None = None

    def add_diagonal(self, intercept=0, slope=1, *,
                     spin: float, parity: float,
                     points = [],
                     thickness: float = 10):
        # TODO ensure that the order of Eg is preserved
        if not points:
            points = points_from_ab(intercept, slope)
        elif len(points) == 1:
            points = points_from_partial(slope, points[0])

        diagonal = empty_like(self.matrix)
        diagonal.values = diagonal_stripe(self.matrix, *points, thickness)
        self.diagonals.append(Diagonal(diagonal, spin, parity))

    def compute_gsf(self, kind='log') -> Vector:
        unsewed: List[(Vector, Vector)] = self.compute_gsf_unsewed()
        self.gsf, *self.gsfs = self.sew(unsewed, kind=kind)
        return self.gsf

    def compute_gsf_unsewed(self) -> List[Tuple[Vector, Vector]]:
        gsfs: List[Tuple[Vector, Vector]] = []
        for diagonal in self.diagonals:
            gsf = diagonal.compute_gsf(self.spinmodel)
            gsfs.append(gsf)

        return gsfs

    def sew(self, gsfs: List[Tuple[Vector, Vector]], kind='log') -> Tuple[Vector, Vector, Vector]:
        # TODO Prettify
        (diagonal_1, Egs_1), (diagonal_2, Egs_2) = gsfs
        # Assumes all have same Ex, diagonal_1.E == diagonal_2.E
        Ex_dim = len(diagonal_1.E)

        Eg_final = []
        gsf_final = []

        if kind == 'log':
            interpolate = lambda *x: log_interp1d([x[0], x[1]], [x[2], x[3]]) # noqa
        elif kind == 'linear':
            interpolate = lin_interp1d
        else:
            raise ValueError(f"Unsupported interpolation {kind}")

        j = 0
        while j < Ex_dim:
            i = j
            # Find a pair that has no zeros or NaNs
            i, j = good_pair(i, Ex_dim, Egs_1, Egs_2, diagonal_1, diagonal_2)
            if j >= Ex_dim:
                break

            Eg1 = Egs_1[i], Egs_1[j]
            Eg2 = Egs_2[i], Egs_2[j]
            I1 = diagonal_1[i], diagonal_1[j]
            I2 = diagonal_2[i], diagonal_2[j]

            f1 = interpolate(Eg1[0], Eg2[0], I1[0], I2[0])
            f2 = interpolate(Eg1[1], Eg2[1], I1[1], I2[1])
            middle = (min(Eg1) + max(Eg2)) / 2

            # print(f"{Eg1[0]=}, {Eg2[0]=}, {I1[0]=}, {I2[0]=}")
            # print(f"{Eg1[1]=}, {Eg2[1]=}, {I1[1]=}, {I2[1]=}")
            # print(f"middle: {middle}")
            # print(f"{f1(middle)}")
            # print(f"{f2(middle)}")
            # print(f"Factor: {f1(middle)/f2(middle)}")

            factor = f1(middle)/f2(middle)
            diagonal_1[j] *= factor
            diagonal_2[j] *= factor

            Eg_final.extend([*Eg1, *Eg2])
            gsf_final.extend([diagonal_1[i], diagonal_1[j],
                              diagonal_2[i], diagonal_2[j]])


        E, values = zip(*sorted(zip(Egs_1.values, diagonal_1.values)))
        gsf_1 = Vector(E=E, values=values)

        E, values = zip(*sorted(zip(Egs_2.values, diagonal_2.values)))
        gsf_2 = Vector(E=E, values=values)

        # Sort list by Eg
        gsf_final = [x for _, x in sorted(zip(Eg_final, gsf_final))]
        Eg_final = list(sorted(Eg_final))

        gsf = Vector(E=Eg_final, values=gsf_final)
        return gsf, gsf_1, gsf_2

    def fit(self, region: Interval) -> float:
        self.fit_region = region
        region = self.gsf.loc[region[0]:region[1]]
        E = region.E.reshape(-1, 1)
        logvalues = np.log(region.values)
        reg = LinearRegression().fit(E, logvalues)
        LOG.info("Fit score: %f", reg.score(E, logvalues))
        slope = reg.coef_
        LOG.info("Slope = %f", slope)
        self.reg = reg
        return slope

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.gsf.plot(ax=ax, label="Concatenated gSFs", **kwargs)
        ax.set_xlabel(r"$E_\gamma$ [keV]")
        ax.set_ylabel(r"$counts / E_{\gamma}^3\times p $")

        for i, gsf in enumerate(self.gsfs):
            spin = self.diagonals[i].spin
            pi = self.diagonals[i].parity
            pi = '+' if pi > 0 else '-'
            gsf.plot(ax=ax, label=f"Diagonal ${spin}^{pi}$",
                     **kwargs)

        if self.reg is not None:
            ax.axvspan(self.fit_region[0], self.fit_region[1], color='grey',
                       alpha=0.1, label='Fit limits')
            region = self.gsf.loc[self.fit_region[0]:self.fit_region[1]]
            E = region.E.reshape(-1, 1)
            ax.plot(region.E, np.exp(self.reg.predict(E)),
                    label='Log linear fit', zorder=10)

        ax.set_yscale('log')
        return ax.figure, ax

    def plot_unsewed(self, ax=None, **kwargs):
        fig, ax = plt.subplots()
        values =[]
        for i, (gsf, eg) in enumerate(self.compute_gsf_unsewed()):
            spin = self.diagonals[i].spin
            pi = self.diagonals[i].parity
            pi = '+' if pi > 0 else '-'
            gsf = gsf.copy()
            gsf_final = [x for _, x in sorted(zip(eg.values, gsf.values))]
            Eg_final = list(sorted(eg.values))
            vec = Vector(E=Eg_final, values=gsf_final)
            vec.plot(ax=ax)
            value = [vec.loc[6000].values, vec.loc[6900].values]
            ax.scatter([6000, 6900],
                       value,
                       edgecolor='k', s=80, facecolors='none')
            values.append(value)
            #gsf.plot(ax=ax, label=f'Diagonal ${spin}^{pi}$')
        print("Ratio:", values[1][0]/values[0][1])
        ax.set_xlabel(r"$E_g$ [keV]")
        ax.set_ylabel(r"$counts / E_{\gamma}^3\times p $")
        ax.set_yscale('log')
        ax.legend()
        return ax

    def plot_diagonals(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        values = zeros_like(self.matrix)
        for diagonal in self.diagonals:
            values += np.nan_to_num(diagonal.values)
        values.plot(ax=ax, **kwargs)
        return ax.figure, ax


def points_from_partial(slope: float, point: Point2D) -> Points2D:
    Eg, Ex = point
    intercept = Ex - slope*Eg
    return points_from_ab(intercept, slope)


def points_from_ab(intercept: float, slope: float) -> Points2D:
    Eg0, Eg1 = 0, 5000
    Ex0, Ex1 = intercept, slope*Eg1 + intercept
    return ((Eg0, Ex0), (Eg1, Ex1))


def diagonal_stripe(matrix: Matrix, p1: Point2D, p2: Point2D,
                    thickness: float) -> array:
    delta = thickness / 2
    # Create the upper and lower bounding diagonals
    X1, Y1 = parallel_line(matrix, p1, p2, delta=-delta)
    X2, Y2 = parallel_line(matrix, p1, p2, delta=+delta)
    # All values between them define the diagonal of interest

    M = np.zeros_like(matrix.values) + np.NaN
    X, Y = np.meshgrid(matrix.range_Eg, matrix.range_Ex)
    mask = (Y1 <= Y) & (Y <= Y2)
    X = X[mask]
    Y = Y[mask]
    M[Y, X] = matrix.values[Y, X]
    return M


def parallel_line(mat: Matrix, p1: Point2D, p2: Point2D,
                  delta: float) -> (array, array):
    (Eg0, Ex0), (Eg1, Ex1) = p1, p2
    X0, X1 = mat.indices_Eg([Eg0, Eg1])
    Y0, Y1 = mat.indices_Ex([Ex0, Ex1])

    a = (Y1-Y0)/(X1-X0)
    b = Y0 - a*X0

    f = lambda x: (a*x + b).astype(int)
    X = mat.range_Eg
    Y = f(X)

    norm = np.sqrt(1 / (a**2 + 1))
    dx = a*delta*norm
    dy = delta*norm

    #X = X + dx
    Y = Y + dy

    return X.astype(int), Y.astype(int)


def lin_interp1d(x0: float, x1: float,
                 y0: float, y1: float) -> Callable[[float],
                                                   float]:
    a = (y1 - y0)/(x1 - x0)
    b = y0 - a*x0
    return lambda x: a*x + b


def allnonzero(*X):
    for x in X:
        if not np.isfinite(x) or x <= 0:
            return False

    return True


def good_pair(start: int, N: int, Eg_1, Eg_2, diagonal_1, diagonal_2):
    for i in range(start, N):
        if allnonzero(Eg_1[i], Eg_2[i], diagonal_1[i], diagonal_2[i]):
            break
        LOG.debug("Bad pair for i = %i", i)

    # Find the next pair that has no zeros or NaNs
    j = i+1
    for j in range(i+1, N):
        if allnonzero(Eg_1[j], Eg_2[j],
                      diagonal_1[j], diagonal_2[j]):
            break
        LOG.debug("Bad pair for j = %i", j)
    # Returns i and/or j > N in case of failure
    return i, j


def normalize_to_shape(extractor: Extractor, shape: Union[float, Shape],
                       region: Interval | None = None):
    if isinstance(shape, (int, float)):
        slope = shape
        if region is None:
            raise ValueError("Supply the fit `region`")
    else:
        slope = shape.reg.coef_[0]
        if region is None:
            region = shape.fit_region

    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[1].axvspan(region[0], region[1], color='grey',
                  alpha=0.1, label='Fit limits')
    N = 10
    alphas = []
    for nld, gsf in zip(extractor.nld, extractor.gsf):
        alpha = fit_to_region(gsf, slope, region)
        nld.transform(const=0.15, alpha=alpha, inplace=False).plot(ax=ax[0], scale='log',
                                                       kind='step',
                                                       color='k', alpha=1/N)
        gsf.transform(alpha=alpha, inplace=False).plot(ax=ax[1], scale='log',
                                                       kind='step',
                                                       color='k', alpha=1/N)
        alphas.append(alpha)

    ax[0].set_title("NLD")
    ax[1].set_title("γSF")
    return alphas, (fig, ax)


def fit_to_region(gsf: Vector, slope: float, region: Interval):
    region = gsf.loc[region[0]:region[1]]
    E = region.E.reshape(-1, 1)

    def err(x) -> float:
        alpha = x[0]
        if alpha <= 0 or alpha > 0.01:
            return np.inf

        trans = region.transform(alpha=alpha, inplace=False)
        logvalues = np.log(trans.values)
        reg = LinearRegression().fit(E, logvalues)
        error = abs(reg.coef_[0] - slope)
        return error

    res = minimize(err, [0.001], method='Powell')
    return float(res.x)
