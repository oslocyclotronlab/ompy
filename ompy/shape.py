# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
from .turbo import *
from typing import Optional, Tuple, List, Iterable, Callable
from .matrix import Matrix, zeros_like, empty_like
from .vector import Vector

logger = logging.getLogger(__name__)
logging.captureWarnings(True)

Point2D = Tuple[float, float]
Points2D = Tuple[Point2D, Point2D]
array = np.ndarray


class Diagonal:
    def __init__(self, diagonal: Matrix, spin: float, parity: int):
        self.matrix = diagonal
        self.spin = spin
        self.parity = parity

    def compute_gsf(self, spinmodel) -> Tuple[Vector]:
        values = np.nan_to_num(self.values)
        centroid = self.Eg@values
        # Sum along Eg
        summed = values.sum(axis=0)
        centroid /= summed
        #centroid = np.nan_to_num(centroid)
        gsf = summed / (centroid**3 * self.spinfactor(spinmodel))
        #gsf = np.nan_to_num(gsf)
        return Vector(E=self.Ex, values=gsf), Vector(E=self.Ex, values=centroid)

    def spinfactor(self, spinmodel) -> array:
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
        return S + spinmodel.distribution()

    @property
    def values(self) -> array:
        return self.matrix.values

    @property
    def Eg(self) -> array:
        return self.matrix.Eg

    @property
    def Ex(self) -> array:
        return self.matrix.Eg


class Shape:
    """Implements the shape method

    """
    def __init__(self, matrix: Matrix):
        self.diagonals: List[Diagonal] = []
        self.matrix: Matrix = matrix
        self.spinmodel = None

    def add_diagonal(self, intercept=0, slope=1, *,
                     spin: float, parity: float,
                     points: Iterable[Point2D] = [],
                     thickness: float = 10):
        # TODO ensure that the order of Eg is preserved
        if not points:
            points = points_from_ab(intercept, slope)
        elif len(points) == 1:
            points = points_from_partial(slope, points[0])

        diagonal = empty_like(self.matrix)
        diagonal.values = diagonal_stripe(self.matrix, *points, thickness)
        self.diagonals.append(Diagonal(diagonal, spin, parity))

    def compute_gsf(self) -> Vector:
        unsewed: List[Tuple[Vector, Vector]] = self.compute_gsf_unsewed()
        return self.sew(unsewed)

    def compute_gsf_unsewed(self) -> List[Tuple[Vector, Vector]]:
        gsfs: List[Tuple[Vector, Vector]] = []
        for diagonal in self.diagonals:
            gsf = diagonal.compute_gsf(self.spinmodel)
            gsfs.append(gsf)

        return gsfs

    def sew(self, gsfs: List[Tuple[Vector, Vector]]) -> Vector:
        (diagonal_1, Egs_1), (diagonal_2, Egs_2) = gsfs
        # Assumes all have same E = Ex
        Ex_bins = diagonal_1.E
        Eg_final = []
        gsf_final = []
        for i in range(len(Ex_bins)-1):
            Eg1 = Egs_1[i], Egs_1[i+1]
            Eg2 = Egs_2[i], Egs_2[i+1]
            I1 = diagonal_1[i], diagonal_1[i+1]
            I2 = diagonal_2[i], diagonal_2[i+1]

            if not allnonzero(*Eg1, *Eg2, *I1, *I2):
                continue

            f1 = interpolate(Eg1[0], Eg2[0], I1[0], I2[0])
            f2 = interpolate(Eg1[1], Eg2[1], I1[1], I2[1])
            middle = (min(Eg1) + max(Eg2)) / 2

            factor = f1(middle)/f2(middle)
            diagonal_1[i+1] *= factor
            diagonal_2[i+1] *= factor

            Eg_final.extend([*Eg1, *Eg2])
            gsf_final.extend([diagonal_1[i], diagonal_1[i+1],
                              diagonal_2[i], diagonal_2[i+1]])

        gsf_final = [x for _, x in sorted(zip(Eg_final, gsf_final))]
        Eg_final = list(sorted(Eg_final))

        gsf = Vector(E=Eg_final, values=gsf_final)
        return gsf

    def plot(self, ax=None, **kwargs):
        fig, ax = plt.subplots()
        gsf = self.compute_gsf()
        gsf.plot(ax=ax, **kwargs)
        ax.set_xlabel(r"$E_\gamma$ [keV]")
        ax.set_ylabel(r"$counts / E_{\gamma}^3\times p $")
        ax.set_yscale('log')
        return ax

    def plot_unsewed(self, ax=None, **kwargs):
        fig, ax = plt.subplots()
        for i, (gsf, _) in enumerate(self.compute_gsf_unsewed()):
            spin = self.diagonals[i].spin
            pi = self.diagonals[i].parity
            pi = '+' if pi > 0 else '-'
            gsf.plot(ax=ax, label=f'Diagonal ${spin}^{pi}$')
        ax.set_xlabel(r"$E_x$ [keV]")
        ax.set_ylabel(r"$counts / E_{\gamma}^3\times p $")
        ax.set_yscale('log')
        ax.legend()
        return ax

    def plot_diagonals(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        values = zeros_like(self.matrix)
        for diagonal in self.diagonals:
            values += np.nan_to_num(diagonal.values)
        values.plot(ax=ax, **kwargs)
        return ax


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
                  delta: float) -> Tuple[array, array]:
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


def interpolate(x0: float, x1: float,
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
