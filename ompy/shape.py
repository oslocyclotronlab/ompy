# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
from .turbo import *
from typing import Optional, Tuple, List, Iterable
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

    def compute_gsf(self, spinmodel) -> Vector:
        values = np.nan_to_num(self.values)
        centroid = self.Eg@values
        # Sum along Eg
        summed = values.sum(axis=0)
        centroid /= summed
        centroid = np.nan_to_num(centroid)
        gsf = summed / (centroid**3 * self.spinfactor(spinmodel))
        gsf = np.nan_to_num(gsf)
        return Vector(E=self.Ex, values=gsf)

    def spinfactor(self, spinmodel) -> float:
        spinmodel.Ex = self.matrix.Ex
        spinmodel.J = self.spin + 1
        S = spinmodel.distribution()
        if self.spin < 1/2:
            return S
        spinmodel.J = self.spin + 1
        S += spinmodel.distribution()
        if 1/2 <= self.spin < 1:
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
        if not points:
            points = points_from_ab(intercept, slope)
        elif len(points) == 1:
            points = points_from_partial(slope, points[0])

        diagonal = empty_like(self.matrix)
        diagonal.values = diagonal_stripe(self.matrix, *points, thickness)
        self.diagonals.append(Diagonal(diagonal, spin, parity))

    def compute_gsf(self) -> List[Vector]:
        gsfs: List[Vector] = []
        for diagonal in self.diagonals:
            gsf = diagonal.compute_gsf(self.spinmodel)
            gsfs.append(gsf)

        return gsfs

    def plot(self, ax=None, **kwargs):
        fig, ax = plt.subplots()
        for i, gsf in enumerate(self.compute_gsf()):
            spin = self.diagonals[i].spin
            pi = self.diagonals[i].parity
            pi = '+' if pi > 0 else '-'
            gsf.plot(ax=ax, label=f'Diagonal ${spin}^{pi}$')
        ax.set_xlabel(r"$E_\gamma$ [keV]")
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
