from .. import Vector, AsymmetricVector
from typing import TypeVar
import numpy as np


T = TypeVar('T',Vector, AsymmetricVector, list[Vector])
def transform(vec: T,
              const: float = 1,
              alpha: float = 0) -> T:
    """Apply a normalization transformation::

        vector -> const * vector * exp(alpha*energy)

    Args:
        const (float, optional): The constant. Defaults to 1.
        alpha (float, optional): The exponential coefficient.
            Defaults to 0.
    Returns:
        Vector: The transformed vector
    """
    match vec:
        case AsymmetricVector():
            return transform_avec(vec, const, alpha)
        case Vector():
            return transform_vec(vec, const, alpha)
        case list():
            return transform_list(vec, const, alpha)
        case x:
            raise TypeError(f"Can't transform {type(x)}")


def transform_vec(vec: Vector, const: float, alpha: float) -> Vector:
    vec1 = vec.to_mid().to_unit('MeV')
    transformed = const * vec1* np.exp(alpha * vec1.E)
    return transformed.to_same(vec)


def transform_avec(vec: AsymmetricVector, const: float, alpha: float) -> AsymmetricVector:
    vec1 = vec.to_mid().to_unit('MeV')
    scaling = const * np.exp(alpha * vec1.E)
    transformed = scaling * vec1.values
    # First order appoximation of error propagation
    lerr = scaling * vec1.lerr
    uerr = scaling * vec1.uerr
    return vec.clone(values=transformed, lerr=lerr, uerr=uerr)

def transform_list(vec: list[Vector], const: float, alpha: float) -> list[Vector]:
    vec1 = vec[0].to_mid().to_unit('MeV')
    e = vec1.E
    X = np.stack([v.values for v in vec])
    transformed = const * X * np.exp(alpha * e)
    return [vec[0].clone(values=values) for values in transformed]
