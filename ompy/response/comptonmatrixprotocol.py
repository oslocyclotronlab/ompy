from .. import MatrixProtocol
from typing import Protocol, TypeGuard
from .. import Vector


class ComptonMatrix(MatrixProtocol, Protocol):
    true_index: Vector
    observed_index: Vector

def is_compton_matrix(matrix: object) -> TypeGuard[ComptonMatrix]:
    return matrix is not None and hasattr(matrix, 'true_index') and hasattr(matrix, 'observed_index')
