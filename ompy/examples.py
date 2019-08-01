import os
from typing import Tuple, Union
from .matrix import Matrix
import numpy as np

EXAMPLES = {'Dy164': {'raw': "../../data/Dy164_raw.m",
                      'response': "../../data/Dy164_response_matrix.m"},
            'Si28': {'raw': "../../data/Si28_raw_matrix_compressed.m"}
            }
DATAPATH = os.path.abspath(__file__)


def get_path(path):
    return os.path.abspath(os.path.join(DATAPATH, path))


def list_examples():
    return list(EXAMPLES.keys())


def load_example_raw(name: str) -> Matrix:
    path = get_path(EXAMPLES[name]['raw'])
    return Matrix(filename=path)


def disjoint_rows(shape: Tuple[int, int]) -> Matrix:
    """ Creates a mock matrix with disjoin rows

    A 5×6 looks like this:
        ░░░░░░░░░░

        ░░░░░░

        ░░

    Args:
        shape: The shape of the matrix
    Returns:
        The matrix
    """
    mat = np.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            if row % 2 and col < row:
                mat[row, col] = 1
    return Matrix(values=mat)


def ones(shape: Tuple[int, int]) -> Matrix:
    """ Creates a mock matrix with ones in the upper diagonal

    A 5×5 looks like this:
        ░░░░░░░░░░
        ░░░░░░░░
        ░░░░░░
        ░░░░
        ░░
    Args:
        shape: The shape of the matrix
    Returns:
        The matrix
    """
    mat = np.ones(shape)
    mat = np.tril(mat)
    return Matrix(values=mat)


def all_generations_trivial(shape: Tuple[int, int],
         ret_firstgen: bool = False) -> Union[Tuple[Matrix, Matrix], Matrix]:
    """ Creates a mock all generations matrix

    A 5×5 looks like this:
        ██▓▓▒▒░░░░
        ▓▓▒▒░░░░
        ▒▒░░░░
        ░░░░
        ░░
    Args:
        shape: The shape of the matrix
    Returns:
        The matrix
    """
    mat = Matrix(shape=shape)
    for row, col in mat.iter():
        if col <= row:
            mat[row, col] = row - col + 1
    if ret_firstgen:
        return mat, ones(shape)
    return mat

