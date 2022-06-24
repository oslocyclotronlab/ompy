import os
from typing import Tuple, List
from .matrix import Matrix
import numpy as np

EXAMPLES = {'dy164': {'raw': "../../example_data/Dy164_raw.m",
                      # 'response': "../../data/Dy164_response_matrix.m"
                      },
            'si28': {'raw': "../../example_data/Si28_raw_matrix_compressed.m"}
            }
DATAPATH = os.path.abspath(__file__)


def get_path(path):
    return os.path.abspath(os.path.join(DATAPATH, path))


def list_examples() -> List[str]:
    """ List examples """
    return list(EXAMPLES.keys())


def example_raw(name: str) -> Matrix:
    """ Load example raw data. To list examples, use `list_examples` """
    name = name.lower()
    path = get_path(EXAMPLES[name]['raw'])
    return Matrix(path=path)


def example_response(name: str) -> Matrix:
    name = name.lower()
    path = get_path(EXAMPLES[name]['response'])
    return Matrix(path=path)


def disjoint_rows(shape: (int, int)) -> Matrix:
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
