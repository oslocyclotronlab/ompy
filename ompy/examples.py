from .matrix import Matrix
import os

EXAMPLES = {'Dy164': {'raw': "../../data/Dy164_raw.m",
                      'response': "../../data/Dy164_response_matrix.m"}
            }
DATAPATH = os.path.abspath(__file__)


def get_path(path):
    return os.path.abspath(os.path.join(DATAPATH, path))


def list_examples():
    return list(EXAMPLES.keys())


def load_example_raw(name):
    path = get_path(EXAMPLES[name]['raw'])
    return Matrix(filename=path)
