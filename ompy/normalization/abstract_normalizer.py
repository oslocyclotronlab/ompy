from __future__ import annotations
import logging
from pathlib import Path
from typing import Union, TypeVar
import numpy as np

from .. import Vector, AsymmetricVector


class AbstractNormalizer():
    """ Abstract class for Normalizers.

        Do not initialize itself.

    Attributes:
        path (Union[str, Path] | None): Path to save/load
        regenerate (bool): If `True`, calculates again instead of
            loading from `path`.
        _save_instance (bool): Can oversteer saving
    """
    LOG = logging.getLogger(__name__)

    def __init__(self, regenerate: bool = True):
        self.regenerate = regenerate
        self._save_instance: bool = True

    def save(self, path: Union[str, Path] | None = None,
             overwrite: bool = True):
        """ Save (pickels) the instance

        Such that it can be loaded, and enabling the `regenerate` later.

        Args:
            path: The path to the save directory. If the
                value is None, 'self.path' will be used.
            overwrite: Overwrite file if existent
        """
        if not self._save_instance:
            return
        path = Path(path) if path is not None else Path(self.path)
        mode = 'wb' if overwrite else 'xb'
        fname = (path / type(self).__name__).with_suffix('.pkl')
        self.LOG.info(f"Saving to {fname}")
        with open(fname, mode) as fobj:
            dill.dump(self, fobj)

    def load(self, path: Union[str, Path] | None = None):
        """ Loads (pickeled) instance.

        Such that it can be loaded if `regenerate = False`.
        Note that if any modifications of the __getstate__ method are present,
        these will effect what attributes are pickeled.

        Args:
            path: The path to the directoryto load file. If the
                value is None, 'self.path' will be used.

        Raises:
            FileNotFoundError: If file is not found
        """
        path = Path(path) if path is not None else Path(self.path)
        fname = (path / type(self).__name__).with_suffix('.pkl')

        self.LOG.info(f"Try loading from {fname}")
        with open(fname, "rb") as fobj:
            saved = dill.load(fobj)
        self.__dict__.update(saved.__dict__)
        self.LOG.info(f"Loaded")

    def save_results_txt(self, path: Union[str, Path] | None = None,
                         nld: Vector = None,
                         gsf: Vector = None,
                         samples: dict = None,
                         suffix: str = None):
        """ Save results as txt

        Uses a folder to save nld, gsf, and the samples (converted to an array)

        Args:
            path: The path to the save directory. If the
                value is None, 'self.path' will be used.
            nld: (unnormalized) NLD to save
            gsf: (unnormalized) gSF to save
            samples: samples to use for normalization
            suffix: suffix to append to filename, eg. iteration number
        """
        path = Path(path) if path is not None else Path(self.path)

        fname = (path / f"nld_{suffix}").with_suffix('.txt')
        self.LOG.debug(f"Saving nld to {fname}")
        nld.save(fname)
        fname = (path / f"gsf_{suffix}").with_suffix('.txt')
        self.LOG.debug(f"Saving gsf to {fname}")
        gsf.save(fname)

        # copy dict to an array
        first_values = next(iter(samples.items()))[1]
        array = np.zeros((len(samples), len(first_values)))
        for i, values in enumerate(samples.values()):
            array[i, :] = values
        seperator = " "
        header = seperator.join(samples.keys())

        fname = (path / f"samples_{suffix}").with_suffix('.txt')
        self.LOG.debug(f"Saving samples to {fname}")
        np.savetxt(fname, array, header=header)


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
