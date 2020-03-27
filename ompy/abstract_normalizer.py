import logging
from pathlib import Path
from typing import Optional, Union
import dill
import numpy as np

from .vector import Vector


class AbstractNormalizer():
    """ Abstract class for Matrix and Vector.

        Do not initialize itself.

    Attributes:
        path (Optional[Union[str, Path]]): Path to save/load
        regenerate (bool): If `True`, calculates again instead of
            loading from `path`.
        _save_instance (bool): Can oversteer saving
    """
    LOG = logging.getLogger(__name__)

    def __init__(self, regenerate: bool = True):
        self.regenerate = regenerate
        self._save_instance: bool = True

    def save(self, path: Optional[Union[str, Path]] = None,
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

    def load(self, path: Optional[Union[str, Path]] = None):
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

    def save_results_txt(self, path: Optional[Union[str, Path]] = None,
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
