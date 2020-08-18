import warnings
import logging
from pathlib import Path
from typing import Optional, Union, Any, Tuple, List


class AbstractLoadSaver(object):
    """ An abstract super class to define the standard
        way for derived classes to save and load data from disk.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None,
                 is_dir: bool = True):
        """ Set the default save position at this point.
            Args:
                path: Where to save and/or load from disk.
                is_dir: If the target path is a folder or a file. Defaults to
                    True.
        """
        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.is_dir = is_dir
            try:
                self.load(self.path)
            except ValueError:
                if is_dir:
                    self.path.mkdir(exist_ok=True, parents=True)

    def load(self, path: Union[str, Path],
             filetype: Optional[str] = None):
        """ Method for loading the file or folder. To be implemented
            by the subclasses.

            Args:
                path: Path to where to find the object(s).
                filetype: Type of the file to load. Will be deduced the
                    filetype from suffix if not given. Only applicable for
                    subclasses that loads a single file.
        """
        raise NotImplementedError

    def save(self, path: Union[str, Path],
             filetype: Optional[str] = None,
             **kwargs):
        """ Method for saving to file or folder. to be implemented
            by the subclasses.

            Args:
                path: Path to where to store the object(s).
                filetype: Type of the file to store. Will be deduced the
                    filetype from suffix if not given. Only applicable for
                    subclasses that saves a single file.
        """
        raise NotImplementedError
