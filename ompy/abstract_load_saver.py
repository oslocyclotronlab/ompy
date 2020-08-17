import warnings
import logging
from pathlib import Path
from typing import Optional, Union, Any, Tuple, List


class abstract_load_saver:
    """ An abstract super class to define the standard
        way for derived classes to save and load data from disk.
    """

    def __init__(self, path: Union[Union[str, Path], None], is_dir: bool = True):
        """ Set the default save position at this point.
            Args:
                path: Where to save and/or load from disk.
                is_dir: If the target path is a folder or a file.
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
                else:
                    pass

        def try_load(self):
            """ Tries to load from file or folder set in self.path.
            """

            if is_dir:
                try:
                    self.load(self.path)
                except ValueError:
                    self.path.mkdir(exist_ok=True, parents=True)
            else:  # For files the exception should be handled elsewhere.
                self.load(self.path)

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
        raise NotImplemented

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
        raise NotImplemented
