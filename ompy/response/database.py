from ..stubs import Pathlike
from typing import Any
from pathlib import Path
from .io import Format, verify_path, save


class Database:
    """ Handling the database of the response functions. """
    def export(self, path: Pathlike, format: Format = 'ompy', exist_ok: bool = False) -> None:
        raise NotImplementedError()

    def import_(self, path: Pathlike, format: Format = 'mama') -> None:
        path = Path(path)
        match format:
            case 'mama':
                raise NotImplementedError()
            case 'ompy':
                raise NotImplementedError()
            case 'feather':
                raise NotImplementedError()
            case 'root':
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unknown format {format}. Available formats are: {Format}")


