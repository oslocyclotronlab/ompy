from typing import TypeAlias, Literal
from pathlib import Path

ResponseName: TypeAlias = Literal['OSCAR2017', 'OSCAR2020']


def get_response_path(name: ResponseName) -> Path:
    """ Returns the path to the response in the database. """
    name = name.upper()
    if name not in ResponseName.__args__:
        raise ValueError(f"Unknown response name {name}. Must be one of {ResponseName}.")
    return Path(__file__).parent.parent.parent / 'data' / 'response' / name
