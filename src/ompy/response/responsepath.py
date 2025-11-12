from typing import TypeAlias, Literal
from pathlib import Path

from ..data_paths import get_response_path as _get_response_path

ResponseName: TypeAlias = Literal["OSCAR2017", "OSCAR2020", "CACTUS"]


def get_response_path(name: ResponseName) -> Path:
    """Return the path to a response dataset."""
    name = name.upper()
    if name not in ResponseName.__args__:
        raise ValueError(
            f"Unknown response name {name}. Must be one of '{', '.join(ResponseName.__args__)}'."
        )

    cache_path = _get_response_path() / name
    if cache_path.exists():
        return cache_path

    #file_path = Path(__file__).resolve()
    #for parent in file_path.parents:
    #    candidate = parent / "data" / "response" / name
    #    if candidate.exists():
    #        return candidate

    raise FileNotFoundError(
        f"Could not locate response data for {name!r}. "
        f"Tried cache location {cache_path} and package relatives under 'data/response'."
    )
