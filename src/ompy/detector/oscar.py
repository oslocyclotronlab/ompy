from __future__ import annotations
from ..response import Response
from typing import TypeAlias, Literal, Self
from .responsedetector import ResponseDetector

OSCARPath: TypeAlias = Literal['OSCAR2017', 'OSCAR2020']

def refine_name(name: str) -> OSCARPath:
    name = name.upper()
    if not name.startswith('OSCAR'):
        name = 'OSCAR' + name
    if name not in OSCARPath.__args__:
        raise ValueError(f"Unknown OSCAR version {name}. Must be one of '{', '.join(OSCARPath.__args__)}'.")
    return name




class OSCAR(ResponseDetector):
    """
    OSCAR is a detector response function for the OSCAR detector.
    Is just a thin wrapper around the ResponseDetector class with OSCAR as the response function.
    """
    @classmethod
    def from_str(cls, version: str) -> Self:
        name = refine_name(version)
        return cls(Response.from_db(name), title=name)

    @classmethod
    def from_default(cls) -> Self:
        return cls.from_str('OSCAR2020')

    def __str__(self) -> str:
        return f"OSCAR with response {self.response}"

