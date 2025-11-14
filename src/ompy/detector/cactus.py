from __future__ import annotations
from ..response import Response
from typing import Literal, Self, TypeAlias
from .responsedetector import ResponseDetector

CACTUSPath: TypeAlias = Literal['CACTUS']

def refine_name(name: str) -> CACTUSPath:
    name = name.upper()
    if not name.startswith('CACTUS'):
        name = 'CACTUS' + name
    if name not in CACTUSPath.__args__:
        raise ValueError(f"Unknown CACTUS version {name}. Must be one of '{', '.join(CACTUSPath.__args__)}'.")
    return name




class CACTUS(ResponseDetector):
    """
    CACTUS is a detector response function for the CACTUS detector.
    Is just a thin wrapper around the ResponseDetector class with CACTUS as the response function.
    """
    def __init__(self, response: Response | None = None, title: str = '', **kwargs):
        # Since we have one CACTUS version, we can just use the response from the database.
        if response is None:
            response = Response.from_db('CACTUS')
        super().__init__(response, title=title or 'CACTUS', **kwargs)

    @classmethod
    def from_str(cls, version: str = '') -> Self:
        name = refine_name(version)
        return cls(Response.from_db(name), title=name)

    @classmethod
    def from_default(cls) -> Self:
        return cls.from_str('CACTUS')

    def __str__(self) -> str:
        return f"CACTUS with response {self.response}"

