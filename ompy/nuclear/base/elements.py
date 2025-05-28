from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeAlias, Literal

ELEMS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "119",
    "120",
    "121",
    "122",
    "123",
    "124",
    "125",
]

ELEMENT: TypeAlias = Literal[
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def get_element_symbol(Z: int) -> ELEMENT:
    return ELEMS[Z - 1]


@dataclass(kw_only=True)
class Element:
    A: int
    Z: int
    symbol: ELEMENT = field(init=False)

    def __post_init__(self):
        if self.A < self.Z:
            raise ValueError(f"A must be greater than Z: {self.A} < {self.Z}")
        if self.A < 1:
            raise ValueError(f"A must be greater than 0: {self.A} < 1")
        if self.Z < 1:
            raise ValueError(f"Z must be greater than 0: {self.Z} < 1")
        if self.Z > 118:
            raise ValueError(f"Z must be less than 118: {self.Z} > 118")
        symbol = get_element_symbol(self.Z)
        self.symbol = symbol

    @classmethod
    def from_str(cls, symbol: str) -> Element:
        """Create an Element from a symbol

        Parameters
        ----------
        symbol : str
            Symbol of the nucleus, e.g. "22Mg"

        Returns
        -------
        Nucleus
            An Element object with the A and Z numbers parsed from the symbol
        """
        A, Z = parse_symbol(symbol)
        return cls(A=A, Z=Z)

    @property
    def mass_symbol(self) -> str:
        return f"{self.A}{self.symbol}"

    def __repr__(self) -> str:
        return f"{self.A}{self.symbol}"

    @property
    def N(self) -> int:
        return self.A - self.Z

    def _repr_latex_(self) -> str:
        return f"$^{{{self.A}}}_{{{self.Z}}}\\text{{{self.symbol}}}_{{{self.N}}}$"

    def __hash__(self) -> int:
        return hash((self.A, self.Z))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Element):
            return False
        return self.A == other.A and self.Z == other.Z


def parse_symbol(symbol: str) -> tuple[int, int]:
    """Parse a symbol into A and Z numbers

    Parameters
    ----------
    symbol : str
        Symbol of the nucleus, e.g. "22Mg"

    Returns
    -------
    tuple[int, int]
        A tuple of the A and Z numbers

    Raises
    ------
    ValueError
        If the symbol is not in the correct format

    Examples
    --------
    >>> parse_symbol("22Mg")
    (22, 12)
    >>> parse_symbol("166Ho")
    (166, 67)
    """
    A = ""
    while symbol and symbol[0].isdigit():
        A += symbol[0]
        symbol = symbol[1:]
    A = int(A)
    symbol = symbol.capitalize()
    return A, ELEMS.index(symbol) + 1
