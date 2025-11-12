from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeAlias, Literal, Iterable, Iterator, Callable

Delta: TypeAlias = int | tuple[int, int]

NUCLIDE: TypeAlias = Literal[
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

NUCLIDES: list[NUCLIDE] = [
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


def get_nuclide_symbol(Z: int) -> NUCLIDE:
    return NUCLIDES[Z - 1]


@dataclass(kw_only=True)
class Nuclide:
    A: int
    Z: int
    symbol: NUCLIDE = field(init=False)

    def __post_init__(self):
        if self.A < self.Z:
            raise ValueError(f"A must be greater than Z: {self.A} < {self.Z}")
        if self.A < 1:
            raise ValueError(f"A must be greater than 0: {self.A} < 1")
        if self.Z < 1:
            raise ValueError(f"Z must be greater than 0: {self.Z} < 1")
        if self.Z > 118:
            raise ValueError(f"Z must be less than 118: {self.Z} > 118")
        symbol = get_nuclide_symbol(self.Z)
        self.symbol = symbol

    @classmethod
    def from_str(cls, symbol: str) -> Nuclide:
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

    @classmethod
    def from_any(cls, obj: str | Nuclide) -> Nuclide:
        if isinstance(obj, str):
            return cls.from_str(obj)
        return obj

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
        if not isinstance(other, Nuclide):
            return False
        return self.A == other.A and self.Z == other.Z


def parse_symbol(symbol: str) -> tuple[int, int]:
    """Parse a symbol into A and Z numbers

    Parameters
    ----------
    symbol : str
        Symbol of the nucleus, e.g. "22Mg" or "Mg22"

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
    >>> parse_symbol("Mg22")
    (22, 12)
    >>> parse_symbol("166Ho")
    (166, 67)
    >>> parse_symbol("Ho166")
    (166, 67)
    """
    # Handle both formats: "22Mg" and "Mg22"
    if symbol[0].isdigit():
        # Format: "22Mg"
        A = ""
        while symbol and symbol[0].isdigit():
            A += symbol[0]
            symbol = symbol[1:]
        A = int(A)
        symbol = symbol.capitalize()
    else:
        # Format: "Mg22"
        symbol_part = ""
        while symbol and symbol[0].isalpha():
            symbol_part += symbol[0]
            symbol = symbol[1:]
        A = int(symbol)
        symbol = symbol_part.capitalize()

    return A, NUCLIDES.index(symbol) + 1

@dataclass
class Neighbors:
    nuclide: Nuclide
    neighbors: list[Nuclide]
    all: list[Nuclide]

    def __iter__(self):
        return iter(self.neighbors)

    def __len__(self):
        return len(self.neighbors)

    def _repr_html_(self):
        from .chart import draw_chart

        return draw_chart(
            self.all, {"neighbors": self.neighbors, "focus": [self.nuclide]}
        )

def _coerce_delta(d: Delta) -> tuple[int, int]:
    if isinstance(d, int):
        return d, d
    plus, minus = d
    return plus, minus

def neighbors(
    nuclide: Nuclide,
    *,
    dZ: Delta = 0,
    dN: Delta = 0,
    nuclides: Iterable[Nuclide] | Iterator[Nuclide] | Callable[[], Iterator[Nuclide]] | None = None,
) -> Neighbors:
    """
    If `nuclides` is None: return the entire rectangle of (Z,N) in the window.
    Otherwise: intersect that rectangle with the provided pool efficiently.
    """
    Z0, N0 = nuclide.Z, nuclide.N
    dZ_plus, dZ_minus = _coerce_delta(dZ)
    dN_plus, dN_minus = _coerce_delta(dN)

    Z_min = max(1, Z0 - dZ_minus)
    Z_max = min(118, Z0 + dZ_plus)     # Element enforces Z ≤ 118
    N_min = max(0, N0 - dN_minus)
    N_max = N0 + dN_plus               # no hard upper bound needed; A = Z + N

    # Case 1: synthesize full rectangle (incl. possibly unobserved nuclides)
    if nuclides is None:
        rect = []
        for Z in range(Z_min, Z_max + 1):
            for N in range(N_min, N_max + 1):
                A = Z + N
                rect.append(Nuclide(A=A, Z=Z))
        return Neighbors(nuclide, rect, rect)

    # Case 2: intersect rectangle with provided pool (efficiently)
    pool = list(nuclides() if callable(nuclides) else nuclides)
    by_ZN = {(e.Z, e.N): e for e in pool}

    found: list[Nuclide] = []
    for Z in range(Z_min, Z_max + 1):
        for N in range(N_min, N_max + 1):
            e = by_ZN.get((Z, N))
            if e is not None:
                found.append(e)

    return Neighbors(nuclide, found, pool)