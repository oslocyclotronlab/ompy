from __future__ import annotations
from ..stubs import Axes, keV, Unitlike
from .. import ureg
import matplotlib.pyplot as plt
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd

# A list over the periodic elements
ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def atomic_number(element: str) -> int:
    """Return the atomic number of the element"""
    return ELEMENTS.index(element) + 1


class Nucleus:
    def __init__(self, nucleus: str | None = None, A: int | None = None, Z: int | None = None):
        if nucleus is None and (A is None or Z is None):
            raise ValueError("Specify either the nucleus as a string or A and Z values.")
        if nucleus is not None and (A is not None or Z is not None):
            raise ValueError("Specify either the nucleus as a string or A and Z values.")
        if nucleus is not None:
            element, A, Z = self.parse_nucleus(nucleus)
        else:
            assert Z is not None
            assert A is not None
            element = ELEMENTS[Z - 1]

        self.element: str = element
        self.A: int = A
        self.Z: int = Z
        self.N: int = A-Z
        self.levels = LevelScheme()
        self.gammas = GammaScheme()

    @staticmethod
    def parse_nucleus(s: str) -> tuple[str, int, int]:
        """Parse a string to get the element, A and Z values.

        Accepts "elementA", "Aelement"

        Parameters
        ----------
        s : str
            String to parse.

        Returns
        -------
        tuple[str, int, int]
            Element, A and Z values.
        """
        element = re.findall(r"[A-Z][a-z]?", s)[0]
        # Capitalize the first letter
        element = element[0].upper() + element[1:]

        A = int(re.findall(r"\d+", s)[0])
        Z = A - atomic_number(element)
        return element, A, Z


    @property
    def name(self):
        return f"{self.A}{self.element}"

    def __str__(self):
        return f"{self.name} (A={self.A}, Z={self.Z}, N={self.N}) with {len(self.levels)} levels and {len(self.gammas)} gammas."



@dataclass
class Spin:
    spin: int
    parity: bool

    @classmethod
    def from_string(cls, s: str) -> Spin:
        """Parse a string to get the spin and parity.

        Parameters
        ----------
        s : str
            String to parse.

        Returns
        -------
        Spin
            Spin and parity.
        """
        spin = int(s[:-1])
        parity = s[-1] == "+"
        return Spin(spin, parity)

    def __str__(self):
        return f"{self.spin}{'+' if self.parity else '-'}"


@dataclass
class Level:
    E: keV
    jpi: list[Spin]

    def __init__(self, E: keV, jpi: Spin | list[Spin]):
        if isinstance(jpi, Spin):
            jpi = [jpi]
        self.E = E
        self.jpi = jpi

    def _jpi_str(self) -> str:
        if len(self.jpi) == 1:
            return f"{self.jpi[0]}"
        else:
            return f"({', '.join([str(jpi) for jpi in self.jpi])})"

    def __str__(self):
        return f"{self.E:.2f} keV, {self._jpi_str()}"

@dataclass
class Gamma:
    Ei: keV
    E: keV
    jpi: list[Spin]
    T: keV

    def __init__(self, Ei: keV, E: keV, jpi: Spin | list[Spin], T: keV):
        if isinstance(jpi, Spin):
            jpi = [jpi]
        self.Ei = Ei
        self.E = E
        self.jpi = jpi
        self.T = T

    def _jpi_str(self) -> str:
        if len(self.jpi) == 1:
            return f"{self.jpi[0]}"
        else:
            return f"({', '.join([str(jpi) for jpi in self.jpi])})"

    def __str__(self):
        # Unicode for the arrow
        arrow = "\u2192"
        return f"{self.Ei:.2f} keV {arrow} {self.Ef:.2f} keV: Eg {self.E:.2f} keV, {self._jpi_str()}, T={self.T} keV"

    @property
    def Ef(self) -> keV:
        return self.Ei - self.E

    @property
    def Eg(self) -> keV:
        return self.E


def str_to_spin(s: str) -> Spin | list[Spin]:
    """Parse a string to get a spin or a list of spins.

    BUG: Doesn't handle 1/2-format correctly
    Parameters
    ----------
    s : str
        String to parse.

    Returns
    -------
    Spin | list[Spin]
        A spin or a list of spins.
    """
    matches = re.findall(r"(\d+[+-])", s)
    if len(matches) == 1:
        return Spin.from_string(matches[0])
    else:
        return [Spin.from_string(match) for match in matches]

class LevelScheme:
    def __init__(self, levels: list[Level] = None):
        if levels is None:
            levels = []
        self.levels = levels

    def append(self, level: Level):
        self.levels.append(level)

    def add(self, E: keV, jpi: list[Spin] | Spin | str):
        if isinstance(jpi, str):
            jpi = str_to_spin(jpi)
        self.levels.append(Level(E, jpi))

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        for level in self.levels:
            ax.axhline(level.E, **kwargs)

        return ax

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["E"] = [level.E for level in self.levels]
        df['Jpi'] = [level.jpi for level in self.levels]
        return df

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> LevelScheme:
        levels = LevelScheme()
        for i, row in df.iterrows():
            levels.add(row["E"], row["Jpi"])
        return LevelScheme(levels)

    def query_pandas(self, query: str) -> pd.DataFrame:
        df = self.to_pandas()
        df = df.query(query)
        return df

    def query(self, query: str) -> LevelScheme:
        df = self.query_pandas(query)
        return LevelScheme.from_pandas(df)

    def __getitem__(self, key: int) -> Level:
        return self.levels[key]

    def __len__(self) -> int:
        return len(self.levels)

    def __setitem__(self, key: int, value: Level):
        self.levels[key] = value

    def __str__(self):
        s = ""
        for level in self.levels:
            s += f"{level}\n"

class GammaScheme:
    def __init__(self, gammas: list[Gamma] | None = None):
        if gammas is None:
            gammas = []
        self.gammas = gammas

    def add(self, Ei: keV, E: keV, jpi: list[Spin] | Spin | str, T: keV):
        if isinstance(jpi, str):
            jpi = str_to_spin(jpi)
        self.gammas.append(Gamma(Ei, E, jpi, T))

    def plot(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        for gamma in self.gammas:
            ax.axvline(gamma.E, **kwargs)

        return ax

    def scatter(self, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        for gamma in self.gammas:
            print(gamma.Ei, gamma.E)
            ax.scatter(gamma.E, gamma.Ei, **kwargs)

        return ax

    def append(self, gamma: Gamma):
        self.gammas.append(gamma)

    def __getitem__(self, e: int | str | Unitlike | slice) -> Gamma:
        match e:
            case int():
                return self.gammas[e]
            case str():
                return self.parse_str(e)
            case Unitlike():
                raise NotImplementedError()
            case slice():
                return parse_slice(e)

    def parse_slice(self, s: slice) -> GammaScheme:
        raise NotImplementedError()

    def parse_str(self, s: str) -> GammaScheme:
        # Split a string like 'Ei < 0.5keV' into ['Ei', '<', '0.5keV']
        lhs, op, rhs = re.split(r"([<>=])", s)
        lhs = lhs.strip().lower().capitalize()
        op = op.strip()
        rhs = rhs.strip().lower().capitalize()
        if lhs in {'Ei', 'Ef', 'E', 'Eg'}:
            left = True
        elif rhs in {'Ei', 'Ef', 'E', 'Eg'}:
            left = False
        else:
            raise ValueError(f"Could not parse {s}")

        cond = lhs if left else rhs
        numeric = rhs if left else lhs
        numeric = ureg(numeric)
        raise NotImplementedError()

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df['Ei'] = [gamma.Ei for gamma in self.gammas]
        df['Eg'] = [gamma.E for gamma in self.gammas]
        df['Ef'] = [gamma.Ef for gamma in self.gammas]
        df['jpi'] = [gamma.jpi for gamma in self.gammas]
        df['T'] = [gamma.T for gamma in self.gammas]
        return df

    @staticmethod
    def from_pandas(df: pd.DataFrame, Ei='Ei', Eg='Eg', Ef='Ef', jpi='jpi', T='T') -> GammaScheme:
        gammas = GammaScheme()
        for _, row in df.iterrows():
            gammas.add(row[Ei], row[Eg], row[jpi], row[T])
        return gammas

    def query_pandas(self, s: str) -> pd.DataFrame:
        df1 = self.to_pandas()
        df2 = df1.query(s)
        return df2

    def query(self, s: str) -> GammaScheme:
        df = self.query_pandas(s)
        return self.from_pandas(df)

    def Ei(self) -> np.ndarray:
        return np.array([gamma.Ei for gamma in self.gammas])

    def Ef(self) -> np.ndarray:
        return np.array([gamma.Ef for gamma in self.gammas])

    def E(self) -> np.ndarray:
        return np.array([gamma.E for gamma in self.gammas])

    def Eg(self) -> np.ndarray:
        return self.E()

    def __len__(self) -> int:
        return len(self.gammas)

    def __setitem__(self, key: int, value: Gamma):
        self.gammas[key] = value

    def __str__(self):
        s = ""
        for gamma in self.gammas:
            s += f"{gamma}\n"
        return s


