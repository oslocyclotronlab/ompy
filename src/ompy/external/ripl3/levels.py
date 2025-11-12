from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TextIO, Iterator, Dict, Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

from .fwf import parse_fwf, FwfParseError
from ...nuclear.base.nuclide import Nuclide
from .stubs import Pathlike, RIPL3_LEVELS_PATH

TNuclide: TypeAlias = tuple[int, int]


@dataclass
class RIPL3Record:
    nuclide: Nuclide
    identification: IdentificationRecord
    levels: list[LevelEntry]

    def __repr__(self) -> str:
        return f"RIPL3Record({self.nuclide}, {len(self.levels)} levels)"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    background: linear-gradient(to right, #f8f9fa, #e9ecef);">
            
            <div style="background: linear-gradient(to right, #3a1c71, #d76d77, #ffaf7b); 
                        padding: 15px; color: white;">
                <h2 style="margin: 0; font-weight: 600;">RIPL3 Level Scheme: {self.nuclide}</h2>
                <p style="margin: 5px 0 0 0;">Number of levels: {len(self.levels)}</p>
            </div>

            <div style="padding: 15px;">
                <div style="margin-bottom: 15px;">
                    <strong>Identification:</strong><br>
                    Sn = {self.identification.Sn:.3f} MeV<br>
                    Sp = {self.identification.Sp:.3f} MeV<br>
                    Complete up to level {self.identification.Nmax}<br>
                    Unique spins/parities up to level {self.identification.Nc}
                </div>

                <div style="max-height: 300px; overflow-y: auto; border: 1px solid #dee2e6; 
                            border-radius: 4px; padding: 10px;">
                    <strong>First 10 levels:</strong><br>"""

        for level in self.levels[:10]:
            try:
                s = int(level.level.s)
            except ValueError:
                s = level.level.s
            pi = '⁺' if level.level.p == 1 else '⁻'
            html += f"""
                    <div style="margin: 5px 0; padding: 5px; background: rgba(0,0,0,0.03);">
                        Level {level.level.N1}: E={level.level.Elv:.3f} MeV, Jπ={s}{pi}, T½={level.level.T1_2}
                    </div>"""

        html += """
                </div>
            </div>
        </div>
        """
        return html

    def print(self):
        print(f"Element: {self.nuclide}")
        print(f"Identification: {self.identification}")
        print("Level Scheme:")
        for level in self.levels:
            try:
                s = int(level.level.s)
            except ValueError:
                s = int(level.level.s)
            pi = '⁺' if level.level.p == 1 else '⁻'
            print(
                f"  {level.level.N1}: E={level.level.Elv} MeV, Jπ={s}{pi}, T½={level.level.T1_2}")
            for gamma in level.gammas:
                print(f"    -> {gamma.Nf}: Eγ={gamma.Eg} MeV, Pg={gamma.Pg}, Pe={gamma.Pe}, ICC={gamma.ICC}")
        print("\n")

    def to_pandas(self, drop_unknown_parity: bool = False) -> pd.DataFrame:
        """ Convert the levels to a pandas DataFrame

        The columns are the excitation energy, spin, and parity.

        Returns
        -------
        pd.DataFrame
            The levels as a DataFrame

        """
        Ex = []
        J = []
        pi = []
        T1_2 = []
        for level in self.levels:
            Ex.append(level.level.Ex)
            J.append(level.level.s)
            pi.append(level.level.p)
            T1_2.append(level.level.T1_2)
        df = pd.DataFrame({"Ex": Ex, "J": J, "pi": pi, "T1_2": T1_2})
        if drop_unknown_parity:
            df = df.query("pi != 0")
        return df

    def plot(
        self,
        ax: tuple[Axes, Axes] | None = None,
        cumulative: bool = False,
        bins: int = 20,
        binwidth: float | None = None
    ):

        df = self.to_pandas()

        # Filter out invalid J values
        df = df[df["J"] >= 0]

        # Set up the figure and gridspec for narrow subplot
        if ax is None:
            fig = plt.figure()
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

            ax_main = fig.add_subplot(gs[0])
            ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
        else:
            fig = ax.get_figure()
            ax_main, ax_hist = ax[0], ax[1]

        # Plot energy levels as horizontal lines
        for _, row in df.iterrows():
            color = 'black' if row["pi"] == 1 else 'red'
            ax_main.hlines(row["Ex"], row["J"] - 0.4, row["J"] + 0.4, color=color, lw=2)

        ax_main.set_xlabel("Spin J")
        ax_main.set_ylabel("Excitation Energy Ex [MeV]")
        ax_main.set_title(f"Energy Level Scheme of {self.nuclide}")
        ax_main.grid(True)

        # Histogram data
        energy_data = df["Ex"]

        if binwidth is not None:
            bins = np.arange(0, energy_data.max() + binwidth, binwidth)
        else:
            bins = bins

        ax_hist.hist(energy_data, bins=bins, orientation='horizontal',
                    cumulative=cumulative, color='gray', edgecolor='black', alpha=0.7)

        ax_hist.set_xlabel("Count")
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        ax_hist.grid(True)

        return ax

@dataclass
class LevelEntry:
    level: LevelRecord
    gammas: list[GammaRecord]


@dataclass
class IdentificationRecord:
    symbol: str
    A: int
    Z: int
    Nol: int  # Number of levels in the decay scheme
    Nog: int  # Number of gamma transitions in the decay scheme
    Nmax: int  # maximum number of levels up to which the level scheme is complete
    Nc: int  # Number of a level up to which the spins and parities are unique
    Sn: float  # Neutron separation energy
    Sp: float  # Proton separation energy

    def __post_init__(self):
        # Trim all strings
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())

    # Aliases
    @property
    def number_of_levels(self) -> int:
        return self.Nol


def fwf_identification_record(line: str) -> IdentificationRecord:
    #   SYMB    A     Z     Nol    Nog    Nmax    Nc     Sn[MeV]     Sp[MeV]
    #   22Mg    22    12    17     18      9      4     19.382000    5.497000
    # The corresponding FORTRAN format is (a5,6i5,2f12.6)
    # SYMB   : mass number with symbol of the element
    # A      : mass number
    # Z      : atomic number
    # Nol    : number of levels in the decay scheme
    # Nog    : number of gamma rays in the decay scheme
    # Nmax   : maximum number of levels up to which the level scheme is
    #          complete
    # Nc     : number of a level up to which spins and parities are unique
    # Sn     : neutron separation energy in MeV
    # Sp     : proton separation energy in MeV
    return IdentificationRecord(*parse_fwf(line, "a5, 6i5, 2f12.6"))


@dataclass
class LevelRecord:
    """
    The RIPL format description is contradictory. The third entry is the unqiue spin,
    but the symbols is either s or J.
    This follows the "levels-readme.html" file.
    """
    N1: int  # Serial number of the level
    Elv: float | None  # Energy of the level in MeV
    s: float | None  # Assigned unique spin
    p: int | None  # Assigned unique parity
    T1_2: float | None  # Half-life of the level in seconds
    Ng: int  # Number of gamma rays de-exciting the level
    J: str | None  # Flag for spin estimation method
    unc: str | None  # Flag for an uncertain level energy
    spins: str | None  # Original spins from the ENSDF file
    nd: int | None  # Number of decay modes of the level

    # m: float  # Decay percentage modifier
    # percent: float  # Percentage decay of different decay modes
    # mode: str  # Short indication of decay modes of a level
    # shift: float  # Value assigned to the unknown "X", in MeV
    # band: int  # Integer assigned to band(s) to which the level is part
    def __post_init__(self):
        # Trim all strings
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())
            if getattr(self, field) == '':
                setattr(self, field, None)
        if self.Ng is None:
            self.Ng = 0

    @property
    def Ex(self) -> float:
        return self.Elv

    @property
    def number_of_gammas(self) -> int:
        return self.Ng

    @property
    def level_id(self) -> int:
        return self.N1


def read_level_record(line):
    # N1  Elv[MeV]  s   p   T1/2    Ng  J  unc  spins   nd  m  percent  mode      m   percent   mode    /.../ shift     band
    # 1  0.000000  0.0  1  3.86E+00  0           0+      2  =  8.2000E+01 %IT     =  1.8000E+01 %B-     /.../ 0.031100  2
    # 2  1.246300  2.0  1  2.10E12   1           2+      0
    # (i3,1x,f10.6,1x,f5.1,i3,1x,(e10.3),i3,1x,a1,1x,a4,1x,a18,i3,10(1x,a2,1x,e10.4,1x,a7),f10.6,1x,3(2i))
    # Nl    :  sequential number of a level i3
    # Elv   :  energy of the level in MeV f10.6
    # s     :  level spin (unique). Whenever possible unknown spins up to f5.1
    # p     :  parity (unique). If the parity of the level was unknown, positive or| i3
    # T1/2  :  half-life of the level (if known). All known half-lives or level widths e10.3
    # Ng    :  number of gamma rays de-exciting the level. i3
    # J     :  flag for spin estimation method (see below the list of possible flags). a1
    # unc   :  flag for an uncertain level energy. When impossible to determine, the a4
    # spins :  original spins from the ENSDF file. a18
    # nd    :  number of decay modes of the level (if known). Values from 0 through 10| i3
    # m     :  decay percentage modifier; informs a user about major uncertainties.
    # percent: percentage decay of different decay modes. As a general rule  the
    # mode   : short indication of decay modes of a level (see Table below).
    # shift  : value assigned to the unknown "X", in MeV.
    # band   : integer assigned to band(s) to which the level is part
    # (i3,1x,f10.6,1x,f5.1,i3,1x,(e10.3),i3,1x,a1,1x,a4,1x,a18,i3,10(1x,a2,1x,e10.4,1x,a7),f10.6,1x,3(2i))
    return LevelRecord(*parse_fwf(line, "i3,1x,f10.6,1x,f5.1,i3,1x,e10.3,i3,1x,a1,1x,a4,1x,a18,i3",
                                missing=" "))


@dataclass
class GammaRecord:
    Nf: int  # Sequential number of the final state
    Eg: float  # Gamma-ray energy in MeV
    # Pg    : Probability that the level decays through photon (gamma ray) emission.
    #         If no branching ratio is given in the ENSDF file, Pg=0.
    Pg: float | None  # Probability of the level decaying through photon (gamma ray) emission
    # Pe    : Probability of the electromagnetic transition (photon, conversion electron, pair creation).
    #         The sum of the Pe gives the IT (electromagnetic transition) branching ratio of the level.
    Pe: float | None
    # ICC   : Internal conversion coefficient of the transition.
    ICC: float | None

    def __hash__(self) -> int:
        return hash((self.Nf, self.Eg, self.Pg, self.Pe, self.ICC))


class GammaRecordError(Exception):
    pass


def read_gamma_record(line):
    #     Two examples of the gamma records are given below:
    #   Nf    Eg[MeV]       Pg          Pe          ICC
    #   3      0.055     2.790E-01   2.870E-01    5.130E-03
    #   1      0.113     5.330E-01   5.330E-01    0.000E+00
    # The corresponding FORTRAN format is (39x,i4,1x,f10.4,3(1x,e10.3))
    # Nf    : sequential number of the final state
    # Eg    : gamma-ray energy in MeV
    # Pg    : Probability that the level decays through photon (gamma ray) emission.
    #         If no branching ratio is given in the ENSDF file, Pg=0.
    # Pe    : Probability of the electromagnetic transition (photon, conversion electron, pair creation).
    #         The sum of the Pe gives the IT (electromagnetic transition) branching ratio of the level.
    #         This is 1 unless other decay modes are listed in the level record (see example below).
    # ICC   : Internal conversion coefficient of the transition.
    #  (39x,i4,1x,f10.4,3(1x,e10.3))
    return GammaRecord(*parse_fwf(line, "39x, i4, 1x, f10.4, 1x, e10.3, 1x, e10.3, 1x, e10.3"))


def get_RIPL3_levels(nuclide: Nuclide | str) -> RIPL3Record:
    """ Get the RIPL3 data for a nuclide.

    Parameters
    ----------
    nuclide : Nuclide
        The nuclide to get the data for

    Returns
    -------
    RIPL3Record
        The data from the file
    """
    nuclide = Nuclide.from_any(nuclide)
    return read_RIPL3_levels(nuclide, RIPL3_LEVELS_PATH / f"z{nuclide.Z:03}.dat")


def read_RIPL3_levels(nuclide: Nuclide, path: Pathlike) -> RIPL3Record:
    """ Read a RIPL3 file and return the data as a RIPL3Record.

    Parameters
    ----------
    nuclide : Nuclide
        The nuclide to read the data for
    path : Pathlike
        The path to the file to read

    Returns
    -------
    RIPL3Record
        The data from the file

    Raises
    ------
    ValueError
        If the nuclide is not found in the file
    GammaRecordError
        If an error occurs while reading a gamma record
    """
    path = Path(path)
    with path.open() as f:
        identification = seek_nuclide(f, nuclide)
        if identification is None:
            raise ValueError(f"Element {nuclide} not found in {path}")
        identification = fwf_identification_record(identification)

        levels: list[LevelEntry] = []
        for i in range(identification.number_of_levels):
            levels.append(read_level(f))

    return RIPL3Record(nuclide, identification, levels)


def seek_nuclide(handle: TextIO, nuclide: Nuclide) -> str | None:
    while line := handle.readline():
        if line[:5].strip() == nuclide.mass_symbol:
            return line


def read_level(handle: TextIO) -> LevelEntry:
    level: LevelRecord = read_level_record(handle.readline())
    gammas: list[GammaRecord] = []
    for ng in range(level.number_of_gammas):
        line = handle.readline()
        try:
            gammas.append(read_gamma_record(line))
        except FwfParseError as e:
            raise GammaRecordError(f"Error parsing gamma record {ng} for level {level.level_id}: {line}") from e
    return LevelEntry(level, gammas)
