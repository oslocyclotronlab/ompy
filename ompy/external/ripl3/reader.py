from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TextIO, Iterator, Callable, TYPE_CHECKING
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

from rainiest.base.chart import draw_chart

from .fwf import parse_fwf, FwfParseError

from ...nuclear.base.elements import Element

Pathlike: TypeAlias = str | Path
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"
RIPL3_LEVELS_PATH = DATA_PATH / "RIPL3" / "levels"
RIPL3_LEVEL_DENSITIES_PATH = DATA_PATH / "RIPL3" / "level_densities"

Nuclide: TypeAlias = tuple[int, int]
Delta: TypeAlias = int | tuple[int, int]

@dataclass
class RIPL3Record:
    element: Element
    identification: IdentificationRecord
    levels: list[LevelEntry]

    def __repr__(self) -> str:
        return f"RIPL3Record({self.element}, {len(self.levels)} levels)"

    def print(self):
        print(f"Element: {self.element}")
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
        ax_main.set_title(f"Energy Level Scheme of {self.element}")
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
                                  convert_missing=True))


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


def get_RIPL3_levels(element: Element) -> RIPL3Record:
    """ Get the RIPL3 data for an element.

    Parameters
    ----------
    element : Element
        The element to get the data for

    Returns
    -------
    RIPL3Record
        The data from the file
    """
    return read_RIPL3_levels(element, RIPL3_LEVELS_PATH / f"z{element.Z:03}.dat")


def read_RIPL3_levels(element: Element, path: Pathlike) -> RIPL3Record:
    """ Read a RIPL3 file and return the data as a RIPL3Record.

    Parameters
    ----------
    element : Element
        The element to read the data for
    path : Pathlike
        The path to the file to read

    Returns
    -------
    RIPL3Record
        The data from the file

    Raises
    ------
    ValueError
        If the element is not found in the file
    GammaRecordError
        If an error occurs while reading a gamma record
    """
    path = Path(path)
    with path.open() as f:
        identification = seek_element(f, element)
        if identification is None:
            raise ValueError(f"Element {element} not found in {path}")
        identification = fwf_identification_record(identification)

        levels: list[LevelEntry] = []
        for i in range(identification.number_of_levels):
            levels.append(read_level(f))

    return RIPL3Record(element, identification, levels)


def seek_element(handle: TextIO, element: Element) -> str | None:
    while line := handle.readline():
        if line[:5].strip() == element.mass_symbol:
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

    
@dataclass
class LevelDensityBSFGRecord:
    Z: int  # Atomic number
    A: int  # Mass number
    El: str  # Element symbol
    I0: float  # Spin of the ground state
    Bn: float  # Neutron separation energy
    D0: float  # Evaluated average resonance spacing
    Derr: float  # Uncertainty of the resonance spacing
    Nlow: int  # Lowest level used for the fit
    Ulow: float  # Excitation energy of the level Nlow
    Ntop: int  # Highest level used for the fit
    Utop: float  # Excitation energy of the level Ntop
    dW: float  # Shell correction energy used in the Ignatyuk formula
    gamma: float  # Damping parameter of the Ignatyuk formula
    ainf: float  # Asymptotic level density parameter
    aerr: float  # Uncertainty of the asymptotic level density parameter
    pairing: float  # Effective energy shift

    def __post_init__(self):
        self.El = self.El.strip()

    def _repr_html_(self):
        """
        HTML representation for Jupyter notebook display.
        """
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    background: linear-gradient(to right, #f8f9fa, #e9ecef);">
            
            <div style="background: linear-gradient(to right, #3a1c71, #d76d77, #ffaf7b); 
                        padding: 15px; color: white; display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0; font-weight: 600;">{self.El}-{self.A} BSFG Level Density</h2>
                <div style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">
                    Z={self.Z}, A={self.A}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; padding: 20px;">
                <!-- Nucleus Properties Panel -->
                <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="margin-top: 0; color: #3a1c71; border-bottom: 2px solid #3a1c71; padding-bottom: 8px;">
                        <span style="font-size: 1.2em;">⚛</span> Nucleus Properties
                    </h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee; font-weight: 600; color: #555;">Element:</td>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee;">{self.El} (Z={self.Z}, A={self.A})</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee; font-weight: 600; color: #555;">Ground State Spin (I₀):</td>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee;">{self.I0}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee; font-weight: 600; color: #555;">Neutron Separation Energy:</td>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee;">{self.Bn} MeV</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 5px; font-weight: 600; color: #555;">Effective Pairing Energy:</td>
                            <td style="padding: 8px 5px;">{self.pairing} MeV</td>
                        </tr>
                    </table>
                </div>
                
                <!-- Resonance Data Panel -->
                <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="margin-top: 0; color: #d76d77; border-bottom: 2px solid #d76d77; padding-bottom: 8px;">
                        <span style="font-size: 1.2em;">📊</span> Resonance Data
                    </h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee; font-weight: 600; color: #555;">Avg. Resonance Spacing:</td>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee;">
                                <span style="font-family: 'Courier New', monospace;">D₀ = {self.D0} ± {self.Derr} eV</span>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee; font-weight: 600; color: #555;">Fitted Level Range:</td>
                            <td style="padding: 8px 5px; border-bottom: 1px solid #eee;">
                                N = {self.Nlow} → {self.Ntop}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 5px; font-weight: 600; color: #555;">Excitation Energy Range:</td>
                            <td style="padding: 8px 5px;">
                                U = {self.Ulow} → {self.Utop} MeV
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <!-- Ignatyuk Parameters Panel -->
            <div style="background-color: white; margin: 0 20px 20px; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="margin-top: 0; color: #ffaf7b; border-bottom: 2px solid #ffaf7b; padding-bottom: 8px;">
                    <span style="font-size: 1.2em;">🧮</span> Ignatyuk Formula Parameters
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #3a1c71;">
                        <div style="font-weight: 600; color: #555; font-size: 0.9em;">Asymptotic Level Density Parameter</div>
                        <div style="font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 5px;">a∞ = {self.ainf} ± {self.aerr} MeV⁻¹</div>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #d76d77;">
                        <div style="font-weight: 600; color: #555; font-size: 0.9em;">Shell Correction Energy</div>
                        <div style="font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 5px;">δW = {self.dW} MeV</div>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; border-left: 3px solid #ffaf7b;">
                        <div style="font-weight: 600; color: #555; font-size: 0.9em;">Damping Parameter</div>
                        <div style="font-family: 'Courier New', monospace; font-size: 1.1em; margin-top: 5px;">γ = {self.gamma} MeV⁻¹</div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div style="background-color: #f8f9fa; padding: 10px 20px; border-top: 1px solid #dee2e6; font-size: 0.8em; color: #6c757d; text-align: center;">
                Back-Shifted Fermi Gas Model (BSFG) • Level Density Data
            </div>
        </div>
        """
        return html


def read_RIPL3_level_density_BSFG(element: Element) -> LevelDensityBSFGRecord:
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-bfmeff.dat"
    with path.open() as f:
        line = seek_element_in_level_densities(f, element)
        if line is None:
            raise ValueError(f"Element {element} not found in {path}")
        return read_level_density_BSFG_record(line)


def seek_element_in_level_densities(handle: TextIO, element: Element) -> str | None:
    while line := handle.readline():
        # The first fortran format is 2i4
        try:
            Z, A = int(line[:4]), int(line[4:8])
        except ValueError:
            continue
        
        if Z == element.Z and A == element.A:
            return line

def available_bsfg() -> Iterator[Element]:
    """ List all available elements in the RIPL3 level densities file.
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-bfmeff.dat"
    with path.open() as f:
        while line := f.readline():
            try:
                Z, A = int(line[:4]), int(line[4:8])
            except ValueError:
                continue
            yield Element(A=A, Z=Z)


def read_RIPL3_level_densities_BSFG() -> Iterator[LevelDensityBSFGRecord]:
    """ Read the RIPL3 level densities for an element.

    Returns
    -------
    list[LevelDensityBSFGRecord]
        The level densities for the element
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-bfmeff.dat"
    with path.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            yield read_level_density_BSFG_record(line)

                
def read_level_density_BSFG_record(line: str) -> LevelDensityBSFGRecord:
    """ Read a level density record from a line.

    Parameters
    ----------
    line : str
        The line to read

    Returns
    ------- 
    LevelDensityBSFGRecord
        The level density record
    """
    return LevelDensityBSFGRecord(*parse_fwf(line, "2i4, 1x, a2, 1x, f4.1, 2x, f6.3, 1x, 1pe10.3, 1x, 1pe10.3, 0p, 1x, i3, 2x, f6.3, 2x, i3, 2x, f6.3, 3f10.5, f8.3, f10.5"))

    
@dataclass()
class LevelDensityCTRecord:
    Z: int  # Atomic number of the compound nucleus
    A: int  # Mass number of the compound nucleus
    El: str  # Element symbol of the compound nucleus
    I0: float  # Spin of the ground state of the target nucleus
    Bn: float  # Neutron binding energy of the compound nucleus in MeV
    D0: float  # Evaluated average resonance spacing in eV
    Derr: float  # Uncertainty of the resonance spacing in eV
    Nlow: int  # Lowest level used for the fit
    Ulow: float  # Excitation energy of the level Nlow in MeV
    Ntop: int  # Highest level used for the fit
    Utop: float  # Excitation energy of the level Ntop in MeV
    dW: float  # Shell correction energy used in the Ignatyuk formula
    gamma: float  # Damping parameter of the Ignatyuk formula
    ainf: float  # Asymptotic level density parameter
    aerr: float  # Uncertainty of the asymptotic level density parameter
    pairing: float  # Effective energy shift
    Ematch: float  # Energy at which the low and high energy formulae match in MeV
    E0: float  # Energy shift for the low-energy approach in MeV
    T: float  # Temperature for the low-energy approach in MeV

    def __post_init__(self):
        self.El = self.El.strip()

    def _repr_html_(self):
        """
        HTML representation for Jupyter notebook display.
        """
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3 style="color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Level Density Record: {self.El}-{self.A}</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                <div style="padding: 10px; background-color: #eef; border-radius: 5px;">
                    <h4 style="margin-top: 0; color: #445;">Nucleus Properties</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Z:</td>
                            <td style="padding: 3px;">{self.Z}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">A:</td>
                            <td style="padding: 3px;">{self.A}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Element:</td>
                            <td style="padding: 3px;">{self.El}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Target Spin (I₀):</td>
                            <td style="padding: 3px;">{self.I0}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Neutron Binding Energy (Bn):</td>
                            <td style="padding: 3px;">{self.Bn} MeV</td>
                        </tr>
                    </table>
                </div>
                
                <div style="padding: 10px; background-color: #efe; border-radius: 5px;">
                    <h4 style="margin-top: 0; color: #454;">Resonance Data</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Avg. Resonance Spacing (D₀):</td>
                            <td style="padding: 3px;">{self.D0} ± {self.Derr} eV</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Levels Used:</td>
                            <td style="padding: 3px;">{self.Nlow} → {self.Ntop}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Energy Range:</td>
                            <td style="padding: 3px;">{self.Ulow} → {self.Utop} MeV</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background-color: #fee; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #544;">Ignatyuk Formula Parameters</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 3px; font-weight: bold;">Shell Correction (dW):</td>
                        <td style="padding: 3px;">{self.dW}</td>
                        <td style="padding: 3px; font-weight: bold;">Damping Parameter (γ):</td>
                        <td style="padding: 3px;">{self.gamma}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px; font-weight: bold;">Asymptotic LDP (a∞):</td>
                        <td style="padding: 3px;">{self.ainf} ± {self.aerr}</td>
                        <td style="padding: 3px; font-weight: bold;">Pairing Energy:</td>
                        <td style="padding: 3px;">{self.pairing}</td>
                    </tr>
                </table>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background-color: #eef; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #445;">Low-Energy Formula Parameters</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 3px; font-weight: bold;">Energy Shift (E₀):</td>
                        <td style="padding: 3px;">{self.E0} MeV</td>
                        <td style="padding: 3px; font-weight: bold;">Temperature (T):</td>
                        <td style="padding: 3px;">{self.T} MeV</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px; font-weight: bold;">Matching Energy:</td>
                        <td style="padding: 3px;">{self.Ematch} MeV</td>
                        <td style="padding: 3px;"></td>
                        <td style="padding: 3px;"></td>
                    </tr>
                </table>
            </div>
        </div>
        """
        return html


def read_RIPL3_level_densities_CT() -> Iterator[LevelDensityCTRecord]:
    """ Read the RIPL3 level densities for an element.

    Returns
    -------
    list[LevelDensityCTRecord]
        The level densities for the element
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-ct.dat"
    with path.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            yield read_level_density_CT_record(line)

            
def read_RIPL3_level_density_CT(elem: Element) -> LevelDensityCTRecord:
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-ctmeff.dat"
    with path.open() as f:
        line = seek_element_in_level_densities(f, elem)
        if line is None:
            raise ValueError(f"Element {elem} not found in {path}")
        return read_level_density_CT_record(line)


def read_level_density_CT_record(line: str) -> LevelDensityCTRecord:
    """ Read a level density record from a line.

    Parameters
    ----------
    line : str
        The line to read

    Returns
    -------
    LevelDensityCTRecord
        The level density record
    """
    return LevelDensityCTRecord(*parse_fwf(line, "(2i4,1x,a2,1x,f4.1,2x,f6.3,1x,1pe10.3,1x,1pe10.3,0p,1x,i3,2x,f6.3,2x,i3,2x,f6.3,3f10.5,f8.3,4f10.5)"))


def available_ct() -> Iterator[Element]:
    """ List all available elements in the RIPL3 level densities file.
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / f"level-densities-ctmeff.dat"
    with path.open() as f:
        while line := f.readline():
            try:
                Z, A = int(line[:4]), int(line[4:8])
            except ValueError:
                continue
            yield Element(A=A, Z=Z)

@dataclass
class Neighbors:
    element: Element
    neighbors: list[Element]
    all: list[Element]

    def __iter__(self):
        return iter(self.neighbors)

    def __len__(self):
        return len(self.neighbors)

    def _repr_html_(self):
        return draw_chart(self.all, {'neighbors': self.neighbors, 'focus': [self.element]})


def neighbors(element: Element,
              nuclides: Iterator[Element] | Callable[[], Iterator[Element]],
              dZ: Delta,
              dN: Delta) -> Neighbors:
    """
    Filter and return neighboring nuclides based on allowed differences in proton and neutron numbers.
    
    Parameters:
        element: Element
            The reference element.
        nuclides: Iterator[Element] | Callable[[], Iterator[Element]]
            An iterator of elements.
        dZ: Delta
            Allowed deviation for proton number. If an int is provided, the range is [-dZ, dZ].
            If a tuple (plus, minus) is provided, the allowed range is [ -minus, +plus ].
        dN: Delta
            Allowed deviation for neutron number (calculated as A-Z). Same interpretation as dZ.
    
    Returns:
        Neighbors
            A list of nuclides that lie within the specified neighboring bounds.
    """
    baseZ, baseA = element.Z, element.A
    baseN = baseA - baseZ  # neutron number of the reference nuclide

    # Unpack allowed deviation for Z
    if isinstance(dZ, int):
        dZ_plus = dZ_minus = dZ
    else:
        dZ_plus, dZ_minus = dZ

    # Unpack allowed deviation for N
    if isinstance(dN, int):
        dN_plus = dN_minus = dN
    else:
        dN_plus, dN_minus = dN

    # Filter the candidate nuclides based on the allowed ranges
    if callable(nuclides):
        nuclides = list(nuclides())

    result: list[Element] = []

    for nuclide in nuclides:
        N = nuclide.A - nuclide.Z  # neutron number of candidate nuclide
        if (baseZ - dZ_minus <= nuclide.Z <= baseZ + dZ_plus and
            baseN - dN_minus <= N <= baseN + dN_plus):
            result.append(nuclide)
    return Neighbors(element, result, nuclides)

# Aliases
get_CT = read_RIPL3_level_density_CT
get_BSFG = read_RIPL3_level_density_BSFG

if __name__ == "__main__":
    import doctest

    doctest.testmod()
