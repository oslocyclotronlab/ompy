from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO, Iterator
import re

from .fwf import parse_fwf, FwfParseError
from ...nuclear.base.nuclide import Nuclide
from .stubs import Pathlike, RIPL3_LEVEL_DENSITIES_PATH


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

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the BSFG level density data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line (default: "")
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}Element: {self.El}-{self.A} (Z={self.Z}, A={self.A})")
        lines.append(f"{indent}Ground State Spin (I₀): {self.I0}")
        lines.append(f"{indent}Neutron Binding Energy (Bn): {self.Bn:.3f} MeV")
        lines.append(f"{indent}Effective Pairing Energy: {self.pairing:.3f} MeV")
        lines.append(f"{indent}Avg. Resonance Spacing (D₀): {self.D0} ± {self.Derr} eV")
        lines.append(f"{indent}Fitted Level Range: N={self.Nlow} → {self.Ntop}")
        lines.append(f"{indent}Excitation Energy Range: U={self.Ulow:.3f} → {self.Utop:.3f} MeV")
        lines.append(f"{indent}Asymptotic Level Density (a∞): {self.ainf:.5f} ± {self.aerr:.5f} MeV⁻¹")
        lines.append(f"{indent}Shell Correction (δW): {self.dW:.5f} MeV")
        lines.append(f"{indent}Damping Parameter (γ): {self.gamma:.5f} MeV⁻¹")
        return "\n".join(lines)

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


def read_RIPL3_level_density_BSFG(element: Nuclide) -> LevelDensityBSFGRecord:
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-bfmeff.dat"
    with path.open() as f:
        line = seek_element_in_level_densities(f, element)
        if line is None:
            raise ValueError(f"Element {element} not found in {path}")
        return read_level_density_BSFG_record(line)


def seek_element_in_level_densities(handle: TextIO, element: Nuclide) -> str | None:
    while line := handle.readline():
        # The first fortran format is 2i4
        try:
            Z, A = int(line[:4]), int(line[4:8])
        except ValueError:
            continue
        
        if Z == element.Z and A == element.A:
            return line


def available_bsfg() -> Iterator[Nuclide]:
    """ List all available elements in the RIPL3 level densities file.
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-bfmeff.dat"
    with path.open() as f:
        while line := f.readline():
            try:
                Z, A = int(line[:4]), int(line[4:8])
            except ValueError:
                continue
            yield Nuclide(A=A, Z=Z)


def read_RIPL3_level_densities_BSFG() -> Iterator[LevelDensityBSFGRecord]:
    """ Read the RIPL3 level densities for an element.

    Returns
    -------
    list[LevelDensityBSFGRecord]
        The level densities for the element
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-bfmeff.dat"
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

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the CT level density data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line (default: "")
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}Element: {self.El}-{self.A} (Z={self.Z}, A={self.A})")
        lines.append(f"{indent}Target Spin (I₀): {self.I0}")
        lines.append(f"{indent}Neutron Binding Energy (Bn): {self.Bn:.3f} MeV")
        lines.append(f"{indent}Effective Pairing Energy: {self.pairing:.3f} MeV")
        lines.append(f"{indent}Avg. Resonance Spacing (D₀): {self.D0} ± {self.Derr} eV")
        lines.append(f"{indent}Fitted Level Range: N={self.Nlow} → {self.Ntop}")
        lines.append(f"{indent}Excitation Energy Range: U={self.Ulow:.3f} → {self.Utop:.3f} MeV")
        lines.append(f"{indent}Asymptotic Level Density (a∞): {self.ainf:.5f} ± {self.aerr:.5f} MeV⁻¹")
        lines.append(f"{indent}Shell Correction (δW): {self.dW:.5f} MeV")
        lines.append(f"{indent}Damping Parameter (γ): {self.gamma:.5f} MeV⁻¹")
        lines.append(f"{indent}Matching Energy (Ematch): {self.Ematch:.3f} MeV")
        lines.append(f"{indent}Energy Shift (E₀): {self.E0:.3f} MeV")
        lines.append(f"{indent}Temperature (T): {self.T:.3f} MeV")
        return "\n".join(lines)

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
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-ct.dat"
    with path.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            yield read_level_density_CT_record(line)

            
def read_RIPL3_level_density_CT(elem: Nuclide) -> LevelDensityCTRecord:
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-ctmeff.dat"
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


def available_ct() -> Iterator[Nuclide]:
    """ List all available elements in the RIPL3 level densities file.
    """
    path = RIPL3_LEVEL_DENSITIES_PATH / "level-densities-ctmeff.dat"
    with path.open() as f:
        while line := f.readline():
            try:
                Z, A = int(line[:4]), int(line[4:8])
            except ValueError:
                continue
            yield Nuclide(A=A, Z=Z)


# Aliases
get_CT = read_RIPL3_level_density_CT
get_BSFG = read_RIPL3_level_density_BSFG
