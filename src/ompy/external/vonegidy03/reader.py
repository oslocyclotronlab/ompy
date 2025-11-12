"""Reader for AME2003 atomic mass evaluation data files (vonEgidy03).

This module provides parsers for the three AME2003 data files:
- mass.mas03: Atomic masses
- rct2.mas03: Reaction and separation energies (part 2)
- rct7.mas03: Pairing energies

The data files are from:
"The Ame2003 atomic mass evaluation (II)" by G.Audi, A.H.Wapstra and C.Thibault
Nuclear Physics A729 p. 337-676, December 22, 2003.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TextIO
from pathlib import Path

from ..ripl3.fwf import parse_fwf
from ...nuclear.base.nuclide import Nuclide
from ...units import ureg, Quantity

Pathlike = str | Path

# Default data paths
DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "egidy03"
MASS_TABLE_PATH = DATA_PATH / "mass.mas03"
REACTION_ENERGY_PATH = DATA_PATH / "rct2.mas03"
PAIRING_ENERGY_PATH = DATA_PATH / "rct7.mas03"


@dataclass
class MassRecord:
    """Record from the AME2003 atomic mass table (mass.mas03).
    
    Attributes
    ----------
    cc : str
        Fortran control character (page/line feed)
    NZ : int
        N-Z value
    N : int
        Neutron number
    Z : int
        Atomic number (proton number)
    A : int
        Mass number
    el : str
        Element symbol
    o : str
        Origin of the data (experimental flags)
    mass_excess : Quantity
        Mass excess (with units keV)
    mass_excess_unc : Quantity
        Uncertainty in mass excess (with units keV)
    binding_per_A : Quantity
        Binding energy per nucleon (with units keV)
    binding_per_A_unc : Quantity
        Uncertainty in binding energy per nucleon (with units keV)
    B : str
        Beta decay type indicator
    beta_energy : Quantity
        Beta-decay energy (with units keV)
    beta_energy_unc : Quantity
        Uncertainty in beta-decay energy (with units keV)
    flag : int
        Undocumented flag field (i3 format)
    atomic_mass : Quantity
        Atomic mass (with units micro-u)
    atomic_mass_unc : Quantity
        Uncertainty in atomic mass (with units micro-u)
    """
    cc: str
    NZ: int
    N: int
    Z: int
    A: int
    el: str
    o: str
    mass_excess: Quantity | None
    mass_excess_unc: Quantity | None
    binding_per_A: Quantity | None
    binding_per_A_unc: Quantity | None
    B: str | None
    beta_energy: Quantity | None
    beta_energy_unc: Quantity | None
    flag: int | None
    atomic_mass: Quantity | None
    atomic_mass_unc: Quantity | None

    def __post_init__(self):
        """Clean up string fields and handle empty strings."""
        for field in ['cc', 'el', 'o', 'B']:
            value = getattr(self, field)
            if isinstance(value, str):
                stripped = value.strip()
                setattr(self, field, stripped if stripped else None)

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the mass data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}Element: {self.el}-{self.A} (Z={self.Z}, N={self.N})")
        if self.mass_excess is not None:
            lines.append(f"{indent}Mass Excess: {self.mass_excess.magnitude:.5f} ± {self.mass_excess_unc.magnitude:.5f} {self.mass_excess.units:~P}")
        if self.binding_per_A is not None:
            lines.append(f"{indent}Binding Energy/A: {self.binding_per_A.magnitude:.3f} ± {self.binding_per_A_unc.magnitude:.3f} {self.binding_per_A.units:~P}")
        if self.beta_energy is not None:
            lines.append(f"{indent}Beta-Decay Energy: {self.beta_energy.magnitude:.3f} ± {self.beta_energy_unc.magnitude:.3f} {self.beta_energy.units:~P}")
        if self.atomic_mass is not None:
            lines.append(f"{indent}Atomic Mass: {self.atomic_mass.magnitude:.5f} ± {self.atomic_mass_unc.magnitude:.5f} {self.atomic_mass.units:~P}")
        return "\n".join(lines)

    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="background: linear-gradient(to right, #1e3c72, #2a5298); 
                        padding: 15px; color: white;">
                <h2 style="margin: 0;">{self.el}-{self.A} Mass Data</h2>
                <div style="font-size: 0.9em; opacity: 0.9;">Z={self.Z}, N={self.N}</div>
            </div>
            <div style="padding: 20px; background: white;">
                <table style="width: 100%; border-collapse: collapse;">
        """
        
        if self.mass_excess is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Mass Excess:</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.mass_excess.magnitude:.5f} ± {self.mass_excess_unc.magnitude:.5f} keV</td>
                    </tr>
            """
        
        if self.binding_per_A is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Binding Energy/A:</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.binding_per_A.magnitude:.3f} ± {self.binding_per_A_unc.magnitude:.3f} keV</td>
                    </tr>
            """
        
        if self.beta_energy is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Beta-Decay Energy ({self.B}):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.beta_energy.magnitude:.3f} ± {self.beta_energy_unc.magnitude:.3f} keV</td>
                    </tr>
            """
        
        if self.atomic_mass is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; font-weight: 600;">Atomic Mass:</td>
                        <td style="padding: 8px;">{self.atomic_mass.magnitude:.5f} ± {self.atomic_mass_unc.magnitude:.5f} µu</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </div>
        """
        return html


@dataclass
class ReactionEnergyRecord:
    """Record from the AME2003 reaction energies table (rct2.mas03).
    
    Attributes
    ----------
    cc : str
        Fortran control character (page/line feed)
    A : int
        Mass number
    el : str
        Element symbol
    Z : int
        Atomic number
    S_n : Quantity
        Neutron separation energy S(n) (with units keV)
    S_n_unc : Quantity
        Uncertainty in S(n) (with units keV)
    S_p : Quantity
        Proton separation energy S(p) (with units keV)
    S_p_unc : Quantity
        Uncertainty in S(p) (with units keV)
    Q_4B_minus : Quantity
        Q-value for 4 beta-minus decay (with units keV)
    Q_4B_minus_unc : Quantity
        Uncertainty in Q(4B-) (with units keV)
    Q_d_a : Quantity
        Q-value for (d,alpha) reaction (with units keV)
    Q_d_a_unc : Quantity
        Uncertainty in Q(d,a) (with units keV)
    Q_p_a : Quantity
        Q-value for (p,alpha) reaction (with units keV)
    Q_p_a_unc : Quantity
        Uncertainty in Q(p,a) (with units keV)
    Q_n_a : Quantity
        Q-value for (n,alpha) reaction (with units keV)
    Q_n_a_unc : Quantity
        Uncertainty in Q(n,a) (with units keV)
    """
    cc: str
    A: int
    el: str
    Z: int
    S_n: Quantity | None
    S_n_unc: Quantity | None
    S_p: Quantity | None
    S_p_unc: Quantity | None
    Q_4B_minus: Quantity | None
    Q_4B_minus_unc: Quantity | None
    Q_d_a: Quantity | None
    Q_d_a_unc: Quantity | None
    Q_p_a: Quantity | None
    Q_p_a_unc: Quantity | None
    Q_n_a: Quantity | None
    Q_n_a_unc: Quantity | None

    def __post_init__(self):
        """Clean up string fields."""
        for field in ['cc', 'el']:
            value = getattr(self, field)
            if isinstance(value, str):
                stripped = value.strip()
                setattr(self, field, stripped if stripped else None)

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the reaction energy data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}Element: {self.el}-{self.A} (Z={self.Z})")
        
        if self.S_n is not None:
            lines.append(f"{indent}S(n): {self.S_n.magnitude:.2f} ± {self.S_n_unc.magnitude:.2f} {self.S_n.units:~P}")
        if self.S_p is not None:
            lines.append(f"{indent}S(p): {self.S_p.magnitude:.2f} ± {self.S_p_unc.magnitude:.2f} {self.S_p.units:~P}")
        if self.Q_4B_minus is not None:
            lines.append(f"{indent}Q(4B-): {self.Q_4B_minus.magnitude:.2f} ± {self.Q_4B_minus_unc.magnitude:.2f} {self.Q_4B_minus.units:~P}")
        if self.Q_d_a is not None:
            lines.append(f"{indent}Q(d,a): {self.Q_d_a.magnitude:.2f} ± {self.Q_d_a_unc.magnitude:.2f} {self.Q_d_a.units:~P}")
        if self.Q_p_a is not None:
            lines.append(f"{indent}Q(p,a): {self.Q_p_a.magnitude:.2f} ± {self.Q_p_a_unc.magnitude:.2f} {self.Q_p_a.units:~P}")
        if self.Q_n_a is not None:
            lines.append(f"{indent}Q(n,a): {self.Q_n_a.magnitude:.2f} ± {self.Q_n_a_unc.magnitude:.2f} {self.Q_n_a.units:~P}")
        
        return "\n".join(lines)

    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="background: linear-gradient(to right, #134e5e, #71b280); 
                        padding: 15px; color: white;">
                <h2 style="margin: 0;">{self.el}-{self.A} Reaction Energies</h2>
                <div style="font-size: 0.9em; opacity: 0.9;">Z={self.Z}</div>
            </div>
            <div style="padding: 20px; background: white;">
                <table style="width: 100%; border-collapse: collapse;">
        """
        
        if self.S_n is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">S(n):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.S_n.magnitude:.2f} ± {self.S_n_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.S_p is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">S(p):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.S_p.magnitude:.2f} ± {self.S_p_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Q_4B_minus is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Q(4B-):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.Q_4B_minus.magnitude:.2f} ± {self.Q_4B_minus_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Q_d_a is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Q(d,a):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.Q_d_a.magnitude:.2f} ± {self.Q_d_a_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Q_p_a is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Q(p,a):</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.Q_p_a.magnitude:.2f} ± {self.Q_p_a_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Q_n_a is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; font-weight: 600;">Q(n,a):</td>
                        <td style="padding: 8px;">{self.Q_n_a.magnitude:.2f} ± {self.Q_n_a_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </div>
        """
        return html


@dataclass
class PairingEnergyRecord:
    """Record from the AME2003 pairing energies table (rct7.mas03).
    
    Attributes
    ----------
    cc : str
        Fortran control character (page/line feed)
    A : int
        Mass number
    el : str
        Element symbol
    Z : int
        Atomic number
    Pa : Quantity
        Pairing energy Pa (with units keV)
    Pa_unc : Quantity
        Uncertainty in Pa (with units keV)
    Dnn : Quantity
        Neutron-neutron pairing difference (with units keV)
    Dnn_unc : Quantity
        Uncertainty in Dnn (with units keV)
    Dpp : Quantity
        Proton-proton pairing difference (with units keV)
    Dpp_unc : Quantity
        Uncertainty in Dpp (with units keV)
    """
    cc: str
    A: int
    el: str
    Z: int
    Pa: Quantity | None
    Pa_unc: Quantity | None
    Dnn: Quantity | None
    Dnn_unc: Quantity | None
    Dpp: Quantity | None
    Dpp_unc: Quantity | None

    def __post_init__(self):
        """Clean up string fields."""
        for field in ['cc', 'el']:
            value = getattr(self, field)
            if isinstance(value, str):
                stripped = value.strip()
                setattr(self, field, stripped if stripped else None)

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the pairing energy data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}Element: {self.el}-{self.A} (Z={self.Z})")
        
        if self.Pa is not None:
            lines.append(f"{indent}Pa: {self.Pa.magnitude:.2f} ± {self.Pa_unc.magnitude:.2f} {self.Pa.units:~P}")
        if self.Dnn is not None:
            lines.append(f"{indent}Dnn: {self.Dnn.magnitude:.2f} ± {self.Dnn_unc.magnitude:.2f} {self.Dnn.units:~P}")
        if self.Dpp is not None:
            lines.append(f"{indent}Dpp: {self.Dpp.magnitude:.2f} ± {self.Dpp_unc.magnitude:.2f} {self.Dpp.units:~P}")
        
        return "\n".join(lines)

    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="background: linear-gradient(to right, #8e2de2, #4a00e0); 
                        padding: 15px; color: white;">
                <h2 style="margin: 0;">{self.el}-{self.A} Pairing Energies</h2>
                <div style="font-size: 0.9em; opacity: 0.9;">Z={self.Z}</div>
            </div>
            <div style="padding: 20px; background: white;">
                <table style="width: 100%; border-collapse: collapse;">
        """
        
        if self.Pa is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Pa:</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.Pa.magnitude:.2f} ± {self.Pa_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Dnn is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 600;">Dnn:</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{self.Dnn.magnitude:.2f} ± {self.Dnn_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        if self.Dpp is not None:
            html += f"""
                    <tr>
                        <td style="padding: 8px; font-weight: 600;">Dpp:</td>
                        <td style="padding: 8px;">{self.Dpp.magnitude:.2f} ± {self.Dpp_unc.magnitude:.2f} keV</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </div>
        """
        return html


def read_mass_record(line: str) -> MassRecord:
    """Parse a single line from mass.mas03 file.
    
    Parameters
    ----------
    line : str
        A line from the mass.mas03 file
        
    Returns
    -------
    MassRecord
        Parsed mass data record with Pint quantities
        
    Notes
    -----
    Format: a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.3,1x
    """
    # Use missing='*' to handle asterisks as missing values
    parsed = list(parse_fwf(line, "a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.3,1x", missing='*'))
    
    # Unpack and apply units
    cc, NZ, N, Z, A, el, o, mass_excess, mass_excess_unc, binding_per_A, binding_per_A_unc, B, beta_energy, beta_energy_unc, flag, atomic_mass, atomic_mass_unc = parsed
    
    # Apply units to energy values (keV) and atomic mass (micro-u)
    if mass_excess is not None:
        mass_excess = mass_excess * ureg.keV
        mass_excess_unc = mass_excess_unc * ureg.keV
    if binding_per_A is not None:
        binding_per_A = binding_per_A * ureg.keV
        binding_per_A_unc = binding_per_A_unc * ureg.keV
    if beta_energy is not None:
        beta_energy = beta_energy * ureg.keV
        beta_energy_unc = beta_energy_unc * ureg.keV
    if atomic_mass is not None:
        atomic_mass = atomic_mass * ureg.microamu
        atomic_mass_unc = atomic_mass_unc * ureg.microamu
    
    return MassRecord(cc, NZ, N, Z, A, el, o, mass_excess, mass_excess_unc, binding_per_A, binding_per_A_unc, B, beta_energy, beta_energy_unc, flag, atomic_mass, atomic_mass_unc)


def read_reaction_energy_record(line: str) -> ReactionEnergyRecord:
    """Parse a single line from rct2.mas03 file.
    
    Parameters
    ----------
    line : str
        A line from the rct2.mas03 file
        
    Returns
    -------
    ReactionEnergyRecord
        Parsed reaction energy data record with Pint quantities
        
    Notes
    -----
    Format: a1,i3,1x,a3,i3,1x,6(f10.2,f8.2)
    """
    # Expand the repeated format to avoid parentheses issue with rstrip
    parsed = list(parse_fwf(line, "a1,i3,1x,a3,i3,1x,f10.2,f8.2,f10.2,f8.2,f10.2,f8.2,f10.2,f8.2,f10.2,f8.2,f10.2,f8.2"))
    
    # Unpack and apply units
    cc, A, el, Z, S_n, S_n_unc, S_p, S_p_unc, Q_4B_minus, Q_4B_minus_unc, Q_d_a, Q_d_a_unc, Q_p_a, Q_p_a_unc, Q_n_a, Q_n_a_unc = parsed
    
    # Apply units - all are energies in keV
    if S_n is not None:
        S_n = S_n * ureg.keV
        S_n_unc = S_n_unc * ureg.keV
    if S_p is not None:
        S_p = S_p * ureg.keV
        S_p_unc = S_p_unc * ureg.keV
    if Q_4B_minus is not None:
        Q_4B_minus = Q_4B_minus * ureg.keV
        Q_4B_minus_unc = Q_4B_minus_unc * ureg.keV
    if Q_d_a is not None:
        Q_d_a = Q_d_a * ureg.keV
        Q_d_a_unc = Q_d_a_unc * ureg.keV
    if Q_p_a is not None:
        Q_p_a = Q_p_a * ureg.keV
        Q_p_a_unc = Q_p_a_unc * ureg.keV
    if Q_n_a is not None:
        Q_n_a = Q_n_a * ureg.keV
        Q_n_a_unc = Q_n_a_unc * ureg.keV
    
    return ReactionEnergyRecord(cc, A, el, Z, S_n, S_n_unc, S_p, S_p_unc, Q_4B_minus, Q_4B_minus_unc, Q_d_a, Q_d_a_unc, Q_p_a, Q_p_a_unc, Q_n_a, Q_n_a_unc)


def read_pairing_energy_record(line: str) -> PairingEnergyRecord:
    """Parse a single line from rct7.mas03 file.
    
    Parameters
    ----------
    line : str
        A line from the rct7.mas03 file
        
    Returns
    -------
    PairingEnergyRecord
        Parsed pairing energy data record with Pint quantities
        
    Notes
    -----
    Format: a1,i3,1x,a3,i3,1x,3(f10.2,f8.2)
    """
    # Expand the repeated format to avoid parentheses issue with rstrip
    parsed = list(parse_fwf(line, "a1,i3,1x,a3,i3,1x,f10.2,f8.2,f10.2,f8.2,f10.2,f8.2"))
    
    # Unpack and apply units
    cc, A, el, Z, Pa, Pa_unc, Dnn, Dnn_unc, Dpp, Dpp_unc = parsed
    
    # Apply units - all are energies in keV
    if Pa is not None:
        Pa = Pa * ureg.keV
        Pa_unc = Pa_unc * ureg.keV
    if Dnn is not None:
        Dnn = Dnn * ureg.keV
        Dnn_unc = Dnn_unc * ureg.keV
    if Dpp is not None:
        Dpp = Dpp * ureg.keV
        Dpp_unc = Dpp_unc * ureg.keV
    
    return PairingEnergyRecord(cc, A, el, Z, Pa, Pa_unc, Dnn, Dnn_unc, Dpp, Dpp_unc)


def seek_mass_nuclide(handle: TextIO, nuclide: Nuclide) -> str | None:
    """Seek to a specific nuclide in the mass.mas03 file.
    
    Parameters
    ----------
    handle : TextIO
        File handle to the mass.mas03 file, positioned after the header
    nuclide : Nuclide
        The nuclide to find
        
    Returns
    -------
    str | None
        The line containing the nuclide data, or None if not found
    """
    while line := handle.readline():
        if len(line) < 100:
            continue
        # Extract Z and A from positions based on format: a1,i3,i5,i5,i5
        # cc(1), NZ(3), N(5), Z(5), A(5)
        try:
            Z = int(line[9:14])  # Z position
            A = int(line[14:19])  # A position
            if Z == nuclide.Z and A == nuclide.A:
                return line
        except ValueError:
            # Skip lines that don't have valid Z and A
            continue
    return None


def seek_reaction_energy_nuclide(handle: TextIO, nuclide: Nuclide) -> str | None:
    """Seek to a specific nuclide in the rct2.mas03 file.
    
    Parameters
    ----------
    handle : TextIO
        File handle to the rct2.mas03 file, positioned after the header
    nuclide : Nuclide
        The nuclide to find
        
    Returns
    -------
    str | None
        The line containing the nuclide data, or None if not found
    """
    while line := handle.readline():
        if len(line) < 50:
            continue
        # Extract A and Z from positions based on format: a1,i3,1x,a3,i3
        # cc(1), A(3), space(1), el(3), Z(3)
        try:
            A = int(line[1:4])  # A position
            Z = int(line[8:11])  # Z position
            if Z == nuclide.Z and A == nuclide.A:
                return line
        except ValueError:
            # Skip lines that don't have valid Z and A
            continue
    return None


def seek_pairing_energy_nuclide(handle: TextIO, nuclide: Nuclide) -> str | None:
    """Seek to a specific nuclide in the rct7.mas03 file.
    
    Parameters
    ----------
    handle : TextIO
        File handle to the rct7.mas03 file, positioned after the header
    nuclide : Nuclide
        The nuclide to find
        
    Returns
    -------
    str | None
        The line containing the nuclide data, or None if not found
    """
    while line := handle.readline():
        if len(line) < 50:
            continue
        # Extract A and Z from positions based on format: a1,i3,1x,a3,i3
        # cc(1), A(3), space(1), el(3), Z(3)
        try:
            A = int(line[1:4])  # A position
            Z = int(line[8:11])  # Z position
            if Z == nuclide.Z and A == nuclide.A:
                return line
        except ValueError:
            # Skip lines that don't have valid Z and A
            continue
    return None


def read_mass_table(path: Pathlike | None = None) -> Iterator[MassRecord]:
    """Read the complete AME2003 mass table.
    
    Parameters
    ----------
    path : Pathlike, optional
        Path to the mass.mas03 file. If None, uses the default path.
        
    Yields
    ------
    MassRecord
        Mass data records
    """
    if path is None:
        path = MASS_TABLE_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        # Read all data lines
        for line in f:
            # Skip lines that don't contain data (check if line is long enough)
            if len(line) < 100:
                continue
            yield read_mass_record(line)


def read_reaction_energies(path: Pathlike | None = None) -> Iterator[ReactionEnergyRecord]:
    """Read the complete AME2003 reaction energies table.
    
    Parameters
    ----------
    path : Pathlike, optional
        Path to the rct2.mas03 file. If None, uses the default path.
        
    Yields
    ------
    ReactionEnergyRecord
        Reaction energy data records
    """
    if path is None:
        path = REACTION_ENERGY_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        # Read all data lines
        for line in f:
            # Skip lines that don't contain data
            if len(line) < 50:
                continue
            yield read_reaction_energy_record(line)


def read_pairing_energies(path: Pathlike | None = None) -> Iterator[PairingEnergyRecord]:
    """Read the complete AME2003 pairing energies table.
    
    Parameters
    ----------
    path : Pathlike, optional
        Path to the rct7.mas03 file. If None, uses the default path.
        
    Yields
    ------
    PairingEnergyRecord
        Pairing energy data records
    """
    if path is None:
        path = PAIRING_ENERGY_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        # Read all data lines
        for line in f:
            # Skip lines that don't contain data
            if len(line) < 50:
                continue
            yield read_pairing_energy_record(line)


def get_mass_data(nuclide: Nuclide, path: Pathlike | None = None) -> MassRecord:
    """Get mass data for a specific nuclide.
    
    Parameters
    ----------
    nuclide : Nuclide
        The nuclide to look up
    path : Pathlike, optional
        Path to the mass.mas03 file. If None, uses the default path.
        
    Returns
    -------
    MassRecord
        Mass data for the requested nuclide
        
    Raises
    ------
    ValueError
        If the nuclide is not found in the table
    """
    if path is None:
        path = MASS_TABLE_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        line = seek_mass_nuclide(f, nuclide)
        if line is None:
            raise ValueError(f"Nuclide {nuclide} not found in mass table")
        
        return read_mass_record(line)


def get_reaction_energy_data(nuclide: Nuclide, path: Pathlike | None = None) -> ReactionEnergyRecord:
    """Get reaction energy data for a specific nuclide.
    
    Parameters
    ----------
    nuclide : Nuclide
        The nuclide to look up
    path : Pathlike, optional
        Path to the rct2.mas03 file. If None, uses the default path.
        
    Returns
    -------
    ReactionEnergyRecord
        Reaction energy data for the requested nuclide
        
    Raises
    ------
    ValueError
        If the nuclide is not found in the table
    """
    if path is None:
        path = REACTION_ENERGY_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        line = seek_reaction_energy_nuclide(f, nuclide)
        if line is None:
            raise ValueError(f"Nuclide {nuclide} not found in reaction energies table")
        
        return read_reaction_energy_record(line)


def get_pairing_energy_data(nuclide: Nuclide, path: Pathlike | None = None) -> PairingEnergyRecord:
    """Get pairing energy data for a specific nuclide.
    
    Parameters
    ----------
    nuclide : Nuclide
        The nuclide to look up
    path : Pathlike, optional
        Path to the rct7.mas03 file. If None, uses the default path.
        
    Returns
    -------
    PairingEnergyRecord
        Pairing energy data for the requested nuclide
        
    Raises
    ------
    ValueError
        If the nuclide is not found in the table
    """
    if path is None:
        path = PAIRING_ENERGY_PATH
    
    path = Path(path)
    with path.open('r') as f:
        # Skip the 39-line header
        for _ in range(39):
            f.readline()
        
        line = seek_pairing_energy_nuclide(f, nuclide)
        if line is None:
            raise ValueError(f"Nuclide {nuclide} not found in pairing energies table")
        
        return read_pairing_energy_record(line)


@dataclass
class VonEgidy03:
    """Complete AME2003 data for a specific nuclide.
    
    This class combines mass, reaction energy, and pairing energy data
    for a single nuclide from the AME2003 atomic mass evaluation.
    
    Attributes
    ----------
    nuclide : Nuclide
        The nuclide this data represents
    mass : MassRecord
        Mass and binding energy data
    reaction : ReactionEnergyRecord
        Reaction and separation energy data
    pairing : PairingEnergyRecord
        Pairing energy data
    """
    nuclide: Nuclide
    mass: MassRecord
    reaction: ReactionEnergyRecord
    pairing: PairingEnergyRecord
    
    @classmethod
    def from_nuclide(cls, nuclide: Nuclide) -> 'VonEgidy03':
        """Load all AME2003 data for a specific nuclide.
        
        Parameters
        ----------
        nuclide : Nuclide
            The nuclide to look up
            
        Returns
        -------
        VonEgidy03
            Complete data for the nuclide
            
        Raises
        ------
        ValueError
            If the nuclide is not found in any of the tables
            
        Examples
        --------
        >>> from ompy.nuclear.base.nuclide import Nuclide
        >>> fe57 = Nuclide(A=57, Z=26)
        >>> data = VonEgidy03.from_nuclide(fe57)
        >>> print(data.mass.mass_excess)
        """
        mass = get_mass_data(nuclide)
        reaction = get_reaction_energy_data(nuclide)
        pairing = get_pairing_energy_data(nuclide)
        return cls(nuclide=nuclide, mass=mass, reaction=reaction, pairing=pairing)
    
    @classmethod
    def from_any(cls, input: str | Nuclide) -> 'VonEgidy03':
        """Load AME2003 data from either a string or Nuclide.
        
        Parameters
        ----------
        input : str | Nuclide
            Either a Nuclide object or a string representation like "57Fe"
            
        Returns
        -------
        VonEgidy03
            Complete data for the nuclide
            
        Raises
        ------
        ValueError
            If the input cannot be parsed or the nuclide is not found
            
        Examples
        --------
        >>> data = VonEgidy03.from_any("57Fe")
        >>> from ompy.nuclear.base.nuclide import Nuclide
        >>> data = VonEgidy03.from_any(Nuclide(A=57, Z=26))
        """
        # Use Nuclide.from_any which handles both string and Nuclide inputs
        nuclide = Nuclide.from_any(input)
        return cls.from_nuclide(nuclide)
    
    def summary(self, indent: str = "") -> str:
        """Generate a comprehensive text summary of all data.
        
        Parameters
        ----------
        indent : str, optional
            Indentation string to prepend to each line
            
        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append(f"{indent}{'=' * 60}")
        lines.append(f"{indent}AME2003 Data for {self.nuclide}")
        lines.append(f"{indent}{'=' * 60}")
        lines.append("")
        
        lines.append(f"{indent}MASS DATA:")
        lines.append(f"{indent}{'-' * 60}")
        if self.mass.mass_excess is not None:
            lines.append(f"{indent}  Mass Excess:      {self.mass.mass_excess.magnitude:12.5f} ± {self.mass.mass_excess_unc.magnitude:10.5f} {self.mass.mass_excess.units:~P}")
        if self.mass.binding_per_A is not None:
            lines.append(f"{indent}  Binding/A:        {self.mass.binding_per_A.magnitude:12.3f} ± {self.mass.binding_per_A_unc.magnitude:10.3f} {self.mass.binding_per_A.units:~P}")
        if self.mass.beta_energy is not None:
            lines.append(f"{indent}  Beta Energy ({self.mass.B}): {self.mass.beta_energy.magnitude:12.3f} ± {self.mass.beta_energy_unc.magnitude:10.3f} {self.mass.beta_energy.units:~P}")
        if self.mass.atomic_mass is not None:
            lines.append(f"{indent}  Atomic Mass:      {self.mass.atomic_mass.magnitude:12.5f} ± {self.mass.atomic_mass_unc.magnitude:10.5f} {self.mass.atomic_mass.units:~P}")
        lines.append("")
        
        lines.append(f"{indent}SEPARATION ENERGIES:")
        lines.append(f"{indent}{'-' * 60}")
        if self.reaction.S_n is not None:
            lines.append(f"{indent}  S(n):             {self.reaction.S_n.magnitude:12.2f} ± {self.reaction.S_n_unc.magnitude:10.2f} {self.reaction.S_n.units:~P}")
        if self.reaction.S_p is not None:
            lines.append(f"{indent}  S(p):             {self.reaction.S_p.magnitude:12.2f} ± {self.reaction.S_p_unc.magnitude:10.2f} {self.reaction.S_p.units:~P}")
        lines.append("")
        
        lines.append(f"{indent}REACTION Q-VALUES:")
        lines.append(f"{indent}{'-' * 60}")
        if self.reaction.Q_4B_minus is not None:
            lines.append(f"{indent}  Q(4B-):           {self.reaction.Q_4B_minus.magnitude:12.2f} ± {self.reaction.Q_4B_minus_unc.magnitude:10.2f} {self.reaction.Q_4B_minus.units:~P}")
        if self.reaction.Q_d_a is not None:
            lines.append(f"{indent}  Q(d,a):           {self.reaction.Q_d_a.magnitude:12.2f} ± {self.reaction.Q_d_a_unc.magnitude:10.2f} {self.reaction.Q_d_a.units:~P}")
        if self.reaction.Q_p_a is not None:
            lines.append(f"{indent}  Q(p,a):           {self.reaction.Q_p_a.magnitude:12.2f} ± {self.reaction.Q_p_a_unc.magnitude:10.2f} {self.reaction.Q_p_a.units:~P}")
        if self.reaction.Q_n_a is not None:
            lines.append(f"{indent}  Q(n,a):           {self.reaction.Q_n_a.magnitude:12.2f} ± {self.reaction.Q_n_a_unc.magnitude:10.2f} {self.reaction.Q_n_a.units:~P}")
        lines.append("")
        
        lines.append(f"{indent}PAIRING ENERGIES:")
        lines.append(f"{indent}{'-' * 60}")
        if self.pairing.Pa is not None:
            lines.append(f"{indent}  Pa:               {self.pairing.Pa.magnitude:12.2f} ± {self.pairing.Pa_unc.magnitude:10.2f} {self.pairing.Pa.units:~P}")
        if self.pairing.Dnn is not None:
            lines.append(f"{indent}  Dnn:              {self.pairing.Dnn.magnitude:12.2f} ± {self.pairing.Dnn_unc.magnitude:10.2f} {self.pairing.Dnn.units:~P}")
        if self.pairing.Dpp is not None:
            lines.append(f"{indent}  Dpp:              {self.pairing.Dpp.magnitude:12.2f} ± {self.pairing.Dpp_unc.magnitude:10.2f} {self.pairing.Dpp.units:~P}")
        
        return "\n".join(lines)
    
    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 20px auto; 
                    border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; color: white; text-align: center;">
                <h1 style="margin: 0; font-size: 2em; font-weight: 600;">
                    {self.mass.el}-{self.mass.A}
                </h1>
                <div style="font-size: 1.2em; opacity: 0.95; margin-top: 8px;">
                    AME2003 Atomic Mass Evaluation
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; 
                            border-radius: 20px; display: inline-block; margin-top: 12px; font-size: 0.95em;">
                    Z={self.mass.Z}, N={self.mass.N}
                </div>
            </div>
            
            <div style="background: #f8f9fa; padding: 30px;">
                
                <!-- Mass Data Panel -->
                <div style="background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <h2 style="margin-top: 0; color: #667eea; border-bottom: 3px solid #667eea; 
                               padding-bottom: 10px; font-size: 1.4em;">
                        Mass Properties
                    </h2>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        """
        
        if self.mass.mass_excess is not None:
            mass_excess_str = f"{self.mass.mass_excess.magnitude:,.5f}".replace(',', "'")
            mass_excess_unc_str = f"{self.mass.mass_excess_unc.magnitude:,.5f}".replace(',', "'")
            html += f"""
                        <tr style="border-bottom: 1px solid #e9ecef;">
                            <td style="padding: 12px 10px; font-weight: 600; color: #495057; width: 40%;">
                                Mass Excess
                            </td>
                            <td style="padding: 12px 10px; font-family: 'Courier New', monospace; color: #212529;">
                                {mass_excess_str} ± {mass_excess_unc_str} keV
                            </td>
                        </tr>
            """
        
        if self.mass.binding_per_A is not None:
            binding_str = f"{self.mass.binding_per_A.magnitude:,.3f}".replace(',', "'")
            binding_unc_str = f"{self.mass.binding_per_A_unc.magnitude:,.3f}".replace(',', "'")
            html += f"""
                        <tr style="border-bottom: 1px solid #e9ecef;">
                            <td style="padding: 12px 10px; font-weight: 600; color: #495057;">
                                Binding Energy per Nucleon
                            </td>
                            <td style="padding: 12px 10px; font-family: 'Courier New', monospace; color: #212529;">
                                {binding_str} ± {binding_unc_str} keV
                            </td>
                        </tr>
            """
        
        if self.mass.beta_energy is not None:
            beta_str = f"{self.mass.beta_energy.magnitude:,.3f}".replace(',', "'")
            beta_unc_str = f"{self.mass.beta_energy_unc.magnitude:,.3f}".replace(',', "'")
            html += f"""
                        <tr style="border-bottom: 1px solid #e9ecef;">
                            <td style="padding: 12px 10px; font-weight: 600; color: #495057;">
                                Beta-Decay Energy ({self.mass.B if self.mass.B else "?"})
                            </td>
                            <td style="padding: 12px 10px; font-family: 'Courier New', monospace; color: #212529;">
                                {beta_str} ± {beta_unc_str} keV
                            </td>
                        </tr>
            """
        
        if self.mass.atomic_mass is not None:
            atomic_mass_str = f"{self.mass.atomic_mass.magnitude:,.5f}".replace(',', "'")
            atomic_mass_unc_str = f"{self.mass.atomic_mass_unc.magnitude:,.5f}".replace(',', "'")
            html += f"""
                        <tr>
                            <td style="padding: 12px 10px; font-weight: 600; color: #495057;">
                                Atomic Mass
                            </td>
                            <td style="padding: 12px 10px; font-family: 'Courier New', monospace; color: #212529;">
                                {atomic_mass_str} ± {atomic_mass_unc_str} μu
                            </td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
                
                <!-- Two column layout for reaction and pairing -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        """
        
        # Separation Energies Panel
        html += """
                    <div style="background: white; border-radius: 10px; padding: 20px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <h2 style="margin-top: 0; color: #f093fb; border-bottom: 3px solid #f093fb; 
                                   padding-bottom: 10px; font-size: 1.3em;">
                            Separation Energies
                        </h2>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        """
        
        if self.reaction.S_n is not None:
            sn_str = f"{self.reaction.S_n.magnitude:,.2f}".replace(',', "'")
            sn_unc_str = f"{self.reaction.S_n_unc.magnitude:,.2f}".replace(',', "'")
            html += f"""
                            <tr style="border-bottom: 1px solid #e9ecef;">
                                <td style="padding: 10px 8px; font-weight: 600; color: #495057;">S(n)</td>
                                <td style="padding: 10px 8px; font-family: 'Courier New', monospace; 
                                           font-size: 0.9em; color: #212529;">
                                    {sn_str} ± {sn_unc_str} keV
                                </td>
                            </tr>
            """
        
        if self.reaction.S_p is not None:
            sp_str = f"{self.reaction.S_p.magnitude:,.2f}".replace(',', "'")
            sp_unc_str = f"{self.reaction.S_p_unc.magnitude:,.2f}".replace(',', "'")
            html += f"""
                            <tr>
                                <td style="padding: 10px 8px; font-weight: 600; color: #495057;">S(p)</td>
                                <td style="padding: 10px 8px; font-family: 'Courier New', monospace; 
                                           font-size: 0.9em; color: #212529;">
                                    {sp_str} ± {sp_unc_str} keV
                                </td>
                            </tr>
            """
        
        html += """
                        </table>
                        
                        <h3 style="margin-top: 20px; margin-bottom: 10px; color: #667eea; 
                                   font-size: 1.1em;">Reaction Q-values</h3>
                        <table style="width: 100%; border-collapse: collapse;">
        """
        
        q_values = [
            ("Q(4B-)", self.reaction.Q_4B_minus, self.reaction.Q_4B_minus_unc),
            ("Q(d,α)", self.reaction.Q_d_a, self.reaction.Q_d_a_unc),
            ("Q(p,α)", self.reaction.Q_p_a, self.reaction.Q_p_a_unc),
            ("Q(n,α)", self.reaction.Q_n_a, self.reaction.Q_n_a_unc),
        ]
        
        for i, (label, value, unc) in enumerate(q_values):
            if value is not None:
                value_str = f"{value.magnitude:,.2f}".replace(',', "'")
                unc_str = f"{unc.magnitude:,.2f}".replace(',', "'")
                border_style = "" if i == len([v for v in q_values if v[1] is not None]) - 1 else "border-bottom: 1px solid #e9ecef;"
                html += f"""
                            <tr style="{border_style}">
                                <td style="padding: 8px; font-weight: 600; color: #495057; font-size: 0.9em;">{label}</td>
                                <td style="padding: 8px; font-family: 'Courier New', monospace; 
                                           font-size: 0.85em; color: #212529;">
                                    {value_str} ± {unc_str} keV
                                </td>
                            </tr>
                """
        
        html += """
                        </table>
                    </div>
        """
        
        # Pairing Energies Panel
        html += """
                    <div style="background: white; border-radius: 10px; padding: 20px; 
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <h2 style="margin-top: 0; color: #a8edea; border-bottom: 3px solid #a8edea; 
                                   padding-bottom: 10px; font-size: 1.3em;">
                            Pairing Energies
                        </h2>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        """
        
        pairing_values = [
            ("Pa", self.pairing.Pa, self.pairing.Pa_unc, "Total pairing"),
            ("Dnn", self.pairing.Dnn, self.pairing.Dnn_unc, "Neutron-neutron"),
            ("Dpp", self.pairing.Dpp, self.pairing.Dpp_unc, "Proton-proton"),
        ]
        
        for i, (label, value, unc, desc) in enumerate(pairing_values):
            if value is not None:
                value_str = f"{value.magnitude:,.2f}".replace(',', "'")
                unc_str = f"{unc.magnitude:,.2f}".replace(',', "'")
                border_style = "" if i == len([v for v in pairing_values if v[1] is not None]) - 1 else "border-bottom: 1px solid #e9ecef;"
                html += f"""
                            <tr style="{border_style}">
                                <td style="padding: 12px 10px;">
                                    <div style="font-weight: 600; color: #495057;">{label}</div>
                                    <div style="font-size: 0.85em; color: #6c757d; margin-top: 2px;">{desc}</div>
                                </td>
                                <td style="padding: 12px 10px; font-family: 'Courier New', monospace; 
                                           font-size: 0.9em; color: #212529;">
                                    {value_str} ± {unc_str} keV
                                </td>
                            </tr>
                """
        
        html += """
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; text-align: center; color: white; font-size: 0.9em;">
                Data from AME2003 (Audi, Wapstra & Thibault, Nuclear Physics A729, 2003)
            </div>
        </div>
        """
        
        return html
