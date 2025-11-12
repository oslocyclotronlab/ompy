from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO, Iterator, Optional, TYPE_CHECKING
from collections import namedtuple

if TYPE_CHECKING:
    from .references import Reference
from .fwf import parse_fwf, FwfParseError
from ...nuclear.base.nuclide import Nuclide
from .stubs import RIPL3_RESONANCE_SPACING_PATH


@dataclass
class ResonanceSpacingRecord:
    """Record for resonance spacing data from RIPL3 files."""
    Z: int  # Charge number of the target nucleus
    El: str  # Element symbol of the target nucleus
    A: int  # Mass number of the target nucleus
    Io: float  # Spin of the ground state of the target nucleus
    Bn: float  # Neutron binding energy for the corresponding compound nucleus in MeV
    D: float  # Average resonance spacing in keV (D0 for s-wave, D1 for p-wave)
    dD: float  # Uncertainty of D in keV
    S: float  # Neutron strength function in 10**(-4) (S0 for s-wave, S1 for p-wave)
    dS: float  # Uncertainty of S in 10**(-4)
    Gg: float | None  # Average radiative width in meV
    dG: float | None  # Uncertainty of Gg in meV
    ComRef: str  # Reference to the work in which the analysis was performed
    wave_type: str  # Type of wave: "s-wave" or "p-wave"

    def __post_init__(self):
        # Trim all strings
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())
            if getattr(self, field) == '':
                setattr(self, field, None)

    def get_reference(self) -> Optional[Reference]:
        
        """Get the reference information for this record.
        
        Returns
        -------
        Optional[Reference]
            The reference information, or None if not found
        """
        from .references import get_reference_by_code
        return get_reference_by_code(self.ComRef)

    def summary(self, indent: str = "") -> str:
        """Generate a text summary of the resonance spacing data.
        
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
        lines.append(f"{indent}Wave Type: {self.wave_type}")
        lines.append(f"{indent}Ground State Spin (I₀): {self.Io}")
        lines.append(f"{indent}Neutron Binding Energy (Bn): {self.Bn:.3f} MeV")
        lines.append(f"{indent}Resonance Spacing (D): {self.D:.3f} ± {self.dD:.3f} keV")
        lines.append(f"{indent}Strength Function (S): {self.S:.3f} ± {self.dS:.3f} × 10⁻⁴")
        if self.Gg is not None and self.dG is not None:
            lines.append(f"{indent}Radiative Width (Γγ): {self.Gg:.3f} ± {self.dG:.3f} meV")
        else:
            lines.append(f"{indent}Radiative Width (Γγ): Not available")
        if self.ComRef:
            lines.append(f"{indent}Reference: {self.ComRef}")
        return "\n".join(lines)

    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        # Get reference information
        ref = self.get_reference()
        
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3 style="color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Resonance Spacing Record: {self.El}-{self.A} ({self.wave_type})</h3>
            
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
                            <td style="padding: 3px; font-weight: bold;">Ground State Spin (I₀):</td>
                            <td style="padding: 3px;">{self.Io}</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Neutron Binding Energy:</td>
                            <td style="padding: 3px;">{self.Bn} MeV</td>
                        </tr>
                    </table>
                </div>
                
                <div style="padding: 10px; background-color: #efe; border-radius: 5px;">
                    <h4 style="margin-top: 0; color: #454;">Resonance Parameters</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Resonance Spacing (D):</td>
                            <td style="padding: 3px;">{self.D} ± {self.dD} keV</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Strength Function (S):</td>
                            <td style="padding: 3px;">{self.S} ± {self.dS} × 10⁻⁴</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Radiative Width:</td>
                            <td style="padding: 3px;">{self.Gg if self.Gg is not None else 'N/A'} ± {self.dG if self.dG is not None else 'N/A'} meV</td>
                        </tr>
                        <tr>
                            <td style="padding: 3px; font-weight: bold;">Reference:</td>
                            <td style="padding: 3px;">{ref.title if ref else self.ComRef}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        """
        return html


def read_resonance_spacing_record(line: str, wave_type: str) -> ResonanceSpacingRecord:
    """Read a resonance spacing record from a line.
    
    Parameters
    ----------
    line : str
        The line to read
    wave_type : str
        The type of wave: "s-wave" or "p-wave"
        
    Returns
    -------
    ResonanceSpacingRecord
        The resonance spacing record
    """
    # Format: (i3,1x,a2,1x,i3,2x,f3.1,2x,f6.3,2x,2(e8.2,2x),1x,2(f4.2,2x),2(f4.1,1x),2x,a4)
    # This format handles the repeated groups properly now
    try:
        # Use the proper FWF format with repeated groups
        values = list(parse_fwf(line, "i3,1x,a2,1x,i3,2x,f3.1,2x,f6.3,2x,2(e8.2,2x),1x,2(f4.2,2x),2(f4.1,1x),2x,a4", 
                              missing="-"))
        
        # Unpack the values according to the actual parsed format
        # Based on the debug output, we have 12 values total
        Z = values[0]
        El = values[1]
        A = values[2]
        Io = values[3]
        Bn = values[4]
        D = values[5]   # First e8.2 value
        dD = values[6]  # Second e8.2 value
        S = values[7]   # First f4.2 value
        dS = values[8]  # Second f4.2 value
        Gg = values[9]  # First f4.1 value (None if missing)
        dG = values[10] # Second f4.1 value (None if missing)
        ComRef = values[11] if len(values) > 11 else ''  # a4 field
        
        return ResonanceSpacingRecord(Z=Z, El=El, A=A, Io=Io, Bn=Bn, D=D, dD=dD, S=S, dS=dS, Gg=Gg, dG=dG, ComRef=ComRef, wave_type=wave_type)
        
    except FwfParseError:
        # Fall back to manual parsing if FWF parsing fails
        parts = line.split()
        
        if len(parts) < 8:
            raise FwfParseError(f"Not enough fields in line: {line}")
        
        Z = int(parts[0])
        El = parts[1]
        A = int(parts[2])
        Io = float(parts[3])
        Bn = float(parts[4])
        D = float(parts[5])
        dD = float(parts[6])
        S = float(parts[7])
        
        # Handle optional fields
        dS = float(parts[8]) if len(parts) > 8 and parts[8] != '' else 0.0
        Gg = None
        dG = None
        ComRef = ''
        
        # Check if we have more fields
        if len(parts) > 9:
            # Handle Gg field
            if len(parts) > 9 and parts[9] != '' and parts[9] != '-':
                try:
                    Gg = float(parts[9])
                except ValueError:
                    Gg = None
            
            # Handle dG field
            if len(parts) > 10 and parts[10] != '' and parts[10] != '-':
                try:
                    dG = float(parts[10])
                except ValueError:
                    dG = None
            
            # Handle reference field
            if len(parts) > 11:
                ComRef = parts[11]
            elif len(parts) > 10 and parts[10] != '' and not parts[10].replace('.', '').replace('-', '').isdigit():
                # Sometimes the reference is in the dG position
                ComRef = parts[10]
            elif len(parts) > 9 and parts[9] != '' and not parts[9].replace('.', '').replace('-', '').isdigit():
                # Sometimes the reference is in the Gg position
                ComRef = parts[9]
        
        return ResonanceSpacingRecord(Z=Z, El=El, A=A, Io=Io, Bn=Bn, D=D, dD=dD, S=S, dS=dS, Gg=Gg, dG=dG, ComRef=ComRef, wave_type=wave_type)


def read_RIPL3_resonance_spacing_s_wave(element: Nuclide) -> ResonanceSpacingRecord:
    """Read s-wave resonance spacing data for an element.
    
    Parameters
    ----------
    element : Nuclide
        The element to read the data for
        
    Returns
    -------
    ResonanceSpacingRecord
        The resonance spacing record
        
    Raises
    ------
    ValueError
        If the element is not found in the file
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances0.dat"
    with path.open() as f:
        line = seek_element_in_resonance_spacing(f, element)
        if line is None:
            raise ValueError(f"Element {element} not found in {path}")
        return read_resonance_spacing_record(line, "s-wave")


def read_RIPL3_resonance_spacing_p_wave(element: Nuclide) -> ResonanceSpacingRecord:
    """Read p-wave resonance spacing data for an element.
    
    Parameters
    ----------
    element : Nuclide
        The element to read the data for
        
    Returns
    -------
    ResonanceSpacingRecord
        The resonance spacing record
        
    Raises
    ------
    ValueError
        If the element is not found in the file
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances1.dat"
    with path.open() as f:
        line = seek_element_in_resonance_spacing(f, element)
        if line is None:
            raise ValueError(f"Element {element} not found in {path}")
        return read_resonance_spacing_record(line, "p-wave")


def seek_element_in_resonance_spacing(handle: TextIO, element: Nuclide) -> str | None:
    """Seek an element in the resonance spacing file.
    
    Parameters
    ----------
    handle : TextIO
        File handle to search
    element : Nuclide
        Element to find
        
    Returns
    -------
    str | None
        The line containing the element data, or None if not found
    """
    while line := handle.readline():
        # Skip comment lines
        if line.startswith('#'):
            continue
        try:
            # Parse Z and A from the line
            Z, A = int(line[:3]), int(line[7:10])
            if Z == element.Z and A == element.A:
                return line
        except (ValueError, IndexError):
            continue
    return None


def read_RIPL3_resonance_spacing_s_wave_all() -> Iterator[ResonanceSpacingRecord]:
    """Read all s-wave resonance spacing records.
    
    Returns
    -------
    Iterator[ResonanceSpacingRecord]
        Iterator over all s-wave resonance spacing records
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances0.dat"
    with path.open() as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                yield read_resonance_spacing_record(line, "s-wave")
            except FwfParseError:
                continue


def read_RIPL3_resonance_spacing_p_wave_all() -> Iterator[ResonanceSpacingRecord]:
    """Read all p-wave resonance spacing records.
    
    Returns
    -------
    Iterator[ResonanceSpacingRecord]
        Iterator over all p-wave resonance spacing records
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances1.dat"
    with path.open() as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                yield read_resonance_spacing_record(line, "p-wave")
            except FwfParseError:
                continue


def available_resonance_spacing_s_wave() -> Iterator[Nuclide]:
    """List all available elements in the s-wave resonance spacing file.
    
    Returns
    -------
    Iterator[Nuclide]
        Iterator over available elements
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances0.dat"
    with path.open() as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                Z, A = int(line[:3]), int(line[7:10])
                yield Nuclide(A=A, Z=Z)
            except (ValueError, IndexError):
                continue


def available_resonance_spacing_p_wave() -> Iterator[Nuclide]:
    """List all available elements in the p-wave resonance spacing file.
    
    Returns
    -------
    Iterator[Nuclide]
        Iterator over available elements
    """
    path = RIPL3_RESONANCE_SPACING_PATH / "resonances1.dat"
    with path.open() as f:
        for line in f:
            if line.startswith('#'):
                continue
            try:
                Z, A = int(line[:3]), int(line[7:10])
                yield Nuclide(A=A, Z=Z)
            except (ValueError, IndexError):
                continue


ResonanceSpacing = namedtuple('ResonanceSpacing', ['s_wave', 'p_wave'])

def read_resonance_spacing(elem: Nuclide) -> ResonanceSpacing:
    """Read both s-wave and p-wave resonance spacing data for an element.
    
    Parameters
    ----------
    elem : Nuclide
        The element to read the data for
        
    Returns
    -------
    ResonanceSpacing
        Named tuple containing s_wave and p_wave ResonanceSpacingRecord objects.
        If either wave type is not found, that field will be None.
    """
    try:
        s_wave = read_RIPL3_resonance_spacing_s_wave(elem)
    except ValueError:
        s_wave = None
        
    try:
        p_wave = read_RIPL3_resonance_spacing_p_wave(elem)
    except ValueError:
        p_wave = None
        
    return ResonanceSpacing(s_wave=s_wave, p_wave=p_wave)



# Aliases for convenience
get_resonance_spacing_s = read_RIPL3_resonance_spacing_s_wave
get_resonance_spacing_p = read_RIPL3_resonance_spacing_p_wave
