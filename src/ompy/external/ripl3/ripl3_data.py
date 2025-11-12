from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

from ...nuclear.base.nuclide import Nuclide
from .levels import RIPL3Record, get_RIPL3_levels
from .level_density import (
    LevelDensityBSFGRecord,
    LevelDensityCTRecord,
    read_RIPL3_level_density_BSFG,
    read_RIPL3_level_density_CT,
)
from .resonance import (
    ResonanceSpacingRecord,
    read_RIPL3_resonance_spacing_s_wave,
    read_RIPL3_resonance_spacing_p_wave,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RIPL3Data:
    """
    Comprehensive RIPL3 data container for a nuclide.
    
    This class provides convenient access to all available RIPL3 data for a given
    nuclide, including:
    - Discrete level scheme and gamma transitions
    - Level density parameters (BSFG and CT models)
    - Resonance spacing data (s-wave and p-wave)
    
    Attributes
    ----------
    element : Nuclide
        The element/nuclide for which data is stored
    levels : Optional[RIPL3Record]
        Discrete level scheme data, or None if not available
    level_density_bsfg : Optional[LevelDensityBSFGRecord]
        Back-Shifted Fermi Gas level density parameters, or None if not available
    level_density_ct : Optional[LevelDensityCTRecord]
        Constant Temperature level density parameters, or None if not available
    s_wave : Optional[ResonanceSpacingRecord]
        S-wave resonance spacing data, or None if not available
    p_wave : Optional[ResonanceSpacingRecord]
        P-wave resonance spacing data, or None if not available
        
    Examples
    --------
    >>> from ompy.external.ripl3 import RIPL3Data
    >>> from ompy.nuclear.base.nuclide import Nuclide
    >>> 
    >>> # Read all available RIPL3 data for Fe-56
    >>> data = RIPL3Data.from_any(Nuclide("Fe56"))
    >>> 
    >>> # Check what data is available
    >>> if data.levels:
    ...     print(f"Found {len(data.levels.levels)} discrete levels")
    >>> 
    >>> if data.level_density_bsfg:
    ...     print(f"BSFG parameter a_inf = {data.level_density_bsfg.ainf}")
    >>> 
    >>> # Access specific data
    >>> if data.has_levels:
    ...     df = data.levels.to_pandas()
    ...     data.levels.plot()
    """
    
    element: Nuclide
    levels: Optional[RIPL3Record] = None
    level_density_bsfg: Optional[LevelDensityBSFGRecord] = None
    level_density_ct: Optional[LevelDensityCTRecord] = None
    s_wave: Optional[ResonanceSpacingRecord] = None
    p_wave: Optional[ResonanceSpacingRecord] = None
    
    @classmethod
    def from_any(cls, nuclide: Nuclide | str, verbose: bool = False) -> RIPL3Data:
        """
        Read all available RIPL3 data for a given element/nuclide.
        
        This method attempts to read all types of RIPL3 data. If a particular
        data type is not available for the nuclide, it will be set to None
        and the loading will continue for other data types.
        
        Parameters
        ----------
        nuclide : Nuclide or str
            The element/nuclide to read data for. Can be an Element object
            or a string like "Fe56" or "56Fe"
        verbose : bool, optional
            If True, print information about which data types were found
            (default: False)
            
        Returns
        -------
        RIPL3Data
            Container with all available RIPL3 data for the nuclide
            
        Examples
        --------
        >>> data = RIPL3Data.from_any("Fe56")
        >>> data = RIPL3Data.from_any(Nuclide(Z=26, A=56))
        >>> data = RIPL3Data.from_any("56Fe", verbose=True)
        """
        nuclide = Nuclide.from_any(nuclide)
        
        # Try to load discrete levels
        levels = None
        try:
            levels = get_RIPL3_levels(nuclide)
            if verbose:
                LOGGER.info("✓ Found discrete levels: %d levels", len(levels.levels))
        except ValueError:
            if verbose:
                LOGGER.info("✗ No discrete levels data available")
        
        # Try to load BSFG level density
        level_density_bsfg = None
        try:
            level_density_bsfg = read_RIPL3_level_density_BSFG(nuclide)
            if verbose:
                LOGGER.info("✓ Found BSFG level density parameters")
        except ValueError:
            if verbose:
                LOGGER.info("✗ No BSFG level density data available")
        
        # Try to load CT level density
        level_density_ct = None
        try:
            level_density_ct = read_RIPL3_level_density_CT(nuclide)
            if verbose:
                LOGGER.info("✓ Found CT level density parameters")
        except ValueError:
            if verbose:
                LOGGER.info("✗ No CT level density data available")
        
        # Try to load s-wave resonance spacing
        s_wave = None
        try:
            s_wave = read_RIPL3_resonance_spacing_s_wave(nuclide)
            if verbose:
                LOGGER.info("✓ Found s-wave resonance spacing data")
        except ValueError:
            if verbose:
                LOGGER.info("✗ No s-wave resonance spacing data available")
        
        # Try to load p-wave resonance spacing
        p_wave = None
        try:
            p_wave = read_RIPL3_resonance_spacing_p_wave(nuclide)
            if verbose:
                LOGGER.info("✓ Found p-wave resonance spacing data")
        except ValueError:
            if verbose:
                LOGGER.info("✗ No p-wave resonance spacing data available")
        
        return cls(
            element=nuclide,
            levels=levels,
            level_density_bsfg=level_density_bsfg,
            level_density_ct=level_density_ct,
            s_wave=s_wave,
            p_wave=p_wave,
        )
    
    @property
    def has_levels(self) -> bool:
        """Check if discrete level data is available."""
        return self.levels is not None
    
    @property
    def has_level_density_bsfg(self) -> bool:
        """Check if BSFG level density data is available."""
        return self.level_density_bsfg is not None
    
    @property
    def has_level_density_ct(self) -> bool:
        """Check if CT level density data is available."""
        return self.level_density_ct is not None
    
    @property
    def has_s_wave(self) -> bool:
        """Check if s-wave resonance spacing data is available."""
        return self.s_wave is not None
    
    @property
    def has_p_wave(self) -> bool:
        """Check if p-wave resonance spacing data is available."""
        return self.p_wave is not None
    
    @property
    def has_any_resonance(self) -> bool:
        """Check if any resonance spacing data is available."""
        return self.has_s_wave or self.has_p_wave
    
    @property
    def has_any_level_density(self) -> bool:
        """Check if any level density data is available."""
        return self.has_level_density_bsfg or self.has_level_density_ct
    
    def __repr__(self) -> str:
        """String representation of the RIPL3Data container."""
        available = []
        if self.has_levels:
            available.append(f"levels({len(self.levels.levels)})")
        if self.has_level_density_bsfg:
            available.append("BSFG")
        if self.has_level_density_ct:
            available.append("CT")
        if self.has_s_wave:
            available.append("s-wave")
        if self.has_p_wave:
            available.append("p-wave")
        
        available_str = ", ".join(available) if available else "no data"
        return f"RIPL3Data({self.element}, {available_str})"

    def has_any_data(self) -> bool:
        """Check if any data is available."""
        return self.has_levels or self.has_level_density_bsfg or self.has_level_density_ct or self.has_s_wave or self.has_p_wave
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebook display."""
        # Build collapsible sections
        sections_html = ""
        
        # Discrete Levels section
        if self.has_levels:
            sections_html += f"""
            <div style="margin-bottom: 15px; border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden;">
                <div style="background: #e9ecef; padding: 12px; font-weight: 600; color: #495057;">
                    <span style="color: #28a745; font-size: 1.2em; margin-right: 8px;">✓</span>
                    Discrete Levels: {len(self.levels.levels)} levels
                </div>
                <div style="padding: 12px; background: white;">
                    Complete up to level: {self.levels.identification.Nmax}<br>
                    Sn = {self.levels.identification.Sn:.3f} MeV<br>
                    Sp = {self.levels.identification.Sp:.3f} MeV
                </div>
            </div>"""
        else:
            sections_html += """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #dee2e6; border-radius: 5px; background: #f8f9fa;">
                <span style="color: #dc3545; font-size: 1.2em; margin-right: 8px;">✗</span>
                <span style="font-weight: 600; color: #6c757d;">Discrete Levels: Not available</span>
            </div>"""
        
        # BSFG Level Density
        if self.has_level_density_bsfg:
            bsfg_details = self.level_density_bsfg.summary(indent="    ").replace("\n", "<br>")
            sections_html += f"""
            <details style="margin-bottom: 15px; border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden;">
                <summary style="background: #e9ecef; padding: 12px; font-weight: 600; color: #495057; cursor: pointer;">
                    <span style="color: #28a745; font-size: 1.2em; margin-right: 8px;">✓</span>
                    BSFG Level Density: a∞={self.level_density_bsfg.ainf:.3f} MeV⁻¹
                </summary>
                <div style="padding: 12px; background: white; font-family: monospace; font-size: 0.9em;">
                    {bsfg_details}
                </div>
            </details>"""
        else:
            sections_html += """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #dee2e6; border-radius: 5px; background: #f8f9fa;">
                <span style="color: #dc3545; font-size: 1.2em; margin-right: 8px;">✗</span>
                <span style="font-weight: 600; color: #6c757d;">BSFG Level Density: Not available</span>
            </div>"""
        
        # CT Level Density
        if self.has_level_density_ct:
            ct_details = self.level_density_ct.summary(indent="    ").replace("\n", "<br>")
            sections_html += f"""
            <details style="margin-bottom: 15px; border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden;">
                <summary style="background: #e9ecef; padding: 12px; font-weight: 600; color: #495057; cursor: pointer;">
                    <span style="color: #28a745; font-size: 1.2em; margin-right: 8px;">✓</span>
                    CT Level Density: T={self.level_density_ct.T:.3f} MeV
                </summary>
                <div style="padding: 12px; background: white; font-family: monospace; font-size: 0.9em;">
                    {ct_details}
                </div>
            </details>"""
        else:
            sections_html += """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #dee2e6; border-radius: 5px; background: #f8f9fa;">
                <span style="color: #dc3545; font-size: 1.2em; margin-right: 8px;">✗</span>
                <span style="font-weight: 600; color: #6c757d;">CT Level Density: Not available</span>
            </div>"""
        
        # S-wave Resonance (collapsible, default collapsed)
        if self.has_s_wave:
            s_wave_details = self.s_wave.summary(indent="    ").replace("\n", "<br>")
            sections_html += f"""
            <details style="margin-bottom: 15px; border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden;">
                <summary style="background: #e9ecef; padding: 12px; font-weight: 600; color: #495057; cursor: pointer;">
                    <span style="color: #28a745; font-size: 1.2em; margin-right: 8px;">✓</span>
                    S-wave Resonance: D={self.s_wave.D:.2f} keV
                </summary>
                <div style="padding: 12px; background: white; font-family: monospace; font-size: 0.9em;">
                    {s_wave_details}
                </div>
            </details>"""
        else:
            sections_html += """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #dee2e6; border-radius: 5px; background: #f8f9fa;">
                <span style="color: #dc3545; font-size: 1.2em; margin-right: 8px;">✗</span>
                <span style="font-weight: 600; color: #6c757d;">S-wave Resonance: Not available</span>
            </div>"""
        
        # P-wave Resonance (collapsible, default collapsed)
        if self.has_p_wave:
            p_wave_details = self.p_wave.summary(indent="    ").replace("\n", "<br>")
            sections_html += f"""
            <details style="margin-bottom: 15px; border: 1px solid #dee2e6; border-radius: 5px; overflow: hidden;">
                <summary style="background: #e9ecef; padding: 12px; font-weight: 600; color: #495057; cursor: pointer;">
                    <span style="color: #28a745; font-size: 1.2em; margin-right: 8px;">✓</span>
                    P-wave Resonance: D={self.p_wave.D:.2f} keV
                </summary>
                <div style="padding: 12px; background: white; font-family: monospace; font-size: 0.9em;">
                    {p_wave_details}
                </div>
            </details>"""
        else:
            sections_html += """
            <div style="margin-bottom: 15px; padding: 12px; border: 1px solid #dee2e6; border-radius: 5px; background: #f8f9fa;">
                <span style="color: #dc3545; font-size: 1.2em; margin-right: 8px;">✗</span>
                <span style="font-weight: 600; color: #6c757d;">P-wave Resonance: Not available</span>
            </div>"""
        
        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 10px; 
                    border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; color: white;">
                <h2 style="margin: 0; font-weight: 600; font-size: 1.8em;">
                    RIPL-3 Nuclear Data
                </h2>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;">
                    {self.element.symbol}-{self.element.A} 
                    <span style="font-size: 0.9em; opacity: 0.85;">(Z={self.element.Z}, A={self.element.A})</span>
                </p>
            </div>
            
            <div style="background: white; padding: 20px;">
                {sections_html}
            </div>
            
            <div style="background: #f8f9fa; padding: 12px 20px; border-top: 1px solid #dee2e6; 
                        font-size: 0.85em; color: #6c757d; text-align: center;">
                Reference Input Parameter Library (RIPL-3) • IAEA Nuclear Data Section
            </div>
        </div>
        """
        return html
    
    def summary(self) -> str:
        """
        Generate a comprehensive summary of all available data.
        
        Returns
        -------
        str
            Summary string describing all available data with full details
        """
        lines = [
            f"RIPL3 Data Summary for {self.element}",
            "=" * 70,
            ""
        ]
        
        # Discrete Levels (only basic info to avoid spam)
        if self.has_levels:
            lines.append(f"✓ Discrete Levels: {len(self.levels.levels)} levels")
            if self.levels.identification:
                lines.append(f"  Complete up to level: {self.levels.identification.Nmax}")
                lines.append(f"  Neutron separation energy (Sn): {self.levels.identification.Sn:.3f} MeV")
                lines.append(f"  Proton separation energy (Sp): {self.levels.identification.Sp:.3f} MeV")
        else:
            lines.append("✗ Discrete Levels: Not available")
        lines.append("")
        
        # BSFG Level Density (full details)
        if self.has_level_density_bsfg:
            lines.append("✓ BSFG Level Density:")
            lines.append(self.level_density_bsfg.summary(indent="  "))
        else:
            lines.append("✗ BSFG Level Density: Not available")
        lines.append("")
        
        # CT Level Density (full details)
        if self.has_level_density_ct:
            lines.append("✓ CT Level Density:")
            lines.append(self.level_density_ct.summary(indent="  "))
        else:
            lines.append("✗ CT Level Density: Not available")
        lines.append("")
        
        # S-wave Resonance Spacing (full details)
        if self.has_s_wave:
            lines.append("✓ S-wave Resonance Spacing:")
            lines.append(self.s_wave.summary(indent="  "))
        else:
            lines.append("✗ S-wave Resonance Spacing: Not available")
        lines.append("")
        
        # P-wave Resonance Spacing (full details)
        if self.has_p_wave:
            lines.append("✓ P-wave Resonance Spacing:")
            lines.append(self.p_wave.summary(indent="  "))
        else:
            lines.append("✗ P-wave Resonance Spacing: Not available")
        
        return "\n".join(lines)
    
    def print(self) -> None:
        """Print a summary of available data to stdout."""
        print(self.summary())

