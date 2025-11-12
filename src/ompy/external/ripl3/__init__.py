from __future__ import annotations

# Import parse_fwf from fwf module
from .fwf import parse_fwf

# Import all classes and functions from the new modules to maintain API compatibility
from .levels import (
    RIPL3Record,
    LevelEntry,
    IdentificationRecord,
    LevelRecord,
    GammaRecord,
    GammaRecordError,
    fwf_identification_record,
    read_level_record,
    read_gamma_record,
    get_RIPL3_levels,
    read_RIPL3_levels,
    seek_nuclide,
    read_level,
)

from .level_density import (
    LevelDensityBSFGRecord,
    LevelDensityCTRecord,
    read_RIPL3_level_density_BSFG,
    seek_element_in_level_densities,
    available_bsfg,
    read_RIPL3_level_densities_BSFG,
    read_level_density_BSFG_record,
    read_RIPL3_level_densities_CT,
    read_RIPL3_level_density_CT,
    read_level_density_CT_record,
    available_ct,
    get_CT,
    get_BSFG,
)

from .resonance import (
    ResonanceSpacingRecord,
    read_resonance_spacing_record,
    read_RIPL3_resonance_spacing_s_wave,
    read_RIPL3_resonance_spacing_p_wave,
    seek_element_in_resonance_spacing,
    read_RIPL3_resonance_spacing_s_wave_all,
    read_RIPL3_resonance_spacing_p_wave_all,
    available_resonance_spacing_s_wave,
    available_resonance_spacing_p_wave,
    get_resonance_spacing_s,
    get_resonance_spacing_p,
    read_resonance_spacing,
)

from .references import (
    Reference,
    parse_references_from_readme,
    get_reference_by_code,
    get_all_references,
)

from .ripl3_data import (
    RIPL3Data,
)

# Import type aliases and constants from stubs
from .stubs import (
    Pathlike,
    Nuclide,
    DATA_PATH,
    RIPL3_LEVELS_PATH,
    RIPL3_LEVEL_DENSITIES_PATH,
    RIPL3_RESONANCE_SPACING_PATH,
)

# Maintain all the original aliases
get_CT = read_RIPL3_level_density_CT
get_BSFG = read_RIPL3_level_density_BSFG
get_resonance_spacing_s = read_RIPL3_resonance_spacing_s_wave
get_resonance_spacing_p = read_RIPL3_resonance_spacing_p_wave

if __name__ == "__main__":
    import doctest
    doctest.testmod()