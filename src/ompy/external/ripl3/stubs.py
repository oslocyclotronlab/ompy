from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from ...data_paths import data_root

# Type aliases
Pathlike: TypeAlias = str | Path
Nuclide: TypeAlias = tuple[int, int]

# Data paths
DATA_PATH = data_root() / "ripl3"
RIPL3_LEVELS_PATH = DATA_PATH / "levels"
RIPL3_LEVEL_DENSITIES_PATH = DATA_PATH / "densities"
RIPL3_RESONANCE_SPACING_PATH = DATA_PATH / "resonances"
