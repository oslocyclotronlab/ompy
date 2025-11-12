"""AME2003 atomic mass evaluation data parsers (vonEgidy03).

This module provides parsers for the AME2003 atomic mass evaluation data files:
- mass.mas03: Atomic masses
- rct2.mas03: Reaction and separation energies
- rct7.mas03: Pairing energies

Reference:
"The Ame2003 atomic mass evaluation (II)" by G.Audi, A.H.Wapstra and C.Thibault
Nuclear Physics A729 p. 337-676, December 22, 2003.

Examples
--------
Get mass data for a specific nuclide:

>>> from ompy.external.vonegidy03 import get_mass_data
>>> from ompy.nuclear.base.nuclide import Nuclide
>>> fe57 = Nuclide(A=57, Z=26)
>>> mass_data = get_mass_data(fe57)
>>> print(f"Mass Excess: {mass_data.mass_excess} keV")

Read all records from a file:

>>> from ompy.external.vonegidy03 import read_mass_table
>>> for record in read_mass_table():
...     print(f"{record.el}-{record.A}: {record.mass_excess} keV")
...     break  # Just show first one

Get multiple data types for a nuclide:

>>> from ompy.external.vonegidy03 import (
...     get_mass_data, get_reaction_energy_data, get_pairing_energy_data
... )
>>> from ompy.nuclear.base.nuclide import Nuclide
>>> fe57 = Nuclide(A=57, Z=26)
>>> mass = get_mass_data(fe57)
>>> reaction = get_reaction_energy_data(fe57)
>>> pairing = get_pairing_energy_data(fe57)
>>> print(mass.summary())
"""

from .reader import (
    # Dataclasses
    MassRecord,
    ReactionEnergyRecord,
    PairingEnergyRecord,
    VonEgidy03,
    
    # Record parsers
    read_mass_record,
    read_reaction_energy_record,
    read_pairing_energy_record,
    
    # Seek functions
    seek_mass_nuclide,
    seek_reaction_energy_nuclide,
    seek_pairing_energy_nuclide,
    
    # File readers
    read_mass_table,
    read_reaction_energies,
    read_pairing_energies,
    
    # Lookup functions
    get_mass_data,
    get_reaction_energy_data,
    get_pairing_energy_data,
    
    # Constants
    DATA_PATH,
    MASS_TABLE_PATH,
    REACTION_ENERGY_PATH,
    PAIRING_ENERGY_PATH,
)

__all__ = [
    # Dataclasses
    'MassRecord',
    'ReactionEnergyRecord',
    'PairingEnergyRecord',
    'VonEgidy03',
    
    # Record parsers
    'read_mass_record',
    'read_reaction_energy_record',
    'read_pairing_energy_record',
    
    # Seek functions
    'seek_mass_nuclide',
    'seek_reaction_energy_nuclide',
    'seek_pairing_energy_nuclide',
    
    # File readers
    'read_mass_table',
    'read_reaction_energies',
    'read_pairing_energies',
    
    # Lookup functions
    'get_mass_data',
    'get_reaction_energy_data',
    'get_pairing_energy_data',
    
    # Constants
    'DATA_PATH',
    'MASS_TABLE_PATH',
    'REACTION_ENERGY_PATH',
    'PAIRING_ENERGY_PATH',
]

