from pathlib import Path
from typing import TypeAlias, Literal
import pandas as pd
import xarray as xr
import numpy as np

Pathlike: TypeAlias = str | Path
Population: TypeAlias = xr.DataArray


def read_population(path: Pathlike) -> Population:
    """ Read TALYS population file

    !Only supports equpartioned spins!

    Args:
        path (Path): Path to the TALYS population file
    Returns:
        xr.DataArray: DataArray with Ex and J as dimensions.

    """
    # copy the "Population of Z= 60 N= 84 (144Nd) before decay" section
    # from TALYS File:
    #   projectile p
    #   element nd
    #   mass 144
    #   energy 10
    #   outpopulation y
    #   bins 54
    #   maxlevelstar 16

    # bin    Ex     Popul.    J= 0.0    J= 1.0    J= 2.0    J= 3.0    J= 4.0    J= 5.0    J= 6.0    J= 7.0    J= 8.0
    # ...
    df: pd.DataFrame = pd.read_fwf(Path(path))
    # We do not need Popul.
    df = df.drop(['bin', 'Popul.'], axis=1)
    Js = [int(float(s.split('= ')[1])) for s in df.columns[1:]]
    values = df.drop(['Ex'], axis=1).to_numpy()
    values = values[:, :, np.newaxis] # Fake parity
    values = np.repeat(values, 2, axis=-1)
    values[:, :, 0] = 2*values[:, :, 0] # Negative parity gets everything
    values[:, :, 1] = 0.0
    ex_js = xr.DataArray(values, dims=['Ex', 'J', 'pi'],
                         coords={'Ex': df['Ex'],
                                 'J': Js,
                                 'pi': ['-', '+']})

    return ex_js
