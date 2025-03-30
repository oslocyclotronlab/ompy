from typing import TypeAlias, Literal
from .. import Matrix, Vector
import numpy as np

UnfoldingMatrix: TypeAlias = Literal['D', 'G', 'GD', 'DG']
Space: TypeAlias = Literal['mu', 'eta']
# We can plot in a specified space, or in 'base' which is the default Space 
# of the unfolder.
PlotSpace: TypeAlias = Space | Literal['base']
Mask1D: TypeAlias = np.ndarray | Vector | Literal['last nonzero']
Mask2D: TypeAlias = np.ndarray | Matrix | Literal['tril', 'last nonzero']
Mask: TypeAlias = Mask1D | Mask2D