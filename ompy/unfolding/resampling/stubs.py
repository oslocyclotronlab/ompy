from typing import TypeAlias, Literal, TypeVar
from ... import Matrix, Vector

VSpace: TypeAlias = Literal["mu", "eta", "nu"]
MV = TypeVar("MV", bound=Matrix | Vector)
CI_Method: TypeAlias = Literal[
    "standard",
    "poisson",
    "bca",
    "supremum",
    "studentized supremum",
    "bonferroni percentile",
    "bonferroni bca",
    "hotelling T2",
]

SaveFormat: TypeAlias = Literal["hdf5", "npz"]