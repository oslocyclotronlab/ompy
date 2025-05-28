from __future__ import annotations
import numpy as np
from .stubs import SamplingType, SamplingFunction, DensityFunction, Levels, VectorizedFunction
from .stubs import SpinDensityFunction, ParityDistributionFunction
from typing import Callable, Literal, overload, TypeAlias, Self, get_type_hints, Any
import xarray as xr
from dataclasses import dataclass
from .stubs import Spin
from .model import Model, ModelParameters, Parameters, ParametersBase
from ..base.elements import Element
from ..readers.ripl3.reader import get_CT, get_BSFG


def poisson_sampling(levels: float | np.ndarray) -> float | np.ndarray:
    return np.random.poisson(levels)


def wigner_sampling(levels: float | np.ndarray) -> float | np.ndarray:
    raise NotImplementedError("Wigner sampling not implemented yet")


def evaluate_density(Ex: np.ndarray, J: np.ndarray, pi: np.ndarray,
density: DensityFunction) -> Levels:

    # Test for constant bin width
    de = np.diff(Ex)
    if not np.allclose(de, de[0]):
        raise ValueError("Excitation energies must have constant bin width")

    # Requires constant bin width
    de = Ex[1] - Ex[0]
    # Generate levels for each bin, spin, and parity
    levels = de * density(Ex, J, pi)
    levels = xr.DataArray(levels, dims=["Ex", "J", "pi"],
                          coords={"Ex": Ex,
                                  "J": J,
                                  "pi": ["+", "-"]})
    levels.assign_attrs(content="nld")

    return levels


def evaluate_density_like(population, density: DensityFunction,
                          binwidth: float | None = None) -> Levels:
    """ Evaluate level density with binning like the given population

    Parameters
    ----------
    population : Population
        The population to sample like
    density : DensityFunction
        The nuclear level density function
    binwidth : float | None, optional
        The bin width, by default None. If set, overrides the bin with of Ex

    Returns
    -------
    Levels
        The sampled levels as a xarray
    """
    Ex = population.Ex.values
    if binwidth is None:
        binwidth = np.diff(Ex).min()
    Ex = np.arange(Ex.min(), Ex.max(), binwidth)
    J = population.J.values
    pi = [0, 1]
    return evaluate_density(Ex, J, pi, density)


def ct(e: float, shift: float, T: float) -> float:
    eff = e - shift
    return np.exp(eff / T) / T


def bsfg(e: float, e1: float, a: float, spincut2: Callable[[float], float]) -> float:
    eff = e - e1
    spincut_2 = spincut2(e)
    denominator = np.exp(2 * np.sqrt(a * eff))
    numerator = 12 * np.sqrt(2 * spincut_2) * a * (1 / 4) * eff ** (5 / 4)
    return numerator / denominator


# Spin cutoff  models
def vonEgidy05(e: float, eshift: float, a: float, A: float) -> float:
    """
    
    Args:
        e (float): The excitation energy
        eshift (float): The back-shift parameter of the BSFG
        a (float): level density parameter of the BSFG
        A (float): The atomic mass number

    Returns:
        float: the spin-cutoff parameter σ^2
    """
    e_eff = e - eshift
    return 0.0146 * A ** (5 / 3) * (1 + np.sqrt(1 + 4 * a * e_eff)) / (2 * a)


def vonEgidy09(e: float, Pa: float, A: float) -> float:
    b = max(0.0, e - 0.5 * Pa)
    return 0.391 * A ** 0.675 * b ** 0.312


def rigid_body(e: float, a: float, A: float) -> float:
    return 0.0138 * A ** (2 / 3) * np.sqrt(e / a)


def rigid_sphere(e: float, eshift: float, a: float, A: float) -> float:
    return 0.0145 * A ** (5 / 3) * np.sqrt((e - eshift) / a)


def gilbert_cameron(e: float, eshift: float, a: float, A: float) -> float:
    return 0.0888 * A ** (2 / 3) * a * np.sqrt((e - eshift)/ a)

def get_record(element: Element | str, source: Literal["ct", "bsfg"] = "ct"):
    if isinstance(element, str):
        element = Element.from_str(element)
    if source == "ct":
        return get_CT(element)
    elif source == "bsfg":
        return get_BSFG(element)
    else:
        raise ValueError(f"Invalid source: {source}")

@dataclass(slots=True, frozen=True)
class NLDatSnParameters(Parameters):
    Jtarget: Spin  # Spin of the target level [hbar]
    D0: float  # s-wave neutron resonance spacing at neutron separation energy [eV]
    Sn: float  # Neutron separation energy [MeV]

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("Jtarget", self.Jtarget, "Spin of the target level [hbar]"),
            ("D0", self.D0, "s-wave neutron resonance spacing at neutron separation energy [eV]"),
            ("Sn", self.Sn, "Neutron separation energy [MeV]"),
        ]

    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(Jtarget=record.I0, D0=record.D0, Sn=record.Bn)


@dataclass(slots=True, frozen=True)
class SpincutParameters(ModelParameters):
    pass



@dataclass(slots=True, frozen=True)
class GilbertCameronParameters(SpincutParameters):
    eshift: float  # Back-shift parameter [MeV]
    a: float  # Level density parameter [1/MeV]
    A: float  # Atomic mass number

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("eshift", self.eshift, "Back-shift parameter [MeV]"),
            ("a", self.a, "Level density parameter [1/MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]

    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(eshift=record.pairing, a=record.ainf, A=record.A)

@dataclass(slots=True, frozen=True)
class VonEgidy05Parameters(SpincutParameters):
    eshift: float  # Back-shift parameter [MeV]
    a: float  # Level density parameter [1/MeV]
    A: float  # Atomic mass number

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("eshift", self.eshift, "Back-shift parameter [MeV]"),
            ("a", self.a, "Level density parameter [1/MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]

    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(eshift=record.pairing, a=record.ainf, A=record.A)

@dataclass(slots=True, frozen=True)
class VonEgidy09Parameters(SpincutParameters):
    Pa: float  # Back-shift parameter [MeV]
    A: float  # Atomic mass number

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("Pa", self.Pa, "Back-shift parameter [MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]
    
    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(Pa=record.pairing, A=record.A)
    
@dataclass(slots=True, frozen=True)
class RigidBodyParameters(SpincutParameters):
    a: float  # Level density parameter [1/MeV]
    A: float  # Atomic mass number

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("a", self.a, "Level density parameter [1/MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]
    
    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(a=record.ainf, A=record.A)

@dataclass(slots=True, frozen=True)
class RigidSphereParameters(SpincutParameters):
    eshift: float  # Back-shift parameter [MeV]
    a: float  # Level density parameter [1/MeV]
    A: float  # Atomic mass number

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("eshift", self.eshift, "Back-shift parameter [MeV]"),
            ("a", self.a, "Level density parameter [1/MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]
    
    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        record = get_record(element, source)
        return cls(eshift=record.pairing, a=record.ainf, A=record.A)

@dataclass(slots=True, frozen=True)
class NLDModelParameters(ModelParameters):
    pass

@dataclass(slots=True, frozen=True)
class BSFGParameters(NLDModelParameters):
    e1: float  # Back-shift parameter [MeV]
    a: float  # Level density parameter [1/MeV]
    A: float  # Atomic mass number

    @classmethod
    def from_element(cls, element: Element | str) -> BSFGParameters:
        if isinstance(element, str):
            element = Element.from_str(element)
        record = get_BSFG(element)
        return cls(e1=record.pairing, a=record.ainf, A=element.A)

    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("e1", self.e1, "Back-shift parameter [MeV]"),
            ("a", self.a, "Level density parameter [1/MeV]"),
            ("A", self.A, "Atomic mass number"),
        ]

@dataclass(slots=True, frozen=True)
class CTParameters(NLDModelParameters):
    T: float  # Temperature [MeV]
    shift: float  # Back-shift parameter [MeV]

    @classmethod
    def from_element(cls, element: Element | str) -> CTParameters:
        if isinstance(element, str):
            element = Element.from_str(element)
        record = get_CT(element)
        return cls(T=record.T, shift=record.E0)
        
    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        return [
            ("T", self.T, "Temperature [MeV]"),
            ("shift", self.shift, "Back-shift parameter [MeV]"),
        ]


@dataclass(slots=True, frozen=True)
class SpincutModel(Model):
    parameters: SpincutParameters
    callback: VectorizedFunction

    @classmethod
    def from_element(cls, element: Element | str, source: Literal["ct", "bsfg"] = "ct") -> Self:
        parameters_type: type = get_type_hints(cls)['parameters']
        parameters = parameters_type.from_element(element, source)
        return cls(parameters=parameters)

        
@dataclass(slots=True, frozen=True)
class GilbertCameron(SpincutModel):
    parameters: GilbertCameronParameters
    callback: VectorizedFunction = gilbert_cameron

@dataclass(slots=True, frozen=True)
class NLDModel(Model):
    parameters: NLDModelParameters
    callback: VectorizedFunction

    @classmethod
    def from_element(cls, element: Element) -> Self:
        parameters_type: type = get_type_hints(cls)['parameters']
        parameters = parameters_type.from_element(element)
        return cls(parameters=parameters)


@dataclass(slots=True, frozen=True)
class CT(NLDModel):
    parameters: CTParameters
    callback: VectorizedFunction = ct



@dataclass(slots=True, frozen=True)
class BSFG(NLDModel):
    parameters: BSFGParameters
    callback: VectorizedFunction = bsfg


def guttormsen(e: float, e_d: float, sigma_d: float, 
                Sn: float,
               spincut2: VectorizedFunction) -> float:
    """ Spin-cutoff parameterization by Guttormsen et al.

    Linear interpolation between the spin-cutoff parameter at the discrete
    energy Ed and the spin-cutoff parameter at the neutron separation energy Sn.
    
    Args:
        e (float): The excitation energy
        e_d (float): The discrete excitation energy
        Sn (float): The neutron separation energy
        sigma_d (float): discrete spin-cutoff parameter determined
            by fitting the spin distribution of known discrete levels
            at Ex = Ed
        spincut2 (VectorizedFunction): The spin cutoff function for
            the spin-cutoff parameter as Sn

    Returns:
        float: The spin cutoff function
    """
    return sigma_d + (e - e_d)/(Sn - e_d)*(spincut2(Sn) - sigma_d)


def ericson_spin_distribution(J: float | np.ndarray, sigma2: float | np.ndarray) -> float | np.ndarray:
    """ The Ericson spin distribution

    
    
    Args:
        J (float | np.ndarray): The spin
        sigma2 (float | np.ndarray): The spin-cutoff parameter σ²

    Returns:
        float | np.ndarray: The spin distribution
    """
    if not np.isscalar(J) and not np.isscalar(sigma2):
        # Fermi gas model spin distribution/density
        sigma2 = np.atleast_1d(sigma2)[:, np.newaxis]
        J = np.atleast_1d(J)  # [np.newaxis, :]

    # return np.squeeze((2*J_ + 1)/(2*spincut_2)*np.exp(-J_*(J_+1/2)/(2*spincut_2)))
    density = (J + 1/2) / (sigma2) * np.exp(-(J + 1/2) ** (2) / (2 * sigma2))
    return density


def equiparity_distribution(e: float | np.ndarray, pi: Literal[0, 1] | bool | np.ndarray) -> float | np.ndarray:
    """ Parity distribtion for equiparity """
    e = np.atleast_1d(e)
    pi = np.atleast_1d(pi)
    return np.full((len(e), len(pi)), fill_value=0.5)


def non_equiparity_distribution(e: float | np.ndarray, pi: Literal[0, 1] | bool,
                                A: int, C: float, D: float) -> float | np.ndarray:
    """ Parity distribution for non-equiparity

    Parameters
    ----------
    e : float | np.ndarray
        Excitation energy
    pi : Literal[0, 1] | bool
        Parity of the level
    A : int
        Atomic number
    C : float
        Constant
    D : float
        Constant

    Returns
    -------
    float | np.ndarray
        Parity distribution
    """
    iseven = A % 2
    sign = 1 if (pi == 1) == iseven else -1
    return 0.5 * (1 + sign / (1 + np.exp(C * (e - D))))


def density(e: float | np.ndarray, J: float | np.ndarray, pi: Literal[0, 1] | bool | np.ndarray,
            nld: VectorizedFunction,
            spincut2: VectorizedFunction,
            parity_distribution: ParityDistributionFunction = equiparity_distribution,
            spin_density: SpinDensityFunction = ericson_spin_distribution,
            ) -> float | np.ndarray:
    """ The total level density

    Broadcasting is probably not working. Broadcasting over `e` and `J` works as long as
    their density functions also play along

    TODO: Implement proper broadcasting over `e`, `J`, and `pi`.
          Spin density is moved to a callback, but takes spincut2 as an argument. Might
          not be general. Eject spincut2 into the defintion of the spin density for the 
          use to decide.

    Parameters
    ----------
    e : np.ndarray
        Excitation energy
    J : np.ndarray
        Spin
    pi : np.ndarray
        Parity [0, 1]
    nld : VectorizedFunction
        The nuclear level density function
    spincut2 : VectorizedFunction
        The spin cutoff function
    parity_distribution : VectorizedFunction, optional
        The parity distribution function, by default equiparity_distribution

    Returns
    -------
    float | np.ndarray
        The total level density
    """
    e = np.atleast_1d(e)
    J = np.atleast_1d(J)
    pi = np.atleast_1d(pi)
    energy_density = nld(e)[:, np.newaxis, np.newaxis]
    sigma2 = spincut2(e)
    J_density = spin_density(J, sigma2)[:, :, np.newaxis]
    parity_density = np.atleast_1d(parity_distribution(e, pi))[:, np.newaxis, :]
    return np.squeeze(energy_density * J_density * parity_density)



def nld_at_Sn_from_D0(D0: float, J: float, g: VectorizedFunction) -> float:
    """ The level density \rho(S_n) at neutron separation energy 
    
    1/D0 = nld(Sn) * ( g(Jtarget+1/2, pi_target)
                        + g(Jtarget1/2, pi_target) )
    Here we assume equal parity, g(J,pi) = g(J)/2 and
    nld(Sn) = 1/D0 * 2/(g(Jtarget+1/2) + g(Jtarget-1/2))
    For the case Jtarget = 0, the g(Jtarget-1/2) = 0

    Args:
        D0 (float): The s-wave neutron resonance spacing
            at the neutron separation energy [eV]
        J (float): The spin of the target level [hbar]
        g(VectorizedFunction): The spin distribution function,
            parameterized only by the spin. Assumes equal parity.

    Returns:
        float: The level density at the neutron separation energy [1/MeV]
    """
    ev_to_MeV = 1e-6
    if J == 0:
        return 1/(D0*ev_to_MeV) * 2 / g(J + 1/2)
    else:
        return 1/(D0*ev_to_MeV) * 2 / (g(J + 1/2) + g(J - 1/2))
