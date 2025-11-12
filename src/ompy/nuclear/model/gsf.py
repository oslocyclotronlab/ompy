from __future__ import annotations
import numpy as np
from enum import IntEnum
from typing import Literal, TypeAlias, Callable
from typing import Union
from numpy import pi
import numpy as np
from .stubs import GammaFunction
from .model import Model, ModelParameters
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
from textwrap import dedent
from ...accel import jax_available, numba_available, numba_cuda_available

_HAS_JAX = jax_available()
_HAS_NUMBA = numba_available()
_HAS_NUMBA_CUDA = numba_cuda_available()

if _HAS_JAX:
    from jax import jit
    import jax.numpy as jnp

if _HAS_NUMBA:
    from numba import types
    from numba import njit
else:
    # nop decorator
    
    def nop(*aargs, **kkwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    njit = nop
    types = None

if _HAS_NUMBA_CUDA:
    from numba import cuda
else:
    cuda = None



class TransitionType(IntEnum):
    """ Enum for transition types """
    Impossible = 0
    E1 = 2
    M1 = 3
    E2 = 5
    M2 = 7
    XL = 11  # Higher order transitions
    M1E2 = 15


def transition_type(J_i: float, pi_i: Literal[-1, 1],
                    J_f: float, pi_f: Literal[-1, 1],
                    max_J: float, A: float) -> TransitionType:
    """ Determine the transition type

    """
    if J_f > max_J or J_f < 0:
        return TransitionType.Impossible

    if J_i < 0:
        raise ValueError("Initial spin must be positive")

    ΔJ = abs(J_f - J_i)
    Δpi = pi_f - pi_i
    is_even = A % 2 == 0

    # Determine transition type based on ΔJ, Δpi, and A (even or odd)
    if is_even:
        if ΔJ == 0:
            if J_f > 0 and J_i > 0:
                if Δpi == 0:
                    return TransitionType.M1E2  # M1+E2
                else:
                    return TransitionType.E1  # E1
            else:
                return TransitionType.Impossible  # no 0 -> 0 with gammas, ignore E0 Internal Conversion
        elif ΔJ == 1:
            if J_f > 0 and J_i > 0:  # triangle check
                if Δpi == 0:
                    return TransitionType.M1E2  # M1+E2
                else:
                    return TransitionType.E1  # E1
            else:
                if Δpi == 0:  # 0+ -> 1+ or 0- -> 1- or 1- -> 0- or 1+ -> 0+
                    return TransitionType.M1  # M1 Pure
                else:  # 0+ -> 1- or 0- -> 1+ or 1+ -> 0- or 1- -> 0+
                    return TransitionType.E1  # E1
        elif ΔJ == 2:
            if Δpi == 0:
                return TransitionType.E2  # E2
            else:
                return TransitionType.Impossible  # no M2,E3
        else:  # ΔJ > 2
            return TransitionType.Impossible  # no Octupole
    else:  # odd A
        if ΔJ == 0:
            if J_f > 0 and J_i > 0:
                if Δpi == 0:
                    return TransitionType.M1E2  # M1+E2
                else:
                    return TransitionType.E1  # E1
            else:
                if Δpi == 0:
                    return TransitionType.M1  # 1/2+ -> 1/2+ pure M1, no E2 via triangle condition
                else:
                    return TransitionType.E1  # 1/2+ -> 1/2- E1
        elif ΔJ == 1:
            if J_f > 0 and J_i > 0:  # triangle check
                if Δpi == 0:
                    return TransitionType.M1E2  # M1+E2
                else:
                    return TransitionType.E1  # E1
            else:
                if Δpi == 0:  # 1/2+ -> 3/2+ can be quadrupole unlike 0+ -> 1-
                    return TransitionType.M1E2  # M1+E2
                else:  # 1/2+ -> 3/2- , etc.
                    return TransitionType.E1  # E1
        elif ΔJ == 2:
            if Δpi == 0:
                return TransitionType.E2  # E2
            else:
                return TransitionType.Impossible  # no M2,E3
        else:  # ΔJ > 2
            return TransitionType.Impossible  # no Octupole

    match (is_even_A, ΔJ, Δpi, J_f > 0, J_i > 0):
        case (True, 0, _, True, True) | (False, 0, _, True, True):
            return TransitionType.M1E2 if Δpi == 0 else TransitionType.E1
        case (True, 1, 0, True, True) | (False, 1, 0, True, True):
            return TransitionType.M1E2
        case (True, 1, _, False, False) | (False, _, 0, False, False):
            return TransitionType.M1
        case (True, 1, _, _, _) | (False, 1, _, _, _):
            return TransitionType.E1
        case (True, 2, 0, _, _) | (False, 2, 0, _, _):
            return TransitionType.E2
        case (True, 0, _, False, False) | (True, _, _, _, _) | (False, _, _, _, _):
            return TransitionType.Impossible

    return TransitionType.Impossible


GSF: TypeAlias = Callable[[float, float], float]


# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08


@njit
def sp(value: float) -> float:
    """
    Single particle strength (constant strength function)

    Parameters:
        value (float]):
            Single particle strength

    Returns:
        float: strength function
    """
    return value

@njit
def slo(E: Union[np.ndarray, float],
         E0: float, sigma0: float, Gamma0: float
         ) -> Union[np.ndarray, float]:
    """
    Standard Lorentzian function f(E; E0, sigma0, Gamma0)
    adapted from Kopecky & Uhl (1989) eq. (2.1)

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value [MeV]
        E0 (float):
            Location parameter [MeV]
        sigma0 (float):
            Scale factor [mb]
        Gamma0 (float):
            Width parameter [MeV]

    Returns:
        Union[np.ndarray, float]: strength function
    """
    f = strength_factor * sigma0 * E * Gamma0**2 / \
        ((E**2 - E0**2)**2 + E**2 * Gamma0**2)
    return f

@njit
def glo_ct(E: Union[np.ndarray, float],
            E0: float, sigma0: float, Gamma0: float,
            T: float) -> Union[np.ndarray, float]:
    """Generalized Lorentzian with constant temperature of final states
    adapted from Kopecky & Uhl (1989) eq. (2.3-2.4)

    Note: Modified to have constant temperature of final states

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value [MeV]
        E0 (float):
            Location parameter [MeV]
        sigma0 (float):
            Scale factor [mb]
        Gamma0 (float):
            Width parameter [MeV]
        T (float):
            Temperature parameter [MeV]

    Returns:
        Union[np.ndarray, float]: strength function
    """
    gc = gamma_glo(E, E0=E0, Gamma0=Gamma0, T=T)
    gc0 = gamma_glo(E=0, E0=E0, Gamma0=Gamma0, T=T)


    f1 = (E * gc) / ((E**2 - E0**2)**2 + E**2 * gc**2)
    f2 = 0.7 * gc0 / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)
    return f

@njit
def gamma_glo(E: Union[np.ndarray, float], E0: float, Gamma0: float,
              T: float) -> Union[np.ndarray, float]:
    """ Width parameter in GLO model

    Note:
        See GLO for documentation

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
    """
    return Gamma0 / E0**2 * (E**2 + (2. * 3.141592 * T)**2)

@njit
def gamma_eglo(E: Union[np.ndarray, float], E0: float, Gamma0: float,
               T: float,
               epsilon_0: float, k: float) -> Union[np.ndarray, float]:
    """ Width parameter in GLO model

    Note:
        See EGLO and GLO for documentation

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        epsilon_0 (float):
            reference/"critical" energy for enhanced width. Free parameter
        k (float):
            enhancement factor; free parameter of e.g. from Fermi gas.
            For k=1 , this is equivalent to Gamma_GLO.
    """
    chi = k + (1.0 - k) * (E - epsilon_0) / (E0 - epsilon_0)
    return chi * gamma_glo(E=E, E0=E0, Gamma0=Gamma0, T=T)

@njit
def eglo_ct(E: Union[np.ndarray, float],
             E0: float, sigma0: float, Gamma0: float,
             T: float, epsilon_0: float = 0, k: float = 1
             ) -> Union[np.ndarray, float]:
    """Enhanced Generalized Lorentzian with CT Temperature

    Modified to have with constant temperature of final states

    adapted from
    - J. Kopecky and M. Uhl, in Capture Gamma Ray Spectroscopy,
      Proceedings of the Seventh International Symposium on Capture Gamma-ray
      Spectroscopy and Related Topics, edited by R. W. Ho6; AIP Conf. Proc. No.
      238 (AIP, New York, 1991), p. 607. DOI: 10.1063/1.41227

    See also:
        - S.G. Kadmenskii, V.P. Markushev, and V.I. Furman. Radiative width of
          neutron reson- ances. Giant dipole resonances. Sov. J. Nucl. Phys.
        - J. Kopecky, M. Uhl and R.E. Chrien, Phys. Rev. C47, 312 (1993)
          DOI: 10.1103/physrevc.47.312
        - RIPL3

    Note:
        - This was modified for a constant temperature dependece
        - RIPL3 provides emperical parametrization of k and epsilon_0,
          but as stated in the original work, they may be changed

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        epsilon_0 (float, optional):
            reference/"critical" energy for enhanced width. Free parameter
        k (float, optional):
            enhancement factor; free parameter of e.g. from Fermi gas.
            For the default value, k=1 , this is equivalent to GLO

    Deleted Parameters:
        A: int
            mass number

    Returns:
        Union[np.ndarray, float]: strength function
    """

    # # (MeV); adopted from RIPL, "depends on model for state density"
    # epsilon_0 = 4.5
    # if A < 148:
    #     k = 1.0
    # if(A >= 148):
    #     k = 1. + 0.09 * (A - 148)**2 * np.exp(-0.18 * (A - 148))

    Gamma_k_E = gamma_eglo(E=0, E0=E0, Gamma0=Gamma0, T=T,
                            epsilon_0=epsilon_0, k=k)

    Gamma_k_0 = gamma_eglo(E=0, E0=E0, Gamma0=Gamma0, T=T,
                            epsilon_0=epsilon_0, k=k)

    f1 = (E * Gamma_k_E) / ((E**2 - E0**2)**2 + E**2 * Gamma_k_E**2)
    f2 = 0.7 * Gamma_k_0 / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f

@njit
def fmglo_ct(E: Union[np.ndarray, float],
             E0: float, sigma0: float, Gamma0: float,
             T: float, epsilon_0: float = 0, k: float = 1
             ) -> Union[np.ndarray, float]:
    """Modified Generalized Lorentzian with CT Temperature

    Modified to have with constant temperature of final states

    adapted from
    - J. Kroll et al., “Strength of the scissors mode in odd-mass Gd isotopes
      from the radiative capture of resonance neutrons,” Physical Review C,
      vol. 88, no. 3, Art. no. 3, Sep. 2013, doi: 10.1103/physrevc.88.034317.

    Note:
        - This was modified for a constant temperature dependece

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        epsilon_0 (float, optional):
            reference/"critical" energy for enhanced width. Free parameter
        k (float, optional):
            enhancement factor; free parameter of e.g. from Fermi gas.
            For the default value, k=1 , this is equivalent to GLO

    Deleted Parameters:
        A: int
            mass number

    Returns:
        Union[np.ndarray, float]: strength function
    """

    # # (MeV); adopted from RIPL, "depends on model for state density"
    # epsilon_0 = 4.5
    # if A < 148:
    #     k = 1.0
    # if(A >= 148):
    #     k = 1. + 0.09 * (A - 148)**2 * np.exp(-0.18 * (A - 148))

    gamma_e = gamma_glo(E=0, E0=E0, Gamma0=Gamma0, T=T)
    gamma_0 = gamma_eglo(E=0, E0=E0, Gamma0=Gamma0, T=T, epsilon_0=epsilon_0, k=k)

    f1 = (E * gamma_e) / ((E**2 - E0**2)**2 + E**2 * gamma_e**2)
    f2 = 0.7 * gamma_0 / E0**3

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f

@njit
def gh_ct(E: Union[np.ndarray, float],
           E0: float, sigma0: float, Gamma0: float,
           T: float, k: float = 0.63) -> Union[np.ndarray, float]:
    """Goriely's Hybrid model, but with constant temperature of final states

    adapted from:
        - S. Goriely, Radiative neutron captures by neutron-rich nuclei and
        the r-process nucleosynthesis, Physics Letters B, Elsevier BV, 1998,
        436, 10-18 DOI: 10.1016/s0370-2693(98)00907-1
        - RIPL3 eq. (144-145)

    Note: Modified to have constant temperature of final states

    Parameters:
        E (Union[np.ndarray, float]):
            Input E value
        E0 (float):
            Location parameter
        sigma0 (float):
            Scale factor
        Gamma0 (float):
            Width parameter
        T (float):
            Temperature parameter
        k (float, optional):
            enhancement factor

    Returns:
        Union[np.ndarray, float]: strength function
    """
    Gamma = k * Gamma0 * (E**2 + 4 * pi**2 * T**2) / (E*E0)
    f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)

    f = strength_factor * sigma0 * Gamma0 * f1
    return f

@jit
def sp_jax(E: jnp.ndarray, value: float) -> jnp.ndarray:
    """JAX-compatible Single Particle model"""
    return jnp.full_like(E, value)

@jit
def slo_jax(E: jnp.ndarray, E0: float, sigma0: float, Gamma0: float) -> jnp.ndarray:
    """JAX-compatible Standard Lorentzian model"""
    return strength_factor * sigma0 * E * Gamma0**2 / ((E**2 - E0**2)**2 + (E * Gamma0)**2)

@jit
def glo_ct_jax(E: jnp.ndarray, E0: float, sigma0: float, Gamma0: float, T: float) -> jnp.ndarray:
    """JAX-compatible Generalized Lorentzian model with Constant Temperature"""
    Gamma = Gamma0 * (E**2 + 4 * jnp.pi**2 * T**2) / E0**2
    f1 = E * Gamma / ((E**2 - E0**2)**2 + (E * Gamma)**2)
    f2 = 0.7 * Gamma0 / E0**3
    return strength_factor * sigma0 * Gamma0 * (f1 + f2)

@jit
def eglo_ct_jax(E: jnp.ndarray, E0: float, sigma0: float, Gamma0: float, T: float, epsilon_0: float = 0, k: float = 1) -> jnp.ndarray:
    """JAX-compatible Enhanced Generalized Lorentzian model with Constant Temperature"""
    Gamma_GLO = lambda E: Gamma0 * (E**2 + 4 * jnp.pi**2 * T**2) / E0**2
    Gamma_EGLO = lambda E: Gamma_GLO(E) * (E - epsilon_0) / E + k * 4 * jnp.pi**2 * T**2 / E0

    gamma_e = Gamma_GLO(E)
    gamma_0 = Gamma_EGLO(jnp.zeros_like(E))

    f1 = (E * gamma_e) / ((E**2 - E0**2)**2 + E**2 * gamma_e**2)
    f2 = 0.7 * gamma_0 / E0**3

    return strength_factor * sigma0 * Gamma0 * (f1 + f2)

@jit
def fmglo_ct_jax(E: jnp.ndarray, E0: float, sigma0: float, Gamma0: float, T: float, epsilon_0: float = 0, k: float = 1) -> jnp.ndarray:
    """JAX-compatible Flexible Modified Generalized Lorentzian model with Constant Temperature"""
    Gamma_GLO = lambda E: Gamma0 * (E**2 + 4 * jnp.pi**2 * T**2) / E0**2
    Gamma_EGLO = lambda E: Gamma_GLO(E) * (E - epsilon_0) / E + k * 4 * jnp.pi**2 * T**2 / E0

    gamma_e = Gamma_GLO(E)
    gamma_0 = Gamma_EGLO(jnp.zeros_like(E))

    f1 = (E * gamma_e) / ((E**2 - E0**2)**2 + E**2 * gamma_e**2)
    f2 = 0.7 * gamma_0 / E0**3

    return strength_factor * sigma0 * Gamma0 * (f1 + f2)

@jit
def gh_ct_jax(E: jnp.ndarray, E0: float, sigma0: float, Gamma0: float, T: float, k: float = 0.63) -> jnp.ndarray:
    """JAX-compatible Goriely's Hybrid model with Constant Temperature"""
    Gamma = k * Gamma0 * (E**2 + 4 * jnp.pi**2 * T**2) / (E*E0)
    f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)
    return strength_factor * sigma0 * Gamma0 * f1

if _HAS_NUMBA_CUDA:
    from numba import cuda

    @cuda.jit
    def sp_cuda(value: float) -> float:
        """
        Single particle strength (constant strength function)

        Parameters:
            value (float]):
                Single particle strength

        Returns:
            float: strength function
        """
        return value

    @cuda.jit
    def slo_cuda(E: Union[np.ndarray, float],
            E0: float, sigma0: float, Gamma0: float
            ) -> Union[np.ndarray, float]:
        """
        Standard Lorentzian function f(E; E0, sigma0, Gamma0)
        adapted from Kopecky & Uhl (1989) eq. (2.1)

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value [MeV]
            E0 (float):
                Location parameter [MeV]
            sigma0 (float):
                Scale factor [mb]
            Gamma0 (float):
                Width parameter [MeV]

        Returns:
            Union[np.ndarray, float]: strength function
        """
        f = strength_factor * sigma0 * E * Gamma0**2 / \
            ((E**2 - E0**2)**2 + E**2 * Gamma0**2)
        return f

    @cuda.jit
    def glo_ct_cuda(E: Union[np.ndarray, float],
                E0: float, sigma0: float, Gamma0: float,
                T: float) -> Union[np.ndarray, float]:
        """Generalized Lorentzian with constant temperature of final states
        adapted from Kopecky & Uhl (1989) eq. (2.3-2.4)

        Note: Modified to have constant temperature of final states

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value [MeV]
            E0 (float):
                Location parameter [MeV]
            sigma0 (float):
                Scale factor [mb]
            Gamma0 (float):
                Width parameter [MeV]
            T (float):
                Temperature parameter [MeV]

        Returns:
            Union[np.ndarray, float]: strength function
        """
        gc = gamma_glo_cuda(E, E0, Gamma0, T)
        gc0 = gamma_glo_cuda(0, E0, Gamma0, T)


        f1 = (E * gc) / ((E**2 - E0**2)**2 + E**2 * gc**2)
        f2 = 0.7 * gc0 / E0**3

        f = strength_factor * sigma0 * Gamma0 * (f1 + f2)
        return f

    @cuda.jit
    def gamma_glo_cuda(E: Union[np.ndarray, float], E0: float, Gamma0: float,
                T: float) -> Union[np.ndarray, float]:
        """ Width parameter in GLO model

        Note:
            See GLO for documentation

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value
            E0 (float):
                Location parameter
            Gamma0 (float):
                Width parameter
            T (float):
                Temperature parameter
        """
        return Gamma0 / E0**2 * (E**2 + (2. * 3.141592 * T)**2)

    @cuda.jit
    def gamma_eglo_cuda(E: Union[np.ndarray, float], E0: float, Gamma0: float,
                T: float,
                epsilon_0: float, k: float) -> Union[np.ndarray, float]:
        """ Width parameter in GLO model

        Note:
            See EGLO and GLO for documentation

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value
            E0 (float):
                Location parameter
            Gamma0 (float):
                Width parameter
            T (float):
                Temperature parameter
            epsilon_0 (float):
                reference/"critical" energy for enhanced width. Free parameter
            k (float):
                enhancement factor; free parameter of e.g. from Fermi gas.
                For k=1 , this is equivalent to Gamma_GLO.
        """
        chi = k + (1.0 - k) * (E - epsilon_0) / (E0 - epsilon_0)
        return chi * gamma_glo_cuda(E, E0, Gamma0, T)

    @cuda.jit
    def eglo_ct_cuda(E: Union[np.ndarray, float],
                E0: float, sigma0: float, Gamma0: float,
                T: float, epsilon_0: float = 0, k: float = 1
                ) -> Union[np.ndarray, float]:
        """Enhanced Generalized Lorentzian with CT Temperature

        Modified to have with constant temperature of final states

        adapted from
        - J. Kopecky and M. Uhl, in Capture Gamma Ray Spectroscopy,
        Proceedings of the Seventh International Symposium on Capture Gamma-ray
        Spectroscopy and Related Topics, edited by R. W. Ho6; AIP Conf. Proc. No.
        238 (AIP, New York, 1991), p. 607. DOI: 10.1063/1.41227

        See also:
            - S.G. Kadmenskii, V.P. Markushev, and V.I. Furman. Radiative width of
            neutron reson- ances. Giant dipole resonances. Sov. J. Nucl. Phys.
            - J. Kopecky, M. Uhl and R.E. Chrien, Phys. Rev. C47, 312 (1993)
            DOI: 10.1103/physrevc.47.312
            - RIPL3

        Note:
            - This was modified for a constant temperature dependece
            - RIPL3 provides emperical parametrization of k and epsilon_0,
            but as stated in the original work, they may be changed

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value
            E0 (float):
                Location parameter
            sigma0 (float):
                Scale factor
            Gamma0 (float):
                Width parameter
            T (float):
                Temperature parameter
            epsilon_0 (float, optional):
                reference/"critical" energy for enhanced width. Free parameter
            k (float, optional):
                enhancement factor; free parameter of e.g. from Fermi gas.
                For the default value, k=1 , this is equivalent to GLO

        Deleted Parameters:
            A: int
                mass number

        Returns:
            Union[np.ndarray, float]: strength function
        """

        # # (MeV); adopted from RIPL, "depends on model for state density"
        # epsilon_0 = 4.5
        # if A < 148:
        #     k = 1.0
        # if(A >= 148):
        #     k = 1. + 0.09 * (A - 148)**2 * np.exp(-0.18 * (A - 148))

        def _Gamma_k(E):
            return gamma_eglo_cuda(E, E0, Gamma0, T,
                            epsilon_0, k)

        f1 = (E * _Gamma_k(E)) / ((E**2 - E0**2)**2 + E**2 * _Gamma_k(E)**2)
        f2 = 0.7 * _Gamma_k(0) / E0**3

        f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

        return f

    @cuda.jit
    def fmglo_ct_cuda(E: Union[np.ndarray, float],
                E0: float, sigma0: float, Gamma0: float,
                T: float, epsilon_0: float = 0, k: float = 1
                ) -> Union[np.ndarray, float]:
        """Modified Generalized Lorentzian with CT Temperature

        Modified to have with constant temperature of final states

        adapted from
        - J. Kroll et al., “Strength of the scissors mode in odd-mass Gd isotopes
        from the radiative capture of resonance neutrons,” Physical Review C,
        vol. 88, no. 3, Art. no. 3, Sep. 2013, doi: 10.1103/physrevc.88.034317.

        Note:
            - This was modified for a constant temperature dependece

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value
            E0 (float):
                Location parameter
            sigma0 (float):
                Scale factor
            Gamma0 (float):
                Width parameter
            T (float):
                Temperature parameter
            epsilon_0 (float, optional):
                reference/"critical" energy for enhanced width. Free parameter
            k (float, optional):
                enhancement factor; free parameter of e.g. from Fermi gas.
                For the default value, k=1 , this is equivalent to GLO

        Deleted Parameters:
            A: int
                mass number

        Returns:
            Union[np.ndarray, float]: strength function
        """

        # # (MeV); adopted from RIPL, "depends on model for state density"
        # epsilon_0 = 4.5
        # if A < 148:
        #     k = 1.0
        # if(A >= 148):
        #     k = 1. + 0.09 * (A - 148)**2 * np.exp(-0.18 * (A - 148))

        gamma_e = gamma_glo_cuda(0, E0, Gamma0, T)
        gamma_0 = gamma_eglo_cuda(0, E0, Gamma0, T, epsilon_0, k)

        f1 = (E * gamma_e) / ((E**2 - E0**2)**2 + E**2 * gamma_e**2)
        f2 = 0.7 * gamma_0 / E0**3

        f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

        return f

    @cuda.jit
    def gh_ct_cuda(E: Union[np.ndarray, float],
            E0: float, sigma0: float, Gamma0: float,
            T: float, k: float = 0.63) -> Union[np.ndarray, float]:
        """Goriely's Hybrid model, but with constant temperature of final states

        adapted from:
            - S. Goriely, Radiative neutron captures by neutron-rich nuclei and
            the r-process nucleosynthesis, Physics Letters B, Elsevier BV, 1998,
            436, 10-18 DOI: 10.1016/s0370-2693(98)00907-1
            - RIPL3 eq. (144-145)

        Note: Modified to have constant temperature of final states

        Parameters:
            E (Union[np.ndarray, float]):
                Input E value
            E0 (float):
                Location parameter
            sigma0 (float):
                Scale factor
            Gamma0 (float):
                Width parameter
            T (float):
                Temperature parameter
            k (float, optional):
                enhancement factor

        Returns:
            Union[np.ndarray, float]: strength function
        """
        Gamma = k * Gamma0 * (E**2 + 4 * pi**2 * T**2) / (E*E0)
        f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)

        f = strength_factor * sigma0 * Gamma0 * f1
        return f



@dataclass(slots=True, frozen=True)
class GSFModelParameters(ModelParameters):
    pass

@dataclass(slots=True, frozen=True)
class SPParameters(GSFModelParameters):
    value: float

@dataclass(slots=True, frozen=True)
class SLOParameters(GSFModelParameters):
    E0: float
    sigma0: float
    Gamma0: float

@dataclass(slots=True, frozen=True)
class GLO_CTParameters(GSFModelParameters):
    E0: float
    sigma0: float
    Gamma0: float
    T: float

@dataclass(slots=True, frozen=True)
class EGLO_CTParameters(GSFModelParameters):
    E0: float
    sigma0: float
    Gamma0: float
    T: float
    epsilon_0: float = 0
    k: float = 1

@dataclass(slots=True, frozen=True)
class fMGLO_CTParameters(GSFModelParameters):
    E0: float
    sigma0: float
    Gamma0: float
    T: float
    epsilon_0: float = 0
    k: float = 1

@dataclass(slots=True, frozen=True)
class GH_CTParameters(GSFModelParameters):
    E0: float
    sigma0: float
    Gamma0: float
    T: float
    k: float = 0.63

@dataclass(slots=True, frozen=True)
class GSFModel(Model):
    parameters: GSFModelParameters
    callback: GammaFunction

    def compile(self, target: Literal['njit', 'cuda'] = 'njit',
                print_body: bool = False):
        if target not in {'cuda', 'njit'}:
            raise ValueError(f"Invalid target: {target}. Must be 'njit' or 'cuda'")

        callback = self.callback
        if target == 'njit':
            compile = njit
        else:
            if cuda is None:
                raise RuntimeError("CUDA backend not available (Numba CUDA runtime not detected)")
            compile = cuda.jit(device=True)
            name = callback.__name__
            if 'cuda' not in name:
                name = name + '_cuda'
            callback = globals().get(name)
            if callback is None:
                raise ValueError(f"No CUDA-compatible version found for {self.callback.__name__}")

        params = asdict(self.parameters)

        # We unpack the parameters into a string
        # and splice them literally into the body
        # Cuda does not support keyword arguments
        p = ', '.join([f'float32({float(value)})' for param, value in params.items()])

        body = f"""
        def cb(e):
            return callback(e, {p})
        """
        if print_body:
            print(body)
        # remove leading whitespace but preserve indentation
        body = dedent(body)
        
        float_constructor = types.float32 if types is not None else float
        local_dict = {'callback': callback, 'float32': float_constructor}
        exec(body, local_dict)
        jit_callback = compile(local_dict['cb'])

        return jit_callback 
        

    def compile_cuda_old(self,print_body: bool = False) -> tuple[GammaFunction, dict]:
        cb = self.callback
        compile = cuda.jit(device=True)
        name = cb.__name__
        if 'cuda' not in name:
            cb = globals().get(f"{name}_cuda")

        params = asdict(self.parameters)

        p_list = ', '.join([f'{param}' for param, value in params.items()])

        body = f"""
        def cb(e, {p_list}):
            return {cb.__name__}(e, {p_list})
        """
        body = dedent(body)
        if print_body:
            print(body)

        local_dict = {cb.__name__: cb}
        exec(body, local_dict)
        fn = local_dict['cb']

        return compile(fn), params

    

        

@dataclass(slots=True, frozen=True)
class SP(GSFModel):
    parameters: SPParameters
    callback: GammaFunction = sp

@dataclass(slots=True, frozen=True)
class SLO(GSFModel):
    parameters: SLOParameters
    callback: GammaFunction = slo

@dataclass(slots=True, frozen=True)
class GLO_CT(GSFModel):
    parameters: GLO_CTParameters
    callback: GammaFunction = glo_ct

@dataclass(slots=True, frozen=True)
class EGLO_CT(GSFModel):
    parameters: EGLO_CTParameters
    callback: GammaFunction = eglo_ct

@dataclass(slots=True, frozen=True)
class fMGLO_CT(GSFModel):
    parameters: fMGLO_CTParameters
    callback: GammaFunction = fmglo_ct

@dataclass(slots=True, frozen=True)
class GH_CT(GSFModel):
    parameters: GH_CTParameters
    callback: GammaFunction = gh_ct

@dataclass(slots=True, frozen=True)
class GSFModelJAX(Model):
    parameters: GSFModelParameters
    callback: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __call__(self, e: jnp.ndarray) -> jnp.ndarray:
        return self.callback(e, **asdict(self.parameters))

    
def to_jax_model(model: GSFModel) -> GSFModelJAX:
    jax_callback = globals().get(f"{model.callback.__name__}_jax")
    if jax_callback is None:
        raise ValueError(f"No JAX-compatible version found for {model.__class__.name} : {model.callback.__name__}")
    
    return GSFModelJAX(
        parameters=model.parameters,
        callback=jax_callback
    )

    
@dataclass
class GSFComponent:
    model: GSFModel
    transition: TransitionType
    label: str

@dataclass
class TotalGSF:
    M1: list[GSFComponent] = field(default_factory=list)
    E1: list[GSFComponent] = field(default_factory=list)
    E2: list[GSFComponent] = field(default_factory=list)
    M2: list[GSFComponent] = field(default_factory=list)

    def match_transitions(self, transition: TransitionType) -> list[GSFComponent]:
        match transition:
            case TransitionType.M1:
                return self.M1
            case TransitionType.E1:
                return self.E1
            case TransitionType.E2:
                return self.E2
            case TransitionType.M2:
                return self.M2
            case _:
                raise ValueError(f"Invalid transition type: {transition}")

    def add_component(self, model: GSFModel, transition: TransitionType, label: str):
        components = self.match_transitions(transition)
        components.append(GSFComponent(model, transition, label))

    def __call__(self, E: np.ndarray) -> np.ndarray:
        total = np.zeros_like(E)
        for components in [self.M1, self.E1, self.E2, self.M2]:
            for component in components:
                total += component.model(E)
        return total

    def call_component(self, transition: TransitionType, E: np.ndarray) -> np.ndarray:
        total = np.zeros_like(E)
        components = self.match_transitions(transition)
        for component in components:
            total += component.model(E)
        return total

    def plot(self, E: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        total = self(E)
        ax.plot(E, total, label='Total', color='black', linewidth=2)
        
        for components in [self.M1, self.E1, self.E2, self.M2]:
            for component in components:
                ax.plot(E, component.model(E), label=f"{component.transition.name}: {component.label}")
        
        ax.set_xlabel(r'$E_\gamma$ (MeV)')
        ax.set_ylabel(r'$\gamma$SF (MeV$^{-3}$)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which='both', ls=':')
        
        return ax

    def to_jax(self) -> TotalGSF_JAX:
        return create_jax_total_gsf(self)



@dataclass(frozen=True, slots=True)
class TotalGSF_JAX:
    """ JAX-compatible TotalGSF model

    This is a less user friendly version of the TotalGSF model,
    which is more efficient to call in JAX.

    """
    E1: Callable[[jnp.ndarray], jnp.ndarray]
    M1: Callable[[jnp.ndarray], jnp.ndarray]
    E2: Callable[[jnp.ndarray], jnp.ndarray]
    M2: Callable[[jnp.ndarray], jnp.ndarray]
    total: Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, E: jnp.ndarray) -> jnp.ndarray:
        return self.total(E)

    def call_component(self, transition: TransitionType, E: jnp.ndarray) -> jnp.ndarray:
        # Might be faster to put this into a dict
        # not sure.
        match transition:
            case TransitionType.M1:
                return self.M1(E)
            case TransitionType.E1:
                return self.E1(E)
            case TransitionType.E2:
                return self.E2(E)
            case TransitionType.M2:
                return self.M2(E)
            case TransitionType.M1E2:
                return self.M1(E) + self.E2(E)
            case _:
                raise ValueError(f"Invalid transition type: {transition}")

def _component_function(E: jnp.ndarray, models: list[tuple[Callable, dict]]) -> jnp.ndarray:
    # Might need to turn this into a scalar to skip allocation
    total = jnp.zeros_like(E)
    for callback, params in models:
        total += callback(E, **params)
    return total

def create_jax_total_gsf(original: TotalGSF) -> TotalGSF_JAX:
    # We hoist up the parameters before jitting, so that we can jit the component functions
    def create_component_function(transition: TransitionType) -> Callable:
        components = original.match_transitions(transition)
        jax_models = [(to_jax_model(c.model).callback, asdict(to_jax_model(c.model).parameters)) for c in components]
        
        def component_function(E: jnp.ndarray) -> jnp.ndarray:
            return _component_function(E, jax_models)
        return component_function

    E1 = create_component_function(TransitionType.E1)
    M1 = create_component_function(TransitionType.M1)
    E2 = create_component_function(TransitionType.E2)
    M2 = create_component_function(TransitionType.M2)

    def total_function(E: jnp.ndarray) -> jnp.ndarray:
        return E1(E) + M1(E) + E2(E) + M2(E)

    return TotalGSF_JAX(E1=jit(E1), M1=jit(M1), E2=jit(E2), M2=jit(M2), total=jit(total_function))
