from .. import JAX_AVAILABLE, NUMBA_AVAILABLE, PYMC_AVAILABLE
from typing import Literal, TypeAlias, Callable
import numpy as np
from .modelmeta import Model
from .modelcontext import Constant
from ..stubs import Axes
from ..helpers import make_ax, is_running_in_jupyter, maybe_set
from ..numbalib import njit
from abc import ABC, abstractmethod

"""
TODO:
- [ ] The specification of a model is separate from its
      use. Prevents models in illegal states.
"""

BACKENDS: TypeAlias = Literal['numpy', 'numba', 'jax']
BACKEND = 'numpy'
def set_backend(backend: Literal['numpy', 'numba', 'jax', 'auto'] = 'auto') -> None:
    global BACKEND
    match backend.lower():
        case 'numpy' | 'numba' | 'jax' as back:
            BACKEND = back
        case 'auto':
            if JAX_AVAILABLE:
                BACKEND = 'jax'
            elif NUMBA_AVAILABLE:
                BACKEND = 'numba'
            else:
                BACKEND = 'numpy'
        case _:
            raise ValueError(f"Backend {backend} not supported.")

def validate_backend(backend: str) -> BACKENDS | None:
    if backend.lower() in ('numpy', 'numba', 'jax'):
        return backend.lower()
    return None

set_backend('auto')


if JAX_AVAILABLE:
    from jax import numpy as jnp
    from jax import jit

    @jit
    def EB09CT(E, J, mass):
        return jnp.power(0.98 * (mass**(0.29)), 2)

    @jit
    def EB09Emp(E, J, mass, Pa_prime):
        Eeff = E - 0.5 * Pa_prime
        Eeff = jnp.where(Eeff < 0, 0, Eeff)
        return 0.391 * jnp.power(mass, 0.675) * jnp.power(Eeff, 0.312)

@njit
def CT_fn(E, T, E0):
    return 1/T * np.exp((E - E0) / T) 

@njit
def BSFG_fn(E, a, shift, sigma):
    """ Create model nld from BSFG model, with low energy fix

    Low energy fix from EB05, see Eq. (3) and text below it.
    Alternative from is presented eg. in RIPL3, see eq. (48).

    Note that we use a different convention for U and the excitation energy
    than in EB05.

    Args:
        energy: energy grid on which the nld model is calculated
        a: Level density parameter
        Eshift: Backshift parameter
        sigma: Spincut parameter (has to be same type as energy)

    Returns:
        nld evaluated at energy
    """
    U = E - shift
    # fix for low energies
    U = np.where(U < (25/16)/a, (25/16)/a, U)

    nom = np.exp(2*np.sqrt(a*U))
    denom = 12 * np.sqrt(2) * sigma * a**(1/4) * U**(5/4)
    nld = nom / denom
    return nld

if JAX_AVAILABLE:
    from jax import jit
    from jax import numpy as jnp

    @jit
    def CT_jax(E, T, E0):
        return 1/T * jnp.exp((E - E0) / T)

    @jit
    def BSFG_jax(E, a, shift, sigma):
        U = E - shift
        # fix for low energies
        U = jnp.where(U < (25/16)/a, (25/16)/a, U)

        nom = jnp.exp(2*jnp.sqrt(a*U))
        denom = 12 * jnp.sqrt(2) * sigma * a**(1/4) * U**(5/4)
        nld = nom / denom
        return nld


class SpinModel:
    """
    Model for the spin distribution of a nucleus.
    """
    def __init__(self, D0=0, Gg=0,  name: str = ''):
        #super().__init__(name=name)
        self.D0 = Constant.from_value(D0, name='D0')
        self.Gg = Constant.from_value(Gg, name='Gg')
        try:
            model = Model.get_context()
            print("Spinmodel context", model)
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate distributions. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )


class NormalizationModel(ABC):
    def __init__(self, backend: BACKENDS | None = None):
        self._backend: BACKENDS = BACKEND
        self.backend = backend or BACKEND

    @property
    def backend(self) -> BACKENDS:
        return self._backend

    @backend.setter
    def backend(self, backend: BACKENDS) -> None:
        if validate_backend(backend) is None:
            raise ValueError(f"Backend {backend} not supported.")
        self._backend = backend

    @staticmethod
    @abstractmethod
    def parameter_map() -> dict[str, str]: ...

    def get_fn(self) -> Callable[..., np.ndarray]:
        match self.backend:
            case 'numpy':
                return self.get_numpy()
            case 'numba':
                return self.get_numba()
            case 'jax':
                return self.get_jax()

    def make_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        match self.backend:
            case 'numpy':
                return self.make_numpy()
            case 'numba':
                return self.make_numba()
            case 'jax':
                return self.make_jax()

    @staticmethod
    @abstractmethod
    def get_numpy() -> Callable[..., np.ndarray]: ...

    @classmethod
    def get_numba(cls) -> Callable[..., np.ndarray]:
        if NUMBA_AVAILABLE:
            from numba import njit
            fn = cls.get_numpy()
            if hasattr(fn, '__numba__'):
                return fn
            return njit(fn)
        else:
            raise ImportError("Numba not available.")

    @classmethod
    def get_jax(cls) -> Callable[..., np.ndarray]:
        raise NotImplementedError()

    def make_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        parameters = self.parameter_map()
        arguments = {fn_key: getattr(self, key) for key, fn_key in parameters.items()}
        fn = self.get_numpy()
        def fn1(E):
            return fn(E, **arguments)
        return fn1

    def make_numba(self) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def make_jax(self) -> Callable[[np.ndarray], np.ndarray]:
        if JAX_AVAILABLE:
            parameters = self.parameter_map()
            arguments = [getattr(self, key) for key in parameters]
            from jax import jit
            fn = self.get_jax()
            def fn1(E):
                return fn(E, *arguments)
            return jit(fn1)
        else:
            raise ImportError("Jax not available.")


class NLDModel(NormalizationModel):
    def plot(self, ax: Axes | None = None):
        ax: Axes = make_ax(ax)
        E = np.linspace(0, 20, 1000)
        ax.plot(E, self.make_fn()(E))
        
        maybe_set(ax, xlabel="Excitation energy [keV]", ylabel="Level density [keV⁻¹]")
        return ax


class GSFModel(NormalizationModel):
    pass


class CT_(NLDModel):
    """
    Constant temperature model.
    """
    def __init__(self, T, E0, backend: BACKENDS | None = None):
        super().__init__(backend=backend)
        self.T = T
        self.E0 = E0

    @staticmethod
    def parameter_map() -> dict[str, str]:
        return {'T': 'T', 'E0': 'E0'}

    @classmethod
    def get_numpy(cls) -> Callable[..., np.ndarray]:
        return CT_fn

    @classmethod
    def get_jax(cls) -> Callable[..., np.ndarray]:
        return CT_jax

    def make_numba(self) -> Callable[[np.ndarray], np.ndarray]:
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba not available.")

        from numba import njit
        fn = self.get_numba()
        T = self.T
        E0 = self.E0
        @njit
        def fn1(E):
            return fn(E, T, E0)

        return fn1

    def latex(self) -> str:
        return r"$$\frac{1}{T} \exp\left(\frac{E - E_0}{T}\right)$$"

    def __repr__(self) -> str:
        s = f"CT(T={self.T}, E0={self.E0})"
        return s

    def _repr_markdown_(self) -> str:
        s = f"""Constant temperature model
{self.latex()}
$T$: {self.T}
$E_0$: {self.E0}
        """
        return s
             

class CT(Model):
    def __init__(self, name: str):
        super().__init__(name=name)

class BSFG(NLDModel):
    """ Back shifted fermi gas """
    def __init__(self, a, shift, sigma, backend: BACKENDS | None = None):
        super().__init__(backend=backend)
        self.a = a
        self.shift = shift
        self.sigma = sigma

    @staticmethod
    def parameter_map() -> dict[str, str]:
        return {'a': 'a', 'shift': 'shift', 'sigma': 'sigma'}

    @classmethod
    def get_numpy(cls) -> Callable[..., np.ndarray]:
        return BSFG_fn

    @classmethod
    def get_jax(cls) -> Callable[..., np.ndarray]:
        return BSFG_jax

    def make_numba(self) -> Callable[[np.ndarray], np.ndarray]:
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba not available.")

        from numba import njit
        fn = self.get_numba()
        a = self.a
        shift = self.shift
        sigma = self.sigma
        @njit
        def fn1(E):
            return fn(E, a, shift, sigma)

        return fn1

    @staticmethod
    def latex() -> str:
        return r"$\rho(E; a, E_1, \sigma ) = \frac{1}{12 \sqrt{2} \sigma(E) a^{1/4} (E-E_1))^{5/4}} \exp\left(2 \sqrt{a (E-E_1)}\right)$"

    @staticmethod
    def describe(render=True) -> str:
        s = fr"""*Back shifted fermi gas model*
${BSFG.latex()}$
$a$: Level density parameter

$E_{1}$: Backshift parameter

$\sigma$: Spin-cutoff parameter / distribution"""
        if render:
            if is_running_in_jupyter():
                from IPython.display import Markdown, display
                return display(Markdown(s))
        return s

    def __repr__(self) -> str:
        s = f"BSFG(a={self.a}, shift={self.shift}, sigma={self.sigma})"
        return s

    def _repr_markdown_(self) -> str:
        s = fr"""Back shifted fermi gas model
${self.latex()}$

$a$: {self.a}

$E_{1}$: {self.shift}

$\sigma$: {self.sigma}
        """
        return s

