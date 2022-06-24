# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from .library import call_model
from scipy.interpolate import interp1d
from typing import Optional, Sequence, Tuple, Any, Union, Dict
import matplotlib.pyplot as plt
from .modelcontext import ModelContext
from .validator import Unitful, UnitfulError, Bounded
import inspect


class SpinModel(ModelContext):
    def __init__(self, model: str):
        print(">>>>>")
        self.model_name = model
        model = SpinFunction.get_model(model)
        defaults = SpinModel.get_variables(model)
        self.model = model(*((0.0,)*len(defaults)))
        self.is_set = {v: False for v in defaults}
        self.defaults = defaults
        print("<<<<<")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for key, is_set in self.is_set.items():
                if not is_set:
                    raise RuntimeError(f"Parameter `{key}` must set.")
        return

    def __setattr__(self, attr, val):
        if hasattr(self, 'defaults'):
            if attr not in self.defaults:
                raise AttributeError(f"Parameter `{attr}` is not a parameter"
                                     f"of {self.model_name}. Available "
                                     f"parameters are "
                                     f"{', '.join(self.defaults)}.")

            retval = setattr(self.model, attr, val)
            self.is_set[attr] = True
            return retval
        else:
            self.__dict__[attr] = val

    @staticmethod
    def get_variables(model):
        # Get all member variables
        vals = inspect.getmembers(model, lambda x: not inspect.isroutine(x))
        # Remove all fields starting with '__'
        vals = [v for v in vals if not v[0].startswith('__')]
        # Only want variables having a twin with underscore
        cut_names = {v[0][1:] for v in vals}
        vals = [v[0] for v in vals if v[0] in cut_names]
        return vals


class MetaSpin(type):
    """ Metaclass for SpinFunction

    The metaclass registers subclasses when they are
    created, allowing for SpinFunction.available() and
    similar methods to work.
    """
    def __new__(cls, clsname, superclasses, attributedict):
        #print("clsname:", clsname)
        #print("superclasses:", superclasses)
        if not superclasses == ():
            root = superclasses[0].__mro__[-2]
            root._register(clsname)
        #print("Attributes:", attributedict)
        return super(MetaSpin, cls).__new__(cls, clsname, superclasses,
                                            attributedict)


class SpinFunction(metaclass=MetaSpin):
    _models = []

    def __init__(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.distribution(*args, **kwargs)

    @classmethod
    def _register(cls, name: str):
        cls._models.append(name)

    def distribution(self, Ex, J):
        """ The Ericson Spin distribution

        Adapted from
        https://doi.org/10.1080/00018736000101239
        """
        σ2 = self.spin_cutoff_sq(Ex, J).T
        dist = ((2*J + 1) / (2*σ2)
                * np.exp(-np.power(J+0.5, 2) / (2*σ2)))
        return np.squeeze(dist)

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        raise NotImplementedError()

    def plot(self, Ex, J, ax=None, **kwargs):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        dist = self.distribution(Ex, J)
        ax.plot(J, dist.T)
        return fig, ax

    @classmethod
    def available(cls):
        return cls._models

    @classmethod
    def is_available(cls, model: str) -> bool:
        if model.lower() not in {m.lower() for m in cls._models}:
            return False
        return True

    @classmethod
    def get_model(cls, model: str) -> SpinFunction:
        model = model.lower()
        for m in cls._models:
            if model == m.lower():
                return eval(m)
        raise ValueError(f"'{model}' is not a valid `SpinFunction` model. "
                         "See `SpinFunction.available()`.")


class Const(SpinFunction):
    """
    Constant spin-cutoff parameter

    Attributes:
        sigma (float): Spin cut-off parameter

    """
    sigma = Bounded(0.0, type=float)

    def __init__(self, sigma):
        self.sigma = sigma

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        return np.full_like(Ex, self.sigma**2)


class EB05(SpinFunction):
    """
    Von Egidy & B PRC72,044311(2005), Eq. (4)
    The rigid moment of inertia formula (RMI)
    FG+CT. Values for NLDa and Eshift can be found
    in the aforementioned paper.
    Doi = 10.1103/PhysRevC.72.044311

    Attributes:
        mass (int): The mass number of the residual nucleus
        NLDa (float): Level density parameter [MeV⁻¹]
        Eshift (float): Energy shift [MeV]
    """
    mass = Bounded(0, type=float)
    NLDa = Unitful('0.0 MeV^(-1)')
    Eshift = Unitful('0.0 MeV')

    def __init__(self, mass, NLDa, Eshift):
        self.mass = mass
        self.NLDa = NLDa
        self.Eshift = Eshift

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        a = self.NLDa.magnitude
        Ex = np.atleast_1d(Ex)
        Eeff = Ex - self.Eshift.magnitude
        Eeff[Eeff < 0] = 0
        sigma2 = (0.0146 * np.power(self.mass, 5.0 / 3.0)
                  * (1 + np.sqrt(1 + 4*a*Eeff))
                  / (2*a))
        return sigma2

    def __str__(self) -> str:
        s = inspect.cleandoc(f"""
        Rigid moment of inertia
        Von Egidy & B PRC72,044311(2005), Eq. (4)
        Mass number:
        mass = {self.mass}
        NLD parameter a:
        NLDa = {self.NLDa:~}
        Energy backshift parameter:
        Eshift = {self.Eshift:~}
        """)
        return s


class EB09CT(SpinFunction):
    """
    The constant temperature (CT) formula
    - Von Egidy & B PRC80,054310, see sec. IV, p7 refering to ref. below
    - original ref: Von Egidy et al., NPA 481 (1988) 189, Eq. (3)

    Attributes:
        mass (int): Mass number
    """
    mass = Bounded(0, type=float)

    def __init__(self, mass):
        self.mass = mass

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        sigma2 = np.power(0.98 * (self.mass**(0.29)), 2)
        return sigma2

    def __str__(self) -> str:
        s = inspect.cleandoc(f"""
        Constant temperature
        Von Egidy et al., NPA 481 (1988) 189, Eq. (3)
        Mass number:
        mass = {self.mass}
        """)
        return s


class EB09Emp(SpinFunction):
    """
    von Egidy, T. and Bucurescu, D
    PhysRevC.80.054310, Eq.(16)
    FG+CT

    Attributes:
        mass: mass number
        Pa_prime: Deuteron pairing energy. Can be found
                  tabulated in aforementioned paper.
    """
    mass = Bounded(0.0, type=float)
    Pa_prime = Unitful('0.0 MeV')

    def __init__(self, mass, Pa_prime):
        self.mass = mass
        self.Pa_prime = Pa_prime

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        Ex = np.atleast_1d(Ex)
        Eeff = Ex - 0.5 * self.Pa_prime.magnitude
        Eeff[Eeff < 0] = 0
        sigma2 = 0.391 * np.power(self.mass, 0.675) * np.power(Eeff, 0.312)
        return sigma2

    def __str__(self) -> str:
        s = inspect.cleandoc(f"""
        von Egidy, T. and Bucurescu, D
        PhysRevC.80.054310, Eq.(16)
        Mass number:
        mass = {self.mass}
        Deuteron pairing energy:
        Pa_prime = {self.Pa_prime}
        """)
        return s


class DiscAndEB05(SpinFunction):
    """
    Linear interpolation of the spin-cut between
    a spin cut "from the discrete levels" and EB05
    Reference: Guttormsen et al., 2017, PRC 96, 024313

    Note:
        We set sigma2(E<E_discrete) = sigma2(E_discrete).
        This is not specified in the article, and may have been done
        differently before.

    Args:
        mass (int): The mass number of the residual nucleus
        NLDa (float): Level density parameter
        Eshift (float): Energy shift
        Sn (float): Neutron separation energy
        sigma2_disc (Tuple[float, float]): [float, float]
            [Energy, sigma2] from the discretes
        Ex (float or Sequence, optional):
            Excitation energy. Defaults to self.Ex
    """
    mass = Bounded(0.0, type=float)
    NLDa = Unitful('0.0 MeV^(-1)')
    Eshift = Unitful('0.0 MeV')
    Sn = Unitful('0.0 MeV')

    def __init__(self, mass, NLDa, Eshift, Sn, sigma2_disc):
        self.mass = mass
        self.NLDa = NLDa
        self.Eshift = Eshift
        self.Sn = Sn
        self.sigma2_disc

    def spin_cutoff_sq(self, Ex, J) -> np.ndarray:
        Sn = self.Sn.magnitude
        mass = self.mass
        NLDa = self.NLDa.magnitude
        Eshift = self.Eshift.magntiude
        Ex = np.atleast_1d(Ex)
        sigma2_Sn = self.gEB05(mass, NLDa, Eshift, Ex=Sn)
        sigma2_EB05 = lambda Ex: self.gEB05(mass, NLDa, Eshift, Ex=Ex)
        x = [sigma2_disc[0], Sn]
        y = [sigma2_disc[1], sigma2_EB05(Sn)]
        sigma2 = interp1d(x, y,
                          bounds_error=False,
                          fill_value=(sigma2_disc[1], sigma2_Sn))
        return np.where(Ex < Sn, sigma2(Ex), sigma2_EB05(Ex))
