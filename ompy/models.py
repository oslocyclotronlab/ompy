import numpy as np
import re
import pickle
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Union, Tuple, Any, Dict, Callable, List
from scipy.optimize import curve_fit

from .vector import Vector


def NonTuple2():
    return [None, None]


@dataclass()
class Model:
    """Dataclass for Model

    Attributes:
        Name: Name of the class (for printing etc.)
    """

    name: str
    __isfrozen: bool = False

    def get_parameters(self) -> List[str]:
        """ Returns a list of the names of the paramters """
        parameters = [p for p in self.__dict__.keys()
                      if "__" not in p]
        return parameters

    def is_changed(self, include: List[str] = [],
                   exclude: List[str] = []) -> None:
        """Verify that defaults arguments have been changed

        Args:
            include (List[str], optional): List of attribute names be
                included in the check. Default is all attributes.
            exclude (List[str], optional): List of attribute names to exclude
                be excluded from check. Default is none

        Raises:
            ValueError: If parameters are still the default values
        """
        include = include if include else self.get_parameters()
        keys = [k for k in include if k not in exclude]

        for key in keys:
            # exception for self._Emax: call self.Emax
            if key.startswith("_"):
                key = key[1:]

            val = getattr(self, key)
            if val is None:
                raise ValueError(f"Model `{self.name}` has default (None) "
                                 f"variable `{key}`.")
            if isinstance(val, list):
                if not val or None in val:
                    raise ValueError(f"Model `{self.name}` has default [] "
                                     f"or `None` in variable `{name}`.")

    def asdict(self) -> Dict[str, Any]:
        """ wrapper for dataclasses.asdict() """
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save the model parameters to `path`

        Args:
            path (Union[str, Path]): The path
        """
        path = Path(path)
        with path.open('wb') as fopen:
            pickle.dump(self, fopen, -1)

    def load(self, path: Union[str, Path]) -> None:
        """Loads own parameters from `path`

        Args:
            path (Union[str, Path]): Path to pickled file

        Raises:
            IOError: Path doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise IOError(f"The path {path} does not exist.")

        with path.open('rb') as fin:
            obj = pickle.load(fin)
        for k in obj.get_parameters():
            setattr(self, k, getattr(obj, k))

    def __setattr__(self, attr, value) -> None:
        if self.__isfrozen and not hasattr(self, attr):
            raise AttributeError(f"Model `{self.name}` does not"
                                 f" have parameter `{attr}`")
        else:
            super().__setattr__(attr, value)

    def _freeze(self):
        self.__isfrozen = True

    def __post_init__(self):
        self._freeze()

    def __str__(self) -> str:
        string = f'Model {self.name}\n\n'
        for fld in fields(self):
            if fld.name.startswith('_') or fld.name == 'name':
                continue
            if fld.metadata:
                string += str(fld.metadata) + "\n"
            string += f"{fld.name}: {gettype(fld.type)} = "
            string += f"{getattr(self, fld.name)}\n\n"
        return string[:-2]


def gettype(signature):
    signature = str(signature)
    signature = signature.replace('typing.', '')
    # Remove nested types types
    signature = re.sub(r"\w+\.", '', signature)
    if signature[0] == "<":
        return signature.split("'")[1]
    return signature


@dataclass
class AbstractExtrapolationModel(Model):
    """Abstract class for extrapolation models

    Attributes:
        scale (float): Exponential scaling Exp[scale*Eg + shift]
            before normalization in MeV^-1
        shift (float): Exponential shift Exp[scale*Eg + shift]
            before normalization
        shift_after (float): Exponential shift Exp[scale*Eg + shift]'
            after normalization
        Emin (float, optional): Minimal gamma energy to extrapolate from in MeV
        Emax (float, optional): Maximal gamma energy to extrapolate from in MeV
        steps (float, optional): Number of gamma energies to use in
            extrapolation. Defaults to 1001.
        method (float): Method to obtain `scale` and `shift`. Must be in
            either ["fit", "fix"]
        model (Callable[..., Any]): If `method` is `"fit"`, the model to fit
            to the data has to be provided
        Efit (Tuple[float, float]): Fit range (lower, higher).

        TODO:
            - allow for more flexible (more general) models.
    """
    scale: float = field(default=1.0,
            metadata='Exponential scaling Exp[scale*Eg + shift]'
                     'before normalization in MeV^-1')  # noqa
    shift: float = field(default=25.0,
            metadata='Exponential shift Exp[scale*Eg + shift]'
                     'before normalization')  # noqa
    _shift_after: float = field(default=None,
            metadata='Exponential shift Exp[scale*Eg + shift]'
                     'after normalization')  # noqa
    Emin: Optional[float] = field(default=None,
            metadata='Minimal gamma energy to extrapolate from in MeV')  # noqa
    Emax: Optional[float] = field(default=None,
            metadata='Maximal gamma energy to extrapolate from in MeV')  # noqa
    steps: int = field(default=1001,
            metadata='Number of gamma energies to use in extrapolation')  # noqa
    _method: str = field(default="fit",
            metadata='Method for obtaining `scale` and `shift`')  # noqa

    # if method is "fit"
    model: Callable[..., Any] = field(default=None,
            metadata='extrapolation model')  # noqa
    Efit: Optional[Tuple[float, float]] = \
              field(default_factory=NonTuple2,
                    metadata='Fit range')  # noqa

    def range(self) -> np.ndarray:
        """Linearly spaced array from Emin to Emax """
        return np.linspace(self.Emin, self.Emax, self.steps)

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        implemented = ["fit", "fix"]
        if value not in implemented:
            raise NotImplementedError(f"method: {value}"
                                      f"must be in {implemented}")
        else:
            self._method = value

    @property
    def shift_after(self) -> str:
        if self._shift_after is None:
            return self.shift
        else:
            return self._shift_after

    @shift_after.setter
    def shift_after(self, value: float) -> None:
        self._shift_after = value

    def norm_to_shift_after(self, norm: float) -> None:
        self.shift_after = self.shift + np.log(norm)

    def fit(self, gsf: Vector, model: Optional[float] = None,
            Emin: Optional[float] = None,
            Emax: Optional[float] = None) -> None:
        """Fits parameters of `model` to the given gsf

        Optional parameters are detemined from the instance attributes if
        not provided.
        Results are stored in the instance attributes.

        Args:
            gsf (Vector): gsf
            model (float, optional): model to fit
            Emin (float, optional): Lower limit on the fit range
            Emax (float, optional): Higher limit on the fit range

        TODO:
            - allow for more flexible (more general) models.
        """
        if model is None:
            model = self.model

        self.autofitrange(gsf)  # sets only if self.Efit[i] is None
        if Emin is None:
            Emin = self.Efit[0]
        if Emax is None:
            Emax = self.Efit[1]

        assert Emin < Emax, f"Emin: {Emin:.1f} should be < Emax: {Emax:.1f}"

        idx1 = gsf.index(Emin)
        idx2 = gsf.index(Emax)
        x = gsf.E[idx1:idx2+1]
        y = gsf.values[idx1:idx2+1]
        if gsf.std is None:
            yerr = None
        else:
            yerr = gsf.std[idx1:idx2+1]
        popt, pcov = curve_fit(model, x, y, sigma=yerr,
                               p0=[1., -20.])
        self.scale = popt[0]
        self.shift = popt[1]

    def extrapolate(self, E: Optional[np.ndarray] = None,
                    scaled: Optional[bool] = True) -> Vector:
        """ Wrapper to extrapolate a model

        Args:
            E (optional): extrapolation energies. If not
            scaled (optional): If gsf is normalized, use same scaling

        Returns:
            The extrapolated values
        """
        shift = self.shift_after if scaled else self.shift
        if E is None:
            E = self.range()
        values = self.model(E, self.scale, shift)
        return Vector(values=values, E=E)

    def autorange(self, *args, **kwargs):
        """ Not implemented for Abstract """
        raise NotImplementedError("Not implemented for Abstract")

    def autofitrange(self, *args, **kwargs):
        """ Not implemented for Abstract """
        raise NotImplementedError("Not implemented for Abstract")


@dataclass
class ExtrapolationModelLow(AbstractExtrapolationModel):
    model: Callable[..., Any] = field(default=None,
            metadata='extrapolation model')  # noqa

    def __post_init__(self):
        self.model = self.__model

    def autorange(self, gsf: Vector):
        """ Set Emin and Emax in MeV from gsf if not set before """
        gsf = gsf.copy()
        gsf.to_MeV()
        self.Emin = 0 if self.Emin is None else self.Emin
        self.Emax = gsf.E[0] if self.Emax is None else self.Emax

    def autofitrange(self, gsf: Vector):
        """ Guess(!) Efit in MeV from gsf if not set before"""
        gsf = gsf.copy()
        gsf.to_MeV()
        fraction = int(len(gsf.E) / 6)
        if self.Efit[0] is None:
            if len(gsf.E) < 8:
                raise NotImplementedError("Set Efit manually")
            self.Efit[0] = gsf.E[2]
        if self.Efit[1] is None:
            if len(gsf.E) < 8:
                raise NotImplementedError("Set Efit manually")
            self.Efit[1] = gsf.E[fraction+2]

    def __model(self, Eg: Optional[float] = None,
                scale: Optional[float] = None,
                shift: Optional[float] = None) -> np.ndarray:
        """ gsf extrapolation at low energies

        Computes Exp[scale·Eg + shift]

        Args:
            Eg: Gamma-ray energies
            scale: The scaling parameter
            shift: The shift parameter
        Returns:
            Extrapolated values
        """
        if Eg is None:
            try:
                Eg = self.range()
            except TypeError:
                raise ValueError("Need to set Eg, or self.Emin and self.Emax")
        if scale is None:
            scale = self.scale
        if shift is None:
            shift = self.shift
        return np.exp(scale*Eg + shift)


@dataclass
class ExtrapolationModelHigh(AbstractExtrapolationModel):
    model: Callable[..., Any] = field(default=None,
            metadata='extrapolation model')  # noqa

    def __post_init__(self):
        self.model = self.__model

    def autorange(self, gsf: Vector):
        """ Set Emin and Emax in MeV from gsf if not set before """
        gsf = gsf.copy()
        gsf.to_MeV()
        self.Emin = 0 if self.Emin is None else self.Emin
        self.Emax = 20 if self.Emax is None else self.Emax

    def autofitrange(self, gsf: Vector, check_existing: bool = True):
        """ Guess(!) Efit in MeV from gsf if not set before"""
        gsf = gsf.copy()
        gsf.to_MeV()
        fraction = int(len(gsf.E) / 6)
        if self.Efit[0] is None:
            if len(gsf.E) < 8:
                raise NotImplementedError("Set Efit manually")
            self.Efit[0] = gsf.E[-fraction-2]
        if self.Efit[1] is None:
            if len(gsf.E) < 8:
                raise NotImplementedError("Set Efit manually")
            self.Efit[1] = gsf.E[-2]

    def __model(self, Eg: Optional[float] = None,
                scale: Optional[float] = None,
                shift: Optional[float] = None) -> np.ndarray:
        """ gsf extrapolation at high energies

        Computes Exp[scale·Eg + shift] / Eg³

        Args:
            Eg: Gamma-ray energies
            scale: The scaling parameter
            shift: The shift parameter
        Returns:
            Extrapolated values
        """
        if Eg is None:
            try:
                Eg = self.range()
            except TypeError:
                raise ValueError("Need to set Eg, or self.Emin and self.Emax")
        if scale is None:
            scale = self.scale
        if shift is None:
            shift = self.shift
        return np.exp(scale*Eg + shift) / np.power(Eg, 3)


@dataclass
class NormalizationParameters(Model):
    """Storage for normalization parameters + some convenience functions
    """

    #: Average s-wave resonance spacing D0 [eV]
    D0: Optional[Tuple[float, float]] = field(default=None,
            metadata='Average s-wave resonance spacing D0 [eV]')  # noqa
    #: Total average radiative width  [meV]
    Gg: Optional[Tuple[float, float]] = field(default=None,
            metadata='Total average radiative width  [meV]')  # noqa
    #: Neutron separation energy [MeV]
    Sn: Optional[Tuple[float, float]] = field(default=None,
            metadata='Neutron separation energy [MeV]')  # noqa
    #: "Target" (A-1 nucleus) ground state spin
    Jtarget: Optional[float] = field(default=None,
            metadata='"Target" (A-1 nucleus) ground state spin')  # noqa
    #: Min energy to integrate <Γγ> from
    Emin: float = field(default=0.0,
            metadata="Min energy to integrate <Γγ> from")  # noqa
    #: Max energy to integrate <Γγ> to
    _Emax: float = field(default=None,
            metadata="Max energy to integrate <Γγ> to")  # noqa
    #: Number of steps in energy grid
    steps: int = field(default=101,
            metadata="Number of steps in energy grid")  # noqa
    #: Spincut model
    spincutModel: str = field(default=None,
            metadata='Spincut model')  # noqa
    #: Parameters necessary for the spin cut model
    spincutPars: Dict[str, Any] = field(default=None,
            metadata='parameters necessary for the spin cut model')  # noqa

    def E_grid(self,
               retstep: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Wrapps np.linspace creates linearly spaced array from Emin to Emax

        Args:
            retstep (bool, optional): If `True` (default), returns stepsize

        Returns:
            Samples of the array. If `retstep` is `True`, returns also spacing
            between the samples.
        """
        return np.linspace(self.Emin, self.Sn[0], num=self.steps,
                           retstep=retstep)

    @property
    def Emax(self) -> float:
        """ Max energy to integrate <Γγ> to """
        if self._Emax is None:
            return self.Sn[0]
        else:
            return self._Emax

    @Emax.setter
    def Emax(self, value: float) -> None:
        assert value > self.Emin, "Emax must be larger then Emin"
        self._Emax = value


@dataclass
class ResultsNormalized(Model):
    """Class to store the results of the Oslo Method

    Attributes:
        nld: see below
        gsf: see below
        pars: see below
        samples: see below
        evidence: see below
        nld_model: see below
        gsf_model_low: see below
        gsf_model_high: see below
    """
    #: (Vector or List[Vector]): normalized or initial, depending on method
    nld: Union[Vector, List[Vector]] = field(default_factory=list,
             metadata='nld (normalized or initial, depending on method)')  # noqa
    #: (Vector or List[Vector]): normalized or initial, depending on method
    gsf: Union[Vector, List[Vector]] = field(default_factory=list,
             metadata='gsf (normalized or initial, depending on method)')  # noqa
    #: (List[Dict[str, Any]]): Parameters for the normalization/models used there
    pars: List[Dict[str, Any]] = field(default_factory=list,
            metadata='Parameters for the normalization/models used there')  # noqa
    #: (List[Dict[str, Any]]): Samples from the posterior of the parameters
    samples: List[Dict[str, Any]] = field(default_factory=list,
            metadata='Samples from the posterior of the parameters')  # noqa
    #: (Tuple of List[Tuple]): Evidence and error in evidence for model
    evidence: Union[Tuple[float, float], List[Tuple[float, float]]] \
        = field(default_factory=list,
                metadata='Global evidence for the model')
    #: (List[Callable[..., Any]]): nld model for each nld
    nld_model: List[Callable[..., Any]] = field(default_factory=list,
            metadata='nld model')  # noqa
    #: List[AbstractExtrapolationModel]: gsf model at low Eγ for each gsf
    gsf_model_low: List[AbstractExtrapolationModel] = \
        field(default_factory=list, metadata='gsf model at low Eγ')  # noqa
    #: List[AbstractExtrapolationModel]: gsf model at high Eγ for each gsf
    gsf_model_high: List[AbstractExtrapolationModel] = \
        field(default_factory=list, metadata='gsf model at high Eγ')  # noqa
