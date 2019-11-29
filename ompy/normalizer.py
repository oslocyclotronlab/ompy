import numpy as np
import logging
import inspect
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
import re
import warnings
import copy
from contextlib import redirect_stdout
from pathlib import Path
from numpy import ndarray
from scipy.optimize import differential_evolution
from typing import Optional, Sequence, Tuple, Any, Union, Callable, Dict, List
from scipy.stats import truncnorm
from tqdm import tqdm
from dataclasses import dataclass, field
from .extractor import Extractor
from .vector import Vector
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized


LOG = logging.getLogger(__name__)

TupleDict = Dict[str, Tuple[float, float]]


class Normalizer:
    """ Normalizes NLD to empirical data

    Takes empirical data in form of an array of discrete levels,
    neutron separation energy Sn, a model to estimate what the
    NLD is at Sn, and several parameters for the model as well
    as bounds on normalization parameters.

    The user can provide either an nld Vector or an instance of an
    Extractor when normalizing. If an Extractor is provided,
    each member will be normalized.

    As a consequence of a complex problem, this class has a complex
    interface. Much of the book-keeping associated with normalization
    has been automated, but there is still a lot of settings and
    parameters for the user to take care of. Some default values
    has been seen, but the user must be _EXTREMELY_ careful when
    evaluating the output.

    Attributes:
        discrete (Vector): The discrete NLD at lower energies. [MeV]
        nld (Vector): The NLD to normalize. Gets converted to [MeV] from [keV].
        D0 (Tuple[float, float]): The (mean, std) average resonance
            spacing from s waves [eV]
        extractor (Extractor): An optional Extractor instance. The
            nld members will all be normalized.
        bounds (Dict[str, Tuple[float, float]): The bounds on each of
            the parameters. Its keys are 'A', 'alpha', 'T', and 'D0'. The
            values are on the form (min, max).
        spin (Dict[str, Any]): The parameters to be used in the spin
            distribution model. Its keys are 'spincutModel', 'spincutPars',
            'J_target', 'Gg' and 'Sn'.
        model (Callable[..., ndarray]): The model to use at high energies
            to estimate the NLD. Defaults to constant temperature model.
        curried_model (Callable[..., ndarray]): Same as model, but
            curried with the spin distribution parameters to reduce
            book keeping. Is set by initial_guess(), _NOT_ the user.
        multinest_path (Path): Where to save the multinest output.
            defaults to 'multinest'.
        res (ResultsNormalized): Results of the normalization
        use_smoothed_levels (bool): Flag for whether to smooth histogram
            over discrete levels when loading from file. Defaults to True.
        path (Path): The path the transformed vectors to.

    TODO:
        - parameter to limit the number of multinest samples to store. Note
        that the samples should be shuffled to retain some "random" samples
        from the pdf (not the importance weighted)
    """
    def __init__(self, *, extractor: Optional[Extractor] = None,
                 nld: Optional[Vector] = None,
                 discrete: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = None) -> None:
        """ Normalizes nld ang gSF.

        The prefered syntax is Normalizer(extractor=...) or Normalizer(nld=...)
        If neither is given, the nld must be explicity be set later by:
            normalizer..renld = nld
            normalizer.extractor = extractor
        or:
            normalizer.normalize(..., nld=nld)
            normalizer.normalize(..., extractor=extractor)

        Note:
            Normalizes nld/gsf according to
                nld' = nld * A * np.exp(alpha * Ex), and
                gsf' = gsf * B * np.exp(alpha * Eg)
            Note: This is the transformation eq (3), Schiller2000

        Args:
            extractor: The extractor to use get nld from
            nld: The nuclear level density vector to normalize.
            discrete: The discrete level density at low energies to
                      normalize to. Provide either histogram as a Vector,
                      or the path to the discrete level file (in keV).
            path: If set, tries to load vectors from path.
        """
        # Create the private variables
        self._discrete = None
        self._discrete_path = None
        self._D0 = None
        self._use_smoothed_levels = None
        self.bounds = {'A': (1, 100), 'alpha': (1e-1, 20), 'T': (0.1, 1),
                       'D0': (None, None)}
        self.spin = {'spincutModel': None,
                     'spincutPars': None,
                     'J_target': None,
                     'Gg': None,
                     'Sn': None}
        self.model: Optional[Callable[..., ndarray]] = constant_temperature
        self.curried_model = lambda *arg: None
        self.multinest_path = Path('multinest')

        # Handle the method parameters
        self.use_smoothed_levels = True
        self.extractor = copy.deepcopy(extractor)
        self.nld = None
        if extractor is not None:
            nld = extractor.nld[0]
        if nld is not None:
            self.nld = nld.copy()
        self.discrete = discrete

        self.res = ResultsNormalized(name="Results NLD")

        LOG.debug("Created Normalizer")

        self.limit_low = None
        self.limit_high = None
        self.std_fake = None  # See `normalize`

        # Load saved transformed vectors
        if path is not None:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)
            try:
                self.load(self.path)
            except AssertionError:
                pass
        else:
            self.path = Path('normalizations')
            self.path.mkdir(exist_ok=True)

    def __call__(self, *args, **kwargs) -> None:
        """ Wrapper around normalize """
        self.normalize(*args, **kwargs)

    def normalize(self, limit_low: Optional[Tuple[float, float]] = None,
                  limit_high: Optional[Tuple[float, float]] = None, *,
                  nld: Optional[Vector] = None,
                  extractor: Optional[Extractor] = None,
                  discrete: Optional[Vector] = None,
                  D0: Optional[Tuple[float, float]] = None,
                  bounds: Optional[TupleDict] = None,
                  spin: Optional[Dict[str, Any]] = None) -> None:
        """ Normalize NLD to a low and high energy region

        Args:
            limit_low: The limits (start, stop) where to normalize
               to discrete levels.
            limit_high: The limits (start, stop) where to normalize to
                a theoretical model and neutron separation energy at high
                energies.
            extractor: The extractor to use get nld from
            nld: The nuclear level density vector to normalize.
            discrete: The discrete level density at low energies to
                normalize to.
            D0: Average resonance spacing from s waves [eV]
            bounds: The bounds of the parameters
            spin: Parameters for use in the high energy model and
                spin cut model.

        """
        # Update internal state
        self.limit_low = self.reset(limit_low)
        self.limit_high = self.reset(limit_high)
        limit_low = self.limit_low
        limit_high = self.limit_high
        discrete = self.reset(discrete)
        discrete.to_MeV()
        nld = self.reset(nld)
        D0 = self.reset(D0)
        self.D0 = D0  # To ensure the bounds are updated
        bounds = self.reset(bounds)
        spin = self.reset(spin)

        # Ensure that spin is set
        for key, val in spin.items():
            if val is None:
                raise ValueError(f"`{key}` has to be set.")
        extractor = self.reset(extractor, nonable=True)

        # Ensure to rerun if called again
        nlds = []
        self.res = ResultsNormalized(name="Results NLD")

        if extractor is not None:
            nlds = extractor.nld
        else:
            nld = self.nld
            nlds = [nld]

        for i, nld in enumerate(tqdm(nlds)):
            LOG.info(f"\n\n---------\nNormalizing nld #{i}")
            nld = nld.copy()
            LOG.debug("Setting NLD, convert to MeV")
            nld.to_MeV()

            # Need to give some sort of standard deviation for sensible results
            # Otherwise deviations at higher level density will have an
            # uncreasonably high weight.
            if self.std_fake is None:
                self.std_fake = False
            if self.std_fake or nld.std is None:
                self.std_fake = True
                nld.std = nld.values * 0.1  # 10% is an arb. choice

            self.nld = nld
            # Use DE to get an inital guess before optimizing
            args, guess = self.initial_guess(limit_low, limit_high)
            # Optimize using multinest
            popt, samples = self.optimize(args, guess)

            transformed = nld.transform(popt['A'][0], popt['alpha'][0],
                                        inplace=False)
            if self.std_fake:
                transformed.std = None

            self.res.nld.append(transformed)
            self.res.pars.append(popt)
            self.res.samples.append(samples)
            ext_model = lambda E: self.curried_model(E, T=popt['T'][0],
                                                     D0=popt['D0'][0])
            self.res.nld_model.append(ext_model)

        self.save()

    def initial_guess(self, limit_low: Optional[Tuple[float, float]] = None,
                      limit_high: Optional[Tuple[float, float]] = None
                      ) -> Tuple[Tuple[float, float, float, float],
                                 Dict[str, float]]:
        """ Find an inital guess for the constant, α, T and D₀

        Uses differential evolution to perform the guessing.

        Args:
            limit_low: The limits (start, stop) where to normalize
               to discrete levels.
            limit_high: The limits (start, stop) where to normalize to
                a theoretical model and neutron separation energy at high
                energies.

        Returns:
           The arguments used for chi^2 minimization and the
           minimizer.
        """
        limit_low = self.reset(limit_low)
        limit_high = self.reset(limit_high)

        bounds = list(self.bounds.values())
        spinstring = json.dumps(self.spin, indent=4, sort_keys=True)

        LOG.debug("Using bounds %s", bounds)
        LOG.debug("Using spin %s", spinstring)

        nld_low = self.nld.cut(*limit_low, inplace=False)
        discrete = self.discrete.cut(*limit_low, inplace=False)
        nld_high = self.nld.cut(*limit_high, inplace=False)

        # We don't want to send unecessary parameters to the minimizer
        model = lambda *args, **kwargs: self.model(*args, **kwargs, spin=self.spin)
        self.curried_model = model
        args = (nld_low, nld_high, discrete, model)
        res = differential_evolution(self.errfn, bounds=bounds, args=args)

        LOG.info("DE results:\n%s", tt.to_string([res.x.tolist()],
            header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]']))

        p0 = dict(zip(["A", "alpha", "T", "D0"], (res.x).T))
        # overwrite result for D0, as we have a "correct" prior for it
        p0["D0"] = self.D0

        return args, p0

    def optimize(self, args,
                 guess: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Find parameters given model constraints and an initial guess

        Employs Multinest

        Args:
            args: The arguments used for finding constraining the modelspace.
            guess: The initial guess of the parameters
        Returns:
            The parameters and the samples found during optimization.
        """
        if guess['alpha'] < 0:
            raise ValueError("Prior selection not implemented for α < 0")
        alpha_exponent = np.log10(guess['alpha'])

        if guess['T'] < 0:
            raise ValueError("Prior selection not implemented for T < 0")
        T_exponent = np.log10(guess['T'])

        A = guess['A']
        D0 = guess['D0']

        # truncations from absolute values
        lower, upper = 0., np.inf
        mu_A, sigma_A = A, 4*A
        a_A = (lower - mu_A) / sigma_A
        mu_D0, sigma_D0 = D0[0], D0[1]
        a_D0 = (lower - mu_D0) / sigma_D0

        def prior(cube, ndim, nparams):
            # NOTE: You may want to adjust this for your case!
            # normal prior
            cube[0] = truncnorm.ppf(cube[0], a_A, upper, loc=mu_A,
                                    scale=sigma_A)
            # log-uniform prior
            # if alpha = 1e2, it's between 1e1 and 1e3
            cube[1] = 10**(cube[1]*2 + (alpha_exponent-1))
            # log-uniform prior
            # if T = 1e2, it's between 1e1 and 1e3
            cube[2] = 10**(cube[2]*2 + (T_exponent-1))
            # normal prior
            cube[3] = truncnorm.ppf(cube[3], a_D0, upper, loc=mu_D0,
                                    scale=sigma_D0)
            if np.isinf(cube[3]):
                LOG.debug("Encountered inf in cube[3]:\n%s", cube[3])

        def loglike(cube, ndim, nparams):
            chi2 = self.errfn(cube, *args)
            loglikelihood = -0.5 * chi2
            return loglikelihood

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / "nld_norm_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        LOG.info("Starting multinest")
        #  Hack where stdout from Multinest is redirected as info messages
        LOG.write = lambda msg: LOG.info(msg) if msg != '\n' else None
        with redirect_stdout(LOG):
            pymultinest.run(loglike, prior, len(guess),
                            outputfiles_basename=str(path),
                            resume=False, verbose=True)

        # Save parameters for analyzer
        names = list(guess.keys())
        json.dump(names, open(str(path) + 'params.json', 'w'))
        analyzer = pymultinest.Analyzer(len(guess),
                                        outputfiles_basename=str(path))

        stats = analyzer.get_stats()

        samples = analyzer.get_equal_weighted_posterior()[:, :-1]
        samples = dict(zip(names, samples.T))

        # Format the output
        popt = dict()
        vals = []
        for name, m in zip(names, stats['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2
            popt[name] = (med, sigma)
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            vals.append(fmts % (med, sigma))

        LOG.info("Multinest results:\n%s", tt.to_string([vals],
                 header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]']))

        return popt, samples

    def plot(self, ax: Any = None) -> Tuple[Any, Any]:
        """ Plot the NLD, discrete levels and result of normalization

        Args:
            ax: The matplotlib axis to plot onto
        Returns:
            The figure and axis created if no ax is supplied.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        for i, transformed in enumerate(self.res.nld):
            label = None
            if i == 0:
                label = "normalized"
            transformed.plot(ax=ax, c='k', alpha=1/len(self.res.nld),
                             label=label)

        self.discrete.plot(ax=ax, label='Discrete levels')

        for i, pars in enumerate(self.res.pars):
            Sn = self.curried_model(T=pars['T'][0],
                                    D0=pars['D0'][0],
                                    E=self.spin['Sn'])

            x = np.linspace(self.limit_high[0], self.spin['Sn'])
            model = self.curried_model(T=pars['T'][0],
                                       D0=pars['D0'][0],
                                       E=x)
            # TODO: plot errorbar
            labelSn = None
            labelModel = None
            if i == 0:
                labelSn = '$S_n$'
                labelModel = 'model'
            ax.scatter(self.spin['Sn'], Sn, label=labelSn)
            ax.plot(x, model, "--", label=labelModel)

        ax.axvspan(self.limit_low[0], self.limit_low[1], color='grey',
                   alpha=0.1, label="fit limits")
        ax.axvspan(self.limit_high[0], self.limit_high[1], color='grey',
                   alpha=0.1)

        ax.set_yscale('log')
        ax.set_ylabel(r"$\rho(E_x) \quad [\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"$E_x \quad [\mathrm{MeV}]$")


        if fig is not None:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def reset(self, variable: Any, nonable: bool = False) -> Any:
        """ Ensures `variable` is not None

        Args:
            variable: The variable to check
            nonable: Does not raise ValueError if
                variable is None.
        Returns:
            The value of variable or self.variable
        Raises:
            ValueError if both variable and
            self.variable are None.
        """
        name = _retrieve_name(variable)
        if variable is None:
            self_variable = getattr(self, name)
            if not nonable and self_variable is None:
                raise ValueError(f"`{name}` must be set")
            return self_variable
        return variable

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """ Load already extracted nld and gsf from file

        Args:
            path: The path to the directory containing the
                files.
        """
        if path is not None:
            path = Path(path)
        else:
            path = Path(self.path)  # TODO: Fix pathing

        if not path.exists():
            raise IOError(f"The path {path} does not exist.")
        if self.res.nld:
            warnings.warn("Loading nld and gsf into non-empty instance")

        LOG.debug("Loading from %s", str(path))

        for fname in path.glob("nld_[0-9]*.npy"):
            self.res.nld.append(Vector(path=fname))

        if not self.res.nld:
            warnings.warn("Found no files")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        if path is not None:
            path = Path(path)
        else:
            path = Path(self.path)  # TODO: Fix pathing

        LOG.debug("Saving to %s", str(path))

        path.mkdir(exist_ok=True)
        for i, nld in enumerate(self.res.nld):
            nld.save(path / f"nld_{i}.npy")

    @staticmethod
    def errfn(x: Tuple[float, float, float, float], nld_low: Vector,
              nld_high: Vector, discrete: Vector,
              model: Callable[..., ndarray]) -> float:
        """ Compute the χ² of the normalization fitting

        Args:
            x: The arguments ordered as A, alpha, T and D0
            nld_low: The lower region where discrete levels will be
                fitted.
            nld_high: The upper region to fit to model.
            discrete: The discrete levels to be used in fitting the
                lower region.
            model: The model to use when fitting the upper region.
                Must support the keyword arguments:
                    model(T=..., D0=..., E=...) -> ndarray
        Returns:
            chi2 (float): The χ² value
        """
        A, alpha, T, D0 = x[:4]
        transformed_low = nld_low.transform(A, alpha, inplace=False)
        transformed_high = nld_high.transform(A, alpha, inplace=False)

        err_low = transformed_low.error(discrete)
        expected = model(T=T, D0=D0, E=transformed_high.E)
        err_high = transformed_high.error(expected)
        return err_low + err_high

    @property
    def discrete(self) -> Optional[Vector]:
        return self._discrete

    @discrete.setter
    def discrete(self, value: Optional[Union[Path, str, Vector]]) -> None:
        if value is None:
            self._discretes = None
            LOG.debug("Set `discrete` to None")
        elif isinstance(value, (str, Path)):
            if self.nld is None:
                raise ValueError(f"`nld` must be set before loading levels")
            nld = self.nld.copy()
            nld.to_MeV()
            if self.use_smoothed_levels:
                self._discrete = load_levels_smooth(value, nld.E)
                LOG.debug("Set `discrete` by loading smooth")
            else:
                self._discrete = load_levels_discrete(value, nld.E)
                LOG.debug("Set `discrete` by loading discrete")
            self._discrete.units = "MeV"
            self._discrete_path = value

        elif isinstance(value, Vector):
            if self.nld is not None and np.any(self.nld.E != value.E):
                raise ValueError("`nld` and `discrete` must"
                                 " have same energy binning")
            self._discrete = value
            LOG.debug("Set `discrete` by Vector")
        else:
            raise ValueError(f"Value {value} is not supported"
                             " for discrete levels")

    @property
    def D0(self) -> Optional[Tuple[float, float]]:
        return self._D0

    @D0.setter
    def D0(self, value: Tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError("D0 must contain (mean, std) [eV] of A-1 nucleus")
        self._D0 = value[0], value[1]
        self.bounds['D0'] = (0.99*value[0], 1.01*value[0])

    @property
    def use_smoothed_levels(self) -> Optional[bool]:
        return self._use_smoothed_levels

    @use_smoothed_levels.setter
    def use_smoothed_levels(self, value: bool) -> None:
        self._use_smoothed_levels = value
        if self._discrete_path is not None:
            self.discrete = self._discrete_path


def load_levels_discrete(path: Union[str, Path], energy: ndarray) -> Vector:
    """ Load discrete levels without smoothing

    Assumes linear equdistant binning

    Args:
        path: The file to load
        energy: The binning to use
    Returns:
        A vector describing the levels
    """
    histogram, _ = load_discrete(path, energy, 0.1)
    return Vector(values=histogram, E=energy)


def load_levels_smooth(path: Union[str, Path], energy: ndarray,
                       resolution: float = 0.1) -> Vector:
    """ Load discrete levels with smoothing

    Assumes linear equdistant binning

    Args:
        path: The file to load
        energy: The binning to use
        resolution: The resolution of the smoothing to use
    Returns:
        A vector describing the smoothed levels
    """
    _, smoothed = load_discrete(path, energy, resolution)
    return Vector(values=smoothed, E=energy)

#################################
# Ugly code below, to be fixed! #
#################################

def constant_temperature(E: ndarray, T: float, D0: float, spin: Dict[str, Any]) -> ndarray:
    Sn = Sn_from_D0(D0, **spin)
    shift = Eshift_from_T(T, Sn)
    return CT(E, T, shift)


def CT(E: ndarray, T: float, Eshift: float) -> ndarray:
    """ Constant Temperature NLD"""
    ct = np.exp((E - Eshift) / T) / T
    return ct


def Eshift_from_T(T, Sn):
    """ Eshift from T for CT formula """
    return Sn[0] - T * np.log(Sn[1] * T)


def Sn_from_D0(D0, Sn, J_target,
               spincutModel=None, spincutPars={},
               **kwargs):
    """ Calculate nld(Sn) from D0

    Parameters:
    -----------
    D0 : (float)
        Average resonance spacing from s waves [eV]
    Sn : (float)
        Separation energy [MeV]
    J_target : (float)
        Target spin
    spincutModel : string
        Model to for the spincut
    spincutPars : dict
        Additional parameters necessary for the spin cut model

    Returns:
    --------
    nld : [float, float]
        Sn, nld at Sn [MeV, 1/MeV]
    """

    def g(J):
        return SpinFunctions(Ex=Sn, J=J,
                             model=spincutModel,
                             pars=spincutPars).distibution()

    if J_target == 0:
        summe = g(J_target + 1 / 2)
    else:
        summe = 1 / 2 * (g(J_target - 1 / 2) + g(J_target + 1 / 2))

    nld = 1 / (summe * D0 * 1e-6)
    return [Sn, nld]


def _retrieve_name(var: Any) -> str:
    """ Finds the source-code name of `var`

        NOTE: Only call from self.reset.

     Args:
        var: The variable to retrieve the name of.
    Returns:
        The variable's name.
    """
    # Retrieve the line of the source code of the third frame.
    # The 0th frame is the current function, the 1st frame is the
    # calling function and the second is the calling function's caller.
    line = inspect.stack()[2].code_context[0].strip()
    match = re.search(r".*\((\w+).*\).*", line)
    assert match is not None, "Retrieving of name failed"
    name = match.group(1)
    return name
