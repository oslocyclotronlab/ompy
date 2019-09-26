import numpy as np
import logging
import inspect
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from pathlib import Path
from numpy import ndarray
from scipy.optimize import differential_evolution
from typing import Optional, Sequence, Tuple, Any, Union, Callable, Dict
from scipy.stats import halfnorm
from .extractor import Extractor
from .vector import Vector
from .multinest_setup import run_nld_2regions
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete


LOG = logging.getLogger(__name__)

TupleDict = Dict[str, Tuple[float, float]]


class Normalizer:
    def __init__(self, *, extractor: Optional[Extractor] = None,
                 nld: Optional[Vector] = None,
                 discrete: Optional[Vector] = None) -> None:
        """ Inits the class

        The prefered syntax is Normalizer(extractor=...) or Normalizer(nld=...)
        If neither is given, the nld must be explicity be set later by:
            normalizer.nld = nld
            normalizer.extractor = extractor
        or:
            normalizer.normalize(..., nld=nld)
            normalizer.normalize(..., extractor=extractor)

        Args:
            extractor: The extractor to use get nld from
            nld: The nuclear level density vector to normalize.
            discrete: The discrete level density at low energies to
                normalize to.
        """
        # Create the private variables
        self._discrete = None
        self._nld = None
        self._D0 = None
        self.bounds = {'A': (1, 100), 'alpha': (1e-2, 5e-3), 'T': (0.1, 1),
                       'D0': (None, None)}
        self.spin = {'spincutModel': 'Disc_and_EB05',
                     'spincutPars': None,
                     'J_target': None,
                     'Gg': None,
                     'Sn': None}
        self.model: Optional[Callable[..., ndarray]] = constant_temperature
        self.multinest_path = Path('multinest')

        # Handle the method parameters
        self.use_smoothed_levels = True
        self.extractor = extractor
        self.nld = nld
        self.discrete = discrete

        LOG.debug("Created Normalizer")

    def __call__(self, *args, **kwargs) -> None:
        """ Wrapper around normalize """
        self.normalize(*args, **kwargs)

    def normalize(self, limit_low: Tuple[float, float],
                  limit_high: Tuple[float, float], *,
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
        nld = self.reset(nld)
        discrete = self.reset(discrete)
        D0 = self.reset(D0)
        self.D0 = D0  # To ensure the bounds are updated
        bounds = self.reset(bounds)
        spin = self.reset(spin)

        # Use DE to get an inital guess before optimizing
        args, guess = self.initial_guess(limit_low, limit_high)
        # Optimize using multinest
        popt, samples = self.optimize(args, guess)

        self.nld_parameters = popt
        self.samples = samples

        return popt, samples

    def initial_guess(self, limit_low: Tuple[float, float],
                      limit_high: Tuple[float, float]
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
        bounds = list(self.bounds.values())
        spinstring = json.dumps(self.spin, indent=4, sort_keys=True)

        LOG.debug("Using bounds %s", bounds)
        LOG.debug("Using spin %s", spinstring)

        nld_low = self.nld.cut(*limit_low, inplace=False)
        discrete = self.discrete.cut(*limit_low, inplace=False)
        nld_high = self.nld.cut(*limit_high, inplace=False)

        # We don't want to send unecessary parameters to the minimizer
        model = lambda *args, **kwargs: self.model(*args, **kwargs, spin=self.spin)
        args = (nld_low, nld_high, discrete, model)
        res = differential_evolution(errfn, bounds=bounds, args=args)

        LOG.info("DE results:\n%s", tt.to_string([res.x.tolist()],
            header=['constant', 'α', 'T', 'D₀']))

        p0 = dict(zip(["A", "alpha", "T", "D0"], (res.x).T))
        # overwrite result for D0, as we have a "correct" prior for it
        p0["D0"] = self.D0

        return args, p0

    def optimize(self, args: Tuple[float, float, float, float],
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

        D0 = guess['D0']
        A = guess['A']

        def prior(cube, ndim, nparams):
            # NOTE: You may want to adjust this for your case!
            # normal prior
            cube[0] = halfnorm.ppf(cube[0], loc=A, scale=4*A)
            # log-uniform prior
            # if alpha = 1e2, it's between 1e1 and 1e3
            cube[1] = 10**(cube[1]*2 + (alpha_exponent-1))
            # log-uniform prior
            # if T = 1e2, it's between 1e1 and 1e3
            cube[2] = 10**(cube[2]*2 + (T_exponent-1))
            # normal prior
            cube[3] = halfnorm.ppf(cube[3], loc=D0[0], scale=D0[1])
            if np.isinf(cube[3]):
                LOG.debug("Encountered inf in cube[3]:\n%s", cube[3])

        def loglike(cube, ndim, nparams):
            chi2 = errfn(cube, *args)
            loglikelihood = -0.5 * chi2
            return loglikelihood

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / "nld_norm"
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
            header=['constant', 'α', 'T', 'D₀']))

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
            fig = None

        self.nld.plot(ax=ax, label='NLD')
        transformed = self.nld.transform(self.nld_parameters['A'][0],
                                         self.nld_parameters['alpha'][0],
                                         inplace=False)
        transformed.plot(ax=ax, label='Transformed')
        self.discrete.plot(ax=ax, label='Discrete levels')
        ax.set_yscale('log')
        if fig is not None:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def reset(self, variable: Any) -> Any:
        """ Ensures `variable` is not None

        Args:
            variable: The variable to check
        Returns:
            The value of variable or self.variable
        Raises:
            ValueError if both variable and
            self.variable are None.
        """
        name = _retrieve_name(variable)
        if variable is None:
            self_variable = getattr(self, name)
            if self_variable is None:
                raise ValueError(f"`{name}` must be set")
            return self_variable
        return variable

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
            if self.use_smoothed_levels:
                self._discrete = load_levels_smooth(value, self.nld.E)
                LOG.debug("Set `discrete` by loading smooth")
            else:
                self._discrete = load_levels_discrete(value, self.nld.E)
                LOG.debug("Set `discrete` by loading discrete")

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
        The χ² value
    """
    A, alpha, T, D0 = x[:4]
    transformed_low = nld_low.transform(A, alpha, inplace=False)
    transformed_high = nld_high.transform(A, alpha, inplace=False)

    err_low = transformed_low.error(discrete)
    expected = model(T=T, D0=D0, E=transformed_high.E)
    err_high = transformed_high.error(expected)
    return err_low + err_high


#################################
# Ugly code below, to be fixed! #
#################################

def constant_temperature(E: ndarray, T: float, D0: float, spin: Dict[str, Any]) -> ndarray:
    Sn = Sn_from_D0(D0, **spin)
    shift = Eshift_from_T(T, Sn)
    return CT(E, T, shift)


def CT(E, T, Eshift):
    """ Constant Temperature NLD"""
    return np.exp((E - Eshift) / T) / T


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
    # The name to use is the name given in self.reset(...)
    name = line.split('(')[1].split(')')[0].strip()
    return name
