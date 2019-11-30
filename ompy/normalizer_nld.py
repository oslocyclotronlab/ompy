import numpy as np
import copy
import logging
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from numpy import ndarray
from scipy.optimize import differential_evolution
from typing import Optional, Tuple, Any, Union, Callable, Dict
from scipy.stats import truncnorm
from .vector import Vector
from .library import self_if_none
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized, NormalizationParameters


LOG = logging.getLogger(__name__)

TupleDict = Dict[str, Tuple[float, float]]


class NormalizerNLD:
    """ Normalizes NLD to empirical data

    Normalizes nld/gsf according to::

        nld' = nld * A * np.exp(alpha * Ex), and

    This is the transformation eq (3), Schiller2000

    Takes empirical data in form of an array of discrete levels,
    neutron separation energy Sn, a model to estimate what the
    NLD is at Sn, and several parameters for the model as well
    as bounds on normalization parameters.

    As a consequence of a complex problem, this class has a complex
    interface. Much of the book-keeping associated with normalization
    has been automated, but there is still a lot of settings and
    parameters for the user to take care of. Some default values
    has been seen, but the user must be _EXTREMELY_ careful when
    evaluating the output.

    Attributes:
        discrete (Vector): The discrete NLD at lower energies. [MeV]
        nld (Vector): The NLD to normalize. Gets converted to [MeV] from [keV].
        norm_pars (NormalizationParameters): Normalization parameters like
            experimental D₀, and spin(-cut) model
        bounds (Dict[str, Tuple[float, float]): The bounds on each of
            the parameters. Its keys are 'A', 'alpha', 'T', and 'D0'. The
            values are on the form (min, max).
        model (Callable[..., ndarray]): The model to use at high energies
            to estimate the NLD. Defaults to constant temperature model.
        curried_model (Callable[..., ndarray]): Same as model, but
            curried with the spin distribution parameters to reduce
            book keeping. Is set by initial_guess(), _NOT_ the user.
        multinest_path (Path): Where to save the multinest output.
            defaults to 'multinest'.
        multinest_kwargs (dict): Additional keywords to multinest. Defaults to
            `{"seed": 65498, "resume": False}`
        res (ResultsNormalized): Results of the normalization
        smooth_levels_fwhm (float): FWHM with which the discrete levels shall
            be smoothed when loading from file. Defaults to 0.1 MeV.
        path (Path): The path the transformed vectors to.

    """
    def __init__(self, *,
                 nld: Optional[Vector] = None,
                 discrete: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = None,
                 norm_pars: Optional[NormalizationParameters] = None) -> None:
        """ Normalizes nld ang gSF.

        Note:
            The prefered syntax is `Normalizer(nld=...)`
            If neither is given, the nld (and other parameters) can be
            explicity
            be set later by::

                `normalizer.normalize(..., nld=...)`

            or::

                `normalizer.nld = ...`

            In the later case you *might* have to send in a copy if it's a
            mutable to ensure it is not changed.

        Args:
            extractor: see above
            nld: see above
            discrete: see above
            path: see above
            norm_pars: see above
        TODO:
            - parameter to limit the number of multinest samples to store. Note
              that the samples should be shuffled to retain some "random"
              samples from the pdf (not the importance weighted)

        """
        # Create the private variables
        self._discrete = None
        self._discrete_path = None
        self._D0 = None
        self._smooth_levels_fwhm = None
        self.norm_pars = norm_pars
        self.bounds = {'A': (1, 100), 'alpha': (1e-1, 20), 'T': (0.1, 1),
                       'D0': (None, None)}  # D0 bounds set later
        self.model: Optional[Callable[..., ndarray]] = constant_temperature
        self.curried_model = lambda *arg: None
        self.multinest_path = Path('multinest')
        self.multinest_kwargs: dict = {"seed": 65498, "resume": False}

        # Handle the method parameters
        self.smooth_levels_fwhm = 0.1
        self.nld = None if nld is None else nld.copy()
        self.discrete = discrete

        self.res = ResultsNormalized(name="Results NLD")

        self.limit_low = None
        self.limit_high = None
        self.std_fake = None  # See `normalize`

        # Load saved transformed vectors
        if path is not None:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True)
            self.load(self.path)
        else:
            self.path = Path('normalizations')
            self.path.mkdir(exist_ok=True)

    def __call__(self, *args, **kwargs) -> None:
        """ Wrapper around normalize """
        self.normalize(*args, **kwargs)

    def normalize(self, *, limit_low: Optional[Tuple[float, float]] = None,
                  limit_high: Optional[Tuple[float, float]] = None,
                  nld: Optional[Vector] = None,
                  discrete: Optional[Vector] = None,
                  bounds: Optional[TupleDict] = None,
                  norm_pars: Optional[NormalizationParameters] = None,
                  num: int = 0) -> None:
        """ Normalize NLD to a low and high energy region

        Args:
            limit_low: The limits (start, stop) where to normalize
               to discrete levels.
            limit_high: The limits (start, stop) where to normalize to
                a theoretical model and neutron separation energy at high
                energies.
            nld: The nuclear level density vector to normalize.
            discrete: The discrete level density at low energies to
                normalize to.
            bounds: The bounds of the parameters
            norm_pars (NormalizationParameters): Normalization parameters like
            experimental D₀, and spin(-cut) model
            num (optional): Loop number, defauts to 0

        """
        # Update internal state
        self.limit_low = self.self_if_none(limit_low)
        self.limit_high = self.self_if_none(limit_high)
        limit_low = self.limit_low
        limit_high = self.limit_high

        discrete = self.self_if_none(discrete)
        discrete.to_MeV()
        nld = self.self_if_none(nld)

        self.norm_pars = self.self_if_none(norm_pars)
        self.norm_pars.is_changed(include=["D0", "Sn", "spincutModel",
                                           "spincutPars"])  # check that set

        self.bounds = self.self_if_none(bounds)
        self.bounds['D0'] = (0.99*self.norm_pars.D0[0],
                             1.01*self.norm_pars.D0[0])

        # ensure that it's updated if running again
        self.res = ResultsNormalized(name="Results NLD")

        LOG.info(f"\n\n---------\nNormalizing nld #{num}")
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
            nld.std = None
            transformed.std = None

        self.res.nld = transformed
        self.res.pars = popt
        self.res.samples = samples
        ext_model = lambda E: self.curried_model(E, T=popt['T'][0],
                                                 D0=popt['D0'][0])
        self.res.nld_model = ext_model

        self.save(num=num)

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
        limit_low = self.self_if_none(limit_low)
        limit_high = self.self_if_none(limit_high)

        bounds = list(self.bounds.values())
        spinParsstring = json.dumps(self.norm_pars.spincutPars, indent=4,
                                    sort_keys=True)

        LOG.debug("Using bounds %s", bounds)
        LOG.debug("Using spincutModel %s", self.norm_pars.spincutModel)
        LOG.debug("Using spincutPars %s", spinParsstring)

        nld_low = self.nld.cut(*limit_low, inplace=False)
        discrete = self.discrete.cut(*limit_low, inplace=False)
        nld_high = self.nld.cut(*limit_high, inplace=False)

        # We don't want to send unecessary parameters to the minimizer
        def model(*args, **kwargs) -> ndarray:
            model = self.model(*args, **kwargs,
                               Jtarget=self.norm_pars.Jtarget,
                               Sn=self.norm_pars.Sn[0],
                               spincutModel=self.norm_pars.spincutModel,
                               spincutPars=self.norm_pars.spincutPars)
            return model

        self.curried_model = model
        args = (nld_low, nld_high, discrete, model)
        res = differential_evolution(self.errfn, bounds=bounds, args=args)

        LOG.info("DE results:\n%s", tt.to_string([res.x.tolist()],
                 header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]']))

        p0 = dict(zip(["A", "alpha", "T", "D0"], (res.x).T))
        # overwrite result for D0, as we have a "correct" prior for it
        p0["D0"] = self.norm_pars.D0

        return args, p0

    def optimize(self, args,
                 guess: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Find parameters given model constraints and an initial guess

        Employs Multinest

        Args:
            num (int): Loop number
            args_nld (Iterable): Additional arguments for the nld errfn
            guess (Dict[str, float]): The initial guess of the parameters

        Returns:
            Tuple:
            - popt (Dict[str, Tuple[float, float]]): Median and 1sigma of the
                parameters
            - samples (Dict[str, List[float]]): Multinest samplesø.
                Note: They are still importance weighted, not random draws
                from the posterior.

        Raises:
            ValueError: Invalid parameters for automatix prior
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
        LOG.debug("with following keywords %s:", self.multinest_kwargs)
        #  Hack where stdout from Multinest is redirected as info messages
        LOG.write = lambda msg: LOG.info(msg) if msg != '\n' else None
        with redirect_stdout(LOG):
            pymultinest.run(loglike, prior, len(guess),
                            outputfiles_basename=str(path),
                            **self.multinest_kwargs)

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

    def plot(self, *, ax: Any = None,
             add_label: bool = True,
             results: Optional[ResultsNormalized] = None,
             add_figlegend: bool = True,
             plot_fitregion: bool = True,
             reset_color_cycle: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """Plot the NLD, discrete levels and result of normalization

        Args:
            ax (optional): The matplotlib axis to plot onto. Creates axis
                is not provided
            add_label (bool, optional): Defaults to `True`.
            add_figlegend (bool, optional):Defaults to `True`.
            results (ResultsNormalized, optional): If provided, nld and model
                are taken from here instead.
            plot_fitregion (Optional[bool], optional): Defaults to `True`.
            reset_color_cycle (Optional[bool], optional): Defaults to `True`
            **kwargs: Description

        Returns:
            fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if reset_color_cycle:
            ax.set_prop_cycle(None)

        res = self.res if results is None else results
        pars = res.pars
        nld = res.nld

        labelNld = None
        labelSn = None
        labelModel = None
        if add_label:
            labelNld = 'normalized'
            labelNldSn = r'$\rho(S_n)$'
            labelModel = 'model'
        nld.plot(ax=ax, label=labelNld, **kwargs)

        if add_label:
            self.discrete.plot(ax=ax, c='k', label='Discrete levels')

        nld_Sn = self.curried_model(T=pars['T'][0],
                                    D0=pars['D0'][0],
                                    E=self.norm_pars.Sn[0])

        x = np.linspace(self.limit_high[0], self.norm_pars.Sn[0])
        model = self.curried_model(T=pars['T'][0],
                                   D0=pars['D0'][0],
                                   E=x)
        # TODO: plot errorbar

        if add_label:
            ax.scatter(self.norm_pars.Sn[0], nld_Sn, label=labelNldSn)
        ax.plot(x, model, "--", label=labelModel,
                c='g', **kwargs)

        if plot_fitregion:
            ax.axvspan(self.limit_low[0], self.limit_low[1], color='grey',
                       alpha=0.1, label="fit limits")
            ax.axvspan(self.limit_high[0], self.limit_high[1], color='grey',
                       alpha=0.1)

        ax.set_yscale('log')
        ax.set_ylabel(r"$\rho(E_x) \quad [\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"$E_x \quad [\mathrm{MeV}]$")
        ax.set_ylim(bottom=0.5/(nld.E[1]-nld.E[0]))

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """ Load already normalized NLD from `path`.

        Args:
            path: The path to the directory containing the
                files.
        TODO:
            - Save/LOAD whole Results class
        """
        raise NotImplementedError()
        if path is not None:
            path = Path(path)
        else:
            path = Path(self.path)  # TODO: Fix pathing

        if not path.exists():
            raise IOError(f"The path {path} does not exist.")
        if self.res.nld:
            warnings.warn("Loading nld and gsf into non-empty instance")

        LOG.debug("Loading from %s", str(path))

        for fname in path.glob("nld_[0-9]+"):
            LOG.debug("Loading %s", fname)
            self.res.load(fname)
            break

        if not self.res.nld:
            warnings.warn("Found no files")

    def save(self, path: Optional[Union[str, Path]] = None,
             num: int = None) -> None:
        """ Save results to `path`.
        TODO:
            - Save/LOAD whole Results class
            - Currently completely broken
        """
        if path is not None:
            path = Path(path)
        else:
            path = Path(self.path)  # TODO: Fix pathing

        LOG.debug("Saving to %s", str(path))

        path.mkdir(exist_ok=True)
        #self.res.save(path / f"nld_{num}")

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
                Must support the keyword arguments
                ``model(T=..., D0=..., E=...) -> ndarray``
        Returns:
            chi2 (float): The χ² value
        """
        A, alpha, T, D0 = x[:4]  # slicing needed for multinest?
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
            LOG.debug("Set `discrete` levels from file with FWHM %s",
                      self.smooth_levels_fwhm)
            self._discrete = load_levels_smooth(value, nld.E,
                                                self.smooth_levels_fwhm)
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
    def smooth_levels_fwhm(self) -> Optional[float]:
        return self._smooth_levels_fwhm

    @smooth_levels_fwhm.setter
    def smooth_levels_fwhm(self, value: float) -> None:
        self._smooth_levels_fwhm = value
        if self._discrete_path is not None:
            self.discrete = self._discrete_path

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)


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
        energy: The binning to use in MeV
        resolution: The resolution (FWHM) of the smoothing to use in MeV
    Returns:
        A vector describing the smoothed levels
    """
    histogram, smoothed = load_discrete(path, energy, resolution)
    return Vector(values=smoothed if resolution > 0 else histogram, E=energy)

#################################
# Ugly code below, to be fixed! #
#################################

def constant_temperature(E: ndarray, *, D0: float,
                         Sn: float, T: float, Jtarget: float,
                         spincutModel: str, spincutPars: Dict[str, Any]) -> ndarray:
    """ CT model "turned around"

    Usually the parameters are the temperature T and a shift Eshift.
    However, you can reparametrise to use T and D0 (from which you
    calculate Eshift)

    Args:
        ... (bla): blub
        D0 (float or Tuple[float, float], optional): If set, this D0 will be
            used instead of the one from spincutPars

    """
    nldSn = nldSn_from_D0(D0=D0,
                          Sn=Sn,
                          Jtarget=Jtarget,
                          spincutModel=spincutModel,
                          spincutPars=spincutPars)
    shift = Eshift_from_T(T, nldSn)
    return CT(E, T, shift)


def CT(E: ndarray, T: float, Eshift: float) -> ndarray:
    """ Constant Temperature NLD"""
    ct = np.exp((E - Eshift) / T) / T
    return ct


def Eshift_from_T(T, nldSn):
    """ Eshift from T for CT formula """
    return nldSn[0] - T * np.log(nldSn[1] * T)


def nldSn_from_D0(D0: float, Sn: float, Jtarget: float,
                  spincutModel: str, spincutPars: Dict[str, Any],
                  **kwargs) -> Tuple[float, float]:
    """Calculate nld(Sn) from D0

    Parameters:
        D0 (float):
            Average resonance spacing from s waves [eV]
        Sn (float):
            Separation energy [MeV]
        Jtarget (float):
            Target spin
        spincutModel (str):
            Model to for the spincut
        spincutPars Dict[str, Any]:
            Additional parameters necessary for the spin cut model
        **kwargs: Description


    Returns:
        nld: Ex=Sn and nld at Sn [MeV, 1/MeV]
    """

    def g(J):
        return SpinFunctions(Ex=Sn, J=J,
                             model=spincutModel,
                             pars=spincutPars).distibution()

    if Jtarget == 0:
        summe = g(Jtarget + 1 / 2)
    else:
        summe = 1 / 2 * (g(Jtarget - 1 / 2) + g(Jtarget + 1 / 2))

    nld = 1 / (summe * D0 * 1e-6)
    return [Sn, nld]
