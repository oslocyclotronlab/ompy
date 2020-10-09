import numpy as np
import copy
import logging
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
from numpy import ndarray
from scipy.optimize import differential_evolution
from typing import Optional, Tuple, Any, Union, Callable, Dict
from pathlib import Path

from scipy.stats import truncnorm
from .vector import Vector
from .library import self_if_none
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized, NormalizationParameters
from .abstract_normalizer import AbstractNormalizer

TupleDict = Dict[str, Tuple[float, float]]


class NormalizerNLD(AbstractNormalizer):
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
        de_kwargs(dict): Additional keywords to differential evolution.
             Defaults to `{"seed": 65424}`. Note that `bounds` has been
             taken out as a separate attribute, but is a keyword of de.
        multinest_path (Path): Where to save the multinest output.
            defaults to 'multinest'.
        multinest_kwargs (dict): Additional keywords to multinest. Defaults to
            `{"seed": 65498, "resume": False}`
        res (ResultsNormalized): Results of the normalization
        smooth_levels_fwhm (float): FWHM with which the discrete levels shall
            be smoothed when loading from file. Defaults to 0.1 MeV.
        path (Path): The path save the results.

    """
    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, *,
                 nld: Optional[Vector] = None,
                 discrete: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False,
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
        super().__init__(regenerate)

        # Create the private variables
        self._discrete = None
        self._discrete_path = None
        self._D0 = None
        self._smooth_levels_fwhm = None
        self.norm_pars = norm_pars
        self.bounds = {'A': [0.1, 1e3], 'alpha': [1e-1, 20], 'T': [0.1, 1],
                       'Eshift': [-5, 5]}  # D0 bounds set later
        self.model: Optional[Callable[..., ndarray]] = self.const_temperature
        # self.curried_model = lambda *arg: None
        self.de_kwargs = {"seed": 65424}
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

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

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
            regenerate: Whether to use already generated files (False) or
                generate them all anew (True).

        """
        if not self.regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

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

        # ensure that it's updated if running again
        self.res = ResultsNormalized(name="Results NLD")

        self.LOG.info(f"\n\n---------\nNormalizing nld #{num}")
        nld = nld.copy()
        self.LOG.debug("Setting NLD, convert to MeV")
        nld.to_MeV()

        # Need to give some sort of standard deviation for sensible results
        # Otherwise deviations at higher level density will have an
        # uncreasonably high weight.
        if self.std_fake is None:
            self.std_fake = False
        if self.std_fake or nld.std is None:
            self.std_fake = True
            nld.std = nld.values * 0.3  # x% is an arb. choice
        self.nld = nld

        # Use DE to get an inital guess before optimizing
        args, guess = self.initial_guess(limit_low, limit_high)
        # Optimize using multinest
        popt, samples = self.optimize(num, args, guess)

        transformed = nld.transform(popt['A'][0], popt['alpha'][0],
                                    inplace=False)
        if self.std_fake:
            nld.std = None
            transformed.std = None

        self.res.nld = transformed
        self.res.pars = popt
        self.res.samples = samples
        ext_model = lambda E: self.model(E, T=popt['T'][0],
                                         Eshift=popt['Eshift'][0])
        self.res.nld_model = ext_model

        self.save()  # save instance

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

        self.LOG.debug("Using bounds %s", bounds)
        self.LOG.debug("Using spincutModel %s", self.norm_pars.spincutModel)
        self.LOG.debug("Using spincutPars %s", spinParsstring)

        nld_low = self.nld.cut(*limit_low, inplace=False)
        discrete = self.discrete.cut(*limit_low, inplace=False)
        nld_high = self.nld.cut(*limit_high, inplace=False)

        nldSn = self.nldSn_from_D0(**self.norm_pars.asdict())[1]
        rel_uncertainty = self.norm_pars.D0[1]/self.norm_pars.D0[0]
        nldSn = np.array([nldSn, nldSn * rel_uncertainty])

        def neglnlike(*args, **kwargs):
            return - self.lnlike(*args, **kwargs)
        args = (nld_low, nld_high, discrete, self.model, self.norm_pars.Sn[0],
                nldSn)
        res = differential_evolution(neglnlike, bounds=bounds, args=args,
                                     **self.de_kwargs)

        self.LOG.info("DE results:\n%s", tt.to_string([res.x.tolist()],
                      header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'Eshift [MeV]']))

        p0 = dict(zip(["A", "alpha", "T", "Eshift"], (res.x).T))
        for key, res in p0.items():
            if res in self.bounds[key]:
                self.LOG.warning(f"DE result for {key} is at edge its bound:"
                                 f"{self.bounds[key]}. This will probably lead"
                                 f"to wrong estimations in multinest, too.")

        return args, p0

    def optimize(self, num: int, args,
                 guess: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Find parameters given model constraints and an initial guess

        Employs Multinest

        Args:
            num (int): Loop number
            args_nld (Iterable): Additional arguments for the nld lnlike
            guess (Dict[str, float]): The initial guess of the parameters

        Returns:
            Tuple:
            - popt (Dict[str, Tuple[float, float]]): Median and 1sigma of the
                parameters
            - samples (Dict[str, List[float]]): Multinest samples.
                Note: They are still importance weighted, not random draws
                from the posterior.

        Raises:
            ValueError: Invalid parameters for automatic prior

        Note:
            You might want to adjust the priors for your specific case! Here
            we just propose a general solution that might often work out of
            the box.
        """
        if guess['alpha'] < 0:
            raise NotImplementedError("Prior selection not implemented for "
                                      "α < 0")
        alpha_exponent = np.log10(guess['alpha'])

        if guess['T'] < 0:
            raise ValueError("Prior selection not implemented for T < 0; "
                             "negative temperature is unphysical")
        T_exponent = np.log10(guess['T'])

        A = guess['A']

        # truncations from absolute values
        lower_A, upper_A = 0., np.inf
        mu_A, sigma_A = A, 10*A
        a_A = (lower_A - mu_A) / sigma_A
        b_A = (upper_A - mu_A) / sigma_A

        lower_Eshift, upper_Eshift = -5., 5
        mu_Eshift, sigma_Eshift = 0, 5
        a_Eshift = (lower_Eshift - mu_Eshift) / sigma_Eshift
        b_Eshift = (upper_Eshift - mu_Eshift) / sigma_Eshift

        def prior(cube, ndim, nparams):
            # NOTE: You may want to adjust this for your case!
            # truncated normal prior
            cube[0] = truncnorm.ppf(cube[0], a_A, b_A, loc=mu_A,
                                    scale=sigma_A)
            # log-uniform prior
            # if alpha = 1e2, it's between 1e1 and 1e3
            cube[1] = 10**(cube[1]*2 + (alpha_exponent-1))
            # log-uniform prior
            # if T = 1e2, it's between 1e1 and 1e3
            cube[2] = 10**(cube[2]*2 + (T_exponent-1))
            # truncated normal prior
            cube[3] = truncnorm.ppf(cube[3], a_Eshift, b_Eshift,
                                    loc=mu_Eshift,
                                    scale=sigma_Eshift)

            if np.isinf(cube[3]):
                self.LOG.debug("Encountered inf in cube[3]:\n%s", cube[3])

        def loglike(cube, ndim, nparams):
            return self.lnlike(cube, *args)

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"nld_norm_{num}_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        self.LOG.info("Starting multinest")
        self.LOG.debug("with following keywords %s:", self.multinest_kwargs)
        #  Hack where stdout from Multinest is redirected as info messages
        self.LOG.write = lambda msg: (self.LOG.info(msg) if msg != '\n'
                                      else None)
        with redirect_stdout(self.LOG):
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

        self.LOG.info("Multinest results:\n%s", tt.to_string([vals],
                      header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'Eshift [MeV]']))

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

        labelNld = '_exp.'
        labelNldSn = None
        labelModel = "_model"
        labelDiscrete = "_known levels"
        if add_label:
            labelNld = 'exp.'
            labelNldSn = r'$\rho(S_n)$'
            labelModel = 'model'
            labelDiscrete = "known levels"
        nld.plot(ax=ax, label=labelNld, **kwargs)

        self.discrete.plot(ax=ax, kind='step', c='k', label=labelDiscrete)

        nldSn = self.nldSn_from_D0(**self.norm_pars.asdict())[1]
        rel_uncertainty = self.norm_pars.D0[1]/self.norm_pars.D0[0]
        nldSn = np.array([nldSn, nldSn * rel_uncertainty])

        x = np.linspace(self.limit_high[0], self.norm_pars.Sn[0])
        model = Vector(E=x, values=self.model(E=x,
                                              T=pars['T'][0],
                                              Eshift=pars['Eshift'][0]))

        ax.errorbar(self.norm_pars.Sn[0], nldSn[0], yerr=nldSn[1],
                    label=labelNldSn, fmt="ks", markerfacecolor='none')
        # workaround for enseble Normalizer; always keep these label
        for i in range(3):
            ax.lines[-(i+1)]._label = "_nld(Sn)"

        ax.plot(model.E, model.values, "--", label=labelModel, markersize=0,
                c='g', **kwargs)

        if plot_fitregion:
            ax.axvspan(self.limit_low[0], self.limit_low[1], color='grey',
                       alpha=0.1, label="fit limits")
            ax.axvspan(self.limit_high[0], self.limit_high[1], color='grey',
                       alpha=0.1)

        ax.set_yscale('log')
        ax.set_ylabel(r"Level density $\rho(E_x)~[\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"Excitation energy $E_x~[\mathrm{MeV}]$")
        ax.set_ylim(bottom=0.5/(nld.E[1]-nld.E[0]))

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    @staticmethod
    def lnlike(x: Tuple[float, float, float, float], nld_low: Vector,
               nld_high: Vector, discrete: Vector,
               model: Callable[..., ndarray],
               Sn, nldSn) -> float:
        """ Compute log likelihood of the normalization fitting

        This is the result up a, which is irrelevant for the maximization

        Args:
            x: The arguments ordered as A, alpha, T and Eshift
            nld_low: The lower region where discrete levels will be
                fitted.
            nld_high: The upper region to fit to model.
            discrete: The discrete levels to be used in fitting the
                lower region.
            model: The model to use when fitting the upper region.
                Must support the keyword arguments
                ``model(E=..., T=..., Eshift=...) -> ndarray``

        Returns:
            lnlike: log likelihood
        """
        A, alpha, T, Eshift = x[:4]  # slicing needed for multinest?
        transformed_low = nld_low.transform(A, alpha, inplace=False)
        transformed_high = nld_high.transform(A, alpha, inplace=False)

        err_low = transformed_low.error(discrete)
        expected = Vector(E=transformed_high.E,
                          values=model(E=transformed_high.E,
                                       T=T, Eshift=Eshift))
        err_high = transformed_high.error(expected)

        nldSn_model = model(E=Sn, T=T, Eshift=Eshift)
        err_nldSn = ((nldSn[0] - nldSn_model)/nldSn[1])**2

        ln_stds = (np.log(transformed_low.std).sum()
                   + np.log(transformed_high.std).sum())

        return -0.5*(err_low + err_high + err_nldSn + ln_stds)

    @staticmethod
    def const_temperature(E: ndarray, T: float, Eshift: float) -> ndarray:
        """ Constant Temperature NLD"""
        ct = np.exp((E - Eshift) / T) / T
        return ct

    @staticmethod
    def nldSn_from_D0(D0: Union[float, Tuple[float, float]],
                      Sn: Union[float, Tuple[float, float]], Jtarget: float,
                      spincutModel: str, spincutPars: Dict[str, Any],
                      **kwargs) -> Tuple[float, float]:
        """Calculate nld(Sn) from D0


        1/D0 = nld(Sn) * ( g(Jtarget+1/2, pi_target)
                         + g(Jtarget1/2, pi_target) )
        Here we assume equal parity, g(J,pi) = g(J)/2 and
        nld(Sn) = 1/D0 * 2/(g(Jtarget+1/2) + g(Jtarget-1/2))
        For the case Jtarget = 0, the g(Jtarget-1/2) = 0

        Parameters:
            D0 (float or [float, float]):
                Average resonance spacing from s waves [eV]. If a tuple,
                it is assumed that it is of the form `[value, uncertainty]`.
            Sn (float or [float, float]):
                Separation energy [MeV]. If a tuple, it is assumed that it is of
                the form `[value, uncertainty]`.
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

        D0 = np.atleast_1d(D0)[0]
        Sn = np.atleast_1d(Sn)[0]

        def g(J):
            return SpinFunctions(Ex=Sn, J=J,
                                 model=spincutModel,
                                 pars=spincutPars).distibution()

        if Jtarget == 0:
            summe = 1 / 2 * g(Jtarget + 1 / 2)
        else:
            summe = 1 / 2 * (g(Jtarget - 1 / 2) + g(Jtarget + 1 / 2))

        nld = 1 / (summe * D0 * 1e-6)
        return [Sn, nld]

    @staticmethod
    def D0_from_nldSn(nld_model: Callable[..., Any],
                      Sn: Union[float, Tuple[float, float]], Jtarget: float,
                      spincutModel: str, spincutPars: Dict[str, Any],
                      **kwargs) -> Tuple[float, float]:
        """Calculate D0 from nld(Sn), assuming equiparity.

        This is the inverse of `nldSn_from_D0`

        Parameters:
            nld_model (Callable[..., Any]): Model for nld above data of the
                from `y = nld_model(E)` in 1/MeV.
            Sn (float or [float, float]):
                Separation energy [MeV]. If a tuple, it is assumed that it is of
                the form `[value, uncertainty]`.
            Jtarget (float):
                Target spin
            spincutModel (str):
                Model to for the spincut
            spincutPars Dict[str, Any]:
                Additional parameters necessary for the spin cut model
            **kwargs: Description


        Returns:
            D0: D0 in eV
        """

        Sn = np.atleast_1d(Sn)[0]
        nld = nld_model(Sn)

        def g(J):
            return SpinFunctions(Ex=Sn, J=J,
                                 model=spincutModel,
                                 pars=spincutPars).distibution()

        if Jtarget == 0:
            summe = 1 / 2 * g(Jtarget + 1 / 2)
        else:
            summe = 1 / 2 * (g(Jtarget - 1 / 2) + g(Jtarget + 1 / 2))

        D0 = 1 / (summe * nld * 1e-6)
        return D0

    @property
    def discrete(self) -> Optional[Vector]:
        return self._discrete

    @discrete.setter
    def discrete(self, value: Optional[Union[Path, str, Vector]]) -> None:
        if value is None:
            self._discretes = None
            self.LOG.debug("Set `discrete` to None")
        elif isinstance(value, (str, Path)):
            if self.nld is None:
                raise ValueError(f"`nld` must be set before loading levels")
            nld = self.nld.copy()
            nld.to_MeV()
            self.LOG.debug("Set `discrete` levels from file with FWHM %s",
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
            self.LOG.debug("Set `discrete` by Vector")
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
