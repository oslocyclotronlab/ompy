import logging
import numpy as np
import copy
import json
import termtables as tt
from numpy import ndarray
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Callable, Dict, Iterable, List
from scipy.stats import truncnorm
import pymultinest
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

from .abstract_normalizer import AbstractNormalizer
from .extractor import Extractor
from .library import log_interp1d, self_if_none
from .models import Model, ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer_nld import NormalizerNLD
from .normalizer_gsf import NormalizerGSF
from .spinfunctions import SpinFunctions
from .vector import Vector


class NormalizerSimultan(AbstractNormalizer):

    """ Simultaneous normalization of nld and gsf. Composed of Normalizer and NormalizerGSF as input, so read more on the normalization there

    Attributes:
        extractor (Extractor): Extractor instance
        gsf (Optional[Vector], optional): gsf to normalize
        multinest_path (Path, optional): Default path where multinest
            saves files
        multinest_kwargs (dict): Additional keywords to multinest. Defaults to
            `{"seed": 65498, "resume": False}`
        nld (Optional[Vector], optional): nld to normalize
        normalizer_nld (NormalizerNLD): `NormalizerNLD` instance to get the normalization paramters
        normalizer_gsf (NormalizerGSF): `NormalizerGSF` instance to get the normalization paramters
        res (ResultsNormalized): Results
        std_fake_gsf (bool): Whether the std. deviation is faked
            (see `normalize`)
        std_fake_nld (bool): Whether the std. deviation is faked
            (see `normalize`)
        path (Path): The path save the results.


    TODO:
        Work with more general models, too, not just CT for nld
    """
    LOG = logging.getLogger(__name__)
    logging.captureWarnings(True)

    def __init__(self, *,
                 gsf: Optional[Vector] = None,
                 nld: Optional[Vector] = None,
                 normalizer_nld: Optional[NormalizerNLD] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False):
        """
        TODO:
            - currently have to set arguments here, an cannot set them in
              "normalize"

        Args:
            gsf (optional): see above
            nld (optional): see above
            normalizer_nld (optional): see above
            normalizer_gsf (optional): see above

        """
        super().__init__(regenerate)
        if normalizer_nld is None:
            self.normalizer_nld = None
        else:
            self.normalizer_nld = copy.deepcopy(normalizer_nld)

        if normalizer_gsf is None:
            self.normalizer_gsf = None
        else:
            self.normalizer_gsf = copy.deepcopy(normalizer_gsf)

        self.gsf = None if gsf is None else gsf.copy()
        self.nld = None if nld is None else nld.copy()

        self.std_fake_nld: Optional[bool] = None  # See `normalize`
        self.std_fake_gsf: Optional[bool] = None  # See `normalize`

        self.res: Optional[ResultsNormalized] = None

        self.multinest_path: Optional[Path] = Path('multinest')
        self.multinest_kwargs: dict = {"seed": 65498, "resume": False}

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

    def normalize(self, *, num: int = 0,
                  gsf: Optional[Vector] = None,
                  nld: Optional[Vector] = None,
                  normalizer_nld: Optional[NormalizerNLD] = None,
                  normalizer_gsf: Optional[NormalizerGSF] = None) -> None:
        """Perform normalization and saves results to `self.res`

        Args:
            num (int, optional): Loop number
            gsf (Optional[Vector], optional): gsf before normalization
            nld (Optional[Vector], optional): nld before normalization
            normalizer_nld (Optional[NormalizerNLD], optional): NormalizerNLD
                instance
            normalizer_gsf (Optional[NormalizerGSF], optional): NormalizerGSF
                instance
        """
        if not self.regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

        # reset internal state
        self.res = ResultsNormalized(name="Results NLD")

        self.normalizer_nld = copy.deepcopy(self.self_if_none(normalizer_nld))
        self.normalizer_gsf = copy.deepcopy(self.self_if_none(normalizer_gsf))
        for norm in [self.normalizer_nld, self.normalizer_gsf]:
            norm._save_instance = False
            norm.regenerate = True

        gsf = self.self_if_none(gsf)
        nld = self.self_if_none(nld)
        nld = nld.copy()
        gsf = gsf.copy()
        nld.to_MeV()
        gsf.to_MeV()

        # Need to give some sort of standard deviation for sensible results
        # Otherwise deviations at higher level density will have an
        # uncreasonably high weight.
        if self.std_fake_nld is None:
            self.std_fake_nld = False
        if self.std_fake_nld or nld.std is None:
            self.std_fake_nld = True
            nld.std = nld.values * 0.3  # x% is an arb. choice
        if self.std_fake_gsf or gsf.std is None:
            self.std_fake_gsf = True
            gsf.std = gsf.values * 0.3  # x% is an arb. choice

        # update
        self.normalizer_nld.nld = nld  # update before initial guess
        self.normalizer_gsf.gsf_in = gsf  # update before initial guess

        # Use DE to get an inital guess before optimizing
        args_nld, guess = self.initial_guess()
        # Optimize using multinest
        popt, samples = self.optimize(num, args_nld, guess)

        self.res.pars = popt
        self.res.samples = samples

        # reset
        if self.std_fake_nld is True:
            self.std_fake_nld = None
            nld.std = None
        if self.std_fake_gsf is True:
            self.std_fake_gsf = None
            gsf.std = None

        self.res.nld = nld.transform(self.res.pars["A"][0],
                                     self.res.pars["alpha"][0], inplace=False)
        self.res.gsf = gsf.transform(self.res.pars["B"][0],
                                     self.res.pars["alpha"][0], inplace=False)

        self.normalizer_gsf.model_low.autorange(self.res.gsf)
        self.normalizer_gsf.model_high.autorange(self.res.gsf)
        self.normalizer_gsf.extrapolate(self.res.gsf)
        self.res.gsf_model_low = self.normalizer_gsf.model_low
        self.res.gsf_model_high = self.normalizer_gsf.model_high
        for model in [self.res.gsf_model_low, self.res.gsf_model_high]:
            model.shift_after = model.shift

        self.save()  # save instance

    def initial_guess(self) -> None:
        """ Find an inital guess for normalization parameters

        Uses guess of normalizer_nld and corresponding normalization of gsf

        Returns:
           The arguments used for chi^2 minimization and the
           minimizer.
        """
        normalizer_nld = self.normalizer_nld
        normalizer_gsf = self.normalizer_gsf

        args_nld, guess = normalizer_nld.initial_guess()
        [A, alpha, T, Eshift] = [guess["A"], guess["alpha"],
                                 guess["T"], guess["Eshift"]]

        nld = normalizer_nld.nld.transform(A, alpha, inplace=False)
        nld_model = lambda E: normalizer_nld.model(E, T=T, Eshift=Eshift)  # noqa

        normalizer_gsf.normalize(nld=nld, nld_model=nld_model, alpha=alpha)
        guess["B"] = normalizer_gsf.res.pars["B"][0]

        guess_print = copy.deepcopy(guess)
        self.LOG.info("DE results/initial guess:\n%s",
                      tt.to_string([list(guess_print.values())],
                                   header=['A', 'α [MeV⁻¹]', 'T [MeV]',
                                           'Eshift [MeV]', 'B']))

        return args_nld, guess

    def optimize(self, num: int,
                 args_nld: Iterable,
                 guess: Dict[str, float]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[float]]]:  # noqa
        """Find parameters given model constraints and an initial guess

        Employs Multinest.

        Args:
            num (int): Loop number
            args_nld (Iterable): Additional arguments for the nld lnlike
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
        B = guess["B"]

        # truncations from absolute values
        lower_A, upper_A = 0., np.inf
        mu_A, sigma_A = A, 10*A
        a_A = (lower_A - mu_A) / sigma_A
        b_A = (upper_A - mu_A) / sigma_A

        lower_Eshift, upper_Eshift = -5., 5
        mu_Eshift, sigma_Eshift = 0, 5
        a_Eshift = (lower_Eshift - mu_Eshift) / sigma_Eshift
        b_Eshift = (upper_Eshift - mu_Eshift) / sigma_Eshift

        lower_B, upper_B = 0., np.inf
        mu_B, sigma_B = B, 10*B
        a_B = (lower_B - mu_B) / sigma_B
        b_B = (upper_B - mu_B) / sigma_B

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
            # truncated normal prior
            cube[4] = truncnorm.ppf(cube[4], a_B, b_B, loc=mu_B,
                                    scale=sigma_B)

            if np.isinf(cube[3]):
                self.LOG.debug("Encountered inf in cube[3]:\n%s", cube[3])

        def loglike(cube, ndim, nparams):
            return self.lnlike(cube, args_nld=args_nld)

        # parameters are changed in the lnlike
        norm_pars_org = copy.deepcopy(self.normalizer_gsf.norm_pars)

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"sim_norm_{num}_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        self.LOG.info("Starting multinest: ")
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
                      header=['A', 'α [MeV⁻¹]', 'T [MeV]',
                              'Eshift [MeV]', 'B']))

        # reset state
        self.normalizer_gsf.norm_pars = norm_pars_org

        return popt, samples

    def lnlike(self, x: Tuple[float, float, float, float, float],
               args_nld: Iterable) -> float:
        """Compute log likelihood  of the normalization fitting

        This is the result up to the constant, which is irrelevant for the
        maximization

        Args:
            x (Tuple[float, float, float, float, float]): The arguments
                ordered as A, alpha, T and Eshift, B
            args_nld (TYPE): Additional arguments for the nld lnlike

        Returns:
            lnlike: log likelihood
        """
        A, alpha, T, Eshift, B = x[:5]  # slicing needed for multinest?

        normalizer_gsf = self.normalizer_gsf
        normalizer_nld = self.normalizer_nld

        err_nld = normalizer_nld.lnlike(x[:4], *args_nld)

        nld = normalizer_nld.nld.transform(A, alpha, inplace=False)
        nld_model = lambda E: normalizer_nld.model(E, T=T, Eshift=Eshift)  # noqa

        normalizer_gsf.nld_model = nld_model
        normalizer_gsf.nld = nld
        # calculate the D0-equivalent of T and Eshift used
        D0 = normalizer_nld.D0_from_nldSn(nld_model,
                                          **normalizer_nld.norm_pars.asdict())
        normalizer_gsf.norm_pars.D0 = [D0, np.nan]  # dummy uncertainty
        normalizer_gsf._gsf = normalizer_gsf.gsf_in.transform(B, alpha,
                                                              inplace=False)
        normalizer_gsf._gsf_low, normalizer_gsf._gsf_high = \
            normalizer_gsf.extrapolate()
        err_gsf = normalizer_gsf.lnlike()
        return err_nld + err_gsf

    def plot(self, ax: Optional[Any] = None, add_label: bool = True,
             add_figlegend: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """Plots nld and gsf

        Args:
            ax (optional): The matplotlib axis to plot onto. Creates axis
                is not provided
            add_label (bool, optional):Defaults to `True`.
            add_figlegend (bool, optional): Defaults to `True`.
            results Optional[ResultsNormalized]: If provided, gsf and model
                are taken from here instead.
            **kwargs: kwargs for plot

        Returns:
            fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax[0].figure

        self.normalizer_nld.plot(ax=ax[0], add_label=True, results=self.res,
                                 add_figlegend=False, **kwargs)
        self.normalizer_gsf.plot(ax=ax[1], add_label=False, results=self.res,
                                 add_figlegend=False, **kwargs)

        ax[0].set_title("Level density")
        ax[1].set_title(r"$\gamma$SF")

        if add_figlegend:
            fig.legend(loc=9, ncol=4, frameon=True)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

        return fig, ax

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)
