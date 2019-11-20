"""Summary

Attributes:
    LOG (TYPE): Description
"""
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

from .extractor import Extractor
from .library import log_interp1d, self_if_none
from .models import Model, ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer_nld import NormalizerNLD
from .normalizer_gsf import NormalizerGSF
from .spinfunctions import SpinFunctions
from .vector import Vector

LOG = logging.getLogger(__name__)


class NormalizerSimultan():

    """ Simultaneous normalization of nld and gsf. Composed of Normalizer and NormalizerGSF as input, so read more on the normalization there

    Attributes:
        extractor (Extractor): Extractor instance
        normalizer_gsf (NormalizerGSF): NormalizerGSF instance
        normalizer_nld (Normalizer): NormalizerNLD instance
        res (ResultsNormalized, optional): Results
        std_fake_gsf (bool): Whether the std. deviation is faked
            (see `normalize`)
        std_fake_nld (bool): Whether the std. deviation is faked
            (see `normalize`)
        multinest_path (Path, optional): Default path where multinest
            saves files
        multinest_kwargs (dict): Additional keywords to multinest. Defaults to
            `{"seed": 65498, "resume": False}`

    TODO: Work with more general models, too, not just CT for nld
    """

    def __init__(self, *,
                 gsf: Optional[Vector] = None,
                 nld: Optional[Vector] = None,
                 normalizer_nld: Optional[NormalizerNLD] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None):
        """
        TODO:
            - currently have to set arguments here, an cannot set them in
              "normalize"

        """
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

    def normalize(self, *, num: Optional[int] = 0,
                  gsf: Optional[Vector] = None,
                  nld: Optional[Vector] = None,
                  normalizer_nld: Optional[NormalizerNLD] = None,
                  normalizer_gsf: Optional[NormalizerGSF] = None):
        """Perform normalization and saves results to `self.res`

        Args:
            num (Optional[int], optional): Loop number
            gsf (Optional[Vector], optional): gsf before normalization
            nld (Optional[Vector], optional): nld before normalization
            normalizer_gsf (NormalizerGSF): NormalizerGSF instance
            normalizer_nld (Normalizer): NormalizerNLD instance
        """
        # reset internal state
        self.res = ResultsNormalized(name="Results NLD")

        normalizer_nld = copy.deepcopy(self.self_if_none(normalizer_nld))
        normalizer_gsf = copy.deepcopy(self.self_if_none(normalizer_gsf))

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
            nld.std = nld.values * 0.1  # 10% is an arb. choice
        if self.std_fake_gsf or gsf.std is None:
            self.std_fake_gsf = True
            gsf.std = gsf.values * 0.1  # 10% is an arb. choice

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


    def initial_guess(self):
        """ Find an inital guess for normalization parameters

        Uses guess of normalizer_nld and corresponding normalization of gsf

        Returns:
           The arguments used for chi^2 minimization and the
           minimizer.
        """
        normalizer_nld = self.normalizer_nld
        normalizer_gsf = self.normalizer_gsf

        args_nld, guess = normalizer_nld.initial_guess()
        [A, alpha, T, D0] = [guess["A"], guess["alpha"],
                             guess["T"], guess["D0"][0]]

        nld = normalizer_nld.nld.transform(A, alpha, inplace=False)
        nld_model = lambda E: normalizer_nld.curried_model(E, T=T, D0=D0)  # noqa

        normalizer_gsf.normalize(nld=nld, nld_model=nld_model)
        guess["B"] = normalizer_gsf.res.pars["B"]

        guess_print = copy.deepcopy(guess)
        guess_print["D0"] = guess_print["D0"][0]
        LOG.info("DE results/initial guess:\n%s\n%s",
                 tt.to_string([list(guess_print.values())],
                 header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]*', 'B']),
                 "*copied from input") # noqa

        return args_nld, guess

    def optimize(self, num: int,
                 args_nld: Iterable,
                 guess: Dict[str, float]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[float]]]:  # noqa
        """Find parameters given model constraints and an initial guess

        Employs Multinest

        Args:
            num (int): Loop number
            args_nld (Iterable): Additional arguments for the nld errfn
            guess (Dict[str, float]): The initial guess of the parameters

        Returns:
            - popt (Dict[str, Tuple[float, float]]): Median and 1sigma of the
                parameters
            - samples (Dict[str, List[float]]): Multinest samplesø.
                Note: They are still importance weighted, not random draws
                from the posterior.

        Raises:
            ValueError: Description
        """
        if guess['alpha'] < 0:
            raise ValueError("Prior selection not implemented for α < 0")
        alpha_exponent = np.log10(guess['alpha'])

        if guess['T'] < 0:
            raise ValueError("Prior selection not implemented for T < 0")
        T_exponent = np.log10(guess['T'])

        A = guess['A']
        D0 = guess['D0']
        B = guess["B"]

        # truncations from absolute values
        lower, upper = 0., np.inf
        mu_A, sigma_A = A, 4*A
        a_A = (lower - mu_A) / sigma_A
        mu_D0, sigma_D0 = D0[0], D0[1]
        a_D0 = (lower - mu_D0) / sigma_D0
        mu_B, sigma_B = B, 4*B
        a_B = (lower - mu_B) / sigma_B

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
            # normal prior
            cube[4] = truncnorm.ppf(cube[4], a_B, upper, loc=mu_B,
                                    scale=sigma_B)

            if np.isinf(cube[3]):
                LOG.debug("Encountered inf in cube[3]:\n%s", cube[3])

        def loglike(cube, ndim, nparams):
            chi2 = self.errfn(cube, args_nld=args_nld)
            loglikelihood = -0.5 * chi2
            return loglikelihood

        # parameters are changed in the errfn
        norm_pars_org = copy.deepcopy(self.normalizer_gsf.norm_pars)

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"sim_norm_{num}_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        LOG.info("Starting multinest: ")
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
                 header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]', 'B']))

        # reset state
        self.normalizer_gsf.norm_pars = norm_pars_org

        return popt, samples

    def errfn(self, x: Tuple[float, float, float, float, float],
              args_nld: Iterable) -> float:
        """Compute the χ² of the normalization fitting

        Args:
            x (Tuple[float, float, float, float, float]): The arguments
                ordered as A, alpha, T and D0, B
            args_nld (TYPE): Additional arguments for the nld errfn

        Returns:
            chi2 (float): The χ² value

        TODO:
            Clean up assignment of D0 (see code)
        """
        A, alpha, T, D0, B = x[:5]

        normalizer_gsf = self.normalizer_gsf
        normalizer_nld = self.normalizer_nld

        err_nld = normalizer_nld.errfn(x[:5], *args_nld)

        nld = normalizer_nld.nld.transform(A, alpha, inplace=False)
        nld_model = lambda E: normalizer_nld.curried_model(E, T=T, D0=D0)  # noqa

        normalizer_gsf.nld_model = nld_model
        normalizer_gsf.nld = nld
        normalizer_gsf.norm_pars.D0 = [D0, np.nan]  # dummy uncertainty
        normalizer_gsf._gsf = normalizer_gsf.gsf_in.transform(B, alpha, inplace=False)
        normalizer_gsf._gsf_low, normalizer_gsf._gsf_high = \
            normalizer_gsf.extrapolate()
        err_gsf = normalizer_gsf.errfn()
        return err_nld + err_gsf

    def plot(self, ax: Any = None, add_label: bool = True,
             add_figlegend: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """ Plots randomly drawn samples
        TODO: Cleanup!
        # TODO:
            - Could not find out how to not plot dublicate legend entries
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax.figure

        self.normalizer_nld.plot(ax=ax[0], add_label=True, results=self.res,
                                 add_figlegend=False, **kwargs)
        self.normalizer_gsf.plot(ax=ax[1], add_label=False, results=self.res,
                                 add_figlegend=False, **kwargs)

        ax[0].set_title("Level density")
        ax[1].set_title("γSF")

        if add_figlegend:
            fig.legend(loc=9, ncol=4, frameon=True)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

        return fig, ax

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)
