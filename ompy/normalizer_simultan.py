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
import pandas as pd
from random import sample

from .extractor import Extractor
from .library import log_interp1d, tranform_nld_gsf
from .models import Model, ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer import Normalizer
from .normalizer_gsf import NormalizerGSF
from .spinfunctions import SpinFunctions
from .vector import Vector

LOG = logging.getLogger(__name__)


class NormalizerSimultan():

    """ Simultaneous normalization of nld and gsf. Composed of Normalizer and NormalizerGSF as input, so read more on the normalization there

    Attributes:
        extractor (Extractor): Extractor instance
        normalizer_gsf (NormalizerGSF): NormalizerGSF instance
        normalizer_nld (Normalizer): Normalizer instance
        res (ResultsNormalized, optional): Results
        std_fake_gsf (bool): Whether the std. deviation is faked
            (see `normalize`)
        std_fake_nld (bool): Whether the std. deviation is faked
            (see `normalize`)
        multinest_path (Path, optional): Default path where multinest
            saves files
        multinest_kwargs (dict): Additional keywords to multinest

    TODO: Work with more general models, too, not just CT for nld
    """

    def __init__(self, *, extractor: Optional[Extractor] = None,
                 # gsf: Optional[Vector] = None,
                 # nld: Optional[Vector] = None,
                 normalizer_nld: Optional[Normalizer] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None):
        """
        TODO:
            - currently have to set arguments here, an cannot set them in
              "normalize"

        """

        self.extractor = extractor

        self.normalizer_nld = copy.deepcopy(normalizer_nld)
        self.normalizer_gsf = copy.deepcopy(normalizer_gsf)

        self.std_fake_nld: Optional[bool] = None  # See `normalize`
        self.std_fake_gsf: Optional[bool] = None  # See `normalize`

        self.res: Optional[ResultsNormalized] = None

        self.multinest_path: Optional[Path] = Path('multinest')
        self.multinest_kwargs: dict = {}

    def normalize(self):
        """ Perform normalization and saves results to `self.res` """
        if self.extractor is not None:
            gsfs = self.extractor.gsf
            nlds = self.extractor.nld

        # Ensure to rerun if called again
        self.res = ResultsNormalized(name="Results NLD")

        # normalization
        lists = zip(nlds, gsfs)
        for i, (nld, gsf) in enumerate(lists):
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
            popt, samples = self.optimize(i, args_nld, guess)

            self.res.pars.append(popt)
            self.res.samples.append(samples)

            # reset
            if self.std_fake_nld is True:
                self.std_fake_nld = None
                nld.std = None
            if self.std_fake_gsf is True:
                self.std_fake_gsf = None
                gsf.std = None
            self.res.nld.append(nld)
            self.res.gsf.append(gsf)

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
        guess["B"] = normalizer_gsf.res.pars[0]["B"]

        guess_print = copy.deepcopy(guess)
        guess_print["D0"] = guess_print["D0"][0]
        LOG.info("DE results/initial guess:\n%s\n%s",
                 tt.to_string([list(guess_print.values())],
                 header=['A', 'α [MeV⁻¹]', 'T [MeV]', 'D₀ [eV]*', 'B']),
                 "*copied from input") # noqa

        return args_nld, guess

    def optimize(self, iteration: int,
                 args_nld: Iterable,
                 guess: Dict[str, float]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[float]]]:  # noqa
        """Find parameters given model constraints and an initial guess

        Employs Multinest

        Args:
            iteration (int): Iteration number
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
        path = self.multinest_path / f"sim_norm_{iteration}_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        # defaults
        kwargs = self.multinest_kwargs
        if "verbose" not in kwargs:
            kwargs["verbose"] = True
        if "False" not in kwargs:
            kwargs["resume"] = False

        LOG.info("Starting multinest: ")
        LOG.debug("with following keywords %s:", kwargs)
        #  Hack where stdout from Multinest is redirected as info messages
        LOG.write = lambda msg: LOG.info(msg) if msg != '\n' else None

        with redirect_stdout(LOG):
            pymultinest.run(loglike, prior, len(guess),
                            outputfiles_basename=str(path),
                            **kwargs)

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
        """
        A, alpha, T, D0, B = x[:5]

        normalizer_gsf = self.normalizer_gsf
        normalizer_nld = self.normalizer_nld

        err_nld = normalizer_nld.errfn(x[:5], *args_nld)

        nld = normalizer_nld.nld.transform(A, alpha, inplace=False)
        nld_model = lambda E: normalizer_nld.curried_model(E, T=T, D0=D0)  # noqa

        normalizer_gsf.nld_model = nld_model
        normalizer_gsf.nld = nld
        normalizer_gsf.norm_pars.D0 = [D0, 5] # CHANGE LATER, but tuple doesn't support item assignment
        normalizer_gsf._gsf = normalizer_gsf.gsf_in.transform(B, alpha, inplace=False)
        normalizer_gsf._gsf_low, normalizer_gsf._gsf_high = \
            normalizer_gsf.extrapolate()
        err_gsf = normalizer_gsf.errfn()
        return err_nld + err_gsf


    def plot(self):
        """ Plots randomly drawn samples
        TODO: Cleanup!
        """
        fig, ax = plt.subplots(1, 2, constrained_layout=True)

        nlds = []
        gsfs = []
        for i in range(len(self.res.nld)):
            nld = self.res.nld[i]
            gsf = self.res.gsf[i]
            samples = self.res.samples[i]
            nlds_, gsfs_ = tranform_nld_gsf(samples, nld, gsf)
            nlds.append(nlds_)
            gsfs.append(gsfs_)

        flatten = lambda l: [item for sublist in l for item in sublist]
        gsfs = flatten(gsfs)
        nlds = flatten(nlds)

        nld_vals = np.zeros((len(nlds), len(nld.E)))
        gsf_vals = np.zeros((len(gsfs), len(gsf.E)))

        for i, (nld, gsf) in enumerate(zip(nlds, gsfs)):
            nld_vals[i, :] = nld.values
            gsf_vals[i, :] = gsf.values

        n_plot = 5
        for i, (nld, gsf) in enumerate(sample(list(zip(nlds, gsfs)), n_plot)):
            nld.std = None  # as they were just faked
            gsf.std = None  # as they were just faked
            if i == 0:
                label = "random sample"
            else:
                label = None
            nld.plot(ax=ax[0], scale="log", color='k', alpha=1/(n_plot),
                     label=label)
            gsf.plot(ax=ax[1], scale="log", color='k', alpha=1/(n_plot),
                     label=label)

        ax[0].set_title("Level density")
        ax[1].set_title("γSF")

        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        stat_nld = pd.DataFrame(nld_vals)
        # Calculate all the desired values
        stat_nld = pd.DataFrame({'mean': stat_nld.mean(), 'median': stat_nld.median(),
            '25%': stat_nld.quantile(0.25, axis=0), '50%': stat_nld.quantile(0.5, axis=0),
            '84%': stat_nld.quantile(0.84, axis=0)})

        stat_gsf = pd.DataFrame(gsf_vals)
        # Calculate all the desired values
        stat_gsf = pd.DataFrame({'mean': stat_gsf.mean(axis=0), 'median': stat_gsf.median(axis=0),
            '16%': stat_gsf.quantile(0.16,axis=0), '50%': stat_gsf.quantile(0.5, axis=0), '84%': stat_gsf.quantile(0.84,axis=0)})

        stat_nld["E"] = nld.E
        stat_gsf["E"] = gsf.E

        stat_nld.plot(x="E", ax=ax[0])
        stat_gsf.plot(x="E", ax=ax[1])

        ax[0].legend(loc="best")
        ax[1].legend(loc="best")

    #     fig2, ax2 = plt.subplots(1, 2, constrained_layout=True)
    #     bins = np.logspace(-8, -6, num=50)

    #     data = gsf_vals[:, 6]
    #     median = np.percentile(data, 50)
    #     low = np.percentile(data, 16)
    #     high = np.percentile(data, 84)
    #     print("low, median, high", low, median, high)
    #     ax2[1].hist(gsf_vals[:, 6], bins=bins)
    #     ax2[1].axvline(median, color="r")
    #     ax2[1].axvline(low, color="g")
    #     ax2[1].axvline(high, color="b")
    #     ax2[1].set_xscale("log")
