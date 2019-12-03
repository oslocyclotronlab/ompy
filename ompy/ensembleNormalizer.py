import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Any, Tuple, Union
from operator import xor
import pandas as pd
import copy
from itertools import repeat

from .models import ResultsNormalized
from .vector import Vector
from .extractor import Extractor
from .normalizer_nld import NormalizerNLD
from .normalizer_gsf import NormalizerGSF
from .normalizer_simultan import NormalizerSimultan

if 'JPY_PARENT_PID' in os.environ:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class EnsembleNormalizer:
    """Normalizes NLD nad γSF extracted from the ensemble

    Usage:
      The calling syntax can be either to normalize simultaneously::

        EnsembleNormalizer(extractor=...,
                           normalizer_simultan=...)

      , or to normalize sequentially::

          EnsembleNormalizer(extractor=...,
                               normalizer_nld=...,
                               normalizer_gsf=...)

    Attributes:
        extractor (Extractor): Extractor instance
        normalizer_nld (NormalizerNLD): NormalizerNLD instance
        normalizer_gsf (NormalizerGSF): NormalizerGSF instance
        normalizer_simultan (NormalizerSimultan): NormalizerSimultan instance
        res (List[ResultsNormalized]): List of the results
    """

    def __init__(self, *, extractor: Extractor,
                 normalizer_nld: Optional[NormalizerNLD] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None,
                 normalizer_simultan: Optional[NormalizerSimultan] = None):
        """
        Args:
            extractor (Extractor): Extractor instance
            normalizer_nld (NormalizerNLD, optional): NormalizerNLD instance
            normalizer_gsf (NormalizerGSF, optional): NormalizerGSF instance
            normalizer_simultan (NormalizerSimultan, optional):
                NormalizerSimultan instance
        """
        self.extractor = extractor

        self.normalizer_nld = normalizer_nld
        self.normalizer_gsf = normalizer_gsf

        self.normalizer_simultan = normalizer_simultan

        self.res: Optional[List[ResultsNormalized]] = None

    def normalize(self) -> None:
        """ Normalize ensemble """
        assert xor((self.normalizer_nld is not None
                    and self.normalizer_gsf is not None),
                   self.normalizer_simultan is not None), \
            "Either 'normalizer_nld' and 'normalizer_gsf' must be set, or " \
            "normalizer_simultan"

        gsfs = self.extractor.gsf
        nlds = self.extractor.nld

        # reset
        self.res = []

        # tqdm should be innermost (see github))
        for i, (nld, gsf) in enumerate(zip(tqdm(nlds), gsfs)):
            LOG.info(f"\n\n---------\nNormalizing #{i}")
            nld.copy()
            nld.cut_nan()

            gsf.copy()
            gsf.cut_nan()

            if self.normalizer_simultan is not None:
                res = self.normalizeSimultan(i, nld=nld, gsf=gsf)
            else:
                res = self.normalizeStagewise(i, nld=nld, gsf=gsf)

            self.res.append(res)

    def normalizeSimultan(self, num: int, *,
                          nld: Vector, gsf: Vector) -> ResultsNormalized:
        """ Wrapper for simultaneous normalization

        Args:
            num (int): Loop number
            nld (Vector): NLD before normalization
            gsf (Vector): gsf before normalization

        Returns:
            res (ResultsNormalized): results (/parameters) of normalization
        """
        self.normalizer_simultan.normalize(gsf=gsf, nld=nld, num=num)
        return self.normalizer_simultan.res

    def normalizeStagewise(self, num: int, *,
                           nld: Vector, gsf: Vector) -> ResultsNormalized:
        """ Wrapper for stagewise normalization

        Args:
            num (int): Loop number
            nld (Vector): NLD before normalization
            gsf (Vector): gsf before normalization

        Returns:
            res (ResultsNormalized): results (/parameters) of normalization
        """
        self.normalizer_nld.normalize(nld=nld, num=num)
        self.normalizer_gsf.normalize(normalizer_nld=self.normalizer_nld,
                                      gsf=gsf)

        # same B for all normalizations of the same nld
        B = self.normalizer_gsf.res.pars["B"]
        N = self.normalizer_gsf.res.samples["A"]
        self.normalizer_gsf.res.samples["B"] = np.full_like(N, B)
        return self.normalizer_gsf.res

    def plot(self, ax: Tuple[Any, Any] = None,
             add_figlegend: bool = True,
             n_plot: bool = 5,
             plot_model_stats: bool = False,
             random_state: Optional[np.random.RandomState] = None,
             **kwargs) -> Tuple[Any, Any]:
        """Plots randomly drawn samples

        Args:
            ax (Tuple[Any, Any], optional): The matplotlib axis to plot onto.
                Creates axis is not provided.
            add_figlegend (bool, optional): Defaults to `True`.
            n_plot (bool, optional): Number of (nld, gsf) samples to plot
            plot_model_stats (bool, optional): Plot stats also for models used
                in normalization
            random_state (np.random.RandomState, optional): random state, set
                by default such that a repeated use of the function gives the
                same results.
            **kwargs: Description

        TODO:
            - Refactor code
            - Could not find out how to not plot dublicate legend entries,
              thus using a workaround
            - Checks if extrapolating where nld or gsf is np.nan

        Returns:
            fig, ax
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax.figure

        norm_sim = self.normalizer_simultan
        if norm_sim is not None:
            normalizer_gsf = copy.deepcopy(norm_sim.normalizer_gsf)
            normalizer_nld = copy.deepcopy(norm_sim.normalizer_nld)
        else:
            normalizer_gsf = copy.deepcopy(self.normalizer_gsf)
            normalizer_nld = copy.deepcopy(self.normalizer_nld)

        if random_state is None:  # cannot move this to definition
            random_state = np.random.RandomState(98765)

        samples = self.samples_from_res(random_state)
        self.plot_selection(ax=ax, samples=samples,
                            normalizer_nld=normalizer_nld,
                            normalizer_gsf=normalizer_gsf)

        # get median, 1 sigma, ...
        percentiles = [0.16, 0.84]
        lines = self.plot_vector_stats(ax, samples, percentiles)

        if plot_model_stats:
            Emin = samples["nld"].iloc[0].E[-1]
            x = np.linspace(Emin, normalizer_nld.norm_pars.Sn[0], num=20)
            color = lines.get_facecolor()
            self.plot_nld_ext_stats(ax[0], x=x, samples=samples,
                                    normalizer_nld=normalizer_nld,
                                    percentiles=percentiles, color=color)

            E = samples["gsf"].iloc[0].E
            xlow = np.linspace(0.001, E[0], num=20)
            xhigh = np.linspace(E[-1], normalizer_gsf.norm_pars.Sn[0], num=20)
            self.plot_gsf_ext_stats(ax[1], xlow=xlow, xhigh=xhigh,
                                    samples=samples,
                                    normalizer_gsf=normalizer_gsf,
                                    percentiles=percentiles, color=color)

        if add_figlegend:
            fig.legend(loc=9, ncol=4, frameon=True)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

        return fig, ax

    def samples_from_res(self,
                         random_state: Optional[np.random.RandomState] = None) -> pd.DataFrame:
        """Draw random samples from results with transformed nld & gsf

        Args:
            random_state (np.random.RandomState, optional): random state,
                set by default such that a repeated use of the function
                gives the same results.

        Returns:
            Samples
        """

        for i in range(len(self.res)):
            nld = self.extractor.nld[i].copy()
            gsf = self.extractor.gsf[i].copy()
            nld.to_MeV()
            gsf.to_MeV()
            samples_ = copy.deepcopy(self.res[i].samples)
            df = tranform_nld_gsf(samples_, nld, gsf,
                                  random_state=random_state)
            if i == 0:
                samples = df
            else:
                samples = samples.append(df)
        return samples

    def plot_selection(self, *, ax: Tuple[Any, Any],
                       samples: pd.DataFrame,
                       normalizer_nld: Optional[NormalizerNLD],
                       normalizer_gsf: Optional[NormalizerGSF],
                       n_plot: Optional[bool] = 5,
                       random_state: Optional[np.random.RandomState] = None) -> None:
        """ Plot some nld and gsf samples

        Args:
            ax (Tuple[Any, Any]): The matplotlib axis to plot onto.
                Creates axis is not provided.
            samples (pd.DataFrame): Random samples from results with
                transformed nld & gsf
            normalizer_nld (NormalizerNLD): NormalizerNLD instance.
                Note: Input a copy as the instance attributes will be changed.
            normalizer_gsf (NormalizerGSF): NormalizerGSF instance.
                Note: Input a copy as the instance attributes will be changed.
            n_plot (bool, optional): Number of (nld, gsf) samples to plot
            random_state (np.random.RandomState, optional): random state, set
                by default such that a repeated use of the function gives the
                same results.

        """
        if random_state is None:  # cannot move this to definition
            random_state = np.random.RandomState(98765)

        res = copy.deepcopy(self.res[0])  # dummy for later
        selection = samples.sample(n=n_plot, random_state=random_state)
        for i, (_, row) in enumerate(selection.iterrows()):
            res.nld = row["nld"]
            res.gsf = row["gsf"]
            res.pars = row.to_dict()

            # workaround for the tuple (currently just a float)
            keys_workaround = ["T", "D0"]
            for key in keys_workaround:
                res.pars[key] = [res.pars[key], np.nan]

            # create extrapolations of gsf
            normalizer_gsf._gsf = row["gsf"]
            normalizer_gsf.extrapolate(row["gsf"])
            res.gsf_model_low = normalizer_gsf.model_low
            res.gsf_model_high = normalizer_gsf.model_high
            for model in [res.gsf_model_low, res.gsf_model_high]:
                model.shift_after = model.shift

            add_label = True if i == 0 else False
            plot_fitregion = True if i == 0 else False
            normalizer_nld.plot(ax=ax[0], results=res,
                                add_label=add_label, alpha=1/n_plot,
                                add_figlegend=False,
                                plot_fitregion=plot_fitregion)
            normalizer_gsf.plot(ax=ax[1], results=res, add_label=False,
                                alpha=1/n_plot,
                                add_figlegend=False,
                                plot_fitregion=plot_fitregion)

    @staticmethod
    def stats_from_df(df, fmap, shape_out, percentiles):
        array = df.to_numpy()
        out = np.zeros(shape_out)
        indexed = enumerate(array)
        # apply fmap to each row of the dataframe
        np.fromiter(map(fmap, indexed, repeat(out)), dtype=float)
        stats = pd.DataFrame(out[:, :])
        stats = pd.DataFrame({'median': stats.median(),
                              'low': stats.quantile(percentiles[0], axis=0),
                              'high': stats.quantile(percentiles[1], axis=0)})
        return stats

    @staticmethod
    def plot_vector_stats(ax: Tuple[Any, Any], samples, percentiles):
        # define helper function
        def vec_to_values(x, out):  # noqa
            idx, vec = x
            out[idx] = vec.values

        df = samples["nld"]
        E = df.iloc[0].E
        stats_nld = EnsembleNormalizer.stats_from_df(df, fmap=vec_to_values,
                                            shape_out=(len(df), len(E)),
                                            percentiles=percentiles)  # noqa
        stats_nld["x"] = E
        stats_nld.plot(x="x", y="median", ax=ax[0], legend=False)

        df = samples["gsf"]
        stats_gsf = EnsembleNormalizer.stats_from_df(df, fmap=vec_to_values,
                                            shape_out=(len(df), len(E)),
                                            percentiles=percentiles)  # noqa
        stats_gsf["x"] = df.iloc[0].E
        stats_gsf.plot(x="x", y="median", ax=ax[1], legend=False)

        pc_diff = percentiles[1] - percentiles[0]
        ax[0].fill_between(stats_nld.x, stats_nld["low"], stats_nld["high"],
                           alpha=0.3,
                           label=f"{(pc_diff)*100:.0f}% credibility interval")
        lines = ax[1].fill_between(stats_gsf.x, stats_gsf["low"],
                                   stats_gsf["high"],
                                   alpha=0.3)
        return lines

    @staticmethod
    def plot_nld_ext_stats(ax: Any, *, x: np.ndarray,
                           samples, normalizer_nld, percentiles,
                           color):
        # define helper function
        def to_values(a, out):  # noqa
            idx, val = a
            out[idx] = normalizer_nld.curried_model(T=val[0],
                                                    D0=val[1],
                                                    E=x)

        df = samples[["T", "D0"]]
        stats = EnsembleNormalizer.stats_from_df(df, fmap=to_values,
                                                 shape_out=(len(df), len(x)),
                                                 percentiles=percentiles)
        ax.fill_between(x, stats["low"], stats["high"],
                        alpha=0.3, color=color)

    @staticmethod
    def plot_gsf_ext_stats(ax: Any, *, xlow: np.ndarray, xhigh: np.ndarray,
                           samples, normalizer_gsf, percentiles,
                           color):
        # define helper function
        def to_values(a, out):  # noqa
            idx, val = a
            low, high = normalizer_gsf.extrapolate(val, E=[xlow, xhigh])
            out[0, idx] = low.values
            out[1, idx] = high.values

        assert len(xlow) == len(xhigh)
        array = samples["gsf"].to_numpy()
        out = np.zeros((2, len(array), len(xlow)))
        indexed = enumerate(array)

        low, high = percentiles
        np.fromiter(map(to_values, indexed, repeat(out)), dtype=float)
        for i, arr in enumerate([out[0, :, :], out[1, :, :]]):
            stat = pd.DataFrame(arr)
            stat = pd.DataFrame({'median': stat.median(),
                                 'low': stat.quantile(low, axis=0),
                                 'high': stat.quantile(high, axis=0)})
            ax.fill_between(xlow if i == 0 else xhigh,
                            stat["low"], stat["high"],
                            alpha=0.3, color=color)


def tranform_nld_gsf(samples: dict, nld=None, gsf=None,
                     N_max: int = 100,
                     random_state=None) -> pd.DataFrame:
    """
    Use a list(dict) of samples of `A`, `B`, and `alpha` parameters from
    multinest to transform a (list of) nld and/or gsf sample(s). Can be used
    to normalize the nld and/or gsf

    Args:
        samples (dict): Multinest samples.
        nld (om.Vector or list/array[om.Vector], optional):
            nld ("unnormalized")
        gsf (om.Vector or list/array[om.Vector], optional):
            gsf ("unnormalized")
        N_max (int, optional): Maximum number of samples returned if `nld`
                               and `gsf` is a list/array
        random_state (optional): random state, set by default such that
                                 a repeted use of the function gives the same
                                 results.

    Returns:
        Dataframe with randomly selected samples of nld, gsf and the
        corresponding parameters. The nld and gsf are transformed

    """

    # Need to sweep though multinest samples in random order
    # as they are ordered with decreasing likelihood by default
    for key, value in samples.items():
        N_multinest = len(value)
        break
    randlist = np.arange(N_multinest)
    if random_state is None:  # cannot move this to definition
        random_state = np.random.RandomState(65489)
    random_state.shuffle(randlist)  # works in-place

    if nld is not None:
        A = samples["A"]
        alpha = samples["alpha"]
        if type(nld) is Vector:
            N = min(N_multinest, N_max)
        else:
            N = len(nld)
    nld_trans = []

    if gsf is not None:
        B = samples["B"]
        alpha = samples["alpha"]
        if type(gsf) is Vector:
            N = min(N_multinest, N_max)
        else:
            N = len(gsf)
    gsf_trans = []

    # transform the list
    for i in range(N):
        i_multi = randlist[i]
        # nld loop
        try:
            if type(nld) is Vector:
                nld_tmp = nld
            else:
                nld_tmp = nld[i]
            nld_tmp = nld_tmp.transform(alpha=alpha[i_multi],
                                        const=A[i_multi], inplace=False)
            nld_trans.append(nld_tmp)
        except:
            pass
        # gsf loop
        try:
            if type(gsf) is Vector:
                gsf_tmp = gsf
            else:
                gsf_tmp = gsf[i]
            gsf_tmp = gsf_tmp.transform(alpha=alpha[i_multi],
                                        const=B[i_multi], inplace=False)
            gsf_trans.append(gsf_tmp)
        except:
            pass

    df = pd.DataFrame()
    df = df.from_dict(samples)
    selected = df.iloc[randlist[:N]]
    selected["nld"] = nld_trans
    selected["gsf"] = gsf_trans

    return selected
