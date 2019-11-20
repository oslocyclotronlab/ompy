import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Union, List, Optional, Any, Tuple
from pathlib import Path
from operator import xor
import pandas as pd
from random import sample
import copy
from itertools import repeat

from .matrix import Matrix
from .models import ResultsNormalized
from .vector import Vector
from .extractor import Extractor
from .normalizer_nld import NormalizerNLD
from .normalizer_gsf import NormalizerGSF
from .normalizer_simultan import NormalizerSimultan

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class EnsembleNormalizer:
    """ Normalizes NLD nad γSF extracted from the ensemble

    Attributes:
        extractor (Extractor): Extractor instance
    """

    def __init__(self, *, extractor: Extractor,
                 normalizer_nld: Optional[NormalizerNLD] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None,
                 normalizer_simultan: Optional[NormalizerSimultan] = None):

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
        self.normalizer_gsf.normalize(nld_normalizer=self.normalizer_nld,
                                      gsf=gsf)

        # same B for all normalizations of the same nld
        B = self.normalizer_gsf.res.pars["B"]
        N = self.normalizer_gsf.res.samples["A"]
        self.normalizer_gsf.res.samples["B"] = np.full_like(N, B)
        return self.normalizer_gsf.res

    def plot(self, ax: Any = None,
             add_figlegend: bool = True,
             n_plot: Optional[bool] = 5,
             **kwargs) -> Tuple[Any, Any]:
        """ Plots randomly drawn samples
        TODO:
            - Could not find out how to not plot dublicate legend entries
            - Checks if extrapolating where nld or gsf is np.nan
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

        for i in range(len(self.res)):
            nld = self.extractor.nld[i].copy()
            gsf = self.extractor.gsf[i].copy()
            nld.to_MeV()
            gsf.to_MeV()
            samples_ = self.res[i].samples
            df = tranform_nld_gsf(samples_, nld, gsf)
            if i == 0:
                samples = df
            else:
                samples = samples.append(df)

        # plot some samples directly
        res = copy.deepcopy(self.res[i])
        for i, (_, row) in enumerate(samples.sample(n=n_plot).iterrows()):
            nld.values *= i
            res.nld = row["nld"]
            res.pars = row.to_dict()

            # workaround for the tuple (currently just a float)
            keys_workaround = ["T", "D0"]
            for key in keys_workaround:
                res.pars[key] = [res.pars[key], np.nan]

            normalizer_gsf._gsf = row["gsf"]
            normalizer_gsf.extrapolate(row["gsf"])
            res.gsf = row["gsf"]
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

        # get median, 1 sigma, ...
        array = samples[["nld", "gsf"]].to_numpy()
        out = np.zeros((2, len(array), len(array[0, 0].E)))
        indexed = enumerate(array)
        def vec_to_values(x, out):  # noqa
            idx, vec = x
            out[0, idx] = vec[0].values
            out[1, idx] = vec[1].values
        np.fromiter(map(vec_to_values, indexed, repeat(out)), dtype=float)
        stat_nld = pd.DataFrame(out[0, :, :])
        stat_gsf = pd.DataFrame(out[1, :, :])

        low, high = [0.16, 0.84]
        stat_nld = pd.DataFrame({'median': stat_nld.median(),
                                 'low': stat_nld.quantile(low, axis=0),
                                 'high': stat_nld.quantile(high, axis=0)})

        stat_gsf = pd.DataFrame({'median': stat_gsf.median(axis=0),
                                 'low': stat_gsf.quantile(low, axis=0),
                                 'high': stat_gsf.quantile(high, axis=0)})

        stat_nld["E"] = nld.E
        stat_gsf["E"] = gsf.E

        stat_nld.plot(x="E", y="median", ax=ax[0], legend=False)
        stat_gsf.plot(x="E", y="median", ax=ax[1], legend=False, label="")
        ax[0].fill_between(stat_nld.E, stat_nld["low"], stat_nld["high"],
                           alpha=0.3,
                           label=f"{(high-low)*100:.0f}% credibility interval")
        lines = ax[1].fill_between(stat_gsf.E, stat_gsf["low"],
                                   stat_gsf["high"],
                                   alpha=0.3)

        # for the extrapolations. Start with nld
        x = np.linspace(nld.E[-1], normalizer_gsf.norm_pars.Sn[0], num=20)
        array = samples[["T", "D0"]].to_numpy()
        out = np.zeros((len(array), len(x)))
        indexed = enumerate(array)
        def to_values(a, out):  # noqa
            idx, val = a
            out[idx] = normalizer_nld.curried_model(T=val[0],
                                                    D0=val[1],
                                                    E=x)
        np.fromiter(map(to_values, indexed, repeat(out)), dtype=float)
        stat = pd.DataFrame(out[:, :])
        stat = pd.DataFrame({'median': stat.median(),
                             'low': stat.quantile(low, axis=0),
                             'high': stat.quantile(high, axis=0)})
        ax[0].fill_between(x, stat["low"], stat["high"],
                           alpha=0.3, color=lines.get_facecolor())

        # gsf extrapolation
        xlow = np.linspace(0.001, gsf.E[0], num=20)
        xhigh = np.linspace(gsf.E[-1], normalizer_gsf.norm_pars.Sn[0], num=20)
        array = samples["gsf"].to_numpy()
        out = np.zeros((2, len(array), len(x)))
        indexed = enumerate(array)
        def to_values(a, out):  # noqa
            idx, val = a
            low, high = normalizer_gsf.extrapolate(val, E=[xlow, xhigh])
            out[0, idx] = low.values
            out[1, idx] = high.values

        np.fromiter(map(to_values, indexed, repeat(out)), dtype=float)
        for i, arr in enumerate([out[0, :, :], out[1, :, :]]):
            stat = pd.DataFrame(arr)
            stat = pd.DataFrame({'median': stat.median(),
                                 'low': stat.quantile(low, axis=0),
                                 'high': stat.quantile(high, axis=0)})
            ax[1].fill_between(xlow if i == 0 else xhigh,
                               stat["low"], stat["high"],
                               alpha=0.3, color=lines.get_facecolor())

        if add_figlegend:
            fig.legend(loc=9, ncol=4, frameon=True)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

        return fig, ax


def tranform_nld_gsf(samples: dict, nld=None, gsf=None,
                     N_max: int = 100,
                     random_state=np.random.RandomState(65489)):
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
        `nld_trans` (list) and/or `gsf_trans` (list)
        and selected (dict[str, float]): Transformed `nld` and or `gsf`, depending on what input is given; and the selected samples.

    """

    # Need to sweep though multinest samples in random order
    # as they are ordered with decreasing likelihood by default
    for key, value in samples.items():
        N_multinest = len(value)
        break
    randlist = np.arange(N_multinest)
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
