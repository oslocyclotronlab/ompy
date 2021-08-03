import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import itertools
from typing import List, Optional, Any, Tuple, Union, Callable
from operator import xor
import pandas as pd
from scipy.stats import norm as scipynorm
import copy
from itertools import repeat
from pathos.multiprocessing import ProcessPool
from pathos.helpers import cpu_count
from pathlib import Path

from .abstract_normalizer import AbstractNormalizer
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


class EnsembleNormalizer(AbstractNormalizer):
    """Normalizes NLD nad γSF extracted from the ensemble

    Usage:
      The calling syntax can be either to normalize simultaneously::

        EnsembleNormalizer(extractor=...,
                           normalizer_simultan=...)

      , or to normalize sequentially::

          EnsembleNormalizer(extractor=...,
                               normalizer_nld=...,
                               normalizer_gsf=...)

    Note:
        If one should add a functionality that depends on random numbers
        withing the parallelized loop make sure to use the random generator
        exposed via the arguments (see Ensemble class for an example). If one
        uses np.random instead, this will be the same an exact copy for each
        process. Note that this is not an issue for multinest seach routine,
        which is anyhow seeded by default as implemented in ompy.

    Attributes:
        extractor (Extractor): Extractor instance
        normalizer_nld (NormalizerNLD): NormalizerNLD instance
        normalizer_gsf (NormalizerGSF): NormalizerGSF instance
        normalizer_simultan (NormalizerSimultan): NormalizerSimultan instance
        res (List[ResultsNormalized]): List of the results
        nprocesses (int): Number of processes for multiprocessing.
            Defaults to number of available cpus-1 (with mimimum 1).
    """
    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, *, extractor: Extractor,
                 normalizer_nld: Optional[NormalizerNLD] = None,
                 normalizer_gsf: Optional[NormalizerGSF] = None,
                 normalizer_simultan: Optional[NormalizerSimultan] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False):
        """
        Args:
            extractor (Extractor): Extractor instance
            normalizer_nld (NormalizerNLD, optional): NormalizerNLD instance
            normalizer_gsf (NormalizerGSF, optional): NormalizerGSF instance
            normalizer_simultan (NormalizerSimultan, optional):
                NormalizerSimultan instance
        """
        super().__init__(regenerate)
        self.extractor = extractor

        self.normalizer_nld = copy.deepcopy(normalizer_nld)
        self.normalizer_gsf = copy.deepcopy(normalizer_gsf)

        self.normalizer_simultan = copy.deepcopy(normalizer_simultan)

        self.nprocesses: int = cpu_count()-1 if cpu_count() > 1 else 1

        self.res: Optional[List[ResultsNormalized]] = None

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

    def normalize(self) -> None:
        """ Normalize ensemble """
        if not self.regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

        assert xor((self.normalizer_nld is not None
                    and self.normalizer_gsf is not None),
                   self.normalizer_simultan is not None), \
            "Either 'normalizer_nld' and 'normalizer_gsf' must be set, or " \
            "normalizer_simultan"

        gsfs = self.extractor.gsf
        nlds = self.extractor.nld

        self.LOG.info(f"Start normalization with {self.nprocesses} cpus")
        pool = ProcessPool(nodes=self.nprocesses)
        N = len(nlds)
        iterator = pool.imap(self.step, range(N), nlds, gsfs)
        self.res = list(tqdm(iterator, total=N))
        pool.close()
        pool.join()
        pool.clear()

        self.save()

    def step(self, i: int, nld: Vector, gsf: Vector):
        """ Normalization step for each ensemble member

        Args:
            i (int): Loop number
            nld (Vector): NLD before normalization
            gsf (Vector): gsf before normalization

        Returns:
            res (ResultsNormalized): results (/parameters) of normalization
        """
        self.LOG.info(f"\n\n---------\nNormalizing #{i}")
        nld.copy()
        nld.cut_nan()

        gsf.copy()
        gsf.cut_nan()

        if self.normalizer_simultan is not None:
            res = self.normalizeSimultan(i, nld=nld, gsf=gsf)
        else:
            res = self.normalizeStagewise(i, nld=nld, gsf=gsf)

        return res

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
        self.normalizer_simultan._save_instance = False
        self.normalizer_simultan.regenerate = True
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
        for norm in [self.normalizer_nld, self.normalizer_gsf]:
            norm._save_instance = False
            norm.regenerate = True

        self.normalizer_nld.normalize(nld=nld, num=num)
        self.normalizer_gsf.normalize(normalizer_nld=self.normalizer_nld,
                                      gsf=gsf, num=num)

        # sample B from the gaussian uncertainty for each nld
        B = self.normalizer_gsf.res.pars["B"]
        N = len(self.normalizer_gsf.res.samples["A"])
        self.normalizer_gsf.res.samples["B"] = scipynorm.rvs(loc=B[0],
                                                             scale=B[1],
                                                             size=N)
        return self.normalizer_gsf.res

    def plot(self, ax: Tuple[Any, Any] = None,
             add_figlegend: bool = True,
             n_plot: bool = 5,
             plot_model_stats: bool = False,
             random_state: Optional[np.random.RandomState] = None,
             return_stats: bool = False,
             **kwargs) -> Union[Tuple[Any, Any],
                                Tuple[Any, Any, Tuple[Any, Any]]]:
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
            return_stats: Whether to return vector stats (percentiles)
            **kwargs: Description

        TODO:
            - Refactor code
            - Could not find out how to not plot dublicate legend entries,
              thus using a workaround
            - Checks if extrapolating where nld or gsf is np.nan

        Returns:
            Tuple: If `return_stats=False`, returns `fig, ax`,
                otherwise `fig, ax, (stats_nld, stats_gsf)`
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax[0].figure

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
                            normalizer_gsf=normalizer_gsf,
                            n_plot=n_plot)

        # unify Egrid (and values) in case vectors are not equally long
        self.samples_unify_E(samples["nld"])
        self.samples_unify_E(samples["gsf"])

        # get median, 1 sigma, ...
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        percentiles = [0.16, 0.84]
        _, stats_nld, stats_gsf = self.plot_vector_stats(ax, samples,
                                                         percentiles,
                                                         colors[1])

        if plot_model_stats or return_stats:
            if plot_model_stats:
                ax_stats = ax
            else:
                fig, ax_stats = plt.subplots(2, 1)  # dummy
            Emin = samples["nld"].iloc[0].E[-1]
            x = np.linspace(Emin, normalizer_nld.norm_pars.Sn[0], num=20)
            stats_nld_model = \
                self.plot_nld_ext_stats(ax_stats[0], x=x, samples=samples,
                                        normalizer_nld=normalizer_nld,
                                        percentiles=percentiles,
                                        color=colors[2],
                                        label="model")

            E = samples["gsf"].iloc[0].E
            xlow = np.linspace(0.001, E[0], num=20)
            xhigh = np.linspace(E[-1], normalizer_gsf.norm_pars.Sn[0], num=20)
            stats_gsf_model = \
                self.plot_gsf_ext_stats(ax_stats[1], xlow=xlow, xhigh=xhigh,
                                        samples=samples,
                                        normalizer_gsf=normalizer_gsf,
                                        percentiles=percentiles,
                                        color=colors[2])

        if add_figlegend:
            fig.legend(loc=9, ncol=4, frameon=True)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

        if return_stats:
            return fig, ax, (stats_nld, stats_gsf, stats_nld_model,
                             stats_gsf_model)
        else:
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

        # dummy to draw axis
        n_plot_ = n_plot if n_plot > 0 else 1

        markers = itertools.cycle(('o', 'x', 'P', 'v', '^', '<', '>', '8',
                                   's', 'p', '*', 'h', 'H', 'D', 'd', 'X'))

        res = copy.deepcopy(self.res[0])  # dummy for later
        selection = samples.sample(n=n_plot_, random_state=random_state)
        for i, (_, row) in enumerate(selection.iterrows()):
            res.nld = row["nld"]
            res.gsf = row["gsf"]
            res.pars = row.to_dict()

            # workaround for the tuple (currently just a float)
            keys_workaround = ["T", "Eshift"]
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

            marker = next(markers)
            normalizer_nld.plot(ax=ax[0], results=res,
                                add_label=add_label, alpha=1/n_plot_,
                                add_figlegend=False,
                                plot_fitregion=plot_fitregion,
                                marker=marker, linestyle="--")
            normalizer_gsf.plot(ax=ax[1], results=res, add_label=False,
                                alpha=1/n_plot_,
                                add_figlegend=False,
                                plot_fitregion=plot_fitregion,
                                marker=marker, linestyle="--")

        if n_plot == 0:  # remove lines if dummy only
            l_keep = []
            for l in ax[0].lines:
                if l._label in ["known levels", "_nld(Sn)"]:
                    l_keep.append(l)
            ax[0].lines = [*l_keep]
            ax[1].lines = []

    def samples_unify_E(self, df: pd.DataFrame) -> None:
        """ Get nlds (or gsfs) on common energy grid, if diff. lengths

        After applying, DataFrame with vectors are on common
        energy grid. Missing values filled with np.nan.

        Args:
            df: DataFrame collumn with vectors to be put on unified energy grid
        """
        extend = np.array(list(map(vec_extend, df)))

        # if equal already: no need to proceede
        if np.equal(extend[0], extend).all():
            return None

        iEmin = np.argmin(extend[:, 0])
        iEmax = np.argmax(extend[:, 1])

        Eunion = np.union1d(df.iloc[iEmin].E, df.iloc[iEmax].E)

        # define helper function
        def vec_extend_values(vec, Eunion):  # noqa
            Eold = vec.E
            interEunion = np.in1d(Eunion, Eold, assume_unique=True)
            interEold = np.in1d(Eold, Eunion, assume_unique=True)
            vec.E = Eunion
            val_union = np.full_like(Eunion, np.nan)
            val_union[interEunion] = vec.values[interEold]
            vec.values = val_union

        # map function to each element
        array = df.to_numpy()
        np.array([vec_extend_values(xi, Eunion) for xi in array])

    @staticmethod
    def plot_vector_stats(ax: Tuple[Any, Any],
                          samples: pd.DataFrame,
                          percentiles: Tuple[float, float],
                          color: Any) -> Tuple[Any,
                                               pd.DataFrame, pd.DataFrame]:
        """ Helper for plotting of stats from a vector

        Args:
            ax: Axes to plot on
            samples: Samples of (nld, gsf, transfromation parameters)
            percentiles: Lower and upper percentile to plot the shading
            color (Any): Color of nld and gsf

        Returns:
            Lines of fill between, and stats DataFrame of nld and gsf
        """

        # workaround as DataFrame changes limits & labels
        lim_ax0 = [ax[0].get_xlim(), ax[0].get_ylim()]
        lim_ax1 = [ax[1].get_xlim(), ax[1].get_ylim()]
        label_ax0 = [ax[0].get_xlabel(), ax[0].get_ylabel()]
        label_ax1 = [ax[1].get_xlabel(), ax[1].get_ylabel()]

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
        stats_nld.plot(x="x", y="median", ax=ax[0], legend=False,
                       color=color)

        df = samples["gsf"]
        E = df.iloc[0].E
        stats_gsf = EnsembleNormalizer.stats_from_df(df, fmap=vec_to_values,
                                            shape_out=(len(df), len(E)),
                                            percentiles=percentiles)  # noqa
        stats_gsf["x"] = df.iloc[0].E
        stats_gsf.plot(x="x", y="median", ax=ax[1], legend=False,
                       color=color)

        pc_diff = percentiles[1] - percentiles[0]
        label = fr"{(pc_diff)*100:.0f}\% credibility interval"
        ax[0].fill_between(stats_nld.x, stats_nld["low"], stats_nld["high"],
                           alpha=0.3,
                           label=label)
        lines = ax[1].fill_between(stats_gsf.x, stats_gsf["low"],
                                   stats_gsf["high"],
                                   alpha=0.3)

        ax[0].set_xlim(lim_ax0[0])
        ax[0].set_ylim(lim_ax0[1])
        ax[1].set_xlim(lim_ax1[0])
        ax[1].set_ylim(lim_ax1[1])

        ax[0].set_xlabel(label_ax0[0])
        ax[0].set_ylabel(label_ax0[1])
        ax[1].set_xlabel(label_ax1[0])
        ax[1].set_ylabel(label_ax1[1])

        return lines, stats_nld, stats_gsf

    @staticmethod
    def stats_from_df(df: pd.DataFrame,
                      fmap: Callable[[Vector, np.array], None],
                      shape_out: Tuple[int, int],
                      percentiles: Tuple[float, float]) -> pd.DataFrame:
        """Helper to get median, 68% or similar from a collection of Vectors

        Args:
            df: DataFrame of Vectors
            fmap: Applied to each row of df
            shape_out: output shape
            percentiles: Upper and lower percentiles for the stats
                (eg. 16 and 84% for something like 1 sigma)

        Returns:
            DataFrame with collumns ['median', 'low', 'high'] and entries for
            each energy of the Vectors.
        """
        array = df.to_numpy()
        out = np.zeros(shape_out)
        indexed = enumerate(array)
        # apply fmap to each row of the DataFrame
        np.fromiter(map(fmap, indexed, repeat(out)), dtype=float)
        stats = pd.DataFrame(out[:, :])
        stats = pd.DataFrame({'median': stats.median(),
                              'low': stats.quantile(percentiles[0], axis=0),
                              'high': stats.quantile(percentiles[1], axis=0)})
        return stats

    @staticmethod
    def plot_nld_ext_stats(ax: Any, *, x: np.ndarray,
                           samples: pd.DataFrame,
                           normalizer_nld: NormalizerNLD,
                           percentiles: Tuple[float, float],
                           **kwargs) -> pd.DataFrame:
        """Helper for plotting statistics of the nld extrapolation

        Args:
            ax: The matplotlib axis to plot onto.
            x: x-axis values (Energies)
            samples: Samples of (nld, gsf, transfromation parameters)
            normalizer_nld: NormalizerNLD instance.
            percentiles: Lower and upper percentile to plot
                the shading
            **kwargs: Additional keyword arguments for the plotting

        Returns:
            DataFrame with collumns ['median', 'low', 'high'] and entries for
            each energy of the Vectors.
        """
        # define helper function
        def to_values(a, out):  # noqa
            idx, val = a
            out[idx] = normalizer_nld.model(E=x, T=val[0], Eshift=val[1])

        df = samples[["T", "Eshift"]]
        stats = EnsembleNormalizer.stats_from_df(df, fmap=to_values,
                                                 shape_out=(len(df), len(x)),
                                                 percentiles=percentiles)
        ax.plot(x, stats["median"], **kwargs)
        ax.fill_between(x, stats["low"], stats["high"],
                        alpha=0.3, **kwargs)
        return stats

    @staticmethod
    def plot_gsf_ext_stats(ax: Any, *, xlow: np.ndarray, xhigh: np.ndarray,
                           samples: pd.DataFrame,
                           normalizer_gsf: NormalizerGSF,
                           percentiles: Tuple[float, float],
                           color: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Helper for plotting statistics of the gsf extrapolations

        Args:
            ax: The matplotlib axis to plot onto.
            xlow: x-axis values (Energies) of the lower extrapolation
            xhigh: x-axis values (Energies) of the higher extrapolation
            samples: Samples of (nld, gsf, transfromation parameters)
            normalizer_gsf: NormalizerNLD instance.
            percentiles: Lower and upper percentile to plot
                the shading
            **kwargs: Additional keyword arguments for the plotting

        Returns:
            Tuple of DataFrames with collumns ['median', 'low', 'high'] and
            entries for each energy of the Vectors. First entry is for the
            lower extrapolation, secondentry is for the higher extrapolation
        """

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
        # stats for upper and lower model
        stats = []
        for i, arr in enumerate([out[0, :, :], out[1, :, :]]):
            stat = pd.DataFrame(arr)
            stat = pd.DataFrame({'median': stat.median(),
                                 'low': stat.quantile(low, axis=0),
                                 'high': stat.quantile(high, axis=0)})
            ax.plot(xlow if i == 0 else xhigh, stat["median"], color=color)
            ax.fill_between(xlow if i == 0 else xhigh,
                            stat["low"], stat["high"],
                            alpha=0.3, color=color)
            stats.append(stat)
        return stats

    def save_results_txt(self, path: Optional[Union[str, Path]] = None,
                         suffix: str = None):
        """ Save results as txt

        Uses a folder to save nld, gsf, and the samples (converted to an array)

        Args:
            path: The path to the save directory. If the
                value is None, 'self.path' will be used.
        """
        path = Path(path) if path is not None else Path(self.path)
        path.mkdir(exist_ok=True, parents=True)
        for i, res in enumerate(self.res):
            super().save_results_txt(path, nld=res.nld, gsf=res.gsf,
                                     samples=res.samples, suffix=i)


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
        DataFrame with randomly selected samples of nld, gsf and the
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


def vec_extend(vector: Vector) -> Tuple[float, float]:
    """ Get the lowest and highest energy of the vector

    Assumes that the energy array is sorted.

    Args:
        vector: input Vector

    Returns:
        Tuple of lowest and highest energies of the Vector"""
    return vector.E[0], vector.E[-1]
