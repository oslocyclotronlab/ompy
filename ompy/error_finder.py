import logging
import numpy as np
import pymc3 as pm
import termtables as tt

from numpy import ndarray
from typing import Optional, Union, Any, Tuple, List, Dict


from .vector import Vector
from .matrix import Matrix
from .dist import FermiDirac


def all_equal(iterator):
    """ Check if all elements in an iterator are equal.
    """

    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


class ErrorFinder:
    """ Find the relative errors from an ensemble of NLDs and γSFs.

    This class uses pyMC3 to calculate the relative errors of the data points
    in the extracted NLDs and γSFs. The class implements two different models,
    one logarithmic and one linear. The logarithmic model will usually be more
    stable and should be used if the linear doesn't converge.

    Attributes:
        algorithm (string): Indicate what algorithm to use, either 'log' or
            'lin'. Default is 'lin'
        seed (int): Seed used in the pyMC3 sampling.
        options (dict): The pymc3.sample options to use.
        testvals (dict): A dictionary with testvalues to feed to pymc3 when
            declaring the prior distributions. This can probably be left to
            the default, but could be changed if needed.
    """

    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, algorithm: Optional[str] = 'log',
                 options: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = 7392):
        """ Initialize ErrorFinder. See attributes above for arguments.

        Raises:
            NotImplementedError if algorithm isn't recognized.

        """

        if not (algorithm.lower() == 'lin' or algorithm.lower() == 'log'):
            raise NotImplementedError(
                f"Algorithm '{algorithm}' not implemented")

        self.algorithm = algorithm.lower()
        self.options = {} if options is None else options
        self.options.setdefault("return_inferencedata", False)
        self.seed = seed
        self.testvals = {'σ_D': 0.25, 'σ_F': 0.25, 'σ_α': 0.05,
                         'σ_A': 0.25, 'σ_B': 0.25, 'A': 1.0, 'B': 1.0}

    def evaluate(self, nlds: List[Vector],
                 gsfs: List[Vector],
                 median: Optional[bool] = False,
                 trace: Optional[bool] = False
                 ) -> Union[Tuple[Vector, Vector], Any]:
        """ Evaluate the model and find the relative errors
        of the NLDs and γSFs passed to the function.

        Args:
            nlds (List[Vector]): List of the NLDs in an ensemble
            gsfs (List[Vector]): List of the γSFs in an ensemble
            median (bool, optional): If the mean of the relative error should
                be used rather than the mean. Default is False.
            trace (bool, optional): If the sample trace should
                be returned or not.
        Returns:
            The relative error of the nuclear level density and
            gamma strength function.
            Optionally returns the full sample trace if `trace` is true.

        Raises:
            ValueError: If length of `nlds` doesn't match the length of `gsfs`.
            pyMC3 may raise errors. Please report if such errors occurs.
        """
        algo = None
        if self.algorithm == 'lin':
            algo = self.linear
            return self.linear(*self.condition_data(nlds, gsfs), trace)
        elif self.algorithm == 'log':
            algo = self.logarithm
            return self.logarithm(*self.condition_data(nlds, gsfs), trace)
        else:
            raise NotImplementedError(
                f"Algorithm '{algorithm}' not implemented")

        return algo(*self.condition_data(nlds, gsfs),
                    median=median, trace=trace)

    def logarithm(self, E_nld: ndarray, E_gsf: ndarray,
                  nld_obs: ndarray, gsf_obs: ndarray,
                  coef_nld: ndarray, coef_gsf: ndarray,
                  vals_nld: ndarray, vals_gsf: ndarray,
                  median: Optional[bool] = False, trace: Optional[bool] = False
                  ) -> Union[Tuple[Vector, Vector], Any]:
        """ Calculate the relative errors of the NLD and γSF points using a
        logarithmic model.

        Args:
            E_nld (ndarray): Energy points of the NLD
            E_gsf (ndarray): Energy points of the γSF
            nld_obs (ndarray): The observation tensor of the NLD
            gsf_obs (ndarray): The observation tensor of the γSF
            coef_nld (ndarray): Indices needed for broadcasting
            coef_gsf (ndarray): Indices needed for broadcasting
            vals_nld (ndarray): Indices needed for broadcasting
            vals_gsf (ndarray): Indices needed for broadcasting
            median (bool, optional): If the resulting relative errors should be
                the mean or the median.
            trace (bool, optional): If the trace from the pyMC3 sampling should
                be returned. Useful for debugging.
        """

        N, _, M_nld = nld_obs.shape
        _, _, M_gsf = gsf_obs.shape

        self.LOG.info("Starting pyMC3 - logarithmic model")

        with pm.Model() as model:

            σ_D = pm.HalfFlat("σ_D", testval=self.testvals['σ_D'])
            σ_F = pm.HalfFlat("σ_F", testval=self.testvals['σ_F'])
            σ_α = pm.HalfFlat("σ_α", testval=self.testvals['σ_α'])

            D = pm.Normal("D", mu=0., sigma=σ_D, shape=[N, N-1])[:, coef_nld]
            F = pm.Normal("F", mu=0., sigma=σ_F, shape=[N, N-1])[:, coef_gsf]
            α = pm.Normal("α", mu=0., sigma=σ_α, shape=[N, N-1])

            σ_ρ = FermiDirac("σ_ρ", lam=10., mu=1., shape=M_nld)[vals_nld]
            σ_f = FermiDirac("σ_f", lam=10., mu=1., shape=M_gsf)[vals_gsf]

            μ_ρ = D + α[:, coef_nld] * E_nld[vals_nld]
            μ_f = F + α[:, coef_gsf] * E_gsf[vals_gsf]

            ρ_obs = pm.Normal("ρ_obs", mu=μ_ρ, sigma=σ_ρ,
                              observed=np.log(nld_obs))
            f_obs = pm.Normal("f_obs", mu=μ_f, sigma=σ_f,
                              observed=np.log(gsf_obs))

            # Perform the sampling
            trace = pm.sample(random_seed=self.seed, **self.options)

        self.display_results(trace)

        mid = np.median if median else np.mean
        nld_rel_err = om.Vector(E=E_nld, values=mid(trace['σ_ρ'], axis=0),
                                std=np.std(trace['σ_ρ'], axis=0), units='MeV')
        gsf_rel_err = om.Vector(E=E_gsf, values=mid(trace['σ_f'], axis=0),
                                std=np.std(trace['σ_f'], axis=0), units='MeV')
        if trace:
            return nld_rel_err, gsf_rel_err, trace
        else:
            return nld_rel_err, gsf_rel_err

    def linear(self, E_nld: ndarray, E_gsf: ndarray,
               nld_obs: ndarray, gsf_obs: ndarray,
               coef_nld: ndarray, coef_gsf: ndarray,
               vals_nld: ndarray, vals_gsf: ndarray,
               median: Optional[bool] = False, trace: Optional[bool] = False
               ) -> Union[Tuple[Vector, Vector], Any]:
        """ Calculate the relative errors of the NLD and γSF points using a
        linear model.

        Args:
            E_nld (ndarray): Energy points of the NLD
            E_gsf (ndarray): Energy points of the γSF
            nld_obs (ndarray): The observation tensor of the NLD
            gsf_obs (ndarray): The observation tensor of the γSF
            coef_nld (ndarray): Indices needed for broadcasting
            coef_gsf (ndarray): Indices needed for broadcasting
            vals_nld (ndarray): Indices needed for broadcasting
            vals_gsf (ndarray): Indices needed for broadcasting
            median (bool, optional): If the resulting relative errors should be
                the mean or the median.
            trace (bool, optional): If the trace from the pyMC3 sampling should
                be returned. Useful for debugging.
        """

        N, M_nld = nld_obs.shape
        _, M_gsf = gsf_obs.shape

        nld_obs = np.log(nld_obs)
        gsf_obs = np.log(gsf_obs)

        self.LOG.info("Starting pyMC3 - linear model")

        with pm.Model() as model:

            σ_A = pm.HalfFlat("σ_A", testval=self.testvals['σ_A'])
            σ_B = pm.HalfFlat("σ_B", testval=self.testvals['σ_B'])
            σ_α = pm.HalfFlat("σ_α", testval=self.testvals['σ_α'])

            A = pm.TruncatedNormal("A", mu=0., sigma=σ_A, shape=[N, N-1],
                                   testval=self.testvals['A'])[:, coef_nld]
            B = pm.TruncatedNormal("B", mu=0., sigma=σ_B, shape=[N, N-1],
                                   testval=self.testvals['B'])[:, coef_gsf]
            α = pm.Normal("α", mu=0., sigma=σ_α, shape=[N+1, N])

            σ_ρ = FermiDirac("σ_ρ", lam=10., mu=1., shape=M_nld)[vals_nld]
            σ_f = FermiDirac("σ_f", lam=10., mu=1., shape=M_gsf)[vals_gsf]

            μ_ρ = A * pm.math.exp(α[:, coef_nld] * E_nld[vals_nld])
            μ_f = B * pm.math.exp(α[:, coef_gsf] * E_gsf[vals_gsf])

            ρ_obs = pm.Normal("ρ_obs", mu=μ_ρ, sigma=σ_ρ*μ_ρ,
                              observed=nld_obs)
            f_obs = pm.Normal("f_obs", mu=μ_f, sigma=σ_f*μ_f,
                              observed=gsf_obs)

            # Perform the sampling
            trace = pm.Sample(random_seed=self.seed, **self.options)

        self.display_results(trace)

        mid = np.median if median else np.mean
        nld_rel_err = om.Vector(E=E_nld, values=mid(trace['σ_ρ'], axis=0),
                                std=np.std(trace['σ_ρ'], axis=0), units='MeV')
        gsf_rel_err = om.Vector(E=E_gsf, values=mid(trace['σ_f'], axis=0),
                                std=np.std(trace['σ_f'], axis=0), units='MeV')

        if trace:
            return nld_rel_err, gsf_rel_err, trace
        else:
            return nld_rel_err, gsf_rel_err

    def condition_data(self, _nlds: List[Vector], _gsfs: List[Vector],
                       ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """ Ensures that data are copied and that all values are in correct
        units. It also checks that all lengths are correct.

        Args:
            nlds (List[Vector]): List of the NLDs in an ensemble
            gsfs (List[Vector]): List of the γSFs in an ensemble

        Returns:
            Tuple of nld energy points, gsf energy points and the observed
            matrix for the NLD and γSF.

        Raises:
            IndexError: If the length of members of the ensemble have an
                equal number of points in the NLD and γSF.

        Warnings:
            Will raise a warning if there are members in the ensemble that
            contains one or more nan's. This is mostly to inform the user
            and shouldn't be an issue later on.

        TODO:
            - Mitigation when cut_nan() results in different length vectors
              of different members of the ensemble. (e.g. len(nld[0]) = 10,
              len(nld[1]) = 9).

        """

        # Ensure that the length of the NLDs and GSFs are equal
        assert len(_nlds) == len(_gsfs), \
            "Length of nlds and gsfs doesn't match"

        N = len(_nlds)

        self.LOG.debug("Processing an ensemble with %d members", N)

        # Make copy to ensure that we don't overwrite anything
        nlds = [nld.copy() for nld in _nlds]
        gsfs = [gsf.copy() for gsf in _gsfs]

        self.LOG.debug("Before removing nan: %d NLD values and %d GSF values",
                       len(nlds[0]), len(gsfs[0]))

        # Ensure that we have in MeV and that there isn't any nan's.
        is_nan = False
        for nld, gsf in zip(nlds, gsfs):
            nld.to_MeV()
            gsf.to_MeV()
            is_nan = np.isnan(nld.values).any() or np.isnan(gsf.values).any()
            nld.cut_nan()
            gsf.cut_nan()
        if is_nan:
            self.LOG.warning(f"NLDs and/or γSFs contains nan's."
                             " They will be removed")

        self.LOG.debug("After removing nan: %d NLD values and %d GSF values",
                       len(nlds[0]), len(gsfs[0]))

        if not (all_equal([len(nld) for nld in nlds]) or
                all_equal([len(gsf) for gsf in gsfs])):
            raise IndexError("NLDs and/or γSFs have different "
                             "length. Consider setting reduce to True "
                             "Note this will result in a partial result")

        # Next we can extract the important bits
        E_nld = nlds[0].E.copy()
        E_gsf = gsfs[0].E.copy()

        M_nld = len(E_nld)
        M_gsf = len(E_gsf)

        nld_obs = []
        gsf_obs = []

        idx_nld = np.tile(np.arange(M_nld), (N-1, 1))
        idx_gsf = np.tile(np.arange(M_gsf), (N-1, 1))

        for n, (nld, gsf) in enumerate(zip(nlds, gsfs)):

            # Make a copy of the values arrays
            nld_array = [nld.values.copy() for nld in nlds]
            gsf_array = [gsf.values.copy() for gsf in gsfs]

            del nld_array[n]
            del gsf_array[n]

            nld_array = np.array(nld_array)
            gsf_array = np.array(gsf_array)

            # Set the observed data
            nld_obs.append(nld.values[idx_nld]/nld_array)
            gsf_obs.append(gsf.values[idx_gsf]/gsf_array)

        nld_obs = np.array(nld_obs)
        gsf_obs = np.array(gsf_obs)

        idx_coef_nld = np.repeat(np.arange(N-1), M_nld).reshape(N-1, M_nld)
        idx_coef_gsf = np.repeat(np.arange(N-1), M_gsf).reshape(N-1, M_gsf)

        idx_vals_nld = np.array([idx_nld] * N)
        idx_vals_gsf = np.array([idx_gsf] * N)

        return E_nld, E_gsf, nld_obs, gsf_obs, \
            idx_coef_nld, idx_coef_gsf, idx_vals_nld, idx_vals_gsf

    def display_results(self, trace: Any) -> None:
        """ Print the results from the pyMC3 inference to the log.
        """

        def line(idx) -> list:
            mean = trace[idx].mean()
            sigma = trace[idx].std()
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            return [idx, fmt % mean, fmt % sigma]

        def lines(idx) -> list:
            _, M = trace[idx].shape
            vals = []
            for m in range(M):
                mean = trace[idx][:, m].mean()
                sigma = trace[idx][:, m].std()
                i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                vals += ['%s[%d]' % (idx, m), fmt % mean, fmt % sigma]
            return vals
        values = []
        try:
            for idx in ['A', 'B', 'α', 'σ_A', 'σ_B', 'σ_α']:
                values += line(idx)
        except KeyError:
            for idx in ['D', 'F', 'α', 'σ_D', 'σ_F', 'σ_α']:
                values += line(idx)

        values += lines('σ_ρ')
        values += lines('σ_f')

        self.LOG.info("Inference results:\n%s",
                      tt.to_string(values, header=['', 'mean', 'std']))
