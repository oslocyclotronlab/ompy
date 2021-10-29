import logging
import numpy as np
import pymc3 as pm
import termtables as tt

from numpy import ndarray
from typing import Optional, Union, Any, Tuple, List, Dict


from .vector import Vector
from .matrix import Matrix
from .dist import FermiDirac


class ErrorFinder:
    """ Find the relative errors from an ensemble of NLDs and γSFs.

    This class uses pyMC3 to calculate the relative errors of the data points
    in the extracted NLDs and γSFs. The class implements two different models,
    one logarithmic and one linear. The logarithmic model will usually be more
    stable and should be used if the linear doesn't converge.

    Attributes:
        algorithm (string): Indicate what algorithm to use Currently only 'log'
            is implemented. Default is 'log'
        seed (int): Seed used in the pyMC3 sampling.
        options (dict): The pymc3.sample options to use.
        testvals (dict): A dictionary with testvalues to feed to pymc3 when
            declaring the prior distributions. This can probably be left to
            the default, but could be changed if needed.
        trace: The trace of the inference data. None if the class hasn't
            ran.
    TODO:
        - Trace should always be saved. Calculations can take hours!
        - Refactor the linear model (maybe remove?)
        - Refactor how data are conditioned
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

        if not (algorithm.lower() == 'log'):
            raise NotImplementedError(
                f"Algorithm '{algorithm}' not implemented")

        self.algorithm = algorithm.lower()
        self.options = {} if options is None else options
        self.options.setdefault("return_inferencedata", False)
        self.seed = seed
        self.testvals = {'σ_D': 0.25, 'σ_F': 0.25, 'σ_α': 0.05,
                         'σ_A': 0.25, 'σ_B': 0.25, 'A': 1.0, 'B': 1.0}
        self.prior_parameters = {'σ_ρ': {'lam': 10., 'mu': 1.},
                                 'σ_f': {'lam': 10., 'mu': 1.}}

    def __call__(self, *args) -> Union[Tuple[Vector, Vector], Any]:
        """ Wrapper for evaluate """
        return self.evaluate(*args)

    def evaluate(self, nlds: List[Vector],
                 gsfs: List[Vector],
                 median: Optional[bool] = False,
                 full: Optional[bool] = False
                 ) -> Union[Tuple[Vector, Vector], Any]:
        """ Evaluate the model and find the relative errors
        of the NLDs and γSFs passed to the function.

        Args:
            nlds (List[Vector]): List of the NLDs in an ensemble
            gsfs (List[Vector]): List of the γSFs in an ensemble
            median (bool, optional): If the mean of the relative error should
                be used rather than the mean. Default is False.
            full (bool, optional): If the sample trace should
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
        if self.algorithm == 'log':
            algo = self.logarithm
        else:
            raise NotImplementedError(
                f"Algorithm '{algorithm}' not implemented")

        return algo(nlds, gsfs, median=median, full=full)

    def logarithm(self, nlds: List[Vector], gsfs: List[Vector],
                  median: Optional[bool] = False, full: Optional[bool] = False
                  ) -> Union[Tuple[Vector, Vector], Any]:
        """Use Bayesian inference to estimate the relative errors of the NLD
        and GSF from an ensemble. See Ingeberg et al., NIM (TBA)

        Args:
            nlds (List[Vector]): Ensemble of NLDs
            gsfs (List[Vector]): Ensemble of GSFs
            median (bool, optional): If the resulting relative errors should be
                the mean or the median.
            full (bool, optional): If the trace from the pyMC3 sampling should
                be returned. Useful for debugging.
        """

        assert len(nlds) == len(gsfs), \
            "Number of ensemble members of the NLD and GSF doesn't match."
        N = len(nlds)

        E_nld, q_nld = format_data(nlds)
        E_gsf, q_gsf = format_data(gsfs)

        M_nld, M_gsf = len(E_nld), len(E_gsf)
        coef_mask_nld, values_mask_nld = format_mask(N, M_nld)
        coef_mask_gsf, values_mask_gsf = format_mask(N, M_gsf)

        self.LOG.info("Starting pyMC3 inference - logarithmic model")
        self.LOG.debug(f"Inference with an ensemble with N={N} members with "
                       f"{M_nld} NLD bins and {M_gsf} GSF bins.")

        with pm.Model() as model:

            σ_D = pm.HalfFlat("σ_D", testval=self.testvals['σ_D'])
            σ_F = pm.HalfFlat("σ_F", testval=self.testvals['σ_F'])
            σ_α = pm.HalfFlat("σ_α", testval=self.testvals['σ_α'])

            D = pm.Normal("D", mu=0, sigma=σ_D, shape=[N, N-1])[:, coef_mask_nld]  # noqa
            F = pm.Normal("F", mu=0, sigma=σ_F, shape=[N, N-1])[:, coef_mask_gsf]  # noqa
            α = pm.Normal("α", mu=0, sigma=σ_α, shape=[N, N-1])

            σ_ρ = FermiDirac("σ_ρ", lam=self.prior_parameters['σ_ρ']['lam'],
                             mu=self.prior_parameters['σ_ρ']['mu'],
                             shape=M_nld)[values_mask_nld]

            σ_f = FermiDirac("σ_f", lam=self.prior_parameters['σ_f']['lam'],
                             mu=self.prior_parameters['σ_f']['mu'],
                             shape=M_gsf)[values_mask_gsf]

            μ_ρ = D + α[:, coef_mask_nld] * E_nld[values_mask_nld]
            μ_f = F + α[:, coef_mask_gsf] * E_gsf[values_mask_gsf]

            q_ρ = pm.Normal("q_ρ", mu=μ_ρ, sigma=np.sqrt(2)*σ_ρ,
                            observed=np.log(q_nld))
            q_f = pm.Normal("q_f", mu=μ_f, sigma=np.sqrt(2)*σ_f,
                            observed=np.log(q_gsf))

            # Perform the sampling
            trace = pm.sample(random_seed=self.seed, **self.options)
        self.trace = trace
        self.display_results(trace)

        mid = np.median if median else np.mean
        nld_rel_err = Vector(E=E_nld, values=mid(trace['σ_ρ'], axis=0),
                             std=np.std(trace['σ_ρ'], axis=0), units='MeV')
        gsf_rel_err = Vector(E=E_gsf, values=mid(trace['σ_f'], axis=0),
                             std=np.std(trace['σ_f'], axis=0), units='MeV')
        if full:
            return nld_rel_err, gsf_rel_err, trace
        else:
            return nld_rel_err, gsf_rel_err

    def display_results(self, trace: Any) -> None:
        """ Print the results from the pyMC3 inference to the log.
        """

        def line(idx) -> list:
            mean = trace[idx].mean()
            sigma = trace[idx].std()
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            return fmts % (mean, sigma)
        header = []
        values = []
        try:
            for idx in ['A', 'B', 'α', 'σ_A', 'σ_B', 'σ_α']:
                values.append(line(idx))
            header = ['A', 'B', 'α', 'σ_A', 'σ_B', 'σ_α']
        except KeyError:
            for idx in ['D', 'F', 'α', 'σ_D', 'σ_F', 'σ_α']:
                values.append(line(idx))
            header = ['D', 'F', 'α', 'σ_D', 'σ_F', 'σ_α']

        errs = []
        _, M_nld = trace['σ_ρ'].shape
        _, M_gsf = trace['σ_f'].shape
        errs = [[f'{m}', '', ''] for m in range(max([M_nld, M_gsf]))]
        for m in range(M_nld):
            mean = trace['σ_ρ'][:, m].mean() * 100.
            sigma = trace['σ_ρ'][:, m].std() * 100.
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            errs[m][1] = fmts % (mean, sigma)
        for m in range(M_gsf):
            mean = trace['σ_f'][:, m].mean() * 100.
            sigma = trace['σ_f'][:, m].std() * 100.
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            errs[m][2] = fmts % (mean, sigma)

        self.LOG.info("Inference results:\n%s\n%s",
                      tt.to_string([values], header=header),
                      tt.to_string(errs, header=['', 'σ_ρ [%]', 'σ_f [%]']))


def all_equal(iterator):
    """ Check if all elements in an iterator are equal.
    """

    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def remove_nans(vecs: List[Vector]) -> List[Vector]:
    """ Remove all points that are nan's for each member of the ensemble"""

    vecs_no_nan = [vec.copy() for vec in vecs]
    for vec in vecs_no_nan:
        vec.cut_nan(inplace=True)

    return vecs_no_nan


def keep_only(vecs: List[Vector]) -> List[Vector]:
    """ Deletes all the points that are not shared by all vectors in the list
    and returns the list with only the points shared among all vectors.

    Args:
        vecs (List): List of similar vectors
    Returns: List of vectors with only the points that share x-value among all
        of the input vectors.
    """
    vecs = remove_nans(vecs)
    E = [vec.E.copy() for vec in vecs]
    energies = {}
    for vec in vecs:
        for E in vec.E:
            if E not in energies:
                energies[E] = [False] * len(vecs)

    # Next we will add if the point is present or not
    for n, vec in enumerate(vecs):
        for E in vec.E:
            energies[E][n] = True

    keep_energy = []
    for key in energies:
        if np.all(energies[key]):
            keep_energy.append(key)

    vec_all_common = [vec.copy() for vec in vecs]
    for vec in vec_all_common:
        E = []
        values = []
        for e, value in zip(vec.E, vec.values):
            if e in keep_energy:
                E.append(e)
                values.append(value)
        vec.E = np.array(E)
        vec.values = np.array(values)

    return vec_all_common


def format_data(vecs: List[Vector]) -> Tuple[ndarray, ndarray]:
    """Find and build required variables for pymc3 model.
    This function takes an ensemble (NLDs or GSFs) and finds the energy
    bins that are not nan and are shared among all members. It then
    builds a tensor:

        q_{r,i,j} = \ln(\frac{v_{r,j}}{v_{i,j}})

    where i and r ≠ i are the ensemble member index and j is the
    bin index.

    Args:
        vecs (List[Vector]): List of vectors. Typically an ensemble
            of NLDs or GSFs.
    Returns:
        E (array): energy bins shared among all of the input vector
        q (array): Observation tensor
    """
    # Remove all bins that contains nan's
    N = len(vecs)
    vecs = keep_only(vecs)

    E = vecs[0].E.copy()
    if vecs[0].units == 'keV':
        E /= 1.0e3
    M = len(E)

    # Masking to get proper broadcasting of shapes
    mask = np.tile(np.arange(M, dtype=int), (N-1, 1))

    # Next we will create the observation tensor
    q = []
    for r in range(N):
        not_r = []
        for i in range(N):
            if i == r:
                continue
            not_r.append(vecs[i].values.copy())
        q.append(vecs[r].values[mask]/not_r)
    q = np.array(q)
    return E, q


def format_mask(N: int, M: int) -> Tuple[ndarray, ndarray]:
    """Setup masking arrays for correct broadcasting.

    Args:
        N (int): Number of ensemble members
        M (int): Number of bins
    Returns:
        coef_mask (array): Masking broadcasting the coefficients.
        values_mask (array): Masking broadcasting the values.
    """

    coef_mask = np.repeat(np.arange(N-1), M).reshape(N-1, M)
    values_mask = np.array([np.tile(np.arange(M, dtype=int), (N-1, 1))] * N)
    return coef_mask, values_mask
