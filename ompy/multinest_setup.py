import json
import numpy
import math
import scipy.stats
import pymultinest
import os
import numpy as np

from .norm_gsf import NormGSF


def run_nld_2regions(p0, chi2_args, errfn):
    """
    Run multinest for the nld normalization on two regions:
    Discrete levels at low energy and nld(Sn) (or model) at high energies
    You might additionally want to run multinest_marginals.py

    Parameters:
    -----------
    p0: (dict of str: float)
        Parameter names and corresponding best fit.
        They will be used to create the priors! You might want to adjust
        them yourself
    chi2_args: (tuple)
        Additional arguments for the Chi2 minimization

    Returns:
    --------
    popt: (dict of str: (float,float)
        Dictionary of median and stddev of the parameters
    samples : dnarray
        Equally weighted samples from the chain

    Note:
    ------------
    multinest-files: (dump on disk)
        dumps multinest files to disk

    """

    assert list(p0.keys()) == ["A", "alpha", "T", "D0"], \
        "check if parameters really differ, if so, need to adjust priors!"

    A = p0["A"]
    alpha = p0["alpha"]
    if alpha >= 0:
        alpha_exponent = math.log(alpha, 10)
    else:
        raise ValueError("Automatic prior selection not implementedfor alpha<0")
        alpha_exponent = math.log(-alpha, 10)
    T = p0["T"]
    assert T > 0, "Automatic prior selection not implementedfor T<0"
    T_exponent = math.log(T, 10)

    D0 = p0["D0"]

    def prior(cube, ndim, nparams):
        # NOTE: You may want to adjust this for your case!
        # normal prior
        cube[0] = scipy.stats.halfnorm.ppf(cube[0], loc=A, scale=4*A)
        # log-uniform prior
        # if alpha = 1e2, it's between 1e1 and 1e3
        cube[1] = 10**(cube[1]*2 + (alpha_exponent-1))
        # log-uniform prior
        # if T = 1e2, it's between 1e1 and 1e3
        cube[2] = 10**(cube[2]*2 + (T_exponent-1))
        # normal prior
        cp = cube[3]
        cube[3] = scipy.stats.halfnorm.ppf(cube[3], loc=D0[0], scale=D0[1])
        if np.isinf(cube[3]):
            print("Cube 3:", cp)

    def loglike(cube, ndim, nparams):
        chi2 = errfn(cube, *chi2_args)
        loglikelihood = -0.5 * chi2
        return loglikelihood

    # number of dimensions our problem has
    n_params = len(p0)

    folder = 'multinest'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(os.getcwd())
    datafile_basename = os.path.join(folder, "nld_norm_")
    # datafile_basename = os.path.join(os.getcwd(), *(folder,"nld_norm_"))
    assert len(datafile_basename) < 60, "Total path length probably too long for multinest (max 80 chars)"

    # run MultiNest
    print("\n Starting multinest: this may take a while. Check also terminal output")
    pymultinest.run(loglike, prior, n_params,
                    outputfiles_basename=datafile_basename,
                    resume = False, verbose = True)
    # save parameter names
    json.dump(list(p0.keys()), open(datafile_basename + 'params.json', 'w'))

    analyzer = pymultinest.Analyzer(n_params,
                                    outputfiles_basename=datafile_basename)
    stats = analyzer.get_stats()
    samples = analyzer.get_equal_weighted_posterior()[:, :-1]
    samples = dict(zip(p0.keys(), samples.T))

    popt = dict()
    for p, m in zip(list(p0.keys()), stats['marginals']):
        lo, hi = m['1sigma']
        med = m['median']
        sigma = (hi - lo) / 2
        popt[p] = (med,sigma)
        i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
        fmt = '%%.%df' % i
        fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
        print(fmts % (p, med, sigma))

    return popt, samples


def run_nld_gsf_simultan(p0, chi2_args):
    """
    Run multinest for simultaneous normaliation of nld and gsf

    Parameters:
    -----------
    p0: (dict of str: float)
        Parameter names and corresponding best fit.
        They will be used to create the priors! You might want to adjust
        them yourself
    chi2_args: (tuple)
        Additional arguments for the Chi2 minimization

    Returns:
    --------
    popt: (dict of str: (float,float)
        Dictionary of median and stddev of the parameters
    samples : dnarray
        Equally weighted samples from the chain

    Note:
    ------------
    multinest-files: (dump on disk)
        dumps multinest files to disk

    """

    assert list(p0.keys()) == ["A", "alpha", "T", "D0", "B"], \
        "check if parameters really differ, if so, need to adjust priors!"

    A = p0["A"]
    alpha = p0["alpha"]
    if alpha >= 0:
        alpha_exponent = math.log(alpha, 10)
    else:
        raise ValueError("Automatic prior selection not implementedfor alpha<0")
        alpha_exponent = math.log(-alpha, 10)
    T = p0["T"]
    assert T > 0, "Automatic prior selection not implementedfor T<0"
    T_exponent = math.log(T, 10)

    D0 = p0["D0"]
    B = p0["B"]

    def prior(cube, ndim, nparams):
        # NOTE: You may want to adjust this for your case!
        # normal prior
        cube[0] = scipy.stats.halfnorm.ppf(cube[0], loc=A, scale=4*A)
        # log-uniform prior
        # if alpha = 1e2, it's between 1e1 and 1e3
        cube[1] = 10**(cube[1]*2 + (alpha_exponent-1))
        # log-uniform prior
        # if T = 1e2, it's between 1e1 and 1e3
        cube[2] = 10**(cube[2]*2 + (T_exponent-1))
        # normal prior
        cube[3] = scipy.stats.halfnorm.ppf(cube[3], loc=D0[0], scale=D0[1])
        # normal prior
        cube[4] = scipy.stats.halfnorm.ppf(cube[4], loc=B, scale=4*B)

    def loglike(cube, ndim, nparams):
        chi2 = NormGSF.chi2_nld_gsf(cube, *chi2_args)
        loglikelihood = -0.5 * chi2
        return loglikelihood

    # number of dimensions our problem has
    n_params = len(p0)

    folder = 'multinest'
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(os.getcwd())
    datafile_basename = os.path.join(folder, "nld_gsf_norm_")
    # datafile_basename = os.path.join(os.getcwd(), *(folder,"nld_norm_"))
    assert len(datafile_basename) < 60, "Total path length probably too long for multinest (max 80 chars)"

    # run MultiNest
    print("\n Starting multinest: this may take a while. Check also terminal output")
    pymultinest.run(loglike, prior, n_params,
                    outputfiles_basename=datafile_basename,
                    resume = False, verbose = True)
    # save parameter names
    json.dump(list(p0.keys()), open(datafile_basename + 'params.json', 'w'))

    analyzer = pymultinest.Analyzer(n_params,
                                    outputfiles_basename=datafile_basename)
    stats = analyzer.get_stats()
    samples = analyzer.get_equal_weighted_posterior()[:, :-1]
    samples = dict(zip(p0.keys(), samples.T))

    popt = dict()
    for p, m in zip(list(p0.keys()), stats['marginals']):
        lo, hi = m['1sigma']
        med = m['median']
        sigma = (hi - lo) / 2
        popt[p] = (med,sigma)
        i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
        fmt = '%%.%df' % i
        fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
        print(fmts % (p, med, sigma))

    return popt, samples

