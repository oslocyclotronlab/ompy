# -*- coding: utf-8 -*-
# cython: profile=True
"""
Script to decompose the frist generations matrix P
into the NLD $\rho$ and transmission coefficient $T$
(or $\gamma$-ray strength function $gsf$) respectivly

to compile, run following:
cython3 rhosig.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.5 -o rhosig.so rhosig.c

Copyright (C) 2018 Fabio Zeiser
University of Oslo
fabiobz [0] fys.uio.no

Slightly modified by Jørgen Eriksson Midtbø and implemented into
oslo_method_python.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from uncertainties import unumpy

cimport cython
cimport numpy as np


def decompose_matrix(P_in, P_err,
                     Emid_Eg, Emid_nld, Emid_Ex, dE_resolution,
                     method="Powell", options={'disp': True},
                     fill_value=0):
    """ routine for the decomposition of the first generations spectrum P_in

    Parameters:
    -----------
    P_in : ndarray
        First generations matrix to be decomposed
    Emid_Eg, Emid_nld, Emid_Ex : ndarray
        Array of middle-bin values for Eg, nld and Ex
    method : str
        minimization method
    options : dict
        minimization methods
    fill_value : currently unused


    Returns:
    --------
    rho_fit: ndarray
        fitted nld
    T_fit: ndarray
        fitted transmission coefficient

    """
    print("attempt decomposition")

    # protect input arrays
    P_in = np.copy(P_in)
    P_err = np.copy(P_err)
    Emid_Eg = np.copy(Emid_Eg)

    Nbins_Ex, Nbins_T = np.shape(P_in)
    Nbins_rho = Nbins_T

    # manipulation to try to improve the fit
    # TODO: imporvement might be to look for elements = 0 only in the trangle Ex<Eg
    #        + automate what value should be filled. Maybe 1/10 of smallest value in matrix?
    # if fill_value!=0:
    #   P_in[np.where(P_in == 0)] = fill_value # fill holes with a really small number
    #   P_in = np.tril(P_in,k=Nbins_T - Nbins_rho) # set lower triangle to 0 -- due to array form <-> where Eg>Ex

    P_in, P_err = normalize(P_in, P_err)

    # initial guesses
    rho0 = np.ones(Nbins_rho)

    T0 = np.zeros(Nbins_T)     # inigial guess for T  following
    for i_Eg in range(Nbins_T): # eq(6) in Schiller2000
        T0[i_Eg] = np.sum(P_in[:,i_Eg]) # no need for i_start; we trimmed the matrix already

    p0 = np.append(rho0,T0) # create 1D array of the initial guess

    # minimization
    res = minimize(objfun1D, x0=p0,
                   args=(P_in, P_err,
                         Emid_Eg, Emid_nld, Emid_Ex, dE_resolution),
                   method=method,
                   options=options)
    # further optimization: eg through higher tolderaced xtol and ftol
    # different other methods tried:
    # res = minimize(objfun1D, x0=p0, args=P_in,
    #   options={'disp': True})
    # res = minimize(objfun1D, x0=p0, args=P_in, method="L-BFGS-B",
    #   options={'disp': True}) # does a bad job when you include the weightings
    # res = minimize(objfun1D, x0=p0, args=P_in, method="Nelder-Mead",
    #   options={'disp': True}) # does a bad job
    # res = minimize(objfun1D, x0=p0, args=P_in, method="BFGS",
    #   options={'disp': True}) # does a bad job

    p_fit = res.x
    rho_fit, T_fit = rhoTfrom1D(p_fit, Nbins_rho)

    return rho_fit, T_fit

def normalize(P_in, P_err=0):
    ##############
    u_oslo_matrix = unumpy.uarray(P_in, P_err)

    # normalize each Ex row to 1 (-> get decay probability)
    for i, normalization in enumerate(np.sum(u_oslo_matrix,axis=1)):
        try:
            u_oslo_matrix[i,:] /= normalization
        except ZeroDivisionError:
            u_oslo_matrix[i,:]=0
    P_in_norm = unumpy.nominal_values(u_oslo_matrix)
    P_err_norm = unumpy.std_devs(u_oslo_matrix)

    return P_in_norm, P_err_norm

def decompose_matrix_with_unc(P_in, P_err, Emid_Eg, Emid_nld, Emid_Ex, N_mc, method="Powell", options={'disp': True}, fill_value=0):
    """
    Routine for the decomposition of the first generations spectrum P_in
    including a simple seach for statistical uncertainties. Perturbes input spectrum N_mc times and finds the mean and stddev. of the resulting fits.


    Parameters:
    -----------
    P_in : ndarray
        First generations matrix to be decomposed
    Emin : ndarray
        Array of middle-bin values
    Emid_Eg, Emid_nld, Emid_Ex : ndarray
        Array of middle-bin values for Eg, nld and Ex
    dE_resolution : nparray
        Detector resolution
    N_mc : int
        Number of iterations for the perturbation
    method : str
        minimization method
    options : dict
        minimization methods
    fill_value : currently unused

    Returns:
    --------
    rho_fit: ndarray
        fitted nld (2D: mean, std)
    T_fit: ndarray
        fitted transmission coefficient (2D: mean, std)
    """
    P_in=np.copy(P_in)

    Nbins_Ex, Nbins_T = np.shape(P_in)
    Nbins_rho = Nbins_T
    rhos = np.zeros((N_mc,Nbins_rho))
    Ts = np.zeros((N_mc,Nbins_T))

    for i_mc in range(N_mc):
        P_in_mc = np.random.poisson(np.where(P_in>0, P_in, 0))
        rhos[i_mc,:], Ts[i_mc,:] = decompose_matrix(P_in_mc, P_err, Emid_Eg,
                                  Emid_nld, Emid_Ex,
                                  method=method,
                                  options=options,
                                  fill_value=fill_value)

    rho_fit = rhos.mean(axis=0)
    rho_fit_err = rhos.std(axis=0)
    rho_fit = np.c_[rho_fit,rho_fit_err]

    T_fit = Ts.mean(axis=0)
    T_fit_err = Ts.std(axis=0)
    T_fit = np.c_[T_fit,T_fit_err]

    return rho_fit, T_fit


def objfun1D(x, *args):
    """
    1D version of the chi2 function (needed for minimize function)
    so x has one dimension only, but may be nested to contain rho and T

    Parameters:
    ----------
    x: ndarray
        workaround: 1D representation of the parameters rho and T
    args: tuple
        tuple of the fixed parameters needed to completely specify the function

    """

    Pexp, Perr, Emid_Eg, Emid_nld, Emid_Ex, dE_resolution = args
    Pexp = np.asarray(Pexp)
    Perr = np.asarray(Perr)
    Emid_Eg = np.asarray(Emid_Eg)
    Emid_nld = np.asarray(Emid_nld)
    Emid_Ex = np.asarray(Emid_Ex)
    dE_resolution = np.asarray(dE_resolution)
    Pexp = Pexp.reshape(-1, Pexp.shape[-1])
    Nbins_Ex, Nbins_T = np.shape(Pexp)
    Nbins_rho = Nbins_T
    rho, T = rhoTfrom1D(x, Nbins_rho)
    return chi2(rho, T, Pexp, Perr, Emid_Eg, Emid_nld, Emid_Ex, dE_resolution)


def chi2(np.ndarray rho, np.ndarray T, np.ndarray Pexp, np.ndarray Perr, np.ndarray Emid_Eg, np.ndarray Emid_nld, np.ndarray Emid_Ex, np.ndarray dE_resolution):
    """ Chi^2 between experimental and fitted first genration matrix"""
    cdef float chi2
    cdef np.ndarray Pfit
    if np.any(rho<0) or np.any(T<0): # hack to implement lower boundary
        chi2 = 1e20
    else:
        Nbins_Ex, Nbins_T = np.shape(Pexp)
        Pfit = PfromRhoT(rho, T, Nbins_Ex, Emid_Eg, Emid_nld, Emid_Ex, dE_resolution)
        # chi^2 = (data - fit)^2 / unc.^2, where unc.^2 = #cnt for Poisson dist.
        chi2 = np.sum( div0((Pexp - Pfit)**2,Perr**2))
    return chi2


@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
def PfromRhoT(np.ndarray rho, np.ndarray T, int Nbins_Ex, np.ndarray Emid_Eg,
              np.ndarray Emid_nld, np.ndarray Emid_Ex, np.ndarray dE_resolution, type="transCoeff"):
    """ Generate a first gernation matrix P from given nld and T (or gsf)

    Parameters:
    -----------
    rho: ndarray
        nld
    T: ndarray, optional
        transmission coefficient; either this or gsf must be specified
    gsf: ndarray, optional
        gamma-ray strength function; either this or gsf must be specified
    type: string, optional
        chosen by type= "transCoeff" /or/ "gsfL1"
    Nbins_Ex, Emid_Eg, Emid_nld, Emid_Ex, dE_resolution:
        bin number and bin center values
    Note: rho and T must have the same bin width

    Returns:
    --------
    P: ndarray
        normalized first generations matrix (sum of each Ex bin = 1)
    """

    cdef int Nbins_T = len(T)
    cdef int i_Ex, i_Eg, i_Ef, Nbins
    cdef double Ef ,Ex
    cdef double Eg
    global Emid_Eg
    cdef np.ndarray P = np.zeros((Nbins_Ex,Nbins_T))

    for i_Ex in range(Nbins_Ex):
        Ex = Emid_Ex[i_Ex]
        Eg_max = Ex + dE_resolution[i_Ex]
        Nbins = (np.abs(Emid_Eg - Eg_max)).argmin() + 1
        for i_Eg in range(Nbins):
            Ef = Emid_Ex[i_Ex] - Emid_Eg[i_Eg]
            i_Ef = (np.abs(Emid_nld-Ef)).argmin()
            P[i_Ex,i_Eg] = rho[i_Ef] * T[i_Eg]
            # if input T was a gsf, not transmission coeff: * E^(2L+1)
            if type=="gsfL1":
                Eg = Emid_Eg[i_Eg]
                P[i_Ex,i_Eg] *= np.power(Eg,3.)
    # normalize each Ex row to 1 (-> get decay probability)
    for i, normalization in enumerate(np.sum(P,axis=1)):
        P[i,:] /= normalization
    return P


def div0(np.ndarray a, np.ndarray b ):
    """ division function designed to ignore / 0,
    i.e. div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    cdef np.ndarray c
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def rhoTfrom1D(np.ndarray x1D, int Nbins_rho):
    """ split 1D array to who equal length subarrays """
    cdef np.ndarray rho = x1D[:Nbins_rho]
    cdef np.ndarray T = x1D[Nbins_rho:]
    return rho, T


def fg_cut_matrix(array, Emid, Egmin, Exmin, Emax, **kwargs):
    """ Make the first generation cuts to the matrix
    Parameters:
    -----------
    array : ndarray
        2D Array that will be sliced
    Emid : ndarray
        Array of bin center energies [Note: up to here assumed symetrix for
        both axes]
    Egmin, Exmin, Emax : doubles
        Lower and higher cuts for the gamma-ray and excitation energy axis
    kwargs: optional
        Will be ignored, just for compatibility;
    Returns:
    --------
    array : ndarray
        Sliced array
    Emid_Eg : ndarray
        Bin center energies of the gamma-ray axis
    Emid_Ex : ndarray
        Bin center energies of the excitation energy axis
    Emid_nld : ndarray
        Bin center energies of the nld once extracted
    """

    np.copy(array)

    # Eg
    i_Egmin = (np.abs(Emid-Egmin)).argmin()
    i_Emax = (np.abs(Emid-Emax)).argmin()+1
    # Ex
    i_Exmin = (np.abs(Emid-Exmin)).argmin()

    array = array[i_Exmin:i_Emax,i_Egmin:i_Emax]
    Emid_Ex = Emid[i_Exmin:i_Emax]
    Emid_Eg = Emid[i_Egmin:i_Emax]
    Emid_nld = Emid[:i_Emax-i_Egmin]

    return array, Emid_Eg, Emid_Ex, Emid_nld
