"""
Normalization of NLD with the Oslo method
"""
import ompy.library as lib

import numpy as np
import scipy.stats as stats
import warnings


class NormNLD:
    """ Normalize nld according to nld' = nld * A * np.exp(alpha * Ex)
    Note: This is the transformation eq (3), Schiller2000

    Parameters:
    -----------
    nld : ndarray
        Nuclear level density before normalization, format: [Ex_i, nld_i]
    method : string
        Method for normalization
    pnorm : dict
        Parameters needed for the chosen normalization method
    nldModel : string
        NLD Model for extrapolation
    pext : dict
        Parameters needed for the chosen extrapolation method
    fname_discretes : str
        Path to file with discrete levels

    """

    def __init__(self, nld, method, pnorm, nldModel, pext, fname_discretes):
        self.nld = nld
        self.method = method
        self.pnorm = pnorm
        self.pext = pext
        self.nldModel = nldModel
        self.fname_discretes = fname_discretes

        self.nld_norm = None  # Normalized nld
        self.discretes = None

        if np.max(self.nld[:,0]>1000):
            warnings.warn("Are you sure that all input is in MeV, not keV?")

        if method is "2points":
            pars_req = {"nldE1", "nldE2"}
            nld_norm, A_norm, alpha_norm = lib.call_model(
                self.norm_2points, pnorm, pars_req)
            nld_ext = self.extrapolate()
            levels_smoothed, _ = self.get_discretes(
                Emids=nld[:, 0], resolution=0.1)
            # TODO: FIX THIS!
            raise Exception("Need to work on the selection of levels here")

            levels_smoothed = levels_smoothed[0:13]
            self.discretes = np.c_[nld[0:13, 0], levels_smoothed]

            self.nld_norm = nld_norm
            self.A_norm = A_norm
            self.alpha_norm = alpha_norm
            self.nld_ext = nld_ext

        elif method is "find_norm":
            popt, samples = self.find_norm()
            self.A_norm = popt["A"][0]
            self.alpha_norm = popt["alpha"][0]
            self.T = popt["T"][0]
            self.normalize_scanning_samples(popt, samples)

        else:
            raise TypeError(
                "\nError: Normalization model not supported; check spelling\n")

    def norm_2points(self, **kwargs):
        """ Normalize to two given fixed points within "exp". nld Ex-trange

        Input:
        ------
        nldE1 : np.array([E1, nldE1])
        nldE2 : np.array([E2, nldE2])


        """
        Ex = self.nld[:, 0]
        nld = self.nld[:, 1]
        E1, nldE1 = self.pnorm["nldE1"]
        E2, nldE2 = self.pnorm["nldE2"]

        fnld = lib.log_interp1d(Ex, nld, bounds_error=True)

        # find alpha and A from the normalization points
        alpha = np.log((nldE2 * fnld(E1)) / (nldE1 * fnld(E2))) / (E2 - E1)
        A = nldE2 / fnld(E2) * np.exp(- alpha * E2)
        print(A)
        print(
            "Normalization parameters: \n alpha={0:1.2e} \t A={1:1.2e} ".format(alpha, A))

        # apply the transformation
        nld_norm = nld * A * np.exp(alpha * Ex)
        return nld_norm, A, alpha

    def extrapolate(self):
        """ Get Extrapolation values """

        model = self.nldModel
        pars = self.pext

        # Earr for extrapolation
        Earr = np.linspace(pars["ext_range"][0], pars["ext_range"][1], num=50)

        # # different extrapolation models
        # def CT(T, Eshift, **kwargs):
        #     """ Constant Temperature"""
        #     return np.exp((Earr-Eshift) / T) / T;

        if model == "CT":
            pars_req = {"T", "Eshift"}
            if ("nld_Sn" in pars) and ("Eshift" in pars == False):
                pars["Eshift"] = self.EshiftFromT(pars["T"], pars["nld_Sn"])
            pars["Earr"] = Earr
            values = lib.call_model(self.CT, pars, pars_req)
        else:
            raise TypeError(
                "\nError: NLD model not supported; check spelling\n")

        extrapolation = np.c_[Earr, values]
        return extrapolation

    @staticmethod
    def normalize(nld, A, alpha):
        """ Normalize nld

        Parameters:
        -----------
        nld : Unnormalized nld, format [Ex, nld]_i
        A : Transformation parameter
        alpha : Transformation parameter

        Returns:
        --------
        nld_norm : Normalized NLD
        """
        Ex = nld[:, 0]
        nld_val = nld[:,1]
        if nld.shape[1] == 3:
            rel_unc = nld[:,2] / nld[:,1]
        nld_norm = nld_val * A * np.exp(alpha * Ex)
        if nld.shape[1] == 3:
            nld_norm = np.c_[nld_norm, nld_norm * rel_unc]
        nld_norm = np.c_[Ex, nld_norm]
        return nld_norm

    @staticmethod
    def CT(Earr, T, Eshift, **kwargs):
        """ Constant Temperature nld"""
        return np.exp((Earr - Eshift) / T) / T

    @staticmethod
    def EshiftFromT(T, nld_Sn):
        """ Eshift from T for CT formula """
        return nld_Sn[0] - T * np.log(nld_Sn[1] * T)

    def find_norm(self):
        """
        Automatically find best normalization taking into account
        discrete levels at low energy and extrapolation at high energies
        via chi^2 minimization.

        TODO: Check validity of the assumed chi^2 "cost" function

        Returns:
        --------
        res.x: ndarray
            Best fit normalization parameters `[A, alpha, T]`
        """

        from scipy.optimize import curve_fit

        nld = self.nld
        # further parameters
        pnorm = self.pnorm
        E1_low = pnorm["E1_low"]
        E2_low = pnorm["E2_low"]

        E1_high = pnorm["E1_high"]
        E2_high = pnorm["E2_high"]
        nld_Sn = pnorm["nld_Sn"]

        # slice out comparison regions
        idE1 = np.abs(nld[:,0] - E1_low).argmin()
        idE2 = np.abs(nld[:,0] - E2_low).argmin()
        data_low = nld[idE1:idE2, :]

        # Get discretes (for the lower energies)
        levels_smoothed, _ = self.get_discretes(Emids=nld[:,0], resolution=0.1)
        levels_smoothed = levels_smoothed[idE1:idE2]
        self.discretes = np.c_[nld[idE1:idE2, 0], levels_smoothed]

        idE1 = np.abs(nld[:,0] - E1_high).argmin()
        idE2 = np.abs(nld[:,0] - E2_high).argmin()
        data_high = nld[idE1:idE2, :]

        if self.nldModel == "CT":
            nldModel = self.CT
        else:
            print("Other models not yet supported in this fit")

        from scipy.optimize import differential_evolution
        chi2_args = (nldModel, nld_Sn, data_low, data_high, levels_smoothed)

        bounds = pnorm["bounds_diff_evo"]
        res = differential_evolution(self.chi2_disc_ext,
                                     bounds=bounds,
                                     args=chi2_args)
        print("Result from find_norm / differential evolution:\n", res)

        from .multinest_setup import run_nld_2regions
        p0 = dict(zip(["A", "alpha", "T"], (res.x).T))
        popt, samples = run_nld_2regions(p0=p0,
                                         chi2_args=chi2_args)

        # set extrapolation as the median values used
        self.pext["T"] = popt["T"][0]
        self.pext["Eshift"] = self.EshiftFromT(popt["T"][0],
                                               self.pnorm["nld_Sn"])
        self.nld_ext = self.extrapolate()

        return popt, samples

    def get_discretes(self, Emids, fname=None, resolution=0.1):
        if fname is None:
            fname = self.fname_discretes
        return lib.get_discretes(Emids, fname, resolution)

    @staticmethod
    def chi2_disc_ext(x,
                      nldModel, nld_Sn, data_low, data_high, levels_smoothed):
        """
        Chi^2 between discrete levels at low energy and extrapolation at high energies

        Note: Currently working with CT extrapolation only, but should be little effort to change.

        Parameters:
        -----------
        x : ndarray
            Optimization argument in form of a 1D array
        nldModel : string
            NLD Model for extrapolation
        nld_Sn : tuple
            nld at Sn of form `[Ex, value]`
        data_low : ndarray
            Unnormalized nld at lower energies to be compared to discretes of form `[Ex, value]`
        data_high : ndarray
            Unnormalized nld at higher energies to be compared to `nldModel` of form `[Ex, value]`
        levels_smoothed: ndarray
            Discrete levels smoothed by experimental resolution of form `[value]`

        """
        A, alpha = x[:2]
        T = x[2]

        data = NormNLD.normalize(data_low, A, alpha)
        # n_low = len(data)
        chi2 = (data[:,1] - levels_smoothed)**2.
        if data.shape[1] == 3:  # weight with uncertainty, if existent
            chi2 /= data[:,2]**2
        chi2_low = np.sum(chi2)

        data = NormNLD.normalize(data_high, A, alpha)
        # n_high = len(data)
        Eshift = NormNLD.EshiftFromT(T, nld_Sn)
        chi2 = (data[:,1] - nldModel(data[:,0], T, Eshift)) ** 2.
        if data.shape[1] == 3:  # weight with uncertainty, if existent
            chi2 /= (data[:,2])**2
        chi2_high = np.sum(chi2)

        chi2 = (chi2_low + chi2_high)
        return chi2

    def normalize_scanning_samples(self, popt, samples):
        """
        Normalize NLD given the transformation parameter samples from multinest

        Parameters:
        -----------
        popt: (dict of str: (float,float)
            Dictionary of median and stddev of the parameters
        samples : dnarray
            Equally weighted samples from the chain
        """
        nld = self.nld
        Ex = self.nld[:, 0]

        # self.A_norm = self.popt["A"][0]
        # self.alpha_norm = self.popt["alpha"][0]
        # self.T = self.popt["T"][0]

        # combine uncertainties from nld (from 1Gen fit) and transformation
        if nld.shape[1] == 3:
            N_samples_max = 100
            N_loop = min(N_samples_max, len(samples["A"]))
            nld_samples = np.zeros((N_loop, len(Ex)))
            for i in range(N_loop):
                nld_tmp = stats.norm.rvs(self.nld[:,1], self.nld[:,2])
                nld_tmp = self.normalize(np.c_[Ex, nld_tmp],
                                         samples["A"][i],
                                         samples["alpha"][i])
                nld_samples[i] = nld_tmp[:,1]
            median = np.median(nld_samples, axis=0)
            std = nld_samples.std(axis=0)
            self.nld_norm = np.c_[Ex, median, std]

        # no uncertainties on nld provided
        if nld.shape[1] == 2:
            self.nld_norm = self.normalize(self.nld, self.A_norm,
                                           self.alpha_norm)
