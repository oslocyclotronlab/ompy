"""
Normalization of GSF with the Oslo method
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import scipy.stats as stats

from .spinfunctions import SpinFunctions
from . import library as lib


class NormGSF:
    """ Normalize GSF to <Gg>

    Attributes:
    -----------
    gsf : ndarray
        gsf before normalization
        format: [E_i, value_i] or [E_i, value_i, yerror_i]
    method : string
        Method for normalization
    extModel : dict of string: string
        Model for extrapolation at ("low", "high") energies
    pext : dict
        Parameters needed for the chosen extrapolation method
    ext_range : ndarrat
        (plot) range for extrapolation ([low1,low2.,high1, high2])
    gsf_norm : ndarray
        normalized/trasformed gsf
        format: [E_i, value_i] or [E_i, value_i, yerror_i]

    nld : ndarray
        normalized NLD
        format: [E_i, value_i] or [E_i, value_i, yerror_i]

    Further inputs & units:
    # Emid_Eg, nld, gsf in MeV, MeV^-1, MeV^-3
        important!!: gsf needs to be "shape corrected" by alpha_norm
    # nld_ext: extrapolation of nld
    # gsf_ext_range: extrapolation ranges of gsf
        important!!: need to be "shape corrected" bs alpha_norm
    # J_target in 1
    # D0 in eV
    # Gg in meV
    # Sn in MeV

    TODO: Propper implementation taking into account uncertainties

    """

    def __init__(self, gsf, method,
                 J_target, D0, Gg, Sn, alpha_norm,
                 pext, ext_range, extModel=None,
                 spincutModel=None, spincutPars={},
                 nld=None, nld_ext=None):
        self.gsf_in = self.transform(
            gsf, B=1, alpha=alpha_norm)  # shape corrected
        self.gsf = np.copy(self.gsf_in)
        self.alpha_norm = alpha_norm
        self.method = method
        self.J_target, self.D0 = J_target, D0
        self.Gg = np.atleast_1d(Gg)
        self.Sn = Sn

        # define defaults
        key = 'gsf_ext_low'
        if key not in pext:
            print("Set {} to a default".format(key))
            pext['gsf_ext_low'] = np.array([1., -25.])
        key = 'gsf_ext_high'
        if key not in pext:
            print("Set {} to a default".format(key))
            pext['gsf_ext_high'] = np.array([1., -25.])
        self.pext = pext

        self.ext_range = ext_range
        self.spincutModel = spincutModel
        self.spincutPars = spincutPars
        self.nld_in = nld
        self.nld_ext = nld_ext

        # initial extrapolation
        gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
        self.gsf_ext_low = gsf_ext_low
        self.gsf_ext_high = gsf_ext_high

    def normalize_fixGg(self):
        """
        Normalize the gsf extracted with the Oslo method
        to the average total radiative width <Gg>
        """
        transform = self.transform

        # "shape" - correction  of the transformation
        gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
        self.gsf_ext_low, self.gsf_ext_high = gsf_ext_low, gsf_ext_high
        norm = self.GetNormFromGgD0()
        gsf = transform(self.gsf_in, B=norm, alpha=0)
        gsf_ext_low = transform(gsf_ext_low, B=norm, alpha=0)
        gsf_ext_high = transform(gsf_ext_high, B=norm, alpha=0)

        self.gsf = gsf
        self.gsf_ext_low = gsf_ext_low
        self.gsf_ext_high = gsf_ext_high

    def normalize_Gg_chi2(self, nld_ens, gsf_ens):
        """
        Test different normalization
        """
        assert self.gsf_in.shape[1] == 3, "Need correct gsf shape: (E,val,unc)"
        #assert type(gsf_samples[0]) == ompy.matrix.Vector

        Gg = self.Gg
        D0 = self.D0
        assert Gg.shape[0] == 2, "Gg doesn't have correct format"
        assert D0.shape[0] == 2, "D0 doesn't have correct format"

        transform = self.transform
        N_samples = len(nld_ens)

        # "shape" - correction  of the transformation
        # gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
        # self.gsf_ext_low, self.gsf_ext_high = gsf_ext_low, gsf_ext_high

        # determine uncertainty of the integral
        # Note: we change the input nld and gsf (`gsf_in`,...),
        # so we need to change it back at the end!
        gsf = np.copy(self.gsf_in)
        nld = np.copy(self.nld_in)
        integral_arr = np.zeros(N_samples)
        for i in range(N_samples):
            gsf[:, 1] = gsf_ens[i].transform(alpha=self.alpha_norm).values
            nld[:, 1] = nld_ens[i].values
            norm, integral_arr[i] = self.GetNormFromGgD0(getIntegral=True,
                                                         nld=nld, gsf=gsf)
        integral_unc_rel = integral_arr.std()/integral_arr.mean()
        # print(integral_unc_rel)
        norm, integral = self.GetNormFromGgD0(getIntegral=True,
                              nld=self.nld_in, gsf=self.gsf_in)
        gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)

        sigma2_rel = (Gg[1]/Gg[0])**2 + (D0[1]/D0[0])**2 + integral_unc_rel**2
        sigma_norm = np.sqrt(sigma2_rel) * norm

        # unc. calc the lazy way: another MC loop
        gsf_samples = []
        for i in range(N_samples):
            gsf[:, 1] = gsf_ens[i].values
            norm_tmp = stats.norm.rvs(norm, sigma_norm)
            gsf_samples.append(gsf_ens[i].transform(alpha=self.alpha_norm,
                                                    const=norm_tmp))

        gsf = transform(self.gsf_in, B=norm, alpha=0)
        gsf[:,2] = np.std([vec.values for vec in gsf_samples], axis=0)
        gsf_ext_low = transform(gsf_ext_low, B=norm, alpha=0)
        gsf_ext_high = transform(gsf_ext_high, B=norm, alpha=0)

        self.gsf = gsf
        self.gsf_ext_low = gsf_ext_low
        self.gsf_ext_high = gsf_ext_high

        return norm, integral_arr

    @staticmethod
    def transform(gsf, B, alpha):
        """ Normalize gsf

        Parameters:
        -----------
        gsf : ndarray
            Unnormalized gsf, [Ex_i, gsf_i] or [Ex_i, gsf_i, yerror_i]
        B, alpha : float
            Transformation parameters

        Returns:
        --------
        gsf_norm : Normalized gsf
        """
        E_array = gsf[:, 0]
        gsf_val = gsf[:, 1]
        if gsf.shape[1] == 3:
            rel_unc = gsf[:,2] / gsf[:,1]
        gsf_norm = gsf_val * B * np.exp(alpha * E_array)
        if gsf.shape[1] == 3:
            gsf_norm = np.c_[gsf_norm, gsf_norm * rel_unc]
        gsf_norm = np.c_[E_array, gsf_norm]
        return gsf_norm


    def gsf_extrapolation(self, pars):
        """finding and plotting extraploation of the gsf
          Parameters:
          -----------
          pars: dictionary with saved parameters

          Returns:
          --------
          gsf_ext_low : np.array of lower extrapolation
          gsf_ext_high : np.array of higher extrapolation

          TODO: Automatic choice of ext model according to method from dict.
        """

        # assert self.extModel==["exp_gsf","exp_trans"]
        ext_range = self.ext_range
        Emin_low, Emax_low, Emin_high, Emax_high = ext_range
        ext_a, ext_b = pars['gsf_ext_high']
        ext_c, ext_d = pars['gsf_ext_low']

        def f_gsf_ext_low(Eg, c, d):
            return np.exp(c * Eg + d)

        def f_gsf_ext_high(Eg, a, b):
            return np.exp(a * Eg + b) / np.power(Eg, 3)

        Emid = np.linspace(Emin_low, Emax_low)
        value = f_gsf_ext_low(Emid, ext_c, ext_d)
        gsf_ext_low = np.c_[Emid, value]

        Emid = np.linspace(Emin_high, Emax_high)
        value = f_gsf_ext_high(Emid, ext_a, ext_b)
        gsf_ext_high = np.c_[Emid, value]

        return gsf_ext_low, gsf_ext_high

    def fnld(self, E, nld=None, nld_ext=None):
        """ compose nld of data & extrapolation

        TODO: Implement uncertainties
        """
        if nld is None:
            nld = self.nld_in
        if nld_ext is None:
            nld_ext = self.nld_ext
        Earr = nld[:, 0]
        nld = nld[:, 1]

        fexp = lib.log_interp1d(Earr, nld)
        fext = lib.log_interp1d(nld_ext[:, 0], nld_ext[:, 1])

        conds = [E <= Earr[-1], E > Earr[-1]]
        funcs = [fexp, fext]
        return np.piecewise(E, conds, funcs)

    def fgsf(self, E, gsf=None, gsf_ext_low=None, gsf_ext_high=None):
        """ compose gsf of data & extrapolation

        TODO: Implement uncertainties
        """
        # gsf = self.gsf
        if gsf is None:
            gsf = self.gsf_in
        if gsf_ext_low is None:
            gsf_ext_low = self.gsf_ext_low
        if gsf_ext_high is None:
            gsf_ext_high = self.gsf_ext_high

        Earr = gsf[:, 0]
        gsf = gsf[:, 1]

        fexp = lib.log_interp1d(Earr, gsf)
        fext_low = lib.log_interp1d(gsf_ext_low[:, 0], gsf_ext_low[:, 1])
        fext_high = lib.log_interp1d(gsf_ext_high[:, 0], gsf_ext_high[:, 1])

        conds = [E < Earr[0], (E >= Earr[0]) & (E <= Earr[-1]), E > Earr[-1]]
        funcs = [fext_low, fexp, fext_high]
        return np.piecewise(E, conds, funcs)

    def spin_dist(self, Ex, J):
        return SpinFunctions(Ex=Ex, J=J,
                             model=self.spincutModel,
                             pars=self.spincutPars).distibution()

    def GetNormFromGgD0(self, getIntegral=False,
                        nld=None, gsf=None,
                        gsf_ext_low=None, gsf_ext_high=None):
        """
        Get the normaliation, see eg. eq (26) in Larsen2011;
        Note however, that we use gsf instead of T

        Returns:
        --------
        norm : float
            Absolute normalization constant by which we need to scale
            the non-normalized gsf, such that we get the correct <Gg>
        integral : (optional) float
            If `getIntegral=True`, returns the integral
        nld : (optional)
            The input nld to the integral for the chosen method
        gsf : (optional)
            The input gsf to the integral for the chosen method
        gsf_ext_low : (optional) np.array
            lower gsf extrapolation
        gsf_ext_high : (optional) np.array
            higher extrapolation

        """
        # setup of the integral (by summation)
        Eint_min = 0
        # TODO: Integrate up to Sn + Exres? And/Or from - Exres
        # Sn + Exres # following routine by Magne; Exres is the energy
        # resolution (of SiRi)
        Eint_max = self.Sn
        Nsteps = 100  # number of interpolation points
        Eintegral, stepSize = np.linspace(Eint_min, Eint_max,
                                          num=Nsteps, retstep=True)

        args = [Eintegral, stepSize, nld, gsf, gsf_ext_low, gsf_ext_high]
        if(self.method == "standard"):
            norm, integral = self.Gg_Norm_standard(*args)
        elif(self.method == "test"):
            norm, integral = self.Gg_Norm_test(*args)

        if getIntegral:
            return norm, integral
        else:
            return norm

    def Gg_Norm_standard(self, Eintegral, stepSize,
                         nld=None, gsf=None,
                         gsf_ext_low=None, gsf_ext_high=None,
                         D0=None):
        """ Compute normalization from Gg integral, the "standard" way

        Equals "old" (normalization.f) version in the Spin sum
        get the normaliation, see eg. eq (26) in Larsen2011; but converted T to gsf
        further assumptions: s-wave (currently) and equal parity

        Parameterts:
        ------------
        Eintegral : ndarray
            (Center)bin of Ex enegeries for the integration
        stepSize : ndarray
            Step size for integration by summation
        nld : (optional)
            The input nld to the integral
        gsf : (optional)
            The input gsf to the integral
        D0 : (optional)
            D0 for the Gg integral
        gsf_ext_low : (optional) np.array
            lower gsf extrapolation
        gsf_ext_high : (optional) np.array
            higher extrapolation

        Returns:
        --------
        norm : float
            Absolute normalization constant
        integral : float
            Value of the Integral
        """
        if nld is None:
            nld = self.nld_in
        if gsf is None:
            gsf = self.gsf_in
        if D0 is None:
            D0 = self.D0
        if gsf_ext_low is None:
            gsf_ext_low = self.gsf_ext_low
        if gsf_ext_high is None:
            gsf_ext_high = self.gsf_ext_high

        J_target = self.J_target

        Gg = self.Gg
        if Gg.shape[0] == 2:
            Gg = Gg[0]
        if D0.shape[0] == 2:
            D0 = D0[0]

        def fnld(E):
            return self.fnld(E, nld)
        def fgsf(E):
            return self.fgsf(E, gsf, gsf_ext_low, gsf_ext_high)

        spin_dist = self.spin_dist
        # lowest energy of experimental nld points
        Enld_exp_min = self.nld_in[0,0]
        Sn = self.Sn

        def SpinSum(Ex, J_target):
            if J_target == 0:
                # if(J_target == 0.0) I_i = 1/2 => I_f = 1/2, 3/2
                return spin_dist(Ex, J_target + 1/2) \
                    + spin_dist(Ex, J_target + 3/2)
            elif J_target == 1/2:
                # if(J_target == 0.5)    I_i = 0, 1  => I_f = 0, 1, 2
                return spin_dist(Ex, J_target - 1/2) \
                    + 2 * spin_dist(Ex, J_target + 1/2) \
                    + spin_dist(Ex, J_target + 3/2)
            elif J_target == 1:
                # if(J_target == 0.5) I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2
                return 2 * spin_dist(Ex, J_target - 1/2) \
                    + 2 * spin_dist(Ex, J_target + 1/2) \
                    + spin_dist(Ex, J_target + 3/2)
            elif J_target > 1:
                # J_target > 1 > I_i = Jt-1/2, Jt+1/2
                #                    => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
                return spin_dist(Ex, J_target - 3/2) \
                    + 2 * spin_dist(Ex, J_target - 1/2) \
                    + 2 * spin_dist(Ex, J_target + 1/2) \
                    + spin_dist(Ex, J_target + 3/2)
            else:
                ValueError("Negative J not supported")

        # perform integration by summation
        # TODO: Revise warnings: do they make sense?
        integral = 0
        for Eg in Eintegral:
            Ex = Sn - Eg
            if Eg <= Enld_exp_min:
                print("warning: Eg < {0}; check rho interpolate"
                      .format(Enld_exp_min))
            if Ex <= Enld_exp_min:
                print("warning: at Eg = {0}: Ex < {1}; " +
                      "check rho interpolate".format(Eg, Enld_exp_min))
            integral += np.power(Eg, 3) * fgsf(Eg) * fnld(Ex) \
                * SpinSum(Ex, J_target)
        integral *= stepSize

        # factor of 2 because of equi-parity (we use total nld in the
        # integral above, instead of the "correct" nld per parity)
        # Units: G / (D) = meV / (eV*1e3) = 1
        integral /= 2
        norm = Gg / (integral * D0 * 1e3)
        return norm, integral


    def Gg_Norm_test(self, Eintegral, stepSize,
                     nld=None, gsf=None, D0=None,
                     gsf_ext_low=None, gsf_ext_high=None):
        """ Compute normalization from Gg integral, "test approach"

        Experimental new version of the spin sum and integration
         similar to (26) in Larsen2011, but derived directly from the definition in Bartholomew ; but converted T to gsf
        Further assumptions: s-wave (currently) and equal parity

        Parameterts:
        ------------
        Eintegral : ndarray
            (Center)bin of Ex enegeries for the integration
        stepSize : ndarray
            Step size for integration by summation
        nld : (optional)
            The input nld to the integral
        gsf : (optional)
            The input gsf to the integral
        D0 : (optional)
            D0 for the Gg integral
        gsf_ext_low : (optional) np.array
            lower gsf extrapolation
        gsf_ext_high : (optional) np.array
            higher extrapolation

        Returns:
        --------
        norm : float
            Absolute normalization constant
        integral : float
            Value of the Integral
        """
        if nld is None:
            nld = self.nld_in
        if gsf is None:
            gsf = self.gsf_in
        if gsf_ext_low is None:
            gsf_ext_low = self.gsf_ext_low
        if gsf_ext_high is None:
            gsf_ext_high = self.gsf_ext_high

        if D0 is None:
            D0 = self.D0
        Gg = self.Gg

        if Gg.shape[0] == 2:
            Gg = Gg[0]
        if D0.shape[0] == 2:
            D0 = D0[0]

        def fnld(E):
            return self.fnld(E, nld)
        def fgsf(E):
            return self.fgsf(E, gsf, gsf_ext_low, gsf_ext_high)

        J_target = self.J_target

        spin_dist = self.spin_dist
        # lowest energy of experimental nld points
        Enld_exp_min = self.nld_in[0,0]
        Sn = self.Sn

        # input checks
        rho01plus = 1/2 * fnld(Sn) \
            * (spin_dist(Sn, J_target - 1/2) + spin_dist(Sn, J_target + 1/2))
        D0_from_fnld = 1 / rho01plus * 1e6
        D0_diff = abs((D0 - D0_from_fnld))
        if (D0_diff > 0.1 * D0):
            ValueError("D0 from extrapolation ({}) " +
                       "and from given D0 ({}) don't match"
                       .format(D0_from_fnld, D0))

        # Calculating the nlds per J and parity in the residual nucleus before decay, and the accessible spins
        # (by dipole decay <- assumption)
        if J_target == 0:
            # J_target = 0 > I_i = 1/2 => I_f = 1/2, 3/2
            # I_residual,i = 1/2 -> I_f = 0, 1
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target + 1/2)
            accessible_spin0 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2)
            # only one spin accessible
            rho1pi = None
            accessible_spin1 = lambda Ex, J_target: None
        elif J_target == 1/2:
            # J_target = 1/2  >  I_i = 0, 1  => I_f = 0, 1, 2
            # I_residual,i = 0 -> I_f = 1
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target - 1/2)
            accessible_spin0 = lambda Ex, J_target: \
                spin_dist(Ex, J_target + 1/2)
            # I_residual,i = 1 -> I_f = 0,1,2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target + 1/2)
            accessible_spin1 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2) + spin_dist(Ex, J_target + 3/2)
        elif J_target == 1:
            # J_target = 1 > I_i = 1/2, 3/2    => I_f = 1/2, 3/2, 5/2
            # I_residual,i = 1/2 -> I_f = 1/2, 3/2
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target - 1/2)
            accessible_spin0 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2)
            # I_residual,i = 3/2 -> I_f = 1/2, 3/2, 5/2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target + 1/2)
            accessible_spin1 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2) + spin_dist(Ex, J_target + 3/2)
        elif J_target > 1:
            # J_target > 1 > I_i = Jt-1/2, Jt+1/2
            # => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
            # I_residual,i = Jt-1/2 -> I_f = Jt-3/2, Jt-1/2, Jt+1/2
            rho0pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target - 1/2)
            accessible_spin0 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 3/2) + spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2)
            # I_residual,i = Jt+1/2 -> I_f = Jt-1/2, Jt+1/2, Jt+3/2
            rho1pi = 1/2 * fnld(Sn) * spin_dist(Sn, J_target + 1/2)
            accessible_spin1 = lambda Ex, J_target: \
                spin_dist(Ex, J_target - 1/2) + spin_dist(Ex, J_target + 1/2) + spin_dist(Ex, J_target + 3/2)
        else:
            ValueError("Negative J not supported")

        # perform integration by summation
        # TODO: Revise warnings: do they make sense?
        integral0 = 0
        integral1 = 0
        for Eg in Eintegral:
            Ex = Sn - Eg
            if Eg <= Enld_exp_min:
                print("warning: Eg < {0}; check rho interpolate"
                      .format(Enld_exp_min))
            if Ex <= Enld_exp_min:
                print("warning: at Eg = {0}: Ex <{1}; check rho interpolate"
                      .format(Eg, Enld_exp_min))
            integral0 += np.power(Eg, 3) * fgsf(Eg) * \
                fnld(Ex) * accessible_spin0(Ex, J_target)
            if rho1pi is not None:
                integral1 += np.power(Eg, 3) * fgsf(Eg) * \
                    fnld(Ex) * accessible_spin1(Ex, J_target)
        # simplification: <Gg>_experimental is usually reported as the average over all individual
        # Gg's. Due to a lack of further knowledge, we assume that there are equally many transisions from target states
        # with It+1/2 as from It-1/2 Then we find:
        # <Gg> = ( <Gg>_(I+1/2) + <Gg>_(I+1/2) ) / 2
        if rho1pi is None:
            integral = 1 / rho0pi * integral0
        else:
            integral = (1 / rho0pi * integral0 + 1 / rho1pi * integral1) / 2
        integral *= stepSize
        # factor of 2 because of equi-parity (we use total nld in the
        # integral above, instead of the "correct" nld per parity)
        # Units: G / (integral) = meV / (MeV*1e9) = 1
        integral /= 2
        norm = Gg / (integral * 1e9)
        return norm, integral

    def plot(self, fig, ax, interactive=False, gsf_referece=None):
        """
        Plot gsf, optinally: interactive renormalization

        Note:
        - To be interactive, the reference must be kept alive
        - If interactive, include something like
          `plt.subplots_adjust(left=0.25, bottom=0.35)`
        Parameters:
        fig, ax :
            Matplotlib figure and axes
        interactive : (bool)
            Enable interactive renormalization according to the chosen
            extrapolations
        gsf_referece : ndarray
            Refernce for plotting and normalization during code debugging
        """
        gsf = self.gsf
        gsf_ext_low = self.gsf_ext_low
        gsf_ext_high = self.gsf_ext_high

        # gsf
        [gsf_plot] = ax.plot(gsf[:,0], gsf[:,1], "o")
        [gsf_ext_high_plt] = ax.plot(gsf_ext_high[:,0], gsf_ext_high[:,1],
                                     "r--", label="ext. high")
        [gsf_ext_low_plt] = ax.plot(gsf_ext_low[:,0], gsf_ext_low[:,1],
                                    "b--", label="ext. high")

        Emin_low, Emax_low, Emin_high, Emax_high = self.ext_range
        ax.set_xlim([Emin_low, Emax_high])
        ax.set_yscale('log')
        ax.set_xlabel(r"$E_\gamma \, \mathrm{(MeV)}$")
        ax.set_ylabel(r'$gsf [MeV^-1]$')

        ax.legend()

        # load referece gsf
        if gsf_referece is not None:
            ax.plot(gsf_referece[:, 0], gsf_referece[:, 1])

        if interactive:
            # Define an axes area and draw a slider in it
            axis_color = 'lightgoldenrodyellow'
            ext_a, ext_b = self.pext['gsf_ext_high']
            ext_c, ext_d = self.pext['gsf_ext_low']
            ext_a_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03],
                                           facecolor=axis_color)
            ext_b_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03],
                                           facecolor=axis_color)
            ext_c_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03],
                                           facecolor=axis_color)
            ext_d_slider_ax = fig.add_axes([0.25, 0.20, 0.65, 0.03],
                                           facecolor=axis_color)

            sext_a = Slider(ext_a_slider_ax, 'a', 0., 4., valinit=ext_a)
            sext_b = Slider(ext_b_slider_ax, 'b', -30, 5, valinit=ext_b)
            sext_c = Slider(ext_c_slider_ax, 'c', 0, 4., valinit=ext_c)
            sext_d = Slider(ext_d_slider_ax, 'd', -30, 5, valinit=ext_d)

            def slider_update(val):
                ext_a = sext_a.val
                ext_b = sext_b.val
                ext_c = sext_c.val
                ext_d = sext_d.val
                # save the values
                self.pext['gsf_ext_low'] = np.array([ext_c, ext_d])
                self.pext['gsf_ext_high'] = np.array([ext_a, ext_b])

                # apply
                gsf_ext_low, gsf_ext_high = self.gsf_extrapolation(self.pext)
                self.gsf_ext_low, self.gsf_ext_high = gsf_ext_low, gsf_ext_high
                norm = self.GetNormFromGgD0()
                gsf = self.transform(self.gsf_in, B=norm, alpha=0)
                gsf_ext_low = self.transform(gsf_ext_low, B=norm, alpha=0)
                gsf_ext_high = self.transform(gsf_ext_high, B=norm, alpha=0)

                self.gsf = gsf
                self.gsf_ext_low = gsf_ext_low
                self.gsf_ext_high = gsf_ext_high

                gsf_plot.set_ydata(gsf[:, 1])
                gsf_ext_high_plt.set_ydata(gsf_ext_high[:, 1])
                gsf_ext_low_plt.set_ydata(gsf_ext_low[:, 1])
                fig.canvas.draw_idle()

            sext_a.on_changed(slider_update)
            sext_b.on_changed(slider_update)
            sext_c.on_changed(slider_update)
            sext_d.on_changed(slider_update)

            # make button an attribute, such that it survives for interactive
            # plot
            reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
            self.button = Button(reset_ax, 'Reset', color=axis_color,
                                 hovercolor='0.975')

            def reset(event):
                sext_a.reset()
                sext_b.reset()
                sext_c.reset()
                sext_d.reset()
            self.button.on_clicked(reset)
