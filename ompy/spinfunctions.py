import numpy as np
from .library import call_model
from scipy.interpolate import interp1d

"""
Code by Fabio Zeiser to calculate spin distribution
functions.
"""

class SpinFunctions:
    """ Calculates spin distributions, spin cuts (...)

    Args:
        Ex : double or dnarray
            Excitation energy
        J : double or dnarray
            Spin
        model : string
            Model to for the spincut
        pars : dict
            Additional parameters necessary for the spin cut model
    """

    def __init__(self, Ex, J, model, pars):
        self.Ex = np.atleast_1d(Ex)
        self.J = np.atleast_1d(J)
        self.model = model
        self.pars = pars

    def get_sigma2(self):
        """ Get the square of the spin cut for a specified model """
        Ex = self.Ex
        model = self.model
        pars = self.pars

        # different spin cut models
        def EB05(mass, NLDa, Eshift, Ex=Ex):
            """
            Von Egidy & B PRC72,044311(2005), Eq. (4)
            The rigid moment of inertia formula (RMI)
            FG+CT
            """
            Ex = np.atleast_1d(Ex)
            Eeff = Ex - Eshift
            Eeff[Eeff < 0] = 0
            sigma2 = (0.0146 * np.power(mass, 5.0 / 3.0)
                      * (1 + np.sqrt(1 + 4 * NLDa * Eeff))
                      / (2 * NLDa))
            return sigma2

        def EB09_CT(mass):
            """
            The constant temperature (CT) formula
            - Von Egidy & B PRC80,054310, below Eq. (8)
            - original ref: Von Egidy et al., NPA 481 (1988) 189, Eq. (3)
            """
            sigma2 = np.power(0.98 * (mass**(0.29)), 2)
            return sigma2

        def EB09_emp(mass, Pa_prime, Ex=Ex):
            """
            Von Egidy & B PRC80,054310, Eq.(16)
            FG+CT
            """
            Ex = np.atleast_1d(Ex)
            Eeff = Ex - 0.5 * Pa_prime
            Eeff[Eeff < 0] = 0
            sigma2 = 0.391 * np.power(mass, 0.675) * np.power(Eeff, 0.312)
            return sigma2

        def Disc_and_EB05(mass, NLDa, Eshift, Sn, sigma2_disc, Ex=Ex):
            """
            Linear interpolation of the spin-cut between
            a spin cut "from the discrete levels" and EB05
            See eg:
            Guttormsen et al., 2017, PRC 96, 024313

            Note:
            We set sigma2(E<E_discrete) = sigma2(E_discrete).
            This is not specified in the article, and may have been done
            differently before.

            Parameters:
            -----------v
            sigma2_disc: [float, float]
                [Energy, sigma2] from the discretes
            """
            Ex = np.atleast_1d(Ex)
            sigma2_Sn = EB05(mass, NLDa, Eshift, Ex=Sn)
            x = [sigma2_disc[0], Sn]
            y = [sigma2_disc[1], sigma2_Sn]
            sigma2 = interp1d(x,y,
                              bounds_error=False,
                              fill_value=(sigma2_disc[1],"extrapolate"))
            return sigma2(Ex)

        if model == "EB05":
            pars_req = {"mass", "NLDa", "Eshift"}
            return call_model(EB05, pars, pars_req)
        if model == "EB09_CT":
            pars_req = {"mass"}
            return call_model(EB09_CT, pars, pars_req)
        if model == "EB09_emp":
            pars_req = {"mass", "Pa_prime"}
            return call_model(EB09_emp, pars, pars_req)
        if model == "Disc_and_EB05":
            pars_req = {"mass", "NLDa", "Eshift", "Sn", "sigma2_disc"}
            return call_model(Disc_and_EB05, pars, pars_req)
        else:
            raise TypeError(
                "\nError: Spincut model not supported; check spelling\n")

    def distibution(self):
        """ Get spin distribution

        Note: assuming equal parity

        Returns:
        --------
        spinDist: double or ndarray
          Spin distribution. Shape depends on input Ex and J and is squeezed
          if only one of them is an array. If both are arrays: spinDist[Ex,J]
        """
        Ex = self.Ex
        J = self.J

        sigma2 = self.get_sigma2()
        sigma2 = sigma2[np.newaxis]  # ability to transpose "1D" array

        spinDist = ((2. * J + 1.) / (2. * sigma2.T)
                    * np.exp(-np.power(J + 0.5, 2.) / (2. * sigma2.T)))
        return np.squeeze(spinDist)  # return 1D if Ex or J is single entry
