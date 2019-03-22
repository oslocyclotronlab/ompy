import numpy as np
from .library import call_model


class SpinFunctions:
    """ Get different spin distributions, spin cuts (...)

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
        def EB05(mass, NLDa, Eshift):
            """
            Von Egidy & B PRC72,044311(2005)
            The rigid moment of inertia formula (RMI)
            FG+CT
            """
            Eeff = Ex - Eshift
            Eeff[Eeff<0] = 0
            sigma2 =  (0.0146 * np.power(mass, 5.0/3.0)
                        * (1 + np.sqrt(1 + 4 * NLDa * Eeff))
                        / (2 * NLDa))
            return sigma2

        def EB09_CT(mass):
            """
            The constant temperature (CT) formula Von Egidy & B PRC80,054310 and NPA 481 (1988) 189
            """
            sigma2 =  np.power(0.98*(A**(0.29)),2)
            return sigma2

        def EB09_emp(mass,Pa_prime):
            """
            Von Egidy & B PRC80,054310
            FG+CT
            """
            Eeff = Ex - 0.5 * Pa_prime
            Eeff[Eeff<0] = 0
            sigma2 = 0.391 * np.power(mass, 0.675) * np.power(mass,0.312)
            return sigma2

        if model=="EB05":
            pars_req = {"mass", "NLDa", "Eshift"}
            return call_model(EB05,pars,pars_req)
        if model=="EB09_CT":
            pars_req = {"mass"}
            return call_model(EB09_CT,pars,pars_req)
        if model=="EB09_emp":
            pars_req = {"mass","Pa_prime"}
            return call_model(EB09_emp,pars,pars_req)
        else:
            raise TypeError("\nError: Spincut model not supported; check spelling\n")

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
        sigma2 = sigma2[np.newaxis] # ability to transpose "1D" array

        spinDist = ( (2.*J+1.) / (2.*sigma2.T)
                    * np.exp(-np.power(J+0.5, 2.)/(2.*sigma2.T)) )
        return np.squeeze(spinDist) # return 1D if Ex or J is single entry
