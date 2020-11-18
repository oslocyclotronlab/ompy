import numpy as np
from .library import call_model
from scipy.interpolate import interp1d
from typing import Optional, Sequence, Tuple, Any, Union, Dict


class SpinFunctions:
    """ Calculates spin distributions, spin cuts (...) """

    def __init__(self, Ex: Union[float, Sequence], J: Union[float, Sequence],
                 model: str, pars: Dict[str, Any]):
        """
        Args:
            Ex (Union[float, Sequence]): Excitation energy
            J (Union[float, Sequence]): Spin
            model (str): Modelname for the the spincut
            pars (Dict[str, Any]): Additional parameters necessary for the
                spin cut model
        """
        self.Ex = np.atleast_1d(Ex)
        self.J = np.atleast_1d(J)
        self.model = model
        self.pars = pars

    def get_sigma2(self):
        """ Get the square of the spin cut for a specified model """
        model = self.model
        pars = self.pars

        if model == "const":
            pars_req = {"sigma"}
            return call_model(self.gconst, pars, pars_req)
        elif model == "EB05":
            pars_req = {"mass", "NLDa", "Eshift"}
            return call_model(self.gEB05, pars, pars_req)
        elif model == "EB09_CT":
            pars_req = {"mass"}
            return call_model(self.gEB09_CT, pars, pars_req)
        elif model == "EB09_emp":
            pars_req = {"mass", "Pa_prime"}
            return call_model(self.gEB09_emp, pars, pars_req)
        elif model == "Disc_and_EB05":
            pars_req = {"mass", "NLDa", "Eshift", "Sn", "sigma2_disc"}
            return call_model(self.gDisc_and_EB05, pars, pars_req)
        else:
            raise TypeError(
                "\nError: Spincut model not supported; check spelling\n")

    def distribution(self) -> Tuple[float, np.ndarray]:
        """Get spin distribution

        Note: Assuming equal parity

        Returns:
            spinDist (Tuple[float, np.ndarray]): Spin distribution. Shape
                depends on input Ex and J and is squeezed if only one of them
                is an array. If both are arrays: `spinDist[Ex,J]`
        """
        sigma2 = self.get_sigma2()
        sigma2 = sigma2[np.newaxis]  # ability to transpose "1D" array

        spinDist = ((2. * self.J + 1.) / (2. * sigma2.T)
                    * np.exp(-np.power(self.J + 0.5, 2.) / (2. * sigma2.T)))
        return np.squeeze(spinDist)  # return 1D if Ex or J is single entry

    # different spin cut models

    def gconst(self, sigma: float,
              Ex: Optional[Union[float, Sequence]] = None) -> Union[float, Sequence] : # noqa
        """
        Constant spin-cutoff parameter

        Args:
            sigma (int): Spin cut-off parameter

        Returns:
            Union[float, Sequence]: Squared spincut
        """
        Ex = self.Ex if Ex is None else Ex
        return np.full_like(Ex, sigma**2)

    def gEB05(self, mass: int, NLDa: float, Eshift: float,
              Ex: Optional[Union[float, Sequence]] = None) -> Union[float, Sequence] : # noqa
        """
        Von Egidy & B PRC72,044311(2005), Eq. (4)
        The rigid moment of inertia formula (RMI)
        FG+CT

        Args:
            mass (int): The mass number of the residual nucleus
            NLDa (float): Level density parameter
            Eshift (float): Energy shift
            Ex (float or Sequence, optional):
                Excitation energy. Defaults to self.Ex

        Returns:
            Union[float, Sequence]: Squared spincut
        """
        Ex = self.Ex if Ex is None else Ex
        Ex = np.atleast_1d(Ex)
        Eeff = Ex - Eshift
        Eeff[Eeff < 0] = 0
        sigma2 = (0.0146 * np.power(mass, 5.0 / 3.0)
                  * (1 + np.sqrt(1 + 4 * NLDa * Eeff))
                  / (2 * NLDa))
        return sigma2

    def gEB09_CT(self, mass: int,
                 Ex: Optional[Union[float, Sequence]] = None) -> Union[float, Sequence]:
        """
        The constant temperature (CT) formula
        - Von Egidy & B PRC80,054310, see sec. IV, p7 refering to ref. below
        - original ref: Von Egidy et al., NPA 481 (1988) 189, Eq. (3)

        Args:
            mass (int): Excitation energy

        Returns:
            Union[float, Sequence]: Squared spincut
        """
        Ex = self.Ex if Ex is None else Ex
        sigma2 = np.power(0.98 * (mass**(0.29)), 2)
        return sigma2

    def gEB09_emp(self, mass: int, Pa_prime: float,
                  Ex: Optional[Union[float, Sequence]] = None) -> Union[float, Sequence] : # noqa
        """
        Von Egidy & B PRC80,054310, Eq.(16)
        FG+CT

        Args:
            mass (int): Excitation energy
            Pa_prime (float): Deuteron pairing energy
            Ex (float or Sequence, optional):
                Excitation energy. Defaults to self.Ex

        Returns:
            Union[float, Sequence]: Squared spincut
        """
        Ex = self.Ex if Ex is None else Ex
        Ex = np.atleast_1d(Ex)
        Eeff = Ex - 0.5 * Pa_prime
        Eeff[Eeff < 0] = 0
        sigma2 = 0.391 * np.power(mass, 0.675) * np.power(Eeff, 0.312)
        return sigma2

    def gDisc_and_EB05(self, mass: int, NLDa: float, Eshift: float, Sn: float,
                       sigma2_disc: Tuple[float, float],
                       Ex: Optional[Union[float, Sequence]] = None) -> Union[float, Sequence] : # noqa
        """
        Linear interpolation of the spin-cut between
        a spin cut "from the discrete levels" and EB05
        RReference: Guttormsen et al., 2017, PRC 96, 024313

        Note:
            We set sigma2(E<E_discrete) = sigma2(E_discrete).
            This is not specified in the article, and may have been done
            differently before.

        Args:
            mass (int): The mass number of the residual nucleus
            NLDa (float): Level density parameter
            Eshift (float): Energy shift
            Sn (float): Neutron separation energy
            sigma2_disc (Tuple[float, float]): [float, float]
                [Energy, sigma2] from the discretes
            Ex (float or Sequence, optional):
                Excitation energy. Defaults to self.Ex

        Returns:
            Union[float, Sequence]: Squared spincut
        """
        Ex = self.Ex if Ex is None else Ex
        Ex = np.atleast_1d(Ex)
        sigma2_Sn = self.gEB05(mass, NLDa, Eshift, Ex=Sn)
        sigma2_EB05 = lambda Ex: self.gEB05(mass, NLDa, Eshift, Ex=Ex)
        x = [sigma2_disc[0], Sn]
        y = [sigma2_disc[1], sigma2_EB05(Sn)]
        sigma2 = interp1d(x, y,
                          bounds_error=False,
                          fill_value=(sigma2_disc[1], sigma2_Sn))
        return np.where(Ex < Sn, sigma2(Ex), sigma2_EB05(Ex))
