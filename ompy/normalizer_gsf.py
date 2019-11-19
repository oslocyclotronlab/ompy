import logging
import numpy as np
import copy
from numpy import ndarray
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Callable
import matplotlib.pyplot as plt

from .library import log_interp1d, self_if_none
from .models import Model, ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer import Normalizer
from .spinfunctions import SpinFunctions
from .vector import Vector

LOG = logging.getLogger(__name__)


class NormalizerGSF():
    def __init__(self, *, nld_normalizer: Optional[Normalizer] = None,
                 nld: Optional[Vector] = None,
                 nld_model: Optional[Callable[..., Any]] = None,
                 alpha: Optional[float] = None,
                 gsf: Optional[Vector] = None,
                 norm_pars: Optional[NormalizationParameters] = None,
                 #path: Optional[Union[str, Path]] = None
                 ) -> None:
        """
        Normalize γSF to a given` <Γγ> (Gg)
        """
        # use self._gsf internally, but separate from self._gsf
        # in order to not trasform self.gsf_in 2x, if normalize() is called 2x
        self.gsf_in = None if gsf is None else gsf.copy()

        self.nld = None if nld is None else nld.copy()
        self.nld_normalizer = nld_normalizer
        self.nld_model = nld_model
        self.alpha = alpha if alpha is None else alpha

        self.model_low = ExtrapolationModelLow(name='model_low')
        self.model_high = ExtrapolationModelHigh(name='model_high')

        if norm_pars is None:
            self.norm_pars = NormalizationParameters(name="GSF Normalization")
        else:
            self.norm_pars = norm_pars

        self._gsf: Optional[Vector] = None
        self._gsf_low: Optional[Vector] = None
        self._gsf_high: Optional[Vector] = None

        self.method_Gg = "standard"

        self.res: Optional[ResultsNormalized] = None

    def normalize(self, *, gsf: Optional[Vector] = None,
                  nld_normalizer: Optional[Normalizer] = None,
                  alpha: Optional[float] = None,
                  nld: Optional[Vector] = None,
                  nld_model: Optional[Callable[..., Any]] = None,
                  norm_pars: Optional[NormalizationParameters] = None,
                  num: Optional[int] = 0) -> None:
        """ Normalize gsf to a given <Γγ> (Gg). Saves results to `self.res`.
        """
        # Update internal state
        if gsf is None:
            gsf = self.gsf_in
        assert gsf is not None, "Need to provide gsf as input"

        nld_normalizer = self.self_if_none(nld_normalizer, nonable=True)

        if nld is None:
            LOG.debug("Setting nld from from nld_normalizer")
            self.nld = nld_normalizer.res.nld
        else:
            self.nld = self.self_if_none(nld)

        alpha = self.self_if_none(alpha, nonable=True)
        if alpha is None:
            LOG.debug("Setting alpha from from nld_normalizer")
            alpha = nld_normalizer.res.pars["alpha"][0]
        assert alpha is not None, \
            "Provide alpha or nld_normalizer with alpha"

        self.nld_model = self.self_if_none(nld_model, nonable=True)
        if nld_model is None:
            LOG.debug("Setting nld_model from from nld_normalizer")
            self.nld_model = nld_normalizer.res.nld_model
        assert alpha is not None, \
            "Provide nld_model or nld_normalizer with nld_model"

        norm_pars = self.self_if_none(norm_pars)
        self.norm_pars.verify()  # check that they have been set

        # ensure to rerun
        if nld_normalizer is not None:
            self.res = copy.deepcopy(nld_normalizer.res)
        else:
            self.res = ResultsNormalized(name="Results NLD and GSF, stepwise")

        LOG.info(f"Normalizing #{num}")
        self._gsf = gsf.copy()  # make a copy as it will be transformed
        gsf.to_MeV()
        gsf = gsf.transform(alpha=alpha, inplace=False)
        self._gsf = gsf

        self.model_low.autorange(gsf)
        self.model_high.autorange(gsf)
        self._gsf_low, self._gsf_high = self.extrapolate(gsf)

        # experimental Gg and calc. both in meV
        B_norm = self.norm_pars.Gg[0] / self.Gg_before_norm()

        # apply transformation and export results
        self._gsf.transform(B_norm)
        self.model_low.norm_to_shift_after(B_norm)
        self.model_high.norm_to_shift_after(B_norm)

        # save results
        self.res.gsf = self._gsf
        self.res.gsf_model_low = copy.deepcopy(self.model_low)
        self.res.gsf_model_high = copy.deepcopy(self.model_high)
        if len(self.res.pars) > 0:
            self.res.pars["B"] = B_norm
        else:
            self.res.pars = {"B": B_norm}

    def extrapolate(self,
                    gsf: Optional[Vector] = None,
                    E: Optional[np.ndarray] = [None, None]) -> Tuple[Vector, Vector]:
        """ Extrapolate gsf using given models

        Args:
            gsf (Optional[Vector]): If extrapolation is fit, it will be fit to
                this vector. Default is `self._gsf`.
            E (optional): extrapolation energies [Elow, Ehigh]
        Returns:
            The extrapolated gSF-vectors on the form
            [low region, high region]
        Raises:
            ValueError if the models have any `None` variables.
        """
        if gsf is None:
            gsf = self._gsf

        if self.model_low.method == "fit":
            LOG.debug("Fitting extrapolation parameters")
            self.model_low.fit(gsf)
            self.model_high.fit(gsf)

        LOG.debug("Extrapolating low: %s", self.model_low)
        extrapolated_low = self.model_low.extrapolate(scaled=False, E=E[0])

        LOG.debug("Extrapolating high: %s", self.model_high)
        extrapolated_high = self.model_high.extrapolate(scaled=False, E=E[1])
        return extrapolated_low, extrapolated_high

    def Gg_before_norm(self):
        """ Compute <Γγ> before normalization

        Returns:
            Gg (float): <Γγ> before normalization, in meV
        """
        LOG.debug("Method to compute Gg: %s", self.method_Gg)
        if self.method_Gg == "standard":
            return self.Gg_standard()
        elif self.method_Gg == "FJT":
            raise NotImplementedError("Sorry, not implemented yet")
            return self.Gg_FJT()
        else:
            NotImplementedError(f"Chosen normalization {self.method_Gg}"
                                "method is not known.")

    def Gg_standard(self):
        """ Compute normalization from <Γγ> (Gg) integral, the "standard" way

        Equals "old" (normalization.f) version in the Spin sum get the
        normaliation, see eg. eq (21) and (26) in Larsen2011; but converted
        T to gsf.
        Assumptions: s-wave (current implementation) and equal parity

        Returns:
            Tuple[norm, ] (float, float): normalization

        Note:
        To derive the calculations, we start with
        <Γγ(E,J,π)> = 1/(2π ρ(E,J,π)) ∑_XL ∑_Jf,πf ∫dEγ 2π Eγ³ gsf(Eγ)
                                                        * ρ(E-Eγ, Jf, πf)

        which leads to (eq 26)**

        <Γγ> = 1 / (4π ρ(Sₙ, Jₜ± ½, πₜ)) ∫dEγ 2π Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum
             = 1 / (2  ρ(Sₙ, Jₜ± ½, πₜ)) ∫dEγ    Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum
            (= 1 / (ρ(Sₙ, Jₜ± ½, πₜ)) ∫dEγ Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum(π) )
            (= D0 1 ∫dEγ Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum(π) )

        where the integral runs from 0 to Sₙ, and the spinsum selects the
        available spins in dippole decays j=[-1,0,1]:
        spinsum = ∑ⱼ g(Sₙ-Eγ, Jₜ± ½+j).

        and <Gg> is a shorthand for <Γγ(Sₙ, Jₜ± ½, πₜ)>
        <Γγ(Sₙ, Jₜ± ½, πₜ)> =    <<Γγ(Sₙ, Jₜ+ ½, πₜ)> + <Γγ(Sₙ, Jₜ- ½, πₜ)>>
                            ≃ ½ * (<Γγ(Sₙ, Jₜ+ ½, πₜ)> + <Γγ(Sₙ, Jₜ- ½, πₜ)>)
        [but I'm challenged to derive "additivee" 1/ρ part; this is
         probably an approximation, although a *equal sign* is used in the
         article]

        We can obtain ρ(Sₙ, Jₜ± ½, πₜ) from eq(19) via D0:
        ρ(Sₙ, Jₜ± ½, πₜ) = ρ(Sₙ, Jₜ+ ½, πₜ) + ρ(Sₙ, Jₜ+ ½, πₜ) = 1/D₀
                         = ½ (ρ(Sₙ, Jₜ+ ½) + ρ(Sₙ, Jₜ+ ½))  [equi-parity]

        Equi-parity means further that g(J,π) = 1/2 * g(J), for the calc.
        above: spinsum(π) =  1/2 * spinsum.

        ** We set B to 1, so we get the <Γγ> before normalization. The
        normalization B is then `B = <Γγ>_exp/ <Γγ>_cal`
        """

        def wnld(E) -> ndarray:
            return fnld(E, self.nld, self.nld_model)

        def wgsf(E) -> ndarray:
            return fgsf(E, self._gsf, self._gsf_low, self._gsf_high)

        def integrate() -> ndarray:
            Eg, stepsize = self.norm_pars.E_grid()
            Ex = self.norm_pars.Sn[0] - Eg
            integral = (np.power(Eg, 3) * wgsf(Eg) * wnld(Ex)
                        * self.SpinSum(Ex, self.norm_pars.Jtarget))
            integral = np.sum(integral) * stepsize
            return integral

        integral = integrate()

        # factor of 2 because of equi-parity `spinsum` instead of `spinsum(π)`,
        # see above
        integral /= 2
        Gg = integral * self.norm_pars.D0[0] * 1e3  # [eV] * 1e3 -> [meV]
        return Gg

    def spin_dist(self, Ex, J):
        """ Wrapper for `SpinFunctions` curried with model and pars """
        return SpinFunctions(Ex=Ex, J=J,
                             model=self.norm_pars.spincutModel,
                             pars=self.norm_pars.spincutPars).distibution()

    def SpinSum(self, Ex, J):
        spin_dist = self.spin_dist
        if J == 0:
            # if(J == 0.0) I_i = 1/2 => I_f = 1/2, 3/2
            return spin_dist(Ex, J + 1/2) \
                + spin_dist(Ex, J + 3/2)
        elif J == 1/2:
            # if(J == 0.5)    I_i = 0, 1  => I_f = 0, 1, 2
            return spin_dist(Ex, J - 1/2) \
                + 2 * spin_dist(Ex, J + 1/2) \
                + spin_dist(Ex, J + 3/2)
        elif J == 1:
            # if(J == 0.5) I_i = 1/2, 3/2  => I_f = 1/2, 3/2, 5/2
            return 2 * spin_dist(Ex, J - 1/2) \
                + 2 * spin_dist(Ex, J + 1/2) \
                + spin_dist(Ex, J + 3/2)
        elif J > 1:
            # J > 1 > I_i = Jt-1/2, Jt+1/2
            #                    => I_f = Jt-3/2, Jt-1/2, Jt+3/2, Jt+5/2
            return spin_dist(Ex, J - 3/2) \
                + 2 * spin_dist(Ex, J - 1/2) \
                + 2 * spin_dist(Ex, J + 1/2) \
                + spin_dist(Ex, J + 3/2)
        else:
            ValueError("Negative J not supported")

    # def errfn(x: float, nld_low: Vector,
    #       nld_high: Vector, discrete: Vector,
    #       model: Callable[..., ndarray]) -> float:
    #     """ Compute the χ² of the normalization fitting NLD and gsf simultan

    #     Args:
    #         x: The arguments ordered as A, alpha, T and D0
    #         nld_low: The lower region where discrete levels will be
    #             fitted.
    #         nld_high: The upper region to fit to model.
    #         discrete: The discrete levels to be used in fitting the
    #             lower region.
    #         model: The model to use when fitting the upper region.
    #             Must support the keyword arguments:
    #                 model(T=..., D0=..., E=...) -> ndarray
    #     Returns:
    #         The χ² value
    #     """
    #     A, alpha, T, D0 = x[:4]
    #     transformed_low = nld_low.transform(A, alpha, inplace=False)
    #     transformed_high = nld_high.transform(A, alpha, inplace=False)

    #     err_low = transformed_low.error(discrete)
    #     expected = model(T=T, D0=D0, E=transformed_high.E)
    #     err_high = transformed_high.error(expected)
    #     return err_low + err_high

    def plot(self, ax: Optional[Any] = None, *,
             add_label: Optional[bool] = True,
             add_figlegend: Optional[bool] = True,
             plot_fitregion: Optional[bool] = True,
             results: Optional[ResultsNormalized] = None,
             reset_color_cycle: Optional[bool] = True,
             **kwargs) -> Tuple[Any, Any]:
        """ Plot the gsf and extrapolation normalization

        Args:
            ax: The matplotlib axis to plot onto
            add_figlegend (bool): Whether to add a legend. Workaround
                for `plot_interactive`.
            results Optional[ResultsNormalized]: If provided, gsf and model
                are taken from here instead.
        Returns:
            The figure and axis created if no ax is supplied.

        TODO:
            plot model for each member
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if reset_color_cycle:
            ax.set_prop_cycle(None)
        res = self.res if results is None else results
        gsf = res.gsf
        model_low = res.gsf_model_low
        model_high = res.gsf_model_high

        label_gsf = "normalized" if add_label else None
        label_model = "model" if add_label else None
        label_limits = "fit limits" if add_label else None
        gsf.plot(ax=ax, label=label_gsf, **kwargs)

        # extend the plotting range to the fit range
        xplot = np.linspace(model_low.Emin, model_low.Efit[1])
        gsf_low = model_low.extrapolate(xplot)
        xplot = np.linspace(model_high.Efit[0], self.norm_pars.Sn[0])
        gsf_high = model_high.extrapolate(xplot)

        gsf_low.plot(ax=ax, c='g', linestyle="--", label=label_model,
                     **kwargs)
        gsf_high.plot(ax=ax, c='g', linestyle="--",
                      **kwargs)

        if plot_fitregion:
            ax.axvspan(model_low.Efit[0], model_low.Efit[1],
                       color='grey', alpha=0.1, label=label_limits)
            ax.axvspan(model_high.Efit[0], model_high.Efit[1],
                       color='grey', alpha=0.1)

        ax.set_yscale('log')

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def plot_interactive(self):
        """ Interactive plot to study the impact of different fit regions

        Note: This implementation is not the fastest, however helped to reduce
              the code quite a lot compared to `slider_update`
        """
        from ipywidgets import interact

        self.normalize()  # maybe not needed here
        fig, ax = self.plot()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        if self.model_low.method == "fit":
            def update(Efit_low0, Efit_low1, Efit_high0, Efit_high1):
                ax.cla()
                self.model_low.Efit = [Efit_low0, Efit_low1]
                self.model_high.Efit = [Efit_high0, Efit_high1]
                self.normalize()
                self.plot(ax=ax, add_figlegend=False)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

            interact(update,
                     Efit_low0=self.model_low.Efit[0],
                     Efit_low1=self.model_low.Efit[1],
                     Efit_high0=self.model_high.Efit[0],
                     Efit_high1=self.model_high.Efit[1])

        elif self.model_low.method == "fix":
            def update(scale_low, shift_low, scale_high, shift_high):
                ax.cla()
                self.model_low.scale = scale_low
                self.model_low.shift = shift_low
                self.model_high.scale = scale_high
                self.model_high.shift = shift_high
                self.normalize()
                _, _, legend = self.plot(ax=ax, add_figlegend=False)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

            interact(update,
                     scale_low=self.model_low.scale_low,
                     shift_low=self.model_low.shift_low,
                     scale_high=self.model_high.scale_high,
                     shift_high=self.model_high.shift_high)

    def errfn(self):
        """ Weighted chi2 """
        Gg = self.norm_pars.Gg[0]
        sigma_Gg = self.norm_pars.Gg[1]
        chi2 = (self.Gg_before_norm() - Gg)/(sigma_Gg)
        chi2 = chi2**2
        return chi2

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)


def fnld(E: ndarray, nld: Vector,
         nld_model: Callable) -> ndarray:
    fexp = log_interp1d(nld.E, nld.values)

    conds = [E <= nld.E[-1], E > nld.E[-1]]
    return np.piecewise(E, conds, [fexp, nld_model(E[conds[-1]])])


def fgsf(E: ndarray, gsf: Vector,
         gsf_low: Vector, gsf_high: Vector) -> ndarray:
    exp = log_interp1d(gsf.E, gsf.values)
    ext_low = log_interp1d(gsf_low.E, gsf_low.values)
    ext_high = log_interp1d(gsf_high.E, gsf_high.values)

    conds = [E < gsf.E[0], (E >= gsf.E[0]) & (E <= gsf.E[-1]), E > gsf.E[-1]]
    return np.piecewise(E, conds, [ext_low, exp, ext_high])
