import logging
import numpy as np
import copy
from numpy import ndarray
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Callable
import matplotlib.pyplot as plt

from .extractor import Extractor
from .library import log_interp1d
from .models import Model, ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer import Normalizer
from .spinfunctions import SpinFunctions
from .vector import Vector

LOG = logging.getLogger(__name__)


class NormalizerGSF(Normalizer):
    def __init__(self, *, extractor: Optional[Extractor] = None,
                 nld_normalizer: Optional[Normalizer] = None,
                 nld: Optional[Vector] = None,
                 nld_model: Optional[Callable[..., Any]] = None,
                 alpha: Optional[float] = None,
                 gsf: Optional[Vector] = None,
                 norm_pars: Optional[NormalizationParameters] = None,
                 # discrete: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = None) -> None:
        """

        TODO:
            - Abstract away Normalizer
            - check that norm_pars is set correctly
            - by default, create "good" model_low and model_high

        """
        # super().__init__(extractor, nld, discrete, path)
        self.extractor = extractor

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

    def normalize(self, *, extractor: Optional[Extractor] = None,
                  nld_normalizer: Optional[Normalizer] = None,
                  gsf: Optional[Vector] = None,
                  alpha: Optional[float] = None,
                  nld: Optional[Vector] = None,
                  nld_model: Optional[Callable[..., Any]] = None,
                  norm_pars: Optional[NormalizationParameters] = None
                  # Gg: Optional[float] = None,  # or [Tuple[float, float]]
                  # D0: Optional[float] = None # or [Tuple[float, float]]
                  ):
        """ Normalize gsf to a given <Γγ> (Gg)

        TODO:
            - transform gsf before sending to extrapolation
            - should we have `NormalizationParameters` as input instead of
              all signle inputs?
              Why do we need this: in a common optimization rutine, we want
              to normalize the gsf given Gg, D0 (...). Need to make sure there
              that we don't change the *actual* Gg, but only the parameter
              point.
            - self.extrapolate(gsf) should get the gsf of the extractor, if
            an extractor is specified
            - specify D0 and Gg here ? Otherwise delete the unused keywords
        """
        # Update internal state
        if gsf is None:
            gsf = self.gsf_in
        extractor = self.reset(extractor, nonable=True)
        nld_normalizer = self.reset(nld_normalizer, nonable=True)
        alpha = self.reset(alpha, nonable=True)
        nld = self.reset(nld, nonable=True)
        nld_model = self.reset(nld_model, nonable=True)
        norm_pars = self.reset(norm_pars)
        self.norm_pars.verify()  # check that they have been set

        if gsf is not None and extractor is not None:
            raise ValueError("Cannot use both kewords simultaneously")

        # if several gsfs shall be normalized, get them from extractor
        if extractor is not None:
            assert nld_normalizer is not None,\
                "If extractor is used, need to provide nld_normalizer, too"
            gsfs = extractor.gsf
            nlds = nld_normalizer.res.nld
            assert len(gsfs) == len(nlds), \
                "Extractor and nld_normalizer need to have same size"
            nld_models = nld_normalizer.res.nld_model
            alphas = [pars["alpha"][0] for pars in nld_normalizer.res.pars]
            self.res = copy.deepcopy(nld_normalizer.res)
        else:
            gsfs = [gsf]
            if nld is None:
                raise ValueError("Need to set either nld or extractor")
            nlds = [nld]
            if nld_model is not None:
                nld_models = [nld_model]
            else:
                assert nld_normalizer is not None, \
                    "Provide nld_model or nld_normalizer"
                LOG.debug("Setting nld_model from nld_normalizer")
                nld_models = [nld_normalizer.res.nld_model[0]]
            if alpha is not None:
                alphas = [alpha]
            else:
                assert nld_normalizer is not None, \
                    "Provide alpha or nld_normalizer"
                LOG.debug("Setting alpha from nld_normalizer")
                alphas = [nld_normalizer.res.pars[0]["alpha"][0]]
            self.res = ResultsNormalized(name="Results NLD and GSF, stepwise")

        # normalization
        lists = zip(nlds, gsfs, nld_models, alphas)
        for i, (nld, gsf, nld_model, alpha) in enumerate(lists):
            LOG.info(f"Normalizing #{i}")
            self._gsf = gsf.copy()  # make a copy as it will be transformed
            gsf.to_MeV()
            gsf = gsf.transform(alpha=alpha, inplace=False)
            self._gsf = gsf

            self.nld = nld
            self.nld_model = nld_model

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
            self.res.gsf.append(self._gsf)
            self.res.gsf_model_low.append(copy.deepcopy(self.model_low))
            self.res.gsf_model_high.append(copy.deepcopy(self.model_high))
            if len(self.res.pars) > 0:
                self.res.pars[i]["B"] = B_norm
            else:
                self.res.pars.append({"B": B_norm})

    def extrapolate(self,
                    gsf: Optional[Vector] = None) -> Tuple[Vector, Vector]:
        """ Extrapolate gsf using given models

        Args:
            gsf (Optional[Vector]): If extrapolation is fit, it will be fit to
                this vector. Default is `self._gsf`.
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
        extrapolated_low = self.model_low.extrapolate(scaled=False)

        LOG.debug("Extrapolating high: %s", self.model_high)
        extrapolated_high = self.model_high.extrapolate(scaled=False)
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

    def plot(self, ax: Any = None,
             add_legend: bool = True) -> Tuple[Any, Any]:
        """ Plot the gsf and extrapolation normalization

        Args:
            ax: The matplotlib axis to plot onto
            add_legend (bool): Whether to add a legend. Workaround
                for `plot_interactive`.
        Returns:
            The figure and axis created if no ax is supplied.

        TODO:
            plot model for each member
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        mylist = zip(self.res.gsf, self.res.gsf_model_low,
                     self.res.gsf_model_high)
        for i, (gsf, model_low, model_high) in enumerate(mylist):
            label_gsf = "normalized" if i == 0 else None
            label_model = "model" if i == 0 else None
            label_limits = "fit_limits" if i == 0 else None
            gsf.plot(ax=ax, c='k', alpha=1/len(self.res.gsf),
                     label=label_gsf)

            # extend the plotting range to the fit range
            xplot = np.linspace(model_low.Emin, model_low.Efit[1])
            gsf_low = model_low.extrapolate(xplot)
            xplot = np.linspace(model_high.Efit[0], self.norm_pars.Sn[0])
            gsf_high = model_high.extrapolate(xplot)

            gsf_low.plot(ax=ax, c='g', linestyle="--", label=label_model,
                         alpha=1/len(self.res.gsf))
            gsf_high.plot(ax=ax, c='g', linestyle="--",
                          alpha=1/len(self.res.gsf))

            if i == 0:
                ax.axvspan(model_low.Efit[0], model_low.Efit[1],
                           color='grey', alpha=0.1, label=label_limits)
                ax.axvspan(model_high.Efit[0], model_high.Efit[1],
                           color='grey', alpha=0.1)

        ax.set_yscale('log')

        if fig is not None and add_legend:
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
                self.plot(ax=ax, add_legend=False)
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
                _, _, legend = self.plot(ax=ax, add_legend=False)
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
