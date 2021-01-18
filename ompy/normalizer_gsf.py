import logging
import numpy as np
import copy
from numpy import ndarray
from typing import Optional, Tuple, Any, Callable, Union
from pathlib import Path
import matplotlib.pyplot as plt

from .abstract_normalizer import AbstractNormalizer
from .library import log_interp1d, self_if_none
from .models import ResultsNormalized, ExtrapolationModelLow,\
                    ExtrapolationModelHigh, NormalizationParameters
from .normalizer_nld import NormalizerNLD
from .spinfunctions import SpinFunctions
from .vector import Vector


class NormalizerGSF(AbstractNormalizer):

    """Normalize γSF to a given` <Γγ> (Gg)

    Normalizes nld/gsf according to the transformation eq (3), Schiller2000::

        gsf' = gsf * B * np.exp(alpha * Eg)

    Attributes:
        gsf_in (Vector): gsf to normalize
        nld (Vector): nld
        normalizer_nld (NormalizerNLD): NormalizerNLD to retriev parameters.
        nld_model (Callable[..., Any]): Model for nld above data of the
            from `y = nld_model(E)`
        alpha (float): tranformation parameter α
        model_low (ExtrapolationModelLow): Extrapolation model for the gsf at
            low energies (below data)
        model_high (ExtrapolationModelHigh): Extrapolation model for the gsf at
            high energies (above data)
        norm_pars (NormalizationParameters): Normalization parameters like
            experimental <Γγ>
        method_Gg (str): Method to use for the <Γγ> integral
        path (Path): The path save the results.
        res (ResultsNormalized): Results

    """
    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, *, normalizer_nld: Optional[NormalizerNLD] = None,
                 nld: Optional[Vector] = None,
                 nld_model: Optional[Callable[..., Any]] = None,
                 alpha: Optional[float] = None,
                 gsf: Optional[Vector] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False,
                 norm_pars: Optional[NormalizationParameters] = None,
                 ) -> None:
        """
        Note:
            The prefered syntax is `Normalizer(nld=...)`
            If neither is given, the nld (and other parameters) can be
            explicity
            be set later by::

                `normalizer.normalize(..., nld=...)`

            or::

                `normalizer.nld = ...`

            In the later case you *might* have to send in a copy if it's a
            mutable to ensure it is not changed.

        Args:
            normalizer_nld (Optional[NormalizerNLD], optional): NormalizerNLD
                to retrieve parameters. If `nld` and/or `nld_model` are not
                set, they are taken from `normalizer_nld.res` in `normalize`.
            nld (Optional[Vector], optional): NLD. If not set it is taken from
                `normalizer_nld.res` in `normalize`.
            nld_model (Optional[Callable[..., Any]], optional): Model for nld
                above data of the from `y = nld_model(E)`. If not set it is
                taken from `normalizer_nld.res` in `normalize`.
            alpha (Optional[float], optional): tranformation parameter α
            gsf (Optional[Vector], optional): gsf to normalize.
            norm_pars (Optional[NormalizationParameters], optional):
                Normalization parameters like experimental <Γγ>

        """
        super().__init__(regenerate)

        # use self._gsf internally, but separate from self._gsf
        # in order to not trasform self.gsf_in 2x, if normalize() is called 2x
        self.gsf_in = None if gsf is None else gsf.copy()

        self.nld = None if nld is None else nld.copy()
        self.normalizer_nld = normalizer_nld
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

        self._saved_SpinSum = None
        self._saved_spincutModel = None
        self._saved_spincutPars = None
        self._saved_SpinSum_args = None

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

    def normalize(self, *, gsf: Optional[Vector] = None,
                  normalizer_nld: Optional[NormalizerNLD] = None,
                  alpha: Optional[float] = None,
                  nld: Optional[Vector] = None,
                  nld_model: Optional[Callable[..., Any]] = None,
                  norm_pars: Optional[NormalizationParameters] = None,
                  num: int = 0) -> None:
        """Normalize gsf to a given <Γγ> (Gg). Saves results to `self.res`.

        Args:
            normalizer_nld (Optional[NormalizerNLD], optional): NormalizerNLD
                to retrieve parameters. If `nld` and/or `nld_model` are not
                set, they are taken from `normalizer_nld.res` in `normalize`.
            nld (Optional[Vector], optional): NLD. If not set it is taken from
                `normalizer_nld.res` in `normalize`.
            nld_model (Optional[Callable[..., Any]], optional): Model for nld
                above data of the from `y = nld_model(E)`. If not set it is
                taken from `normalizer_nld.res` in `normalize`.
            alpha (Optional[float], optional): tranformation parameter α
            gsf (Optional[Vector], optional): gsf to normalize.
            norm_pars (Optional[NormalizationParameters], optional):
                Normalization parameters like experimental <Γγ>
            num (Optional[int], optional): Loop number, defaults to 0.
        """
        if not self.regenerate:
            try:
                print("regenerate is self.regenerate")
                self.load()
                return
            except FileNotFoundError:
                pass

        # Update internal state
        if gsf is None:
            gsf = self.gsf_in
        assert gsf is not None, "Need to provide gsf as input"

        normalizer_nld = self.self_if_none(normalizer_nld, nonable=True)

        if nld is None:
            self.LOG.debug("Setting nld from from normalizer_nld")
            self.nld = normalizer_nld.res.nld
        else:
            self.nld = self.self_if_none(nld)

        alpha = self.self_if_none(alpha, nonable=True)
        if alpha is None:
            self.LOG.debug("Setting alpha from from normalizer_nld")
            alpha = normalizer_nld.res.pars["alpha"][0]
        assert alpha is not None, \
            "Provide alpha or normalizer_nld with alpha"

        self.nld_model = self.self_if_none(nld_model, nonable=True)
        if nld_model is None:
            self.LOG.debug("Setting nld_model from from normalizer_nld")
            self.nld_model = normalizer_nld.res.nld_model
        assert alpha is not None, \
            "Provide nld_model or normalizer_nld with nld_model"

        self.norm_pars = self.self_if_none(norm_pars)
        self.norm_pars.is_changed()  # check that they have been set

        # ensure to rerun
        if normalizer_nld is not None:
            self.res = copy.deepcopy(normalizer_nld.res)
        else:
            self.res = ResultsNormalized(name="Results NLD and GSF, stepwise")

        self.LOG.info(f"Normalizing #{num}")
        self._gsf = gsf.copy()  # make a copy as it will be transformed
        gsf.to_MeV()
        gsf = gsf.transform(alpha=alpha, inplace=False)
        self._gsf = gsf

        self.model_low.autorange(gsf)
        self.model_high.autorange(gsf)
        self._gsf_low, self._gsf_high = self.extrapolate(gsf)

        # experimental Gg and calc. both in meV
        B_norm = self.norm_pars.Gg[0] / self.Gg_before_norm()
        # propagate uncertainty of D0
        B_norm_unc = B_norm * self.norm_pars.D0[1] / self.norm_pars.D0[0]

        # apply transformation and export results
        self._gsf.transform(B_norm)
        self.model_low.norm_to_shift_after(B_norm)
        self.model_high.norm_to_shift_after(B_norm)

        # transfer results
        self.res.gsf = self._gsf
        self.res.gsf_model_low = copy.deepcopy(self.model_low)
        self.res.gsf_model_high = copy.deepcopy(self.model_high)
        if len(self.res.pars) > 0:
            self.res.pars["B"] = [B_norm, B_norm_unc]
        else:
            self.res.pars = {"B": [B_norm, B_norm_unc]}

        self.save()  # saves instance

    def extrapolate(self,
                    gsf: Optional[Vector] = None,
                    E: Optional[np.ndarray] = [None, None]
                    ) -> Tuple[Vector, Vector]:
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
            self.LOG.debug("Fitting extrapolation parameters")
            self.model_low.fit(gsf)
            self.model_high.fit(gsf)

        self.LOG.debug("Extrapolating low: %s", self.model_low)
        extrapolated_low = self.model_low.extrapolate(scaled=False, E=E[0])

        self.LOG.debug("Extrapolating high: %s", self.model_high)
        extrapolated_high = self.model_high.extrapolate(scaled=False, E=E[1])
        return extrapolated_low, extrapolated_high

    def Gg_before_norm(self) -> float:
        """ Compute <Γγ> before normalization

        Returns:
            Gg (float): <Γγ> before normalization, in meV
        """
        self.LOG.debug("Method to compute Gg: %s", self.method_Gg)
        if self.method_Gg == "standard":
            return self.Gg_standard()
        else:
            # placeholder if normalizations with
            NotImplementedError(f"Chosen normalization {self.method_Gg}"
                                "method is not known.")

    def Gg_standard(self) -> float:
        """ Compute normalization from <Γγ> (Gg) integral, the "standard" way

        Equals "old" (normalization.f) version in the Spin sum get the
        normaliation, see eg. eq (21) and (26) in Larsen2011; but converted
        T to gsf.

        Assumptions: s-wave (current implementation) and equal parity

        .. parsed-literal:: # better format in shpinx
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

            and <Γγ> is a shorthand for <Γγ(Sₙ, Jₜ± ½, πₜ)>.

            When Jₜ≠0 where we see resonances from both states Jₜ± ½, but
            often only <Γγ> is provided. We assume this is calculated as the
            average over all width: (dropping Sₙ and πₜ in notation)
            <Γγ> = ( N(Jₜ+½) <Γγ(Jₜ+ ½)> + N(Jₜ-½) <Γγ(Jₜ-½)> )
                    / ( N(Jₜ+½) + N(Jₜ-½) ),
            where N(J) is the number of levels with spin J. This is by ρ(J).
            N(Jₜ+½) + N(Jₜ-½) is the same as ρ(Sₙ, Jₜ± ½, πₜ).

            We can obtain ρ(Sₙ, Jₜ± ½, πₜ) from eq(19) via D0:
            ρ(Sₙ, Jₜ± ½, πₜ) = ρ(Sₙ, Jₜ+ ½, πₜ) + ρ(Sₙ, Jₜ+ ½, πₜ) = 1/D₀
                             = ½ (ρ(Sₙ, Jₜ+ ½) + ρ(Sₙ, Jₜ+ ½))  [equi-parity]

            Equi-parity means further that g(J,π) = 1/2 * g(J), for the calc.
            above: spinsum(π) =  1/2 * spinsum.

            For the Jₜ≠0 case, we can now rewrite:
            <Γγ> = D0 * (ρ(Sₙ, Jₜ+ ½) * <Γγ(Jₜ+ ½)>
                         + ρ(Sₙ, Jₜ- ½) * <Γγ(Jₜ- ½)>),
                 = D0 * (  ∫dEγ Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum(Jₜ- ½, π)
                         + ∫dEγ Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) spinsum(Jₜ+ ½, π) )
                 = D0 ∫dEγ Eγ³ gsf(Eγ) ρ(Sₙ-Eγ) (  spinsum(Jₜ- ½, π)
                                                 + spinsum(Jₜ+ ½, π))


            ** We set B to 1, so we get the <Γγ> before normalization. The
            normalization B is then `B = <Γγ>_exp/ <Γγ>_cal`

        Returns:
            Calculated Gg from gsf and nld
        """

        def wnld(E) -> ndarray:
            return self.fnld(E, self.nld, self.nld_model)

        def wgsf(E) -> ndarray:
            return self.fgsf(E, self._gsf, self._gsf_low, self._gsf_high)

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

    def SpinSum(self, Ex: Union[float, np.ndarray],
                J: float,
                Ltransfer: float = 0) -> Union[float, np.ndarray]:
        """Sum of spin distributions of the available states x 2
           (assumes equiparity)

        Note:
            `∑_It ∑_Jr g(Jₜ- ½)`,
            where the first sum is over the available states in the residual '
            nuclus assuming a angular momentum transfer `l`: `It = Jₜ ± ½ ± l`.
            The second sum assumes that we can reach states only b dipole
            radiaton, so the available final states are: `Jr = It ± 1`.

            As g(J) is normalized as `∑_J g(J) = 1`, and not `∑_J g(J, π) = ½`
            the sum calculated here is actually 2x the sum of avaiable states.

        Args:
            Ex (Union[float, np.ndarray]): Excitation energy
            J (float): Target spin (in neutron capture reaction)
            Ltransfer (float, optional): Angular momentum transfer. Default is
                 0, corresponding to "swave" transfer.

        Returns:
            Union[float, np.ndarray]: Sum of spin distributions. If `Ex` is
                and array, this will be an array with the sum for each `Ex`.
        """
        # load saved results if available

        locals_wo_self = locals()
        locals_wo_self.pop("self")
        if self.SpinSum_save_reload(locals_wo_self):
            return self._saved_SpinSum
        else:
            spin_dist = self.spin_dist

            assert J >= 0 and Ltransfer >= 0, \
                f"J={J} and l={Ltransfer} cannot be negative"

            # accessible residual spins
            Jres_min = min(abs(J - 1/2 - Ltransfer), abs(J + 1/2 - Ltransfer),
                           abs(J - 1/2 + Ltransfer))
            Jres_max = J + 1/2 + Ltransfer
            total = 0
            for Jres in np.arange(Jres_min, Jres_max+1):
                # accessible spins in dipole transition
                Jfinal_min = min(abs(Jres - 1), Jres)
                Jfinal_max = Jres + 1
                for Jfinal in np.arange(Jfinal_min, Jfinal_max + 1):
                    if Jres == Jfinal == 0:
                        continue
                    total += spin_dist(Ex, Jfinal)

            # enabling fast reload
            self._saved_SpinSum = total
            self._saved_spincutModel = self.norm_pars.spincutModel
            self._saved_spincutPars = self.norm_pars.spincutPars
            self._saved_SpinSum_args = locals_wo_self

            return total

    def SpinSum_save_reload(self, SpinSum_args) -> bool:
        """ Reload SpinSum if computed with the same parameters before

        Note:
            Note the most beautiful comparison, but we get a
            significant speedup in the multinest calculations

        Args:
            SpinSum_args: Arguments sent to SpinSum

        Return:
            bool: True, if args of SpinSum are the same as before,
                and self.spincutModel and self.spincutPars are unchanged
        """
        saved_args = self._saved_SpinSum_args
        if self._saved_spincutModel is None or self._saved_spincutPars is None:
            return False
        elif (self._saved_spincutModel == self.norm_pars.spincutModel and
              self._saved_spincutPars == self.norm_pars.spincutPars
              and np.all(saved_args["Ex"] == SpinSum_args["Ex"])
              and saved_args["J"] == SpinSum_args["J"]
              and saved_args["Ltransfer"] == SpinSum_args["Ltransfer"]):
            return True
        else:
            return False

    def spin_dist(self, Ex, J):
        """
        Wrapper for :meth:`ompy.SpinFunctions` curried with model and pars
        """
        return SpinFunctions(Ex=Ex, J=J,
                             model=self.norm_pars.spincutModel,
                             pars=self.norm_pars.spincutPars).distribution()

    def plot(self, ax: Optional[Any] = None, *,
             add_label: bool = True,
             add_figlegend: bool = True,
             plot_fitregion: bool = True,
             results: Optional[ResultsNormalized] = None,
             reset_color_cycle: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """Plot the gsf and extrapolation normalization

        Args:
            ax (optional): The matplotlib axis to plot onto. Creates axis
                is not provided
            add_label (bool, optional): Defaults to `True`.
            add_figlegend (bool, optional):Defaults to `True`.
            plot_fitregion (Optional[bool], optional): Defaults to `True`.
            results (ResultsNormalized, optional): If provided, gsf and model
                are taken from here instead.
            reset_color_cycle (Optional[bool], optional): Defaults to `True`
            **kwargs: Additional keyword arguments

        Returns:
            fig, ax
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

        label_gsf = "exp." if add_label else "_exp."
        label_model = "model" if add_label else "_model"
        label_limits = "fit limits" if add_label else "_fit limits"
        gsf.plot(ax=ax, label=label_gsf, **kwargs)

        # extend the plotting range to the fit range
        xplot = np.linspace(model_low.Emin, model_low.Efit[1])
        gsf_low = model_low.extrapolate(xplot)
        xplot = np.linspace(model_high.Efit[0], self.norm_pars.Sn[0])
        gsf_high = model_high.extrapolate(xplot)

        defaults = {"markersize": 0, "linestyle": "--"}
        defaults.update(kwargs)
        gsf_low.plot(ax=ax, c='g', label=label_model, **defaults)
        gsf_high.plot(ax=ax, c='g', **defaults)

        if plot_fitregion:
            ax.axvspan(model_low.Efit[0], model_low.Efit[1],
                       color='grey', alpha=0.1, label=label_limits)
            ax.axvspan(model_high.Efit[0], model_high.Efit[1],
                       color='grey', alpha=0.1)

        ax.set_yscale('log')
        ax.set_xlabel(rf"$\gamma$-ray energy $E_\gamma$~[{gsf.units}]")
        ax.set_ylabel(rf"$\gamma$-SF f($E_\gamma$)~[{gsf.units}$^{{-3}}$]")

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def plot_interactive(self):
        """ Interactive plot to study the impact of different fit regions

        Note:
            - This implementation is not the fastest, however helped to reduce
                the code quite a lot compared to `slider_update`
        """
        from ipywidgets import interact
        from ipywidgets.widgets import SelectionRangeSlider

        self.normalize()  # maybe not needed here
        fig, ax = self.plot()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()

        if self.model_low.method == "fit":
            energies = [x for x in self._gsf.E]
            options = [(x, x) for x in energies]
            defaults = [self._gsf.index(self.model_low.Efit[0]),
                        self._gsf.index(self.model_low.Efit[1])]
            slider_low = SelectionRangeSlider(options=options,
                                              index=defaults,
                                              description='Elow',
                                              disabled=False)
            energies = [x for x in self._gsf.E]
            options = [(x, x) for x in energies]
            defaults = [self._gsf.index(self.model_high.Efit[0]),
                        self._gsf.index(self.model_high.Efit[1])]
            slider_high = SelectionRangeSlider(options=options,
                                               index=defaults,
                                               description='Ehigh',
                                               disabled=False)

            def update(slider_low, slider_high):
                ax.cla()
                self.model_low.Efit = slider_low
                self.model_high.Efit = slider_high
                self.normalize()
                self.plot(ax=ax, add_figlegend=False)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

            interact(update,
                     slider_low=slider_low, slider_high=slider_high)

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

    def lnlike(self):
        """ log likelihood """
        Gg = self.norm_pars.Gg[0]
        sigma_Gg = self.norm_pars.Gg[1]
        chi2 = (self.Gg_before_norm() - Gg)/(sigma_Gg)
        chi2 = chi2**2
        return -0.5*chi2

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)

    @staticmethod
    def fnld(E: ndarray, nld: Vector,
             nld_model: Callable) -> ndarray:
        """Function composed of nld and model, providing `y = nld(E)`

        Args:
            E (ndarray): Energies to evaluate
            nld (Vector): NLD Vector for composition
            nld_model (Callable): NLD model for composition

        Returns:
            ndarray: Composite NLD

        Raises:
            ValueError: For low energies, the nld is not extrapolated. This
                may cause an error, in the calculation of Gg, when
                `nld` does not extend to 0. Please see
                https://github.com/oslocyclotronlab/ompy/issues/170 for more
                info.
        """
        fexp = log_interp1d(nld.E, nld.values)

        conds = [E <= nld.E[-1], E > nld.E[-1]]
        try:
            return np.piecewise(E, conds, [fexp, nld_model(E[conds[-1]])])
        except ValueError as e:
            print(e)
            raise ValueError("Probably your nld does not extend to 0 MeV. Please see https://github.com/oslocyclotronlab/ompy/issues/170 for more info.")  # noqa

    @staticmethod
    def fgsf(E: ndarray, gsf: Vector,
             gsf_low: Vector, gsf_high: Vector) -> ndarray:
        """Function composed of gsf and model, providing `y = gsf(E)`

        Note:
            It will take the extrapolation where no exp. data is available.

        Args:
            E (ndarray): Energies to evaluate
            gsf (Vector): GSF Vector for composition (mid part)
            gsf_low (Vector): GSF Vector for composition (lower part),
                usually an extrapolation
            gsf_high (Vector): GSF Vector for composition (higher part),
                usually an extrapolation

        Returns:
            ndarray: Description
        """
        exp = log_interp1d(gsf.E, gsf.values)
        ext_low = log_interp1d(gsf_low.E, gsf_low.values)
        ext_high = log_interp1d(gsf_high.E, gsf_high.values)

        conds = [E < gsf.E[0],
                 (E >= gsf.E[0]) & (E <= gsf.E[-1]),
                 E > gsf.E[-1]]
        return np.piecewise(E, conds, [ext_low, exp, ext_high])
