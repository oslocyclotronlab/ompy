from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math

# Your reader with Pint units (keV, etc.)
from ...external.vonegidy03 import VonEgidy03  # expects .from_any("57Fe") etc.
from ..base import Nuclide

# ------------------------- enums + parsing -------------------------

class SigmaMode(Enum):
    RMI  = 1   # Egidy & Bucurescu 2006 rigid moment
    GC   = 2   # Gilbert & Cameron 1965
    CT   = 3   # Constant T (E&B 2009)
    EB09 = 4   # Egidy & Bucurescu 2009 (energy-dependent σ)

    @classmethod
    def parse(cls, x: str | int | "SigmaMode") -> "SigmaMode":
        if isinstance(x, SigmaMode):
            return x
        if isinstance(x, int):
            return {1: cls.RMI, 2: cls.GC, 3: cls.CT, 4: cls.EB09}[x]
        s = str(x).strip().lower()
        alias = {
            "1": cls.RMI, "rmi": cls.RMI,
            "2": cls.GC,  "gc":  cls.GC, "g&c": cls.GC, "gilbert-cameron": cls.GC,
            "3": cls.CT,  "ct":  cls.CT, "constant-t": cls.CT,
            "4": cls.EB09,"eb09":cls.EB09,"e&b2009":cls.EB09,"egidy2009":cls.EB09,
        }
        if s not in alias:
            raise ValueError(f"Unknown sigma_mode: {x}")
        return alias[s]


class TempMode(Enum):
    CFG = "cfg"   # T = sqrt(U/a)
    AFG = "afg"   # T = (1 + sqrt(1+4 a U))/(2 a)

    @classmethod
    def parse(cls, x: str | "TempMode") -> "TempMode":
        if isinstance(x, TempMode):
            return x
        s = str(x).strip().lower()
        alias = {"cfg": cls.CFG, "afg": cls.AFG}
        if s not in alias:
            raise ValueError(f"Unknown temp_mode: {x}")
        return alias[s]


# ------------------------- utilities/physics -------------------------

def _parity_class(A: int, Z: int) -> int:
    """0: EE, 1: OE, 2: EO, 3: OO."""
    N = A - Z
    evenN = (N % 2 == 0)
    evenZ = (Z % 2 == 0)
    if evenN and evenZ: return 0
    if (not evenN) and evenZ: return 1
    if evenN and (not evenZ): return 2
    return 3


def _Pa_prime(A: int, Z: int, Pd: float) -> float:
    """ROBIN rule: Pa′ = -Pd for EE/OE ; +Pd for EO/OO."""
    return -Pd if _parity_class(A, Z) in (0, 1) else +Pd


def _temperature(a: float, U: float, mode: TempMode) -> float:
    U = max(U, 0.0)
    if a <= 0.0:
        return 0.0
    if mode is TempMode.CFG:
        return math.sqrt(U / a)
    # AFG
    return (1.0 + math.sqrt(1.0 + 4.0 * a * U)) / (2.0 * a)


def _spin_prob(I: float, sigma2: float) -> float:
    if sigma2 <= 0.0:
        return 0.0
    return ((2.0 * I + 1.0) / (2.0 * sigma2)) * math.exp(-((I + 0.5) ** 2) / (2.0 * sigma2))


# ---- shell correction (match ROBIN macroscopic LD choices) ----

def _shellcorr(A: int, Z: int, mass_excess_keV: float) -> float:
    """S(A,Z) = M_exp(MeV) - M_LD(A,Z)."""
    pi = math.pi
    u = 931.494043  # MeV
    Mn = (8071.323 / 1000.0) + u
    Mp = (7288.969 / 1000.0) + u
    avol = -15.65
    asf  = 17.63
    asym = 27.72
    ass  = -25.60
    r0 = 1.233

    e2 = (1.60217653e-19) ** 2 / (4.0 * pi * 8.854187817e-12)
    e2 *= 6.24150947e+12
    e2 *= 1.0e+15

    rA = float(A); rZ = float(Z); rN = rA - rZ
    Eb_on_A = -(avol
                + asf * (rA ** (-1.0/3.0))
                + ((3.0 * e2) / (5.0 * r0)) * (rZ ** 2) * (rA ** (-4.0/3.0))
                + (asym + ass * (rA ** (-1.0/3.0))) * ((rN - rZ) / rA) ** 2)
    Mtheo = rN * Mn + rZ * Mp - rA * (Eb_on_A + u)
    S = (mass_excess_keV / 1000.0) - Mtheo
    return S


def _dSdA(A: int, Z: int, get_me_keV) -> float:
    S1 = _shellcorr(A+2, Z+1, get_me_keV(A+2, Z+1))
    S2 = _shellcorr(A-2, Z-1, get_me_keV(A-2, Z-1))
    return 0.25*(S1 - S2)


# ---- defaults (E&B 2009 / E&B 2006) ----

def eb2009_defaults(A: int, Z: int, S: float, Pd: float) -> Tuple[float, float]:
    Pa_p = _Pa_prime(A, Z, Pd)
    Sprime = S + 0.5 * Pa_p
    a = (0.199 + 0.0096 * Sprime) * (A ** 0.869)
    E1 = -0.381 + 0.5 * Pa_p
    return a, E1


def eb2006_defaults(A: int, Z: int, S: float, dS: float, Pd: float) -> Tuple[float, float]:
    pc = _parity_class(A, Z)
    if pc == 0:   # EE
        E1 = -0.693 - 0.5 * Pd + 0.5 * dS
        Sprime = S - 0.5 * Pd
    elif pc in (1, 2):  # OE/EO
        E1 = -0.563 - 0.5 * Pd + 0.5 * dS
        Sprime = S
    else:         # OO
        E1 = -0.240 + 0.5 * Pd + 0.29 * dS
        Sprime = S + 0.5 * Pd
    a = float(A) * (0.127 + 4.98e-03 * Sprime - 8.95e-05 * float(A))
    return a, E1


def ct2009_defaults(A: int, Z: int, S: float, Pd: float) -> Tuple[float, float]:
    Pa_p = _Pa_prime(A, Z, Pd)
    Sprime = S + 0.5 * Pa_p
    T = (float(A) ** (-0.66666)) / (0.0597 + 0.00198 * Sprime)
    E0 = -1.004 + 0.5 * Pa_p
    return T, E0


# ---- σ(E) models ----

def sigma2_rmi(A: int, T: float, rmi_reduction: float = 1.0) -> float:
    return rmi_reduction * 0.0146 * (A ** (5.0 / 3.0)) * T


def sigma2_gc(A: int, a: float, T: float) -> float:
    return 0.0888 * (A ** (2.0 / 3.0)) * a * T


def sigma2_eb2009(A: int, E: float, E1: float) -> float:
    x = max(E - (E1 + 0.381), 0.0)
    return 0.391 * (A ** 0.675) * (x ** 0.312)


# ---- ρ(U) normalizations ----

def rho_bsfg(U: float, a: float, sigma2: float) -> float:
    """ROBIN’s BSFG normalization (per MeV)."""
    if a <= 0.0 or sigma2 <= 0.0:
        return 0.0
    uu = max(U, 1e-12)
    return math.exp(2.0 * math.sqrt(a * uu)) / (12.0 * math.sqrt(2.0 * sigma2) * (a ** 0.25) * (uu ** 1.25))


def rho_ct(E: float, T: float, E0: float) -> float:
    return (1.0 / T) * math.exp((E - E0) / T) if (T > 0.0 and (E - E0) > 0.0) else 0.0


# ------------------------- result containers -------------------------

@dataclass
class EvalPoint:
    E: float      # MeV
    U: float      # MeV (intrinsic energy)
    T: float      # MeV
    sigma: float  # dimensionless
    rho: float    # 1/MeV


@dataclass
class Robin:
    # Identity
    A: int
    Z: int
    element: str

    # Inputs (MeV)
    Sn: float
    Sp: float
    Pn: float
    Pp: float
    Pd: float

    # Shell (MeV)
    S: float
    dSdA: float

    # Model
    sigma_mode: SigmaMode
    temp_mode: TempMode

    # Parameters
    a: Optional[float] = None
    E1: Optional[float] = None
    Tct: Optional[float] = None
    E0ct: Optional[float] = None
    rmi_reduction: float = 1.0

    # Derived
    Pa_prime: float = field(default=0.0)
    at_Bn: Optional[EvalPoint] = None
    at_Bp: Optional[EvalPoint] = None

    # -------- construction from tables (using units-aware reader) --------

    @classmethod
    def from_von_egidy(
        cls,
        nuclide: str | Any,
        sigma_mode: SigmaMode | str = "eb09",
        temp_mode: TempMode | str = "afg",
        rmi_reduction: float = 1.0,
        override_params: Optional[Dict[str, float]] = None,
    ) -> "Robin":
        data = VonEgidy03.from_any(nuclide)

        A, Z, el = data.mass.A, data.mass.Z, data.mass.el

        # Quantities -> MeV magnitudes
        def q_mev(q) -> float:
            return float(q.to("MeV").magnitude) if q is not None else 0.0

        Sn = q_mev(data.reaction.S_n)
        Sp = q_mev(data.reaction.S_p)
        Pd = q_mev(data.pairing.Pa)
        Pn = q_mev(data.pairing.Dnn)
        Pp = q_mev(data.pairing.Dpp)

        if data.mass.mass_excess is None:
            raise ValueError("Mass excess missing for this nuclide")
        S = _shellcorr(A, Z, float(data.mass.mass_excess.magnitude))  # keV -> inside fn converts to MeV


        def get_me_keV(a: int, z: int) -> float:
            # Prefer a direct (A,Z) lookup if your reader provides it:
            nbh = VonEgidy03.from_nuclide(Nuclide(A=a, Z=z))       # <-- implement/use this if available
            # Fallback (if needed): scan the table for exact (A,Z)
            # nbh = VonEgidy03.lookup_by_AZ(a, z)
            if nbh.mass.mass_excess is None:
                raise ValueError("Neighbor mass excess missing")
            return float(nbh.mass.mass_excess.to("keV").magnitude)


        try:
            dS = _dSdA(A, Z, get_me_keV)
        except Exception:
            dS = 0.0

        Pa_p = _Pa_prime(A, Z, Pd)

        # Modes
        s_mode = SigmaMode.parse(sigma_mode)
        t_mode = TempMode.parse(temp_mode)

        # Defaults
        a = E1 = Tct = E0ct = None
        if s_mode == SigmaMode.EB09:
            a, E1 = eb2009_defaults(A, Z, S, Pd)
        elif s_mode in (SigmaMode.RMI, SigmaMode.GC):
            a, E1 = eb2006_defaults(A, Z, S, dS, Pd)
        elif s_mode == SigmaMode.CT:
            Tct, E0ct = ct2009_defaults(A, Z, S, Pd)

        # Overrides
        if override_params:
            a   = override_params.get("a", a)
            E1  = override_params.get("E1", E1)
            Tct = override_params.get("T", Tct)
            E0ct= override_params.get("E0", E0ct)

        rob = cls(
            A=A, Z=Z, element=el,
            Sn=Sn, Sp=Sp, Pn=Pn, Pp=Pp, Pd=Pd,
            S=S, dSdA=dS,
            sigma_mode=s_mode, temp_mode=t_mode,
            a=a, E1=E1, Tct=Tct, E0ct=E0ct,
            rmi_reduction=rmi_reduction,
            Pa_prime=Pa_p,
        )
        rob._evaluate()
        return rob

    # -------- core evaluation --------

    def _eval_fg_point(self, E: float) -> EvalPoint:
        assert self.a is not None and self.E1 is not None
        U = E - self.E1
        T = _temperature(self.a, U, self.temp_mode)
        if self.sigma_mode == SigmaMode.RMI:
            sig2 = sigma2_rmi(self.A, T, self.rmi_reduction)
        elif self.sigma_mode == SigmaMode.GC:
            sig2 = sigma2_gc(self.A, self.a, T)
        else:  # EB09
            sig2 = sigma2_eb2009(self.A, E, self.E1)
        rho = rho_bsfg(U, self.a, sig2)
        return EvalPoint(E=E, U=U, T=T, sigma=math.sqrt(max(sig2, 0.0)), rho=rho)

    def _eval_ct_point(self, E: float) -> EvalPoint:
        assert self.Tct is not None and self.E0ct is not None
        sigma = 0.98 * (float(self.A) ** 0.29)
        rho = rho_ct(E, self.Tct, self.E0ct)
        return EvalPoint(E=E, U=E, T=self.Tct, sigma=sigma, rho=rho)

    def _evaluate(self) -> None:
        if self.sigma_mode == SigmaMode.CT:
            self.at_Bn = self._eval_ct_point(self.Sn)
            self.at_Bp = self._eval_ct_point(self.Sp)
        else:
            self.at_Bn = self._eval_fg_point(self.Sn)
            self.at_Bp = self._eval_fg_point(self.Sp)

    # -------- friendly summary/HTML --------

    def _repr_html_(self) -> str:
        def f(x, p=3): return f"{x:.{p}f}"
        bn = self.at_Bn or EvalPoint(self.Sn, 0.0, 0.0, 0.0, 0.0)
        bp = self.at_Bp or EvalPoint(self.Sp, 0.0, 0.0, 0.0, 0.0)

        if self.sigma_mode == SigmaMode.CT:
            params = f"T = {f(self.Tct,3)} MeV, E0 = {f(self.E0ct,3)} MeV"
        else:
            params = f"a = {f(self.a,3)} 1/MeV, E1 = {f(self.E1,3)} MeV"
            if self.sigma_mode == SigmaMode.RMI and self.rmi_reduction != 1.0:
                params += f" (RMI red = {f(self.rmi_reduction,2)})"

        return f"""
        <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto; max-width: 900px; margin: 16px auto;
                    border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.12);">
          <div style="background: linear-gradient(135deg,#1e3c72,#2a5298); color:white; padding: 18px 22px;">
            <h2 style="margin:0">{self.element}-{self.A} &nbsp; (Z={self.Z})</h2>
            <div style="opacity:.9; font-size: 0.95em;">ROBIN-style level density & spin cut-off</div>
          </div>

          <div style="background:#fff; padding: 18px 22px;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">
              <div>
                <h3 style="margin:8px 0; color:#1e3c72;">Shell / Pairing</h3>
                <div>S = {f(self.S,3)} MeV, &nbsp; dS/dA = {f(self.dSdA,3)} MeV</div>
                <div>Pn = {f(self.Pn,3)} MeV, &nbsp; Pp = {f(self.Pp,3)} MeV</div>
                <div>Pd = {f(self.Pd,3)} MeV, &nbsp; Pa′ = {f(self.Pa_prime,3)} MeV</div>
              </div>
              <div>
                <h3 style="margin:8px 0; color:#1e3c72;">Thresholds & Model</h3>
                <div>Bn = {f(self.Sn,3)} MeV &nbsp;&nbsp; Bp = {f(self.Sp,3)} MeV</div>
                <div>Mode: {self.sigma_mode.name} / T-mode: {self.temp_mode.value.upper()}</div>
                <div>Params: {params}</div>
              </div>
            </div>

            <div style="margin-top: 12px; display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">
              <div style="background:#f8fafc; padding:12px; border-radius:8px; border:1px solid #e5e7eb;">
                <h4 style="margin:4px 0; color:#1e3c72;">At Bn</h4>
                <div>T = {f(bn.T,3)} MeV</div>
                <div>σ = {f(bn.sigma,3)}</div>
                <div>ρ = {bn.rho:.5E} 1/MeV</div>
              </div>
              <div style="background:#f8fafc; padding:12px; border-radius:8px; border:1px solid #e5e7eb;">
                <h4 style="margin:4px 0; color:#1e3c72;">At Bp</h4>
                <div>T = {f(bp.T,3)} MeV</div>
                <div>σ = {f(bp.sigma,3)}</div>
                <div>ρ = {bp.rho:.5E} 1/MeV</div>
              </div>
            </div>
          </div>
        </div>
        """

    # -------- plot: spin distribution (what ROBIN writes) --------

    def plot(self, E: Optional[float] = None, sigma: Optional[float] = None,
             Imax: int = 10, dI: float = 0.5, ax=None):
        """
        Plot P(I) = (2I+1)/(2σ²) * exp(-(I+0.5)^2/(2σ²)).
        If E is given (MeV), compute σ at that energy (with current model).
        If sigma is given, use it directly.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if sigma is None:
            if E is None:
                E = self.Sn  # default: neutron threshold
            if self.sigma_mode == SigmaMode.CT:
                sigma = 0.98 * (float(self.A) ** 0.29)
            elif self.sigma_mode == SigmaMode.EB09:
                assert self.a is not None and self.E1 is not None
                sigma = math.sqrt(max(sigma2_eb2009(self.A, E, self.E1), 0.0))
            elif self.sigma_mode == SigmaMode.RMI:
                assert self.a is not None and self.E1 is not None
                U = E - self.E1
                T = _temperature(self.a, U, self.temp_mode)
                sigma = math.sqrt(max(sigma2_rmi(self.A, T, self.rmi_reduction), 0.0))
            elif self.sigma_mode == SigmaMode.GC:
                assert self.a is not None and self.E1 is not None
                U = E - self.E1
                T = _temperature(self.a, U, self.temp_mode)
                sigma = math.sqrt(max(sigma2_gc(self.A, self.a, T), 0.0))
            else:
                sigma = 0.0

        I = np.arange(0.0, Imax + 1e-9, dI)
        P = np.array([_spin_prob(i, sigma * sigma) for i in I])
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.0))
        ax.plot(I, P, label=f"σ = {sigma:.3f}", marker='o', ls=':')
        ax.set_xlabel("Spin I")
        ax.set_ylabel("P(I)")
        ax.set_title(f"{self.element}-{self.A} spin distribution")
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax
