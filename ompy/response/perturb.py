from .discreteinterpolation import DiscreteInterpolation
from .interpolations import FEInterpolation, EscapeInterpolation, AnnihilationInterpolation, LinearInterpolation, FWHMInterpolation
from .responsedata import Components
from .response import Response
import numpy as np


def perturb(R: Response,* , delta: float | None = None,
            delta_c: float | None = None, delta_intp: float | None = None) -> Response:
    if delta is not None:
        delta_c = delta
        delta_intp = delta
    elif delta_c is None or delta_intp is None:
        raise ValueError("Must specify either delta or both delta_c and delta_intp")
    components = uniform_components(delta_c)
    intp = perturb_intp(R.interpolation, delta_intp)
    return R.clone(R=False, interpolation=intp, components=components)


def perturb_intp(intp: DiscreteInterpolation, delta: float):
    """Perturb the interpolation by a small amount

    Args:
        intp (DiscreteInterpolation): The interpolation to perturb
        delta (float): The amount to perturb the interpolation by

    Returns:
        DiscreteInterpolation: The perturbed interpolation
    """
    fe = perturb_FE(intp.FE, delta)
    se = perturb_escape(intp.SE, delta)
    de = perturb_DE(intp.DE, delta)
    ap = perturb_AP(intp.AP, delta)
    return intp.clone(FE=fe, SE=se, DE=de, AP=ap)


def uniform_components(delta: float) -> Components:
    c = np.random.uniform(1/delta, delta, 4)
    # We leave FE unchanged, as the final response is normalized
    return Components(1.0, *c)


def perturb_FE(intp: FEInterpolation, delta: float) -> FEInterpolation:
    """Perturb the interpolation by a small amount

    Args:
        intp (FEInterpolation): The interpolation to perturb
        delta (float): The amount to perturb the interpolation by

    Returns:
        FEInterpolation: The perturbed interpolation
    """
    p = intp.p.copy()
    u = np.random.uniform(-1, 1)
    p[0] = (1 + delta*u*2e-5) * intp.p[0]
    u = np.random.uniform(-1, 1)
    #p[1] = (1 + delta*u*2e-6) * intp.p[1]
    intp = intp.copy(p=p)
    return intp


def perturb_escape(intp: EscapeInterpolation, delta: float) -> EscapeInterpolation:
    """Perturb the interpolation by a small amount

    Args:
        intp (EscapeInterpolation): The interpolation to perturb
        delta (float): The amount to perturb the interpolation by

    Returns:
        EscapeInterpolation: The perturbed interpolation
    """
    p = intp.copy()
    u = np.random.uniform(-1, 1)
    a = (1 + delta*u*1e-2)*p.gf3.a
    gf3 = p.gf3.copy(a = a)
    p.gf3 = gf3
    return p

def perturb_DE(intp: EscapeInterpolation, delta: float) -> EscapeInterpolation:
    return perturb_escape(intp, 2*delta)


def perturb_AP(intp: AnnihilationInterpolation, delta: float) -> AnnihilationInterpolation:
    return perturb_escape(intp, delta)
