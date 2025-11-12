from __future__ import annotations

from pint._typing import UnitLike

# Lazy singleton registry; created on first use.
__ureg = None

def registry():
    """
    Return a Pint UnitRegistry (created on first call).
    Does not import Pint until needed.
    """
    global __ureg
    if __ureg is None:
        try:
            from pint import UnitRegistry  # local import
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Pint is required for unit handling. Install with "
                "`pip install pint` or `pip install 'ompy[full]'`."
            ) from e
        __ureg = UnitRegistry(system="SI")
        # Avoid global side effects like setup_matplotlib() at import time.
        # Provide a helper for users who want it:
    return __ureg

def setup_matplotlib_units():
    """Optional convenience to integrate Pint with Matplotlib."""
    ureg = registry()
    try:
        ureg.setup_matplotlib()
    except Exception:
        # Non-fatal: users may not have matplotlib, or may run headless.
        pass

# Convenience alias often used in Pint code:
u = registry()
Quantity = u.Quantity
Unit = u.Unit
Q_ = Quantity
ureg = u



def from_unit(quantity: UnitLike, default: UnitLike) -> float:
    unit = ureg.Unit(default)
    match quantity:
        case str():
            return ureg.Quantity(quantity).to(unit).magnitude
        case ureg.Quantity():
            return quantity.to(unit).magnitude
        case _:
            return quantity


def into_unit(quantity: UnitLike, default: UnitLike) -> UnitLike:
    value = from_unit(quantity, default)
    return value * ureg.Unit(default)