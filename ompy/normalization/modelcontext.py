from __future__ import annotations
from .. import PYMC_AVAILABLE
from typing import TypeVar, Type, TYPE_CHECKING, Self, NewType, Any
import numpy as np
from .. import ureg, Quantity, Unit
from .modelmeta import Model



TM = TypeVar("TM", bound="Model")
def modelcontext(model: TM | None = None) -> TM:
    """
    Return the given model or, if none was supplied, try to find one in
    the context stack.
    """
    if model is None:
        model = Model.get_context(error_if_none=False)

        if model is None:
            # TODO: This should be a ValueError, but that breaks
            # ArviZ (and others?), so might need a deprecation.
            raise TypeError("No model on context stack.")
    return model

VarName = NewType("VarName", str)
NameLess = VarName('')
def get_var_name(var) -> VarName:
    """Get an appropriate, plain variable name for a variable."""
    return VarName(str(getattr(var, "name", var)))

def Point(*args, filter_model_vars=False, **kwargs) -> dict[VarName, np.ndarray]:
    """Build a point. Uses same args as dict() does.
    Filters out variables not in the model. All keys are strings.

    Parameters
    ----------
    args, kwargs
        arguments to build a dict
    filter_model_vars : bool
        If `True`, only model variables are included in the result.
    """
    model = modelcontext(kwargs.pop("model", None))
    args = list(args)
    try:
        d = dict(*args, **kwargs)
    except Exception as e:
        raise TypeError(f"can't turn {args} and {kwargs} into a dict. {e}")
    return {
        get_var_name(k): np.array(v)
        for k, v in d.items()
        if not filter_model_vars or (get_var_name(k) in map(get_var_name, model.value_vars))
    }

class Registerable:
    def __init__(self, name: VarName | str):
        print("In registerable: ", name)
        self._name = VarName(name)
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate the class. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )
        model.add_named_variable(self)

    @property
    def name(self) -> VarName:
        return self._name


class InfoPoint(Registerable):
    def __init__(self, name: VarName | str):
        super().__init__(name)


class Constant(InfoPoint):
    def __init__(self, name: VarName | str, value: Any, unit=None):
        super().__init__(name)
        self.value = value
        self.unit = unit

    @classmethod
    def from_value(cls, val, name: VarName | str = NameLess) -> Constant:
        match val:
            case Quantity(v, u):
                if name is NameLess:
                    raise ValueError("Name must be specified if value is a Quantity")
                return cls(name, v, u)
            case Constant():
                return val
            case _:
                if name is NameLess:
                    raise ValueError("Name must be specified")
                return cls(name, val)

    def __repr__(self):
        unit = f" [{self.unit}]" if self.unit is not None else ""
        return f"const {self.name} ({self.value}{unit})"


class Variable(InfoPoint):
    def __init__(self, name: VarName | str, value: Any, unit=None):
        super().__init__(name)
        self.value = value
        self.unit = unit

    def __repr__(self):
        unit = f" [{self.unit}]" if self.unit is not None else ""
        return f"var {self.name} ({self.value}{unit})"



class UncertainConstant(Constant):
    def __init__(self, name: VarName | str, value: Any, dist: Distribution, unit=None):
        super().__init__(name, value, unit)
        self.dist = dist

    def __repr__(self):
        unit = f" [{self.unit}]" if self.unit is not None else ""
        return f"{self.name} ({self.value} ± {self.uncertainty}{unit})"

class Distribution(Registerable):
    pass

class Continuous(Distribution):
    pass

class Discrete(Distribution):
    pass

class Uniform(Continuous):
    def __init__(self, name, lower, upper):
        super().__init__(name)
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f"{self.name} ~ Uniform({self.lower}, {self.upper})"

class Normal(Continuous):
    def __init__(self, name, mu, sigma):
        super().__init__(name)
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"{self.name} ~ Normal({self.mu}, {self.sigma})"

class Poisson(Discrete):
    def __init__(self, name, mu):
        super().__init__(name)
        self.mu = mu

    def __repr__(self):
        return f"{self.name} ~ Poisson({self.mu})"
