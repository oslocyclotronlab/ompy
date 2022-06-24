from . import ureg, DimensionalityError, Q_
import re
from typing import Tuple
from .header import Unit


class Validator:
    def __init__(self):
        raise NotImplementedError()

    def __set_name__(self, owner, name):
        self.private_name: str = '_' + name
        self.public_name: str = name
        setattr(owner, self.private_name, self.default)

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    def __get__(self, obj, objtype=None):
        # Autoreload has a bug where it is unable to reload decorators.
        # This seems to (?) circumvent the bug
        if obj is None:
            print("TRIED TO SET TO NONE. Using autoreload?")
            return
        return getattr(obj, self.private_name)

    def validate(self, value):
        raise NotImplementedError()


class Unitful(Validator):
    def __init__(self, default):
        self.default = ureg.Quantity(default)
        self.units = self.default.units

    def validate(self, value):
        value = ureg.Quantity(value)
        if value.dimensionless:
            if not self.dimensionless:
                value *= self.units
            else:
                value = value.magnitude
        else:
            try:
                value = value.to(self.units)
            except DimensionalityError:
                raise DimensionalityError(value.units, self.units) from None
        return value

    @property
    def dimensionless(self) -> bool:
        return self.default.dimensionless


class Bounded(Validator):
    def __init__(self, default, min=None, max=None, type=None):
        self.min = min
        self.max = max
        self.default = default
        self.type = type

    def validate(self, value):
        if self.type is not None:
            if self.type == float:
                if not isinstance(value, (int, float)):
                    raise TypeError(f"{self.public_name} must be {self.type.__name__},"
                                    f" not {type(value).__name__}.")
                value = float(value)
            elif not isinstance(value, self.type):
                raise TypeError(f"{self.public_name} must be {self.type.__name__},"
                                f" not {type(value).__name__}.")
        if self.min is not None and value < self.min:
            raise ValueError(f"Expect {self.public_name} > {self.min}.")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expect {self.public_name} < {self.max}.")
        return value


class Choice(Validator):
    def __init__(self, default, choices, preprocess=lambda x: x):
        self.default = default
        self.choices = choices
        self.preprocess = preprocess

    def validate(self, value):
        try:
            value = self.preprocess(value)
        except:
            pass
        if value not in self.choices:
            raise ValueError(f"{self.public_name} must be one of "
                             f"{self.choices}, not {value}")
        return value


class Toggle(Validator):
    def __init__(self, default):
        self.default = default

    def validate(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"{self.public_name} must be True or False.")
        return value


class UnitfulError(Validator):
    def __init__(self, value, error=None):
        if isinstance(value, str):
            value, error = self.parse(value)
        else:
            if error is None:
                raise ValueError("`error` must be provided.")
            value = Q_(value)
            error = Q_(value)
        error = error.to(value.units)
        self.default = [value, error]
        self.units = self.default[0].units
        if self.dimensionless:
            raise ValueError("UnitfulError can not be dimensionless.")

    @staticmethod
    def parse(string: str) -> Tuple[Unit, Unit]:
        # Matches 5+-6 keV, 5-+0.1 eV and 4±7 eV
        match = re.match(r"(.+)(?:\+\-|\-\+|±)(.+)", string)
        if match is not None:
            val, err = match.groups()
            val = Q_(val)
            err = Q_(val)
            if val.dimensionless and not err.dimensionless:
                val *= err.units
            elif not val.dimensionless and err.dimensionless:
                err *= val.units
        else:
            # Matches 50(6) keV, 50(8 eV) keV
            match = re.match(r"(.+)\((.+)\)\s*(?P<u>.*)", string)
            if match is not None:
                val, err, unit = match.groups()
                val = Q_(val)
                err = Q_(val)
                if unit:
                    unit = Q_(unit)
                    val *= unit
                    if err.dimensionless:
                        err *= unit
            else:
                raise ValueError(f"Could not parse {string}.")
        return val, err

    def validate(self, value) -> Tuple[Unit, Unit]:
        if isinstance(value, str):
            value, error = self.parse(value)
        else:
            value, error = value
            value = Q_(value)
            error = Q_(error)

        if value.dimensionless and error.dimensionless:
            if not self.dimensionless:
                value *= self.units
                error *= self.units
            else:
                value = value.magnitude
                error = error.magnitude
        else:
            try:
                value = value.to(self.units)
            except DimensionalityError:
                raise DimensionalityError(value.units, self.units) from None
            try:
                error = error.to(self.units)
            except DimensionalityError:
                raise DimensionalityError(error.units, self.units) from None
        return (value, error)

    @property
    def dimensionless(self) -> bool:
        return self.default[0].dimensionless
