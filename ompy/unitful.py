from . import ureg, DimensionalityError


class Unitful:
    def __init__(self, default):
        self.default = ureg.Quantity(default)
        self.units = self.default.units

    def __set_name__(self, owner, name):
        self.private_name: str = '_' + name
        self.public_name: str = name

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    def __get__(self, obj, objtype=None):
        if not hasattr(obj, self.private_name):
            setattr(obj, self.private_name, self.default)
        return getattr(obj, self.private_name)

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

    def dimensionless(self) -> bool:
        return self.default.dimensionless
