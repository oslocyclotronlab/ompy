from . import ureg, DimensionalityError


class Unitful:
    def __set_name__(self, owner, name):
        self.private_name: str = '_' + name
        self.public_name: str = name

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def validate(self, value):
        try:
            value = ureg.Quantity(value).to(self.units)
        except DimensionalityError:
            value *= self.units
        return value
