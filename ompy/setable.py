import warnings

class Setable:
    """

    TODO: Something is fishy
    """
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset
        self.name = fget.__name__

    def __get__(self, obj, objtype=None):
        modified = f"_{self.name}_modified"
        value = self.fget(obj)
        if hasattr(obj, modified):
            if not getattr(obj, modified):
                warnings.warn(f"Parameter '{self.name}' should be set."
                              f" Using default value of {value}")
        else:
            setattr(obj, modified, False)
            warnings.warn(f"Parameter '{self.name}' should be set."
                          f" Using default value of {value}")
        return value

    def __set__(self, obj, val):
        modified = f"_{self.name}_modified"
        setattr(obj, modified, True)
        if self.fset is None:
            return setattr(obj, f"_{self.name}", val)
        else:
            return self.fset(obj, val)

    def setter(self, fset):
        return type(self)(self.fget, fset)
