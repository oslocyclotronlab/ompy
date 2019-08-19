from typing import Union
from .matrix import Matrix, Vector
from copy import copy


class Action:
    """ Allows for delayed method calls """
    def __init__(self, base='Matrix'):
        self.calls = []
        if base.lower() == 'matrix':
            self.patch(Matrix)
        elif base.lower() == 'vector':
            self.patch(Vector)
        else:
            raise ValueError("'base' must be 'Matrix' or 'Vector'")

    def __call__(self, target: Union[Matrix, Vector]) -> None:
        return self.act_on(target)

    def patch(self, base):
        for name in base.__dict__:
            obj = getattr(base, name)
            if callable(obj):
                setattr(self, name, wrap(self, name))

    def act_on(self, target: Union[Matrix, Vector]):
        for func, args, kwargs in self.calls:
            getattr(target, func)(*args, **kwargs)


def wrap(self, name):
    def wrapper(*args, **kwargs):
        self.calls.append([name, args, kwargs])
    return wrapper
