from typing import Union, List, Iterable, Any, Callable, Tuple, Dict, TypeVar, Generic
from .matrix import Matrix
from .vector import Vector

T = TypeVar('T', Matrix, Vector)

class Action(Generic[T]):
    """ Allows for delayed method calls """
    def __init__(self, base='Matrix'):
        self.base = base
        self.calls = []
        if base.lower() == 'matrix':
            self.patch(Matrix)
        elif base.lower() == 'vector':
            self.patch(Vector)
        else:
            raise ValueError("'base' must be 'Matrix' or 'Vector'")

    def __call__(self, target: T) -> None:
        return self.act_on(target)

    def patch(self, base):
        for name in base.__dict__:
            obj = getattr(base, name)
            if callable(obj):
                setattr(self, name, wrap(self, name))

    def act_on(self, target: T, **kwargs) -> List[Any]:
        ret_vals: List[Any] = []
        for func, args, _kwargs in self.calls:
            ret = getattr(target, func)(*args, **(_kwargs | kwargs))
            ret_vals.append(ret)
        return ret_vals

    def map(self, collection: Iterable[T]) -> List[List[Any]]:
        ret_vals: List[List[Any]] = []
        for member in collection:
            ret = self.act_on(member)
            ret_vals.append(ret)
        return ret_vals

    def curry(self, **kwargs):
        for call in self.calls:
            call[2].update(kwargs)

    def get_args(self, method: str) -> ([Any, ...], Dict[str, Any]):
        for (name, args, kwargs) in self.calls:
            if name == method:
                return args, kwargs
        raise ValueError(f"{self.base} has not method {method}.")


def wrap(self, name: str) -> Callable:
    """ Saves the function call and arguments for later use

    Returns an instance of self to allow for dot chaining

    Args:
        name: The name of the callable
    Returns:
        The wrapped function
    """
    def wrapper(*args, **kwargs):
        self.calls.append([name, args, kwargs])
        return self
    return wrapper
