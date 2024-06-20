from __future__ import annotations

from contextlib import contextmanager
import logging
import numpy as np
from abc import ABC, abstractmethod
from .index import Index
from .. import Unit, JAX_WORKING
from ..computation_context import new_context, active_context, ComputationContext, delete_context
from .abstractarrayprotocol import AbstractArrayProtocol
from nptyping import NDArray, Shape, Floating
from typing import Iterator, Self, Literal, Callable, overload, TypeAlias


LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

#TODO Implement all of the i-methods and logical methods
# [ ] __imatmul__
# [ ] Make +- work when one index lines inside the other
#     - How to handle different array types, with/without error?
#     - Remember units

if JAX_WORKING:
    import jax
    from jaxlib.xla_extension import ArrayImpl
    from jaxlib.xla_extension import Device as _Device
    Device: TypeAlias = Literal['cpu'] | _Device
else:
    Device: TypeAlias = Literal['cpu']


ARRAY_CLASSES: dict[str, type[AbstractArray]] = {}


class AbstractArray(AbstractArrayProtocol, ABC):
    __default_unit: Unit = Unit('keV')
    _ndim = -1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ARRAY_CLASSES[cls.__name__] = cls

    def __init__(self, values: np.ndarray):
        self.values: NDArray[Shape['*', ...], Floating] = values
        # The context in which the array was created
        # only used for internal bookkeeping. Should not be saved
        # or modified by the user.
        self.__context: ComputationContext | None = active_context()
        if self.__context is not None:
            self.__context.push(self)

    def _disconnect_context(self, context: ComputationContext | None = None):
        if context is not None:
            if context != self.__context:
                raise RuntimeError("Context mismatch")
        self.__context = None

    def _get_context(self) -> ComputationContext | None:
        return self.__context

    @property
    def __array_interface__(self):
        return self.values.__array_interface__

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # TODO Untested. Might summon demons.
        cls = type(self)
        # Replace ArrayWrapper instances with their .values attribute
        inputs = tuple(i.values if isinstance(i, AbstractArray) else i for i in inputs)

        # Perform the operation on the underlying numpy arrays
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # Wrap the result back in an ArrayWrapper (or handle other result types as needed)
        if method == 'at':
            # In-place method, no return value
            return None
        elif isinstance(result, tuple):
            # Multiple return values
            return tuple(self.clone(values=x) for x in result)
        elif method == 'reduceat':
            # reduceat returns a single array
            return self.clone(values=result)
        else:
            # Standard ufunc, single return value
            return self.clone(values=result)

    @abstractmethod
    def is_compatible_with(self, other: AbstractArray | Index) -> bool: ...

    @abstractmethod
    def clone(self, **kwargs) -> Self: ...

    def copy(self, **kwargs) -> Self:
        return self.clone(copy=True, **kwargs)

    def check_or_assert(self, other) -> np.ndarray | float:
        if isinstance(other, AbstractArray):
            if not self.shape == other.shape:
                raise ValueError(f"Incompatible shapes. {self.shape} != {other.shape}")
            if not self.is_compatible_with(other):
                raise ValueError("Incompatible binning.")
            other = other.values
        return other

    def __sub__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values - other, name='')

    def __rsub__(self, other) -> Self:
        result = self.clone(values = other - self.values, name='')
        return result

    def __add__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values + other, name='')

    def __radd__(self, other) -> Self:
        x = self.__add__(other)
        return x

    def __mul__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values * other, name='')

    def __rmul__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = other * self.values, name='')

    def __truediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values / other, name='')

    def __rtruediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = other / self.values, name='')

    def __pow__(self, val: float) -> Self:
        return self.clone(values = self.values ** val)

    def __iand__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values &= other
        return self

    def __and__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values & other, name='')

    def __or__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values | other, name='')

    def __ior__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values |= other
        return self

    def __ixor__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values ^= other
        return self

    def __xor__(self, other: AbstractArrayProtocol | np.ndarray) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values = self.values ^ other, name='')

    def __lshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values=self.values << other, name='')

    def __rshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        return self.clone(values=self.values >> other, name='')

    def __ilshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values <<= other
        return self

    def __irshift__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values >>= other
        return self

    def __iadd__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values += other
        return self

    def __isub__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values -= other
        return self

    def __imul__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values *= other
        return self

    def __itruediv__(self, other) -> Self:
        other = self.check_or_assert(other)
        self.values /= other
        return self

    def __invert__(self):
        return self.clone(values=~self.values)

    @abstractmethod
    def __matmul__(self, other) -> AbstractArray: ...

    #def __rmatmul__(self, other) -> AbstractArray:

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __getitem__(self, key):
        return self.values.__getitem__(key)

    def __setitem__(self, key, item):
        return self.values.__setitem__(key, item)

    def __getattr__(self, attr):
        """ Fallback to numpy if AbstractArray does not have the attribute
        """
        name = self.__class__.__name__
        if attr.startswith("_"):
            raise AttributeError(f"'{name}' object has no attribute {attr}")
        res = getattr(self.__dict__['values'], attr, None)
        if res is not None:
            return res
        # Can't use AttributeError as that is handled exceptionally and
        # causes a really, reeally wierd bug
        raise Exception(f"Neither {name} nor {name}.values has '{attr}'")

    def __len__(self) -> int:
        return len(self.values)

    def __neg__(self):
        return self.clone(values=-self.values)

    def __lt__(self, other):
        return self.values < other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __ge__(self, other):
        return self.values >= other

    def __abs__(self):
        return self.clone(values=np.abs(self.values))

    @property
    def vlabel(self) -> str:
        return self.metadata.vlabel

    @property
    def valias(self) -> str:
        return self.metadata.valias

    @property
    def name(self) -> str:
        return self.metadata.name

    @name.setter
    def name(self, name: str):
        self.metadata = self.metadata.update(name=name)

    @property
    def title(self) -> str:
        return self.name

    @title.setter
    def title(self, title: str):
        self.name = title

    def astype(self, dtype) -> Self:
        return self.clone(values=self.values.astype(dtype))

    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> Self:
        return self.clone(values=func(self.values))

    def sample(self, N: int, mask: np.ndarray | None = None, **kwargs) -> list[Self]:
        """ Draw `N` poisson samples from the array.

        The `mask` specifies values to ignore. If not set, the mask is assumed to be
        all zero elements "after" the diagonal.
        Zeros that are not masked are treated to have Poisson std = 3 to allow for
        non-zero sampling.
        
        Args:
            N (int): The number of samples to generate.
            mask (np.ndarray, optional): A boolean mask array to apply zeros to. If not provided, the last non-zero elements are used.
        
        Returns:
            list[Self]: An iterator that yields `N` new instances of the array, with the sampled values.
        """
        return list(self.sample_it(N, mask, **kwargs))

    def sample_it(self, N: int, mask: np.ndarray | None = None, zero_value: int = 0,
                  zero_limit: int = 0, **kwargs) -> Iterator[Self]:
        """ Draw `N` poisson samples from the array.

        The `mask` specifies values to ignore. If not set, the mask is assumed to be
        all zero elements "after" the diagonal.
        Zeros that are not masked are treated to have Poisson std = 3 to allow for
        non-zero sampling.
        
        Args:
            N (int): The number of samples to generate.
            mask (np.ndarray, optional): A boolean mask array to apply zeros to. If not provided, the last non-zero elements are used.
        
        Returns:
            Iterator[Self]: An iterator that yields `N` new instances of the array, with the sampled values.
        """

        if mask is None:
            if self.ndim == 1:
                mask = np.ones(len(self.values), dtype=bool)
                i = self.last_nonzero(**kwargs)
                mask[i:] = False
            else:
                mask = self.last_nonzeros(**kwargs)
        X = np.where(self.values <= zero_limit, zero_value, self.values)
        X[~mask] = 0
        for _ in range(N):
            sample = self.clone(values=np.random.poisson(X))
            yield sample

    @overload
    def to_gpu(self, inplace: Literal[False] = ..., device=...) -> Self: ...

    @overload
    def to_gpu(self, inplace: Literal[True] = ..., device=...) -> None: ...
            
    def to_gpu(self, inplace: bool = False, device=None) -> Self | None:
        """
        Move the .values to the GPU.

        Args:
            inplace (bool, optional): If True, the operation is performed in-place. Defaults to False.
            device (Device, optional): The device to which the values are moved. Defaults to None.

        Returns:
            Self | None: Returns None if inplace is True, otherwise returns a new instance with values moved to GPU.
        """
        if inplace:
            self.values = to_gpu(self.values, device)
            return None
        else:
            return self.clone(values=to_gpu(self.values, device))

    @overload
    def to_cpu(self, inplace: Literal[False] = ..., device=...) -> Self: ...
    
    @overload
    def to_cpu(self, inplace: Literal[True] = ..., device=...) -> None: ...

    def to_cpu(self, inplace: bool = False, device: Device | None = None) -> Self | None:
        """
        Move the .values to the CPU.

        Args:
            inplace (bool, optional): If True, the operation is performed in-place. Defaults to False.
            device (Device, optional): The device to which the values are moved. Defaults to None.

        Returns:
            Self | None: Returns None if inplace is True, otherwise returns a new instance with values moved to CPU.
        """
        if inplace:
            self.values = to_cpu(self.values, device)
            return None
        else:
            return self.clone(values=to_cpu(self.values, device))

    @overload
    def to_device(self, device: Device | None = ..., inplace: Literal[False] = ...) -> Self: ...

    @overload
    def to_device(self, device: Device | None = ..., inplace: Literal[True] = ...) -> None: ...

    def to_device(self, device: Device | None = None, inplace: bool = False) -> Self | None:
        """
        Move the .values to the specified device.

        Args:
            device (Device, optional): The device to which the values are moved. Defaults to None.
            inplace (bool, optional): If True, the operation is performed in-place. Defaults to False.

        Returns:
            Self | None: Returns None if inplace is True, otherwise returns a new instance with values moved to the specified device.
        """
        if inplace:
            self.values = to_device(self.values, device)
            return None
        else:
            return self.clone(values=to_device(self.values, device))

    @property
    def device(self) -> Device:
        """
        Get the device of the .values.

        Returns:
            Device: The device of the .values.
        """
        return _device(self.values)

    @overload
    def as_numpy(self, inplace: Literal[False] = ...) -> Self: ...

    @overload
    def as_numpy(self, inplace: Literal[True] = ...) -> None: ...

    def as_numpy(self, inplace: bool = False) -> Self | None:
        if inplace:
            self.values = np.asarray(self.values)
            return None
        else:
            return self.clone(values=np.asarray(self.values))


def to_device(array, device: Device):
    match device:
        case 'cpu' | 'gpu?':
            return array
        case _:
            if device != 'cpu':
                raise RuntimeError(f"Only CPU device is supported on this system, not {device}.")


def to_gpu(array, device: Device | None = None):
    raise RuntimeError("JAX is not working on this system.")


def to_cpu(array, device: Device | None = None):
    return array


def _device(array) -> Device: 
    return 'cpu'


def get_default_gpu() -> Device:
    raise RuntimeError("JAX is not working on this system.")


def get_default_cpu() -> Device:
    return 'cpu'


if JAX_WORKING:
    def get_default_gpu() -> Device:
        return jax.devices('gpu')[0]

    def get_default_cpu() -> Device:
        return jax.devices('cpu')[0]

    def to_device(array, device: Device):
        if device == 'cpu':
            device = get_default_cpu()

        if not isinstance(array, ArrayImpl) or not _device(array) == device:
            return jax.device_put(array, device)
        return array

    def to_gpu(array, device: Device | None = None):
        if device is None:
            device = get_default_gpu()
        return to_device(array, device)

    def to_cpu(array, device: Device | None = None):
        if device is None:
            device = get_default_cpu()
        return to_device(array, device)

    def _device(array) -> Device:
        if isinstance(array, ArrayImpl):
            return array.device_buffer.device()
        else:
            return 'cpu'


def on_gpu(*arr: AbstractArray, revert: bool = True, endpoint: Device | Literal['leave', 'numpy'] = 'leave'):
    """
    Context manager that moves the given AbstractArray objects to the GPU,
    and returns them to their original devices. Created variables in the 
    context are not handled.
    
    Args:
        *arr (AbstractArray): Variable number of AbstractArray objects to be moved to the GPU.
        revert (bool):
            If True, the objects are reverted to their original devices after the context is exited.
            If False, the objects are left on the GPU after the context is exited.
        endpoint (Device, optional): The device to which the objects within the
            active context are moved to once it exits. Defaults to 'leave', leaving
            them on the same device.

    Yields:
        None
    
    Raises:
        None
    
    Returns:
        None
    """
    return on_device('gpu', *arr, revert=revert, endpoint=endpoint)


def on_cpu(*arr: AbstractArray, revert: bool = True, endpoint: Device | Literal['leave', 'numpy'] = 'leave'):
    """
    Context manager that moves the given AbstractArray objects to the CPU,
    and returns them to their original devices. Created variables in the
    context are not handled.

    Args:
        *arr (AbstractArray): Variable number of AbstractArray objects to be moved to the GPU.
        revert (bool):
            If True, the objects are reverted to their original devices after the context is exited.
            If False, the objects are left on the GPU after the context is exited.
        endpoint (Device, optional): The device to which the objects within the
            active context are moved to once it exits. Defaults to 'leave', leaving
            them on the same device.

    Yields:
        None

    Raises:
        None

    Returns:
        None
    """
    return on_device('cpu', *arr, revert=revert, endpoint=endpoint)


@contextmanager
def on_device(device: Device, *arr: AbstractArray, revert: bool = True,
              endpoint: Device | Literal['leave', 'numpy'] = 'leave'):
    """
    Context manager that moves the given AbstractArray objects to the specified device,
    and returns them to their original devices. Created variables in the context are not handled.

    Args:
        device (Device): The device to which the objects are moved to.
        *arr (AbstractArray): Variable number of AbstractArray objects to be moved to the GPU.
        revert (bool):
            If True, the objects are reverted to their original devices after the context is exited.
            If False, the objects are left on the GPU after the context is exited.
        endpoint (Device, optional): The device to which the objects within the
            active context are moved to once it exits. Defaults to 'leave', leaving
            them on the same device.

    Yields:
        None

    Raises:
        None

    Returns:
        None
    """
    devices = []
    array_types = []
    match device:
        case 'cpu':
            device = get_default_cpu()
        case 'gpu':
            device = get_default_gpu()
        case 'gpu?':
            try:
                device = get_default_gpu()
            except Exception:
                device = get_default_cpu()


    with new_context() as compute_context:
        try:
            for a in arr:
                devices.append(a.device)
                array_types.append(type(a.values))
                a.to_device(inplace=True, device=device)
            yield
        finally:
            # Handle the original variables
            if revert:
                for a, device, atype in zip(arr, devices, array_types):
                    a.to_device(inplace=True, device=device)
                    if atype == np.ndarray:
                        a.as_numpy(inplace=True)

            # Handle the created variables
            match endpoint:
                case 'leave':
                    pass
                case 'numpy':
                    for a in compute_context:
                        a.to_cpu(inplace=True)
                        a.as_numpy(inplace=True)
                case _:
                    for a in compute_context:
                        a.to_device(inplace=True, device=endpoint)


def to_plot_axis(axis: int | str) -> Literal[1,2,3]:
    """Maps axis to 0, 1 or 2 according to which axis is specified

    Args:
        axis: Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y'), or
              (2, 'both', 'egex', 'exeg', 'xy', 'yx')
    Returns:
        An int describing the axis in the basis of the plot,
        _not_ the values' dimension.

    Raises:
        ValueError if the axis is not supported
    """
    try:
        axis = axis.lower()
    except AttributeError:
        pass

    if axis in (0, 'eg', 'x'):
        return 0
    elif axis in (1, 'ex', 'y'):
        return 1
    elif axis in (2, 'both', 'egex', 'exeg', 'xy', 'yx'):
        return 2
    else:
        raise ValueError(f"Unrecognized axis: {axis}")


def fetch(array, dtype, order) -> np.ndarray:
    order = 'C' if order is None else order
    return np.asarray(array, dtype=dtype, order=order)

if JAX_WORKING:
    import jax.numpy as jnp
    def fetch(array, dtype, order) -> jnp.ndarray:
        if isinstance(array, jnp.ndarray):
            if order is not None:
                # order != 'K' not supported!
                # We let jax throw the error, and perhaps not throw
                # the error in the future
                return jnp.asarray(array, order=order, dtype=dtype)
            return array.astype(dtype)
        else:
            order = 'C' if order is None else order
            return np.asarray(array, dtype=dtype, order=order)