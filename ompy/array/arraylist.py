from __future__ import annotations
from typing import Iterator, TypeVar, Generic, Self, Sequence
from .abstractarray import AbstractArray, ARRAY_CLASSES
from .index import Index
from pathlib import Path
from .. import H5PY_AVAILABLE, __full_version__
import numpy as np
from dataclasses import asdict
import time
import logging
from tqdm.auto import tqdm
from ..version import warn_version

LOG = logging.getLogger(__name__)


if H5PY_AVAILABLE:
    LOG.debug("Defaulting to HDF5")
    from .filehandling import dict_to_hdf5, hdf5_to_dict
    import h5py

""" TODO
-[ ] Add support for saving to npz
-[x] Add support for saving to hdf5
-[ ] Add support for reading from npz
-[x] Add support for reading from hdf5
-[ ] Add support for ROOT
-[ ] Add support for partial loading
"""

T = TypeVar('T', bound=(AbstractArray))


class ArrayList(Sequence[T]):
    """ Wrapper around a list of arrays """

    def __init__(self, prototype: T | None = None):
        self.array: list[T] = []
        self.prototype: T | None = prototype

    def check_item(self, item: T) -> None:
        if self.prototype is not None:
            if not self.prototype.is_compatible_with(item):
                raise TypeError("Incompatible type {type(item)}")
        else:
            raise TypeError("Prototype must be set before appending")

    def append(self, item: T) -> None:
        self.check_item(item)
        self.array.append(item.values)

    def extend(self, items: list[T] | np.ndarray) -> None:
        if isinstance(items, np.ndarray):
            if self.prototype is None:
                raise TypeError("Prototype must be set before extending with numpy arrays")
            # Assume first index is the "list" index
            if items.shape[1:] != self.prototype.values.shape:
                raise TypeError(f"Incompatible shape. Got {items.shape[1:]} expected {self.prototype.values.shape}")
            for i in range(items.shape[0]):
                self.array.append(items[i])
        else:
            for item in items:
                self.check_item(item)
            self.array.extend([item.values for item in items])

    def to_arrays(self, **kwargs) -> Iterator[T]:
        assert self.prototype is not None
        return (self.prototype.clone(values=array, **kwargs) for array in self.array)

    @classmethod
    def from_list(cls, lst: list[T], prototype: T | None = None) -> Self:
        assert len(lst) > 0
        if prototype is None:
            prototype = lst[0]
        array_list = cls(prototype)
        array_list.extend(lst)
        return array_list

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> T:
        return self.array[index]

    def __setitem__(self, index: int, value: T) -> None:
        self.check_item(value)
        self.array[index] = value.values

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        if self.prototype is None:
            return
        # Save as npz or hdf5
        if H5PY_AVAILABLE:
            path = path.with_suffix(".hdf5")
            kwargs.setdefault('compression', 'gzip')
            with h5py.File(path, 'w') as f:
                self.insert_into_tree(f, '/', **kwargs)

    @classmethod
    def from_path(cls, path: Path, root='/', read_only: int | None = None, **kwargs) -> Self:
        path = Path(path)
        LOG.debug(f"Trying to read from {path}")
        if path.suffix == '.hdf5':
            LOG.debug("File has hdf5 suffix")
            return cls.from_hdf5(path, root, **kwargs)
        elif path.suffix == '':
            LOG.debug("File has no suffix")
            if H5PY_AVAILABLE:
                if path.with_suffix('.hdf5').exists():
                    LOG.debug("Assuming hdf5 file")
                    return cls.from_hdf5(path.with_suffix('.hdf5'), root, read_only, **kwargs)
        raise NotImplementedError("Only hdf5 is supported for now")

    @classmethod
    def from_hdf5(cls, path: Path, root='/', read_only: int | None = None, **kwargs) -> Self:
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not available")
        with h5py.File(path, 'r') as f:
            return cls.from_tree(f, root, read_only, **kwargs)

    @classmethod
    def from_tree(cls, tree, root='/', read_only: int | None = None, **kwargs) -> Self:
        """ Read the arrays from a hdf5 tree """
        start = time.time()
        array_type = tree[root].attrs['type']
        array_cls = ARRAY_CLASSES[array_type]
        X_dict= hdf5_to_dict(tree, root + 'X_index/')
        X_index = Index.from_dict(X_dict)
        if array_cls._ndim > 1:
            Y_dict = hdf5_to_dict(tree, root + 'Y_index/')
            Y_index = Index.from_dict(Y_dict)
        meta = hdf5_to_dict(tree, root + 'meta/')
        version = tree[root].attrs['version']
        warn_version(version)
        array_paths = sorted(tree[root + 'array'], key=int)
        arrays = []
        for i, arr_path in enumerate(array_paths):
            if read_only is not None and i >= read_only:
                break
            arrays.append(np.asarray(tree[root + 'array/' + arr_path]))
        LOG.debug(f"Read {len(arrays)} {array_type} in {time.time() - start:.2f} s")
        proto_array = arrays[0]
        if array_cls._ndim == 1:
            proto = array_cls(X=X_index, values=proto_array, metadata=meta)
        elif array_cls._ndim == 2:
            proto = array_cls(X=X_index, Y=Y_index, values=proto_array, metadata=meta)
        else:
            raise NotImplementedError("Only 1D and 2D arrays are supported")

        arraylist = cls(proto)
        arraylist.array.extend(arrays)
        return arraylist

    def insert_into_tree(self, tree, root='/', compact: bool = False, disable_tqdm: bool = False, **kwargs):
        """ Insert the arrays into a hdf5 tree """
        if self.prototype is None:
            return
        if root[-1] != '/':
            root += '/'
        LOG.debug(f"Inserting {len(self.array)} arrays into tree at {root}"
                  f" (compact={compact}, kwargs={kwargs})")
        start = time.time()
        tree.create_group(root + 'X_index')
        dict_to_hdf5(tree, self.prototype.X_index.to_dict(), root + 'X_index/')
        tree.create_group(root + 'Y_index')
        dict_to_hdf5(tree, self.prototype.Y_index.to_dict(), root + 'Y_index/')
        tree.create_group(root + 'meta')
        dict_to_hdf5(tree, asdict(self.prototype.metadata), root + 'meta/')
        tree[root].attrs['version'] = __full_version__
        tree[root].attrs['type'] = self.prototype.__class__.__name__
        tree[root].attrs['compact'] = compact
        if compact:
            array = np.stack(self.array)
            tree.create_dataset(f"{root}/array", data=array, **kwargs)
        else:
            tqdm_ = tqdm if not disable_tqdm else lambda x: x
            for i in tqdm_(range(len(self.array))):
                tree.create_dataset(f"{root}/array/{i}", data=self.array[i], **kwargs)
        LOG.debug(f"Saved {len(self.array)} arrays in {time.time() - start:.2f} s")
