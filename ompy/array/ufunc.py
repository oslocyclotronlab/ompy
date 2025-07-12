from .matrix import Matrix
from .vector import Vector
from .matrixprotocol import MatrixProtocol
from .index import Index, to_index
from .abstractarray import AbstractArray, to_plot_axis
from .abstractarrayprotocol import AbstractArrayProtocol
from typing import Tuple, overload, Literal, Callable, TypeVar
import numpy as np
from ..stubs import array
from .. import ureg, Quantity
from .. import JAX_WORKING
M = TypeVar('M', bound=MatrixProtocol)
#V = TypeVar('V', bound=VectorProtocol)
A = TypeVar('A', bound=AbstractArrayProtocol)

T = TypeVar('T')

def zeros_like(array: A, **kwargs) -> A:
    match array:
        case Vector():
            return array.clone(values=np.zeros_like(array.values), **kwargs)
        case Matrix():
            return array.clone(values=np.zeros_like(array.values), **kwargs)
        case np.ndarray():
            if array.ndim != 1:
                raise ValueError("Numpy array must be vector.")
            return Vector(X=array, values=np.zeros_like(array), **kwargs)
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")


def empty_like(array: A, **kwargs) -> A:
    match array:
        case Vector():
            return array.clone(values=np.empty_like(array.values), **kwargs)
        case Matrix():
            return array.clone(values=np.empty_like(array.values), **kwargs)
        case np.ndarray():
            if array.ndim != 1:
                raise ValueError("Numpy array must be vector.")
            return Vector(X=array, values=np.empty_like(array), **kwargs)
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")

@overload
def empty(ex: ..., eg: None, **kwargs) -> Vector: ...
@overload
def empty(ex: ..., eg: array, **kwargs) -> Matrix: ...

def empty(ex: ..., eg: array | None = None, **kwargs) -> Vector | Matrix:
    if eg is None:
        values = np.empty(len(ex), **kwargs)
        return Vector(E=ex, values=values)
    values = np.empty((len(ex), len(eg)), **kwargs)
    return Matrix(X=ex, Y=eg, values=values)

def eye(array: array, **kwargs) -> Matrix:
    if isinstance(array, Matrix):
        array = array.X
    return Matrix(X=array, Y=array, values=np.eye(len(array), **kwargs))


def zeros(array: array | int | Tuple[int, int],
          **kwargs) -> AbstractArray:
    raise NotImplementedError()
    if isinstance(array, np.ndarray):
        if array.ndim == 1:
            return Vector(values=np.zeros_like(array, **kwargs))
        elif array.ndim == 2:
            return Matrix(values=np.zeros_like(array, **kwargs))
        else:
            raise ValueError("Array must have dimension < 3.")
    elif isinstance(array, (tuple, list)):
        if len(array) == 1:
            return Vector(values=np.zeros(array, **kwargs))
        elif len(array) == 2:
            return Matrix(values=np.zeros(array, **kwargs))
        else:
            raise ValueError("Array must have dimension < 3.")
    else:
        raise ValueError(f"Expected numpy array or iterable, not {type(array)}.")


def linspace_with_units(start, stop, num=50, **kwargs) -> tuple[np.ndarray, Quantity]:
    # Convert inputs to Quantity objects
    start = Quantity(start)
    stop = Quantity(stop)

    match start.dimensionless, stop.dimensionless:
        case True, True:
            start = start * ureg.keV
            stop = stop * ureg.keV
        case True, False:
            start = start * stop.units
        case False, True:
            stop = stop * start.units
        case False, False:
            if start.is_compatible_with(stop):
                if start.units != stop.units:
                    # If the units are not the same, use stop
                    # TODO Bad solution, fix later.
                    start = start.to(stop.units)
            else:
                raise ValueError(f"Units of start [{start.units}] and stop [{stop.units}] are not compatible.")

    # Create the linspace array
    result = np.linspace(start.magnitude, stop.magnitude, num, **kwargs)

    return result, stop.units


def linspace(start, stop, *args, **kwargs) -> Vector:
    bins, units = linspace_with_units(start, stop, *args, **kwargs)
    if units == ureg.dimensionless:
        units = ureg.Unit("keV")
    index = to_index(bins, edge='left', unit=units, label='Energy', enforce_uniform=True)
    return Vector(X=index, values=np.zeros(len(bins), dtype=float), unit=units)

def index_linspace(start, stop, *args, **kwargs) -> Index:
    bins, units = linspace_with_units(start, stop, *args, **kwargs)
    if units == ureg.dimensionless:
        units = ureg.Unit("keV")
    return to_index(bins, edge='left', unit=units, label='Energy', enforce_uniform=True)

@overload
def fmap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def fmap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def fmap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `functor_map`. Applies a function to the values of an array. Equal to a Haskell fmap <&>
    
    Example::
        >>> fmap(vec, lambda x: x + 1)
        # same as vec.clone(values=vec.values + 1) 
    """
    match array:
        case Vector():
            return array.clone(values=func(*args, **kwargs))
        case Matrix():
            return array.clone(values=func(*args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")

@overload
def umap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def umap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def umap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `unwrap_map`. Applies a function to the values of an array. Equal to a Haskell fmap <&>.

    This is the same as fmap, but uses the array as the first argument of the function and
    unwraps other arguments.

    Example::
        >>> umap(vec, lambda x: x + 1)
        # returns a vector with values of vec + 1
    """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    match array:
        case Vector():
            return array.clone(values=func(array.values, *args, **kwargs))
        case Matrix():
            return array.clone(values=func(array.values, *args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")



def omap(func: Callable[[np.ndarray], T], *args, **kwargs) -> T:
    """ `out_of_map`. Applies `func` while unwrapping the arguments. 
    
    Example::
        >>> omap(np.add, vec1, vec2)
        # returns vec1 + vec2 as np.ndarray
    
    """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    return func(*args, **kwargs)


@overload
def xmap(array: Vector, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Vector: ...
@overload
def xmap(array: Matrix, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> Matrix: ...

def xmap(array: AbstractArray, func: Callable[[np.ndarray], np.ndarray], *args, **kwargs) -> AbstractArray:
    """ `x unwrap_map`. Applies a function to the index of an array. Equal to a Haskell fmap <&>.

    This is the same as fmap, but uses the index of the array as the first argument of the function and
    unwraps other arguments.

    Example::
        >>> xmap(vec, lambda x: x + 1)
        # returns a vector with values of vec.X + 1
    """
    args = [a.values if isinstance(a, AbstractArray) else a for a in args]
    kwargs = {k: v.values if isinstance(v, AbstractArray) else v for k, v in kwargs.items()}
    match array:
        case Vector():
            return array.clone(values=func(array.X, *args, **kwargs))
        case Matrix():
            return array.clone(values=func(array.X, *args, **kwargs))
        case _:
            raise ValueError(f"Expected Array, not {type(array)}.")


def unpack_to_vectors(A: Matrix, ax: int | str = 0) -> list[Vector]:
    """ Unpacks a matrix into a list of vectors. """
    ax_: Literal[0, 1] = A.axis_to_int(ax, False)
    if ax_ == 0:
        return [A.iloc[i, :] for i in range(A.shape[0])]
    elif ax_ == 1:
        return [A.iloc[:, i] for i in range(A.shape[1])]
    else:
        raise ValueError(f"Axis must be 0 or 1, not {ax}.")


def pack_into_matrix(vectors: list[Vector], index: Index | array = None) -> Matrix:
    """ Packs a list of vectors into a matrix. """
    from .index import MidNonUniformIndex
    values = np.stack([vec.values for vec in vectors])
    index_x = vectors[0].X_index
    if index is None:
        index = to_index(list(range(len(vectors))), edge='mid', unit='', label='')
    elif not isinstance(index, Index):
        index = to_index(index, edge='mid', unit='', label='Index')
    return Matrix(X=index, Y=index_x, values=values)


def compute_transition_matrix(Ei, Eg, matrix, binwidth='min', cut=False):
    """
    Compute the matrix corresponding to transitions from Ei to final energy levels Ef.

    Isn't this some sort of shear transformation?
   
    Dumb w[i, j] = h[i, i-j] shouldn't work because of bin widths, _but it does_!!
    
    Parameters:
        Ei (array-like): Array of initial energy levels.
        Eg (array-like): Array of emitted gamma ray energies.
        matrix (array-like): Input matrix of shape (len(Ei), len(Eg)) with counts of observed gamma rays.
        
    Returns:
        tuple: Array of final energy levels Ef and the transition matrix of shape (len(Ei), len(Ef)).
    """
    # Calculate dEi, dEg, and dEf
    dEi = Ei[1] - Ei[0]
    dEg = Eg[1] - Eg[0]
    if binwidth == 'sum':
        dEf = dEg + dEi
    elif binwidth == 'min':
        dEf = min(dEg, dEi)
    else:
        dEf = binwidth
    average= False
    
    # Compute the Ef array
    Ef_min = Ei[0] - Eg[-1] # - dEg  # minimum possible final energy
    Ef_max = Ei[-1] + dEi         # maximum possible final energy
    Ef = np.arange(Ef_min, Ef_max + dEf, dEf)
    dEf = Ef[1] - Ef[0] # !! Tiny float error gave strange bug.
    
    # Create a transition matrix initialized with zeros
    transition_matrix = np.zeros((len(Ei), len(Ef)))
    count = np.zeros_like(transition_matrix)
    
    # Populate the transition matrix
    for i in range(len(Ei)):
        for j in range(len(Eg)):
            k = int((Ei[i] - Eg[j] - Ef[0])//dEf) #search(Ef, Ei[i] - Eg[j])
            transition_matrix[i, k] += matrix[i, j]
            count[i, k] += 1

    if average:
        transition_matrix[count > 1] /= count[count > 1]
    # Remove unecessary zeros
    if cut:
        i = np.argmax(transition_matrix.sum(axis=0) > 0)
        return Ef[i:], transition_matrix[:, i:]
    else:
        return Ef, transition_matrix

def transition_matrix(mat, binwidth='min', cut=False) -> Matrix:
    Ef, val = compute_transition_matrix(mat.Ex, mat.Eg, mat.values, binwidth=binwidth, cut=cut)
    return Matrix(Ei=mat.Ex, Ef=Ef, values=val, xlabel='$E_i$', ylabel='$E_f$')

def exeg_to_efeg(mat, cut=False) -> Matrix:
    if JAX_WORKING:
        Ef, val = compute_EfEg_jax(mat.Ex, mat.Eg, mat.values, cut=cut)
        Ef = np.asarray(Ef)
    else:
        Ef, val = compute_EfEg(mat.Ex, mat.Eg, mat.values, cut=cut)
    return Matrix(Ef=Ef, Eg=mat.Eg, values=val, xlabel='$E_f$', ylabel=r'$E_\gamma$')
        
def compute_EfEg(Ex, Eg, matrix, cut=False):
    """
    Compute the matrix corresponding to transitions from Ei to final energy levels Ef.

    Isn't this some sort of shear transformation?
   
    Dumb w[i, j] = h[i, i-j] shouldn't work because of bin widths, _but it does_!!
    
    Parameters:
        Ei (array-like): Array of initial energy levels.
        Eg (array-like): Array of emitted gamma ray energies.
        matrix (array-like): Input matrix of shape (len(Ei), len(Eg)) with counts of observed gamma rays.
        
    Returns:
        tuple: Array of final energy levels Ef and the transition matrix of shape (len(Ei), len(Ef)).
    """
    # Calculate dEi, dEg, and dEf
    dEx = Ex[1] - Ex[0]
    dEg = Eg[1] - Eg[0]
    dEf = min(dEg, dEx)
    average= False
    
    # Compute the Ef array
    Ef_min = Ex[0] - Eg[-1] # - dEg  # minimum possible final energy
    Ef_max = Ex[-1] + dEx         # maximum possible final energy
    Ef = np.arange(Ef_min, Ef_max + dEf, dEf)
    dEf = Ef[1] - Ef[0] # !! Tiny float error gave strange bug.
    
    # Create a transition matrix initialized with zeros
    transition_matrix = np.zeros((len(Ef), len(Eg)))
    count = np.zeros_like(transition_matrix)
    
    # Populate the transition matrix
    for i in range(len(Ex)):
        for j in range(len(Eg)):
            k = int((Ex[i] - Eg[j] - Ef[0])//dEf) #search(Ef, Ex[i] - Eg[j])
            transition_matrix[k, j] += matrix[i, j]
            count[k, j] += 1

    if average:
        transition_matrix[count > 1] /= count[count > 1]
    # Remove unecessary zeros
    if cut:
        x_sum = transition_matrix.sum(axis=1) > 0
        i = np.argmax(x_sum)
        j = len(Ef) - np.argmax(np.flip(x_sum))
        return Ef[i:j], transition_matrix[i:j, :]
    else:
        return Ef, transition_matrix

if JAX_WORKING:
    import jax.numpy as jnp
    from jax import vmap
    import jax
    def compute_EfEg_jax(Ex, Eg, matrix, cut=False):
        """
        JAX version of compute_EfEg. Computes the matrix corresponding to transitions from Ei to final energy levels Ef.
        
        Parameters:
            Ex (jax.Array): Array of initial energy levels.
            Eg (jax.Array): Array of emitted gamma ray energies.
            matrix (jax.Array): Input matrix of shape (len(Ex), len(Eg)) with counts of observed gamma rays.
            cut (bool): Whether to remove unnecessary zeros from the output.
            
        Returns:
            tuple: Array of final energy levels Ef and the transition matrix of shape (len(Ef), len(Eg)).
        """
        
        # Calculate dEi, dEg, and dEf
        dEx = Ex[1] - Ex[0]
        dEg = Eg[1] - Eg[0]
        dEf = jnp.minimum(dEg, dEx)
        
        # Compute the Ef array
        Ef_min = Ex[0] - Eg[-1]  # minimum possible final energy
        Ef_max = Ex[-1] + dEx    # maximum possible final energy
        Ef = jnp.arange(Ef_min, Ef_max + dEf, dEf)
        dEf = Ef[1] - Ef[0]  # Ensure consistent bin width
        
        # Create a function to compute the index for a single (Ex, Eg) pair
        def get_index(Ex_i, Eg_j):
            return jnp.floor((Ex_i - Eg_j - Ef[0]) / dEf).astype(jnp.int32)
        
        # Vectorize the index computation over Ex and Eg
        indices = vmap(lambda x: vmap(lambda y: get_index(x, y))(Eg))(Ex)
        
        # Create the transition matrix using scatter_add
        transition_matrix = jnp.zeros((len(Ef), len(Eg)))
        
        # Use a loop over Ex indices since scatter_add doesn't support 2D indices directly
        def body_fun(i, transition_matrix):
            indices_i = indices[i]
            matrix_i = matrix[i]
            # Add values to the transition matrix using scatter_add
            dimension_numbers = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(1,),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,)
            )
            # Stack indices into a single array
            indices_2d = jnp.stack([indices_i, jnp.arange(len(Eg))], axis=1)
            # Reshape matrix_i to have rank 2
            matrix_i_reshaped = matrix_i.reshape(1, -1)
            return jax.lax.scatter_add(
                transition_matrix,
                indices_2d,
                matrix_i_reshaped,
                dimension_numbers=dimension_numbers,
                indices_are_sorted=False,
                unique_indices=False
            )
        
        # Apply the loop
        transition_matrix = jax.lax.fori_loop(
            0, len(Ex), body_fun, transition_matrix
        )
        
        if cut:
            # Compute row sums and find non-zero rows
            x_sum = jnp.sum(transition_matrix, axis=1) > 0
            i = jnp.argmax(x_sum)
            j = len(Ef) - jnp.argmax(jnp.flip(x_sum))
            return Ef[i:j], transition_matrix[i:j, :]
        else:
            return Ef, transition_matrix

import jax
import jax.numpy as jnp
from jax.ops import segment_sum

def compute_EfEg_jax(Ex, Eg, matrix, cut=False):
    # bin widths
    dEx = Ex[1] - Ex[0]
    dEg = Eg[1] - Eg[0]
    dEf = jnp.minimum(dEg, dEx)

    # Ef array
    Ef_min = Ex[0] - Eg[-1]
    Ef_max = Ex[-1] + dEx
    Ef = jnp.arange(Ef_min, Ef_max + dEf, dEf)
    dEf = Ef[1] - Ef[0]              # re-compute to avoid tiny float errors
    num_Ef = Ef.shape[0]

    # For each (i,j) compute the Ef-bin index K[i,j]
    K = jnp.floor_divide(Ex[:, None] - Eg[None, :] - Ef[0], dEf).astype(int)
    valid = (K >= 0) & (K < num_Ef)
    K_clipped = jnp.clip(K, 0, num_Ef - 1)

    # Define a helper that, for one column j, does:
    #   hist[k] = sum_i matrix[i,j] where K_clipped[i,j] == k  (and valid)
    #   cnt[k]  = count of i where K[i,j] == k
    def hist_and_count(vals_j, Kj, vj):
        hist = segment_sum(vals_j * vj, Kj, num_segments=num_Ef)
        cnt  = segment_sum(vj.astype(vals_j.dtype), Kj, num_segments=num_Ef)
        return hist, cnt

    # vmap over the gamma-ray axis (columns)
    hist_cols, cnt_cols = jax.vmap(hist_and_count, in_axes=(1,1,1))(matrix, K_clipped, valid)

    # hist_cols has shape (len(Eg), num_Ef) → transpose to (num_Ef, len(Eg))
    transition_matrix = hist_cols.T
    count_matrix      = cnt_cols.T

    # (optional) average if you want
    # transition_matrix = jnp.where(count_matrix > 1,
    #                               transition_matrix / count_matrix,
    #                               transition_matrix)

    if cut:
        # drop leading/trailing all-zero Ef rows
        nonzero = transition_matrix.sum(axis=1) > 0
        i0 = jnp.argmax(nonzero)
        i1 = transition_matrix.shape[0] - jnp.argmax(nonzero[::-1])
        return Ef[i0:i1], transition_matrix[i0:i1, :]
    else:
        return Ef, transition_matrix

    
def efeg_to_exeg(mat, cut=False) -> Matrix:
    Ei, val = compute_EiEg_jax(mat.Ef, mat.Eg, mat.values, mat.Ex, cut=cut)
    Ei = np.asarray(Ei)
    return Matrix(Ei=Ei, Eg=mat.Eg, values=val, xlabel='$E_i$', ylabel=r'$E_\gamma$')

        
def compute_EiEg_jax(Ef, Eg, matrix_feg, Ei, cut=False):
    """
    Inverse of compute_EfEg_jax: take a matrix in (Ef, Eg) and
    rebin it back into (Ei, Eg), where Ei is the original Ex axis.
    
    Args:
      Ef         1D array of final energies (length Nf)
      Eg         1D array of gamma energies (length Ng)
      matrix_feg 2D array shape (Nf, Ng): your Ef–Eg histogram
      Ei         1D array of initial energies (length Ni), same as Ex
      cut        if True, drop any leading/trailing all-zero Ei rows
    
    Returns:
      (Ei_axis, transition_matrix) where transition_matrix has shape (Ni, Ng)
    """
    # bin widths
    dEi = Ei[1] - Ei[0]
    dEg = Eg[1] - Eg[0]
    # recompute Ef bin width
    dEf = Ef[1] - Ef[0]
    
    # For each (k,j) in (Ef,Eg), compute the Ei‐bin index I[k,j]
    # Ei = Ef + Eg  =>  I = floor((Ef[k] + Eg[j] - Ei[0]) / dEi)
    I = jnp.floor_divide(
        Ef[:, None] + Eg[None, :] - Ei[0],
        dEi
    ).astype(int)
    valid = (I >= 0) & (I < Ei.shape[0])
    I_clipped = jnp.clip(I, 0, Ei.shape[0] - 1)
    
    # helper: for one Eg‐column j, sum over Ef‐axis
    def hist_and_count(vals_k, Ik, vmask):
        # vals_k:    shape (Nf,), the column of matrix_feg
        # Ik:        shape (Nf,), target bin indices along Ei
        # vmask:     shape (Nf,), bool mask of valid
        hist = segment_sum(vals_k * vmask, Ik, num_segments=Ei.shape[0])
        cnt  = segment_sum(vmask.astype(vals_k.dtype), Ik, num_segments=Ei.shape[0])
        return hist, cnt
    
    # vmap over the Ng columns
    hist_cols, cnt_cols = jax.vmap(
        hist_and_count, in_axes=(1,1,1)
    )(matrix_feg, I_clipped, valid)
    
    # hist_cols is shape (Ng, Ni) → transpose efeg_to_exegto (Ni, Ng)
    transition_matrix = hist_cols.T
    count_matrix      = cnt_cols.T
    
    # if you want to normalize by counts, you can uncomment:
    # transition_matrix = jnp.where(
    #     count_matrix > 1,
    #     transition_matrix / count_matrix,
    #     transition_matrix
    # )
    
    if cut:
        # drop leading/trailing all-zero Ei rows
        nonzero = transition_matrix.sum(axis=1) > 0
        i0 = jnp.argmax(nonzero)
        i1 = transition_matrix.shape[0] - jnp.argmax(nonzero[::-1])
        return Ei[i0:i1], transition_matrix[i0:i1, :]
    else:
        return Ei, transition_matrix