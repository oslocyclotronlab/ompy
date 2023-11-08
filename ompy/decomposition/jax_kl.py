from jax import lax
from jax._src.typing import Array, ArrayLike
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import promote_args_inexact, promote_dtypes_inexact
from jax._src import custom_derivatives
from jax._src.numpy.util import _wraps
from jax._src.ops import special as ops_special
import jax.numpy as jnp
import numpy as np

"""
The following code is copied from jax.scipy.special
Would like to just import kl_div, but the version is not released yet.
It is a good and stable implementation!
"""
@custom_derivatives.custom_jvp
def xlogy(x: ArrayLike, y: ArrayLike) -> Array:
  # Note: xlogy(0, 0) should return 0 according to the function documentation.
  x, y = promote_args_inexact("xlogy", x, y)
  x_ok = x != 0.
  return jnp.where(x_ok, lax.mul(x, lax.log(y)), jnp.zeros_like(x))

def _xlogy_jvp(primals, tangents):
  (x, y) = primals
  (x_dot, y_dot) = tangents
  result = xlogy(x, y)
  return result, (x_dot * lax.log(y) + y_dot * x / y).astype(result.dtype)
xlogy.defjvp(_xlogy_jvp)

@custom_derivatives.custom_jvp
def xlog1py(x: ArrayLike, y: ArrayLike) -> Array:
  # Note: xlog1py(0, -1) should return 0 according to the function documentation.
  x, y = promote_args_inexact("xlog1py", x, y)
  x_ok = x != 0.
  return jnp.where(x_ok, lax.mul(x, lax.log1p(y)), jnp.zeros_like(x))
def _xlog1py_jvp(primals, tangents):
  (x, y) = primals
  (x_dot, y_dot) = tangents
  result = xlog1py(x, y)
  return result, (x_dot * lax.log1p(y) + y_dot * x / (1 + y)).astype(result.dtype)
xlog1py.defjvp(_xlog1py_jvp)

@custom_derivatives.custom_jvp
def _xlogx(x):
  """Compute x log(x) with well-defined derivatives."""
  return xlogy(x, x)

def _xlogx_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  return  _xlogx(x), x_dot * (lax.log(x) + 1)
_xlogx.defjvp(_xlogx_jvp)

def kl_div(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
    p, q = promote_args_inexact("kl_div", p, q)
    zero = _lax_const(p, 0.0)
    both_gt_zero_mask = lax.bitwise_and(lax.gt(p, zero), lax.gt(q, zero))
    one_zero_mask = lax.bitwise_and(lax.eq(p, zero), lax.ge(q, zero))

    safe_p = jnp.where(both_gt_zero_mask, p, 1)
    safe_q = jnp.where(both_gt_zero_mask, q, 1)

    log_val = lax.sub(
        lax.add(
            lax.sub(_xlogx(safe_p), xlogy(safe_p, safe_q)),
            safe_q,
        ),
        safe_p,
    )
    result = jnp.where(
        both_gt_zero_mask, log_val, jnp.where(one_zero_mask, q, np.inf)
    )
    return result
