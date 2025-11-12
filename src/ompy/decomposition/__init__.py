from .product import *
from ..accel import jax_available
if jax_available():
    from .direct import *
