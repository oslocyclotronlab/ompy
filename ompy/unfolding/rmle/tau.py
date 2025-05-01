import jax.numpy as jnp

def to_tau(mu):
    return jnp.sqrt(mu)
    # return jnp.sqrt(mu)
    # return jnp.log(mu + 1e-10)


def from_tau(tau):
    # return jnp.where(tau < 0, tau**2, tau)
    # Need to be careful with the linear term, as it allows for negative values
    return tau**2  # + 1e-3*tau
    # return tau**2
    # return jnp.exp(tau)