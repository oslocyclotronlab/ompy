from __future__ import annotations
from ..array import Matrix, Vector
import numba
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import kl_div
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from typing import Literal
"""
We have a problem with vanishing gradients in the optimization.
[solved?] Perhaps split up the cumulative sum into chunks and backpropagate each.
Can perhaps treat it as a dynamic programming problem, iteratively processing
larger and larger matrices while keeping the preivous results semi-frozen.
Softmax each row to enforce normalization, maybe also stabilize.

[solved] It doesn't seem to be converging to the correct solution.

Weighting the loss function encourages better convergence to the correct solution.
Maybe? Not sure.

[solved] It gives NaN when given the correct solution. Numerical issue, or conceptual?

[solved] What if a state decays to a state that does not exist in the original FG matrix?
There might be, or the error might be the numerical missing of ef = ex - eg.
How to fix? smooth by a small amount? Won't fix the states that truly don't exist.
Need to either compute two matrices: the seen states and the unseen states,
or always normalize by the population.

AG_obs = eta(Ex)*AG(FG)
eta = beta > 0.5
beta(Ex) = sigmoid(beta')
optimzie beta'

It manages optimization when given the correct censor.
Letting it optimize the censor is not obivous. It gives a solution,
but not a very good one.

[ ] Encourage denser solutions
[ ] First make AG, then cut, to avoid wonky AGs
[x] Trilu
"""

def all_generations(FG: Matrix) -> Matrix:
    Ex = FG.Ex
    Eg = FG.Eg

    # Precalculate the final energy levels

    FG = FG.copy()

    AG = FG.values.copy()
    print("Starting all generations")
    _all_generations(FG.values, AG, Ex, Eg)
    print("Finished all generations")
    return FG.copy(values=AG)

@numba.njit
def _all_generations(FG: np.ndarray, AG: np.ndarray, Ex: np.ndarray, Eg: np.ndarray):
    ef_indices = precompute_indices(Ex, Eg)

    for i in range(len(Ex)):
        for j in range(len(Eg)):
            # Final energy level
            ef_idx = ef_indices[i, j]
            if ef_idx == -1:
                break
            # Transition strength
            strength = FG[i, j]
            AG[i, :] += strength*AG[ef_idx, :]

            
@numba.njit(parallel=True)
def _all_generations_2(FG: np.ndarray, AG: np.ndarray, Ex: np.ndarray, Eg: np.ndarray):
    ef_indices = precompute_indices(Ex, Eg)
    # Create temporary array for atomic operations
    temp_AG = np.zeros_like(AG)
    
    # Main computation loop
    for i in numba.prange(len(Ex)):
        for j in range(len(Eg)):
            ef_idx = ef_indices[i, j]
            if ef_idx == -1:
                break
                
            # Compute strength contribution
            strength = FG[i, j]
            
            # Use temporary array for atomic operations
            temp_AG[i, :] += strength * AG[ef_idx, :]
    
    # Combine results
    for i in numba.prange(len(Ex)):
        AG[i, :] += temp_AG[i, :]


@numba.njit
def precompute_indices(Ex: np.ndarray, Eg: np.ndarray) -> np.ndarray:
    n_ex = len(Ex)
    n_eg = len(Eg)
    
    # Pre-calculate all possible final energy levels
    ef_indices = np.zeros((n_ex, n_eg), dtype=np.int64)
    for i in range(n_ex):
        for j in range(n_eg):
            if Eg[j] > Ex[i]:
                ef_indices[i, j:] = -1
                break
            ef = Ex[i] - Eg[j]
            ef_indices[i, j] = binary_search(Ex, ef)
    return ef_indices


@numba.njit
def binary_search(arr: np.ndarray, value: float) -> int:
    """
    Optimized binary search for finding energy level indices.
    Returns the index of the closest value in arr to value.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if abs(arr[mid] - value) < 1e-10:  # Using small epsilon for float comparison
            return mid
        elif arr[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
            
    # Return closest match if exact match not found
    if left >= len(arr):
        return right
    if right < 0:
        return left
    if abs(arr[left] - value) < abs(arr[right] - value):
        return left
    return right

@numba.njit
def index(A, x):
    i = np.searchsorted(A, x, side='right')
    return i



def all_generations_jax(FG: Matrix, **kwargs) -> Matrix:
    # Note! FG *must* be normalized in mu-space
    Ex = jnp.array(FG.Ex)
    Eg = jnp.array(FG.Eg)

    # Precalculate the final energy levels
    Ef_map = precompute_indices(FG.Ex, FG.Eg)
    Ef_map = jnp.array(Ef_map)

    FG = FG.copy()

    fg = jnp.array(FG.values)
    print("Starting all generations")
    AG = compute_ag_2(fg, Ex, Ef_map, **kwargs)
    print("Finished all generations")
    return FG.copy(values=AG)

# Function to compute AG recursively
@jax.jit
def compute_ag(FG, Ex, Eg, Ef_map):
    def update_ag(carry, ex_idx):
        AG = carry
        AG_current = FG[ex_idx]
        
        def body_fun(eg_idx, AG_current):
            ef_idx = Ef_map[ex_idx, eg_idx]
            # Replace if statement with where
            update = FG[ex_idx, eg_idx] * AG[ef_idx, :]
            AG_current = jnp.where(ef_idx >= 0, 
                                AG_current + update,
                                AG_current)
            return AG_current

        # Replace for loop with fori_loop
        AG_current = jax.lax.fori_loop(
            0, len(Eg),
            body_fun,
            AG_current
        )
        
        AG = AG.at[ex_idx].set(AG_current)
        return AG, None

    FG = jax.device_put(FG)
    Ex = jax.device_put(Ex)
    Eg = jax.device_put(Eg)
    Ef_map = jax.device_put(Ef_map)

    # Initialize AG with zeros
    AG = jnp.zeros_like(FG)

    # Use jax.lax.scan to efficiently apply the recursion over all ex
    AG_final, _ = jax.lax.scan(update_ag, AG, jnp.arange(len(Ex)))

    return AG_final


# Function to compute AG recursively
@jax.jit
def compute_ag_2(FG, Ex, Ef_map):
    """
    Optimized version of compute_ag using JAX's advanced features
    
    Key optimizations:
    1. Vectorized operations using vmap
    2. Removed unnecessary device_put operations
    3. Pre-computed masks for valid ef_idx
    4. Batched matrix operations instead of loop
    5. Static argument handling
    """
    # Create mask for valid ef_idx once
    valid_ef_mask = Ef_map >= 0
    
    def update_ag_vectorized(AG, ex_idx):
        # Get current FG row
        FG_row = FG[ex_idx]
        
        # Extract relevant Ef indices for this ex_idx
        ef_indices = Ef_map[ex_idx]
        
        # Gather all relevant AG rows in one operation
        AG_gathered = AG[ef_indices]
        
        # Compute all updates in parallel using broadcasting
        updates = FG_row[:, None] * AG_gathered
        
        # Apply mask for valid ef_idx
        masked_updates = jnp.where(valid_ef_mask[ex_idx, :, None], updates, 0)
        
        # Sum all contributions
        AG_current = FG_row + jnp.sum(masked_updates, axis=0)
        
        # Update AG in place
        return AG.at[ex_idx].set(AG_current), None
    
    # Initialize AG
    AG = jnp.zeros_like(FG)
    
    # Use scan for the main loop
    AG_final, _ = jax.lax.scan(
        update_ag_vectorized,
        AG,
        jnp.arange(len(Ex))
    )
    
    return AG_final

    
@jax.jit
def compute_ag_chunked(FG, Ex, Ef_map, chunk_size=100):
    valid_ef_mask = Ef_map >= 0
    
    def process_chunk(start_idx, end_idx, AG):
        def update_ag_single(AG, ex_idx):
            FG_row = FG[ex_idx]
            ef_indices = Ef_map[ex_idx]
            # Process in smaller chunks
            chunk_updates = []
            for i in range(0, len(ef_indices), chunk_size):
                chunk_ef = ef_indices[i:i + chunk_size]
                chunk_mask = valid_ef_mask[ex_idx, i:i + chunk_size]
                AG_chunk = AG[chunk_ef]
                updates_chunk = FG_row[i:i + chunk_size, None] * AG_chunk
                masked_chunk = jnp.where(chunk_mask[:, None], updates_chunk, 0)
                chunk_updates.append(masked_chunk)
            
            all_updates = jnp.concatenate(chunk_updates, axis=0)
            AG_current = FG_row + jnp.sum(all_updates, axis=0)
            return AG.at[ex_idx].set(AG_current), None
        
        return jax.lax.scan(
            update_ag_single,
            AG,
            jnp.arange(start_idx, end_idx)
        )[0]
    
    AG = jnp.zeros_like(FG)
    num_ex = len(Ex)
    
    # Process main computation in chunks
    for i in range(0, num_ex, chunk_size):
        end_idx = min(i + chunk_size, num_ex)
        AG = process_chunk(i, end_idx, AG)
    
    return AG

@jax.jit
def tau_to_theta(tau):
    theta = tau**2
    # Apply a softmax row-wise
    #ag = jax.nn.softmax(ag**2, axis=1)
    sum = jnp.sum(theta, axis=1, keepdims=True)
    fg = theta / (sum + 1e-8)
    #fg=theta

    return fg
    #return ag**2

def theta_to_tau(theta):
    return jnp.sqrt(theta)

@jax.jit
def censoring_sigmoid(x, temperature=1e-3):
    """
    Sigmoid function with a temperature parameter
    At high temperature: smooth, gradual transition
    At low temperature: sharp, binary
    """
    return jax.nn.sigmoid(x / temperature)

@jax.jit
def loss_fn(params, AG, Ex, Ef_map, temperature, G_g=None, G_in=None, D=None, pop=None):
    fg_hat = tau_to_theta(params[0])
    ag_hat = compute_ag_2(fg_hat, Ex, Ef_map)

    if pop is not None:
        ag_hat = ag_hat*pop

    # The censoring is a probability, so we use a sigmoid to ensure it is between 0 and 1
    censor = censoring_sigmoid(params[1], temperature)
    ag_obs_hat = jnp.einsum('i,ij->ij', censor, ag_hat)
    ag_cens_hat = jnp.einsum('i,ij->ij', 1-censor, ag_hat)
    # We penalize the sum of the censored states.
    # divide by AG to make it comparable to the KL divergence
    censoring_loss = jnp.sum(ag_cens_hat) / jnp.sum(AG)

    # Fold as in a usual unfolding/folding model
    if D is not None:
        ag_obs_hat = ag_obs_hat@D
    if G_g is not None:
        ag_obs_hat = ag_obs_hat@G_g
    if G_in is not None:
        ag_obs_hat = G_in@ag_obs_hat

    kl_loss = jnp.sum(kl(AG, ag_obs_hat))
    loss = kl_loss + 1e1*censoring_loss
    #loss = jnp.where((AG <= 1e-7) & (ag_hat > 1e-7), (1+ag_hat)**2, loss)

    #loss = jnp.sum((AG - ag_hat)**2)
    #loss = kl_div(ag_hat, AG)
    #loss = jnp.sum(jnp.where(jnp.isfinite(loss), loss, 0))
    return loss
    #loss = jnp.sum(jnp.where(jnp.isfinite(loss), loss, 0))
    #print(loss)
    #return loss
    # Weight the sum to more heavily penalize the later rows and early columns
    #N, M = AG.shape
    # Define f(i) and g(j)
    #f = jnp.arange(1, N+1)**0.5         # Linear increasing for rows
    #f = jnp.cumsum(f) 
    #g = (jnp.arange(1, M+1)**1)[::-1]    # Inversely proportional for columns

    # Create the weights matrix
    #weights =  jnp.outer(f, g)
    #weights = weights / jnp.sum(weights)
    #weights = 1
    #return jnp.sum(loss * weights)

@jax.jit
def kl(f, f_hat):
    return f - f_hat + f_hat * jnp.log(f_hat / (f+1e-18) + 1e-18)

def optimize(Y, optimizer, iterations=1000, disable_tqdm=False,
             init: Matrix | None = None,
             G_g: Matrix | None = None, G_in: Matrix | None = None,
             D: Matrix | None = None, pop= None,
             mask: np.ndarray | Matrix | Literal['tril'] = 'tril',
             gradient_weights: np.ndarray | None = None,
             opt_state: optax.OptState | None = None,
             init_censor: Vector | None = None,
             final_temperature: float = 0.1,
             init_temperature: float = 2):
    Ex = jnp.array(Y.Ex)
    Eg = jnp.array(Y.Eg)
    Ef_map = precompute_indices(Y.Ex, Y.Eg)
    Ef_map = jnp.array(Ef_map)
    ag = jnp.array(Y.values)

    match mask:
        case 'tril':
            mask = tril_mask(Y)
        case _:
            mask = np.asarray(mask)
    mask = jnp.array(mask)
    # Apply the mask to the optimizer to ignore the masked elements
    #optimizer = optax.masked(optimizer, mask)
    if D is not None:
        D = jnp.array(D.values)
    if G_g is not None:
        G_g = jnp.array(G_g.values)
    if G_in is not None:
        G_in = jnp.array(G_in.values)


    if init is None:
        fg = jnp.array(Y.values)
    else:
        if isinstance(init, Matrix):
            fg = init.values
        elif isinstance(init, str):
            if init == 'random':
                fg = 1+np.random.rand(*Y.values.shape)
                fg /= fg.sum(axis=0)
            else:
                raise ValueError(f"Unknown initialization method: {init}")
        elif isinstance(init, (int, float)):
            fg = np.full_like(Y.values, init)
        else:
            fg = init
    fg_init = fg
    fg = theta_to_tau(jnp.asarray(fg))

    if init_censor is None:
        censor = jnp.ones(len(Ex), dtype=jnp.float32)
    else:
        censor = jnp.array(init_censor)

    params = (fg, censor)

    if opt_state is None:
        opt_state = optimizer.init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    mask = ~mask
    @jax.jit
    def get_temperature(i):
        return 1
        progress = i / iterations
        #return final_temperature
        return init_temperature + (final_temperature - init_temperature)*progress
    # Update step
    @jax.jit
    def step(i, params, opt_state, gradient_weights):
        t = get_temperature(i)
        loss, grads = loss_and_grad(params, ag, Ex, Ef_map, t, G_g, G_in, D, pop=pop)
        # We scale the gradients by the number of rows to make them comparable
        #grads = (grads[0] * gradient_weights, grads[1])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        fg = params[0]
        fg = fg.at[mask].set(0)
        params = (fg, params[1])
        #grad_norm = jnp.linalg.norm(grads)
        return loss, params, grads

    losses = np.zeros(iterations)
    grad_norms = np.zeros(iterations)
    row_grad_norms = np.zeros((iterations, len(Ex)))
    col_grad_norms = np.zeros((iterations, len(Eg)))
    censor_norms = np.zeros((iterations, len(Ex)))

    if disable_tqdm:
        bar = range(iterations)
    else:
        bar = tqdm(range(iterations))

    if gradient_weights is not None:
        gradient_weights = jnp.array(gradient_weights)
    else:
        gradient_weights = 1
        
    for i in bar:
        loss, params, grads = step(i, params, opt_state, gradient_weights)
        losses[i] = loss
        grad_norm  = global_grad_norm(grads)
        grad_norms[i] = grad_norm
        row_grad_norms[i] = jnp.linalg.norm(grads[0], axis=1)
        col_grad_norms[i] = jnp.linalg.norm(grads[0], axis=0)
        censor_norms[i] = grads[1]
        if not disable_tqdm:
            bar.set_postfix_str(f'loss: {loss:.2e}, grad_norm: {grad_norm:.2e}')

    fig, ax = plt.subplots()
    ax.plot(grad_norms)
    plt.show()

    fig, ax = plt.subplots()
    im = ax.matshow(row_grad_norms, aspect='auto', cmap='turbo')
    plt.colorbar(im)
    plt.show()

    fig, ax = plt.subplots()
    im = ax.matshow(col_grad_norms, aspect='auto', cmap='turbo')
    plt.colorbar(im)
    plt.show()

    fig, ax = plt.subplots()
    im = ax.matshow(censor_norms, aspect='auto', cmap='turbo')
    plt.colorbar(im)
    plt.show()

    fg_hat = tau_to_theta(params[0])
    fg_hat = Y.copy(values=fg_hat)
    censor_hat = np.asarray(params[1])
    censor_hat = Vector(Ex=Y.X_index, values=censor_hat)
    censor_hat.title = r'Censoring'
    censor_hat.ylabel = r'$\hat{\pi}(E_{\text{in}})$'
    return AGResult(AG=Y, FG=fg_hat, optimizer=optimizer, iterations=iterations, losses=losses, initial_fg=fg_init, population=pop,
                    censor=censor_hat), opt_state
    


class AGResult:
    def __init__(self, AG: Matrix, FG: Matrix, optimizer: optax.GradientTransformation, iterations: int, losses: np.ndarray,
                 initial_fg: Matrix, censor: Vector,population: np.ndarray | None = None):
        self.AG = AG.as_numpy()
        self.FG = FG.as_numpy()
        self.optimizer = optimizer
        self.iterations = iterations
        self.losses = losses
        self.initial_fg = FG.copy(values=np.asarray(initial_fg))
        self.censor_weights = censor
        self.censor = censor.clone(values=np.asarray(jax.nn.sigmoid(censor.values)))
        self.population = population

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.losses)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        return fig, ax

    def FG_hat(self, with_censor: bool = False):
        if with_censor:
            if self.censor is None:
                raise ValueError("censor not provided")
            return self.FG.clone(values=np.einsum('i,ij->ij', self.censor, self.FG))
        return self.FG

    def AG_hat(self, with_censor: bool = False):
        AG_hat = all_generations_jax(self.FG.as_numpy()).as_numpy()
        if with_censor:
            if self.censor is None:
                raise ValueError("censor not provided")
            return AG_hat.clone(values=np.einsum('i,ij->ij', self.censor, AG_hat))
        return AG_hat

    def FG_imputed(self):
        return self.FG_hat(with_censor=False) - self.FG_hat(with_censor=True)

    def AG_imputed(self):
        return self.AG_hat(with_censor=False) - self.AG_hat(with_censor=True)


def tril_mask(A: Matrix):
    X = A.X
    Y = A.Y
    return X[:, None] >= Y[None, :] #- 1*(Y[1] - Y[0])

    
def global_grad_norm(grads):
    # Flatten all gradients into a single vector
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    # Concatenate all gradients
    concatenated = jnp.concatenate([g.ravel() for g in flat_grads])
    # Compute L2 norm
    norm = jnp.linalg.norm(concatenated)
    return norm

