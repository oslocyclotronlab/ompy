from __future__ import annotations
from .penalty import Penalty
from .loss import KullbackLeibler, Loss
from .utils import pytree_dataclass
from .stubs import ExpectationParameter, Data
from dataclasses import dataclass
import jax
from typing import Iterable

@dataclass(frozen=True)
class ModelLoss:
    loss: Loss = KullbackLeibler()
    penalty: tuple[Penalty, ...] = ()

    def __init__(self, *args, loss: Loss | None = None,
                 penalty: Penalty | tuple[Penalty, ...] | None = None):
        # Handle positional args
        pos_loss = None
        pos_penalties = []
        
        # Parse positional args
        for arg in args:
            if isinstance(arg, Loss):
                if pos_loss is not None:
                    raise TypeError("Multiple Loss instances provided in positional arguments")
                pos_loss = arg
            elif isinstance(arg, Penalty):
                pos_penalties.append(arg)
            elif isinstance(arg, tuple) and all(isinstance(p, Penalty) for p in arg):
                pos_penalties.extend(arg)
            else:
                raise TypeError(f"Cannot interpret argument of type {type(arg)} as loss or penalty")
        
        # Check for conflicts between positional and keyword arguments
        if pos_loss is not None and loss is not None:
            raise TypeError("Loss specified both positionally and as keyword argument")
        if pos_penalties and penalty is not None:
            raise TypeError("Penalties specified both positionally and as keyword argument")
            
        # Use positional args if provided, otherwise use keyword args
        loss = pos_loss if pos_loss is not None else loss
        penalty = tuple(pos_penalties) if pos_penalties else penalty
            
        # Set defaults if not provided
        if loss is None:
            loss = KullbackLeibler()
            
        if penalty is None:
            penalty = ()
            
        # Ensure penalty is a tuple
        if not isinstance(penalty, Iterable):
            penalty = (penalty,)
        else:
            penalty = tuple(penalty)
            
        # Set attributes
        object.__setattr__(self, 'loss', loss)
        object.__setattr__(self, 'penalty', penalty)

    def __call__(self, alpha: ExpectationParameter, y: Data,
                 *args, **kwargs):
        return self.loss(alpha, y, *args, **kwargs)


def tree_flatten_modelloss(model):
    children = (model.loss, model.penalty)  # arrays/dynamic values
    aux_data = {}  # static values
    return children, aux_data


def tree_unflatten_modelloss(aux_data, children):
    loss, penalty = children
    return ModelLoss(loss=loss, penalty=penalty)


jax.tree_util.register_pytree_node(ModelLoss,
                                  tree_flatten_modelloss,
                                  tree_unflatten_modelloss)