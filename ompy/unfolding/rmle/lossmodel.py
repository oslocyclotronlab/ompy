from .penalty import Penalty
from .loss import KullbackLeibler, Loss
from .utils import pytree_dataclass

@pytree_dataclass
class ModelLoss:
    loss: Loss = KullbackLeibler()
    penalty: tuple[Penalty, ...] = ()

