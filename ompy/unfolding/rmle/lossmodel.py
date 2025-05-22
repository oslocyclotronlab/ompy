from .penalty import Penalty
from .loss import KullbackLeibler, Loss
from .utils import pytree_dataclass
from .stubs import ExpectationParameter, Data

@pytree_dataclass
class ModelLoss:
    loss: Loss = KullbackLeibler()
    penalty: tuple[Penalty, ...] = ()

    def __call__(self, alpha: ExpectationParameter, y: Data,
                 *args, **kwargs):
        return self.loss(alpha, y, *args, **kwargs)


