import numpy as np
from .unfolder import Unfolder, UnfoldedResult1DSimple, Errors1DCovariance, ResultMeta
from .. import Matrix, Vector
from ..stubs import Axes
from ..numbalib import njit, prange, objmode
import time
from iminuit import Minuit
from .loss import loss_factory_bg, loss_factory, LogLike, LossFn, print_minuit_convergence, get_transform
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult

@dataclass
class MLResult1D(Errors1DCovariance, UnfoldedResult1DSimple):
    loss: LossFn
    res: Minuit
    minuit: Minuit


class ML(Unfolder):
    def __init__(self, R: Matrix, G: Matrix):
        super().__init__(R, G)

    def _unfold_vector(self, R: Matrix, data: Vector, background: Vector | None,
                       initial: Vector,
                       mask: Vector,
                       verbose: bool = True, **kwargs):
        transform = kwargs.pop('transform', 'id')
        tmap, imap = get_transform(transform)
        loss: LossFn = kwargs.pop('loss', 'loglike')
        ll: LogLike = kwargs.pop('loglike', 'll')
        mask_ = mask.values
        if background is None:
            loss = loss_factory(loss, R, data,
                                ll, mapfn=tmap, imapfn=imap, mask=mask_, **kwargs)
        else:
            loss = loss_factory_bg(loss, R, data, background,
                                ll, mapfn=tmap, imapfn=imap, mask=mask_, **kwargs)
        mu0: np.ndarray = tmap(initial.values)
        m = Minuit(loss, mu0)
        m.tol = 1e-3
        m.errordef = Minuit.LIKELIHOOD
        start = time.time()
        ret: Minuit = m.migrad(iterate=100)
        ret2 = m.hesse()
        elapsed = time.time() - start
        if verbose:
            print_minuit_convergence(m)
        u: Vector = data.clone(values=imap(np.asarray(m.values)))
        return MLResult1D(meta=ResultMeta(time=elapsed),
                          R=R, raw=data, background=background, initial=initial,
                          mask=mask,
                          u=u, cov=np.asarray(m.covariance), loss=loss,
                          res=ret, minuit=m)


    def supports_background(self) -> bool:
        return True
