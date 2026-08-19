"""Composite PINN loss.

The total loss is a weighted sum of

* the three PDE residual MSEs (Poisson, electron continuity, hole continuity),
  evaluated at interior collocation points, and
* the boundary MSE, enforcing the ohmic-contact ``(u, v, w)`` targets.

Balancing these terms is the crux of training a semiconductor PINN: the Poisson
residual carries the exponential space-charge term while the continuity
residuals are tiny in the neutral bulk, so fixed equal weights leave the
boundary or the continuity equations under-enforced.  :class:`LossWeights` holds
the (possibly adaptive) weights; :func:`total_loss` combines the parts and also
returns the component values for logging and for gradient-based reweighting.
"""

from dataclasses import dataclass

import torch


@dataclass
class LossWeights:
    poisson: float = 1.0
    electron: float = 1.0
    hole: float = 1.0
    boundary: float = 1.0


def _mse(x):
    return torch.mean(x ** 2)


def pde_losses(physics, net, X, C):
    r_u, r_v, r_w = physics.residuals(net, X, C)
    return _mse(r_u), _mse(r_v), _mse(r_w)


def boundary_loss(net, Xb, targets):
    """MSE of the network against the ohmic ``(u,v,w)`` targets at terminals.

    ``targets`` is a ``(B, 3)`` tensor aligned with the boundary coords ``Xb``.
    """
    pred = net(Xb)
    return _mse(pred - targets)


def total_loss(physics, net, X, C, Xb, targets, weights):
    lu, lv, lw = pde_losses(physics, net, X, C)
    lb = boundary_loss(net, Xb, targets)
    total = (weights.poisson * lu + weights.electron * lv
             + weights.hole * lw + weights.boundary * lb)
    return total, {"poisson": lu, "electron": lv, "hole": lw, "boundary": lb}
