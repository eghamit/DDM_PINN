"""Autodiff differential operators.

Thin wrappers around ``torch.autograd.grad`` for the first and second
derivatives the drift-diffusion residuals need.  Keeping them here makes the
residual code read like the maths and keeps the ``create_graph=True`` bookkeeping
(needed so the residual itself remains differentiable w.r.t. the network
weights) in one place.
"""

import torch


def d_dx(y, x):
    """First derivative dy/dx (same shape as ``y``), graph retained."""
    (grad,) = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )
    return grad


def d2_dx2(y, x):
    """Second derivative d2y/dx2 via two applications of :func:`d_dx`."""
    return d_dx(d_dx(y, x), x)
