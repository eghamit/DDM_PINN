"""Physics: autodiff operators, densities, and PDE residuals."""

import numpy as np
import torch

from DDM_PINN.physics.densities import electron_density, hole_density
from DDM_PINN.physics.operators import d2_dx2, d_dx
from DDM_PINN.physics.residuals import DriftDiffusion1D

torch.set_default_dtype(torch.float64)


def test_operators_against_analytic():
    x = torch.linspace(0.1, 1.0, 50, requires_grad=True).reshape(-1, 1)
    y = torch.sin(3.0 * x)
    dy = d_dx(y, x)
    d2y = d2_dx2(y, x)
    assert torch.allclose(dy, 3.0 * torch.cos(3.0 * x), atol=1e-8)
    assert torch.allclose(d2y, -9.0 * torch.sin(3.0 * x), atol=1e-6)


def test_densities():
    u = torch.tensor([0.0, 1.0, -1.0])
    v = torch.zeros(3)
    w = torch.zeros(3)
    assert torch.allclose(electron_density(u, v), torch.exp(u))
    assert torch.allclose(hole_density(u, w), torch.exp(-u))
    # mass action n p = 1 (scaled) when v = w
    assert torch.allclose(electron_density(u, v) * hole_density(u, w),
                          torch.ones(3), atol=1e-12)


def test_poisson_residual_zero_when_charge_matches():
    """Choosing C = n - p (with u_xx = 0) drives the Poisson residual to zero.

    ``u`` is linear (so ``u_xx = 0``); ``v = w`` are given a tiny non-linear
    dependence so the autodiff graph stays connected (an exactly-linear field
    would make the derivative a graph-disconnected constant - a purely
    mathematical edge case never hit by a real network).
    """
    phys = DriftDiffusion1D(1.0, 0.35, poisson_scale=1.0, continuity_scale=1.0)

    class Field(torch.nn.Module):
        def forward(self, x):
            u = 0.3 * x
            v = 1e-4 * x ** 2
            w = 1e-4 * x ** 2                # w = v -> mass action n p = 1
            return torch.cat([u, v, w], dim=1)

    net = Field()
    X = torch.linspace(0.0, 1.0, 20, requires_grad=True).reshape(-1, 1)
    _, _, _, n, p = phys.fields(net, X)
    C = (n - p).detach()                     # choose C so n - p - C = 0
    r_u, _, _ = phys.residuals(net, X, C)
    assert torch.max(torch.abs(r_u)) < 1e-8


def test_current_flux_near_zero_at_equilibrium():
    """With v = w ~ const (no quasi-Fermi split) the net current is negligible."""
    phys = DriftDiffusion1D(1.0, 0.35)

    class EqField(torch.nn.Module):
        def forward(self, x):
            return torch.cat([0.3 * x, 1e-8 * x ** 2, 1e-8 * x ** 2], dim=1)

    net = EqField()
    X = torch.linspace(0.0, 1.0, 20, requires_grad=True).reshape(-1, 1)
    _, _, j = phys.current_flux(net, X)
    assert torch.max(torch.abs(j)) < 1e-6
