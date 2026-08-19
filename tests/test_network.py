"""Neural field and the hard boundary-constraint transform."""

import numpy as np
import torch

from DDM_PINN.nn.constraints import HardBC1D
from DDM_PINN.nn.network import PINNField

torch.set_default_dtype(torch.float64)


def test_field_output_shape():
    net = PINNField(in_dim=1, out_dim=3, seed=0)
    x = torch.linspace(0, 1, 17).reshape(-1, 1)
    y = net(x)
    assert y.shape == (17, 3)


def test_fourier_field_runs():
    net = PINNField(in_dim=1, out_dim=3, fourier=True, seed=0)
    x = torch.linspace(0, 1, 8).reshape(-1, 1)
    assert net(x).shape == (8, 3)


def test_hard_bc_hits_targets_exactly():
    base = PINNField(in_dim=1, out_dim=3, seed=1)
    Ls = 0.05
    net = HardBC1D(base, Ls, junctions=[0.5 * Ls], transition_width=0.1 * Ls)
    left = (-13.8, 0.0, 0.0)
    right = (13.8, 0.2, 0.2)
    net.set_targets(left, right)
    ends = torch.tensor([[0.0], [Ls]])
    out = net(ends).detach().numpy()
    assert np.allclose(out[0], left, atol=1e-9)
    assert np.allclose(out[1], right, atol=1e-9)


def test_hard_bc_potential_baseline_is_monotone_and_bounded_curvature():
    base = PINNField(in_dim=1, out_dim=3, seed=1)
    Ls = 0.05
    net = HardBC1D(base, Ls, junctions=[0.5 * Ls], transition_width=0.004)
    net.set_targets((-13.8, 0.0, 0.0), (13.8, 0.0, 0.0))
    # zero out the network correction so we see the baseline alone
    for ppar in base.parameters():
        torch.nn.init.zeros_(ppar)
    x = torch.linspace(0, Ls, 400, requires_grad=True).reshape(-1, 1)
    u = net(x)[:, 0:1]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux))[0]
    # monotone increasing potential baseline, curvature bounded (no spike)
    assert (ux.detach() >= -1e-6).all()
    assert float(torch.max(torch.abs(uxx))) < 1e7
