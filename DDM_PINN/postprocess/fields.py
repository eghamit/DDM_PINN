"""Evaluate a trained field into physical quantities.

Given the trained network and the scaling, :func:`evaluate` samples the solution
on a physical-coordinate grid and returns the electrostatic potential [V], the
electron/hole quasi-Fermi potentials [V], and the carrier concentrations [m^-3]
- the same per-node fields the FEM solver reports, so the two can be compared
point for point.
"""

import numpy as np
import torch

from DDM_PINN.physics.densities import electron_density, hole_density


class Solution1D:
    def __init__(self, x, potential, phi_n, phi_p, n, p):
        self.x = x                      # [m]
        self.potential = potential      # [V]
        self.electron_fermi_potential = phi_n
        self.hole_fermi_potential = phi_p
        self.electron_density = n       # [m^-3]
        self.hole_density = p           # [m^-3]

    @property
    def built_in_potential(self):
        return float(self.potential.max() - self.potential.min())


def evaluate(net, scaling, length, num=401, dtype=torch.float64):
    """Sample the trained solution on ``num`` points over ``[0, length]`` [m]."""
    x = np.linspace(0.0, length, num)
    X = torch.tensor((x / scaling.length).reshape(-1, 1), dtype=dtype)
    with torch.no_grad():
        uvw = net(X)
    u = uvw[:, 0].cpu().numpy()
    v = uvw[:, 1].cpu().numpy()
    w = uvw[:, 2].cpu().numpy()
    n = scaling.density * np.exp(np.clip(u - v, None, 80.0))
    p = scaling.density * np.exp(np.clip(w - u, None, 80.0))
    VT = scaling.potential
    return Solution1D(x, VT * u, VT * v, VT * w, n, p)
