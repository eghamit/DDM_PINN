"""Carrier densities from the scaled unknowns.

The scaled carrier densities follow algebraically from the quasi-Fermi
variables, exactly as in the FEM reference solver::

    n / n_i = exp(u - v + band_n + Lambda_n)
    p / n_i = exp(w - u + band_p + Lambda_p)

``band_n/band_p`` are the heterojunction band-edge offsets (0 for a single
material) and ``Lambda_n/Lambda_p`` are the additive log-density shifts that
carry the Schrodinger-Poisson quantum correction and/or the Fermi-Dirac
degeneracy shift.  Every physics extension of DDM.SPC enters here as one more
additive term in these exponents - which is precisely why they are trivial to
differentiate through in a PINN.

A clamp on the exponent mirrors the FEM ``safe_exp`` guard: it keeps the density
finite during the first, wild iterations of training without changing the
converged solution (the exponent never approaches the clamp there).
"""

import torch

# matches DDM.SPC safe_exp guard (exp(80) ~ 5.5e34, still finite in float64)
_EXP_CLAMP = 80.0


def electron_density(u, v, shift=0.0):
    return torch.exp(torch.clamp(u - v + shift, max=_EXP_CLAMP))


def hole_density(u, w, shift=0.0):
    return torch.exp(torch.clamp(w - u + shift, max=_EXP_CLAMP))
