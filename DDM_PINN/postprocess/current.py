"""Terminal-current extraction.

At steady state the total current density ``J = J_n + J_p`` is spatially constant
in a 1-D device (``dJ/dx = q(G - R) = 0`` with no net generation), so evaluating
the scaled flux ``j = -(mu_n n v_X + mu_p p w_X)`` at any set of interior points
and averaging gives the terminal current density.  Multiplying by the De Mari
current scale ``J_0`` returns it in physical ``A/m^2``; multiplying further by a
cross-sectional area returns amperes.

Averaging over a spread of interior points (rather than trusting a single one)
also doubles as a solution-quality diagnostic: a well-converged solution has a
nearly flat ``J(x)``, so the spread of ``j`` across the device is small.
"""

import numpy as np
import torch


def current_density(physics, net, scaling, length, num=200, dtype=torch.float64):
    """Total physical current density ``J`` [A/m^2] and its uniformity spread."""
    Ls = length / scaling.length
    x = np.linspace(0.05 * Ls, 0.95 * Ls, num)
    X = torch.tensor(x.reshape(-1, 1), dtype=dtype, requires_grad=True)
    _, _, j_tot = physics.current_flux(net, X)
    j = j_tot.detach().cpu().numpy().reshape(-1)
    J0 = scaling.current
    # total current is spatially constant at steady state; the median is a
    # robust estimate (insensitive to the larger residual near the contacts),
    # and the spread of j(x) doubles as a solution-quality diagnostic.
    J = float(np.median(j)) * J0
    J_spread = float(np.std(j)) * J0
    return J, J_spread
