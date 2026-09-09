"""Terminal-current extraction.

At steady state the total current density ``J = J_n + J_p`` is spatially constant
in a 1-D device (``dJ/dx = q(G - R) = 0`` with no net generation), so in
principle it can be read anywhere.  In practice *where* it is read matters a lot
for a PINN.  In the neutral bulk one carrier density is enormous
(``n ~ N/n_i ~ 1e6``) while its quasi-Fermi gradient is tiny, so the flux
``j_n = -mu_n n v_X`` amplifies any small error in ``v_X`` by that huge density -
and the total current is a near-cancellation of two large, noisy fluxes.  Around
the **metallurgical junction** the two densities are both ~``n_i`` (O(1) scaled),
so there is no density amplification and the total current is by far the cleanest
there.

Hence the terminal current is extracted from a window **around the junction /
depletion region** when the junction location is known, falling back to the whole
interior otherwise.  The spread of ``j`` over the window doubles as a
solution-quality (current-conservation) diagnostic.
"""

import numpy as np
import torch


def current_density(physics, net, scaling, length, num=200, dtype=torch.float64,
                    junctions=None, edge_width=None):
    """Total physical current density ``J`` [A/m^2] and its uniformity spread.

    Parameters
    ----------
    junctions : sequence[float], optional
        Scaled junction coordinate(s).  When given, the current is read from a
        window around them (``+/- edge_width``) - the depletion-edge functional,
        where the carrier densities are O(1) and the flux is not amplified by a
        huge bulk density.
    edge_width : float, optional
        Half-width (scaled) of that window; defaults to a few units.
    """
    Ls = length / scaling.length
    if junctions:
        w = edge_width if edge_width is not None else 0.1 * Ls
        xs = []
        for xj in junctions:
            xs.append(np.linspace(max(0.0, xj - 2.0 * w),
                                  min(Ls, xj + 2.0 * w), num))
        x = np.concatenate(xs)
    else:
        x = np.linspace(0.05 * Ls, 0.95 * Ls, num)
    X = torch.tensor(x.reshape(-1, 1), dtype=dtype, requires_grad=True)
    _, _, j_tot = physics.current_flux(net, X)
    j = j_tot.detach().cpu().numpy().reshape(-1)
    J0 = scaling.current
    J = float(np.median(j)) * J0
    # relative spread of j over the window: a current-conservation diagnostic
    J_spread = float(np.std(j)) * J0
    return J, J_spread
