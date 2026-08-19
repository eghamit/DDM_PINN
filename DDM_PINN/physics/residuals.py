"""Coupled drift-diffusion PDE residuals (strong form, 1-D).

The PINN minimises the strong-form residuals of the same three equations the FEM
reference solver assembles in weak form, written in De Mari scaled variables
``(u, v, w)`` with ``n = exp(u - v)``, ``p = exp(w - u)`` and scaled net doping
``C``::

    Poisson    r_u = u_XX  -  (n - p - C)
    electrons  r_v = d/dX( mu_n n v_X )  +  R
    holes      r_w = d/dX( mu_p p w_X )  -  R

(The signs of the recombination term ``R`` follow the FEM weak form: it is a sink
of electrons and a sink of holes, entering the two continuity residuals with
opposite signs so total current is conserved.  ``R = 0`` is the default
recombination-free ideal-diffusion model.)

The derivatives are taken by autodiff, so no mesh and no analytic Jacobian are
needed - the network *is* the trial solution and the residuals are evaluated
pointwise at collocation coordinates.

The class also exposes the scaled current-density fluxes ``j_n = -mu_n n v_X``,
``j_p = -mu_p p w_X`` (De Mari scaled) for terminal-current post-processing.
"""

import torch

from DDM_PINN.physics.densities import electron_density, hole_density
from DDM_PINN.physics.operators import d_dx


class DriftDiffusion1D:
    def __init__(self, mun_scaled, mup_scaled, recombination=None,
                 poisson_scale=1.0, continuity_scale=1.0):
        self.mun = float(mun_scaled)
        self.mup = float(mup_scaled)
        self.recombination = recombination
        # Residual normalisation.  In De Mari scaling the net doping is huge
        # (C = N / n_i ~ 1e6) and it multiplies *both* the Poisson space-charge
        # term and the carrier density inside the continuity flux, so the raw
        # residuals span ~1e12 and swamp each other in the loss.  Dividing each
        # residual by its natural magnitude (the doping scale) puts all three
        # near O(1) so the optimiser balances Poisson against the two
        # continuity equations.  Scaling a residual never moves its zero, so the
        # converged solution is unchanged.
        self.poisson_scale = float(poisson_scale)
        self.continuity_scale = float(continuity_scale)

    def fields(self, net, X):
        """Return the raw unknowns and densities at collocation coords ``X``.

        ``X`` must be a ``(M, 1)`` tensor with ``requires_grad=True``.
        """
        uvw = net(X)
        u = uvw[:, 0:1]
        v = uvw[:, 1:2]
        w = uvw[:, 2:3]
        n = electron_density(u, v)
        p = hole_density(u, w)
        return u, v, w, n, p

    def residuals(self, net, X, C):
        """Return ``(r_poisson, r_electron, r_hole)`` at ``X``.

        ``C`` is the scaled net doping at ``X`` (same shape as a column).
        """
        u, v, w, n, p = self.fields(net, X)

        u_x = d_dx(u, X)
        u_xx = d_dx(u_x, X)
        # Poisson residual normalised by the (global) doping scale.  Because the
        # net doping magnitude is ~C_max throughout each neutral region, this
        # normalisation simultaneously (i) keeps the space-charge residual O(1)
        # in the depletion region and (ii) reads as the *relative* charge
        # imbalance in the neutral bulk (where |C| ~ C_max), which is the
        # physically meaningful, exponentially sensitive quantity.
        r_poisson = (u_xx - (n - p - C)) / self.poisson_scale

        v_x = d_dx(v, X)
        w_x = d_dx(w, X)
        flux_n = self.mun * n * v_x
        flux_p = self.mup * p * w_x
        r_electron = d_dx(flux_n, X)
        r_hole = d_dx(flux_p, X)

        if self.recombination is not None:
            R = self.recombination(n, p)          # scaled net rate
            r_electron = r_electron + R
            r_hole = r_hole - R

        r_electron = r_electron / self.continuity_scale
        r_hole = r_hole / self.continuity_scale
        return r_poisson, r_electron, r_hole

    def current_flux(self, net, X):
        """Scaled current-density fluxes ``(j_n, j_p, j_total)`` at ``X``.

        Physical current density is ``J_0 * j`` where ``J_0`` is the De Mari
        current scale.  Total current is spatially constant at steady state, so
        this evaluated at any interior point gives the terminal current density.
        """
        u, v, w, n, p = self.fields(net, X)
        v_x = d_dx(v, X)
        w_x = d_dx(w, X)
        j_n = -self.mun * n * v_x
        j_p = -self.mup * p * w_x
        return j_n, j_p, j_n + j_p
