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


# density clamp shared with the direct formulation (mirrors safe_exp)
_S_CLAMP = 80.0


class DriftDiffusionDirect1D:
    """Direct (u, ln n, ln p) formulation of the same drift-diffusion system.

    Instead of the quasi-Fermi potentials, the network parametrises the
    electrostatic potential and the **log carrier densities**
    ``s_n = ln(n/n_i)``, ``s_p = ln(p/n_i)`` (so ``n = n_i e^{s_n}`` is positive
    by construction and spans its ~12-decade range).  This is the primary-
    variable ``(psi, n, p)`` form of the drift-diffusion model used by classical
    device simulators and by DDNet - written here with the drift-diffusion flux
    in its native (drift + diffusion) form rather than the quasi-Fermi flux::

        Poisson    r_u  = u_XX - (n - p - C)
        electrons  r_sn = d/dX[ mu_n n (s_n_X - u_X) ] - R
        holes      r_sp = d/dX[ mu_p p (s_p_X + u_X) ] - R

    (Using ``n_X = n s_n_X`` the electron flux ``mu_n (n_X - n u_X)`` becomes
    ``mu_n n (s_n_X - u_X)``.)  Mathematically this is equivalent to the
    quasi-Fermi form under ``v = u - s_n``, ``w = s_p + u``; the point of keeping
    both is to compare their *conditioning* - here ``s_n, s_p`` swing by the full
    Boltzmann range (~+/-14) tracking the potential, whereas the quasi-Fermi
    ``v, w`` stay ~O(1).  Same residual normalisation as the quasi-Fermi class.
    """

    def __init__(self, mun_scaled, mup_scaled, recombination=None,
                 poisson_scale=1.0, continuity_scale=1.0):
        self.mun = float(mun_scaled)
        self.mup = float(mup_scaled)
        self.recombination = recombination
        self.poisson_scale = float(poisson_scale)
        self.continuity_scale = float(continuity_scale)

    def fields(self, net, X):
        out = net(X)
        u = out[:, 0:1]
        s_n = out[:, 1:2]
        s_p = out[:, 2:3]
        n = torch.exp(torch.clamp(s_n, max=_S_CLAMP))
        p = torch.exp(torch.clamp(s_p, max=_S_CLAMP))
        return u, s_n, s_p, n, p

    def residuals(self, net, X, C):
        u, s_n, s_p, n, p = self.fields(net, X)

        u_x = d_dx(u, X)
        u_xx = d_dx(u_x, X)
        r_poisson = (u_xx - (n - p - C)) / self.poisson_scale

        sn_x = d_dx(s_n, X)
        sp_x = d_dx(s_p, X)
        flux_n = self.mun * n * (sn_x - u_x)      # mu_n (n_X - n u_X)
        flux_p = self.mup * p * (sp_x + u_x)      # mu_p (p_X + p u_X)
        r_electron = d_dx(flux_n, X)
        r_hole = d_dx(flux_p, X)

        if self.recombination is not None:
            R = self.recombination(n, p)
            r_electron = r_electron - R
            r_hole = r_hole - R

        r_electron = r_electron / self.continuity_scale
        r_hole = r_hole / self.continuity_scale
        return r_poisson, r_electron, r_hole

    def current_flux(self, net, X):
        u, s_n, s_p, n, p = self.fields(net, X)
        u_x = d_dx(u, X)
        sn_x = d_dx(s_n, X)
        sp_x = d_dx(s_p, X)
        # J_n = mu_n (n_X - n u_X) = mu_n n (s_n_X - u_X); J_p = -mu_p n(...)
        j_n = self.mun * n * (sn_x - u_x)
        j_p = -self.mup * p * (sp_x + u_x)
        return j_n, j_p, j_n + j_p
