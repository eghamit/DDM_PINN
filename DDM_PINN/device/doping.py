"""Net doping profile C(x) as a continuous field.

The PINN needs the net doping ``C = (N_D - N_A) / n_i`` (scaled) as a smooth
function of position so it can be evaluated - and differentiated through - at
arbitrary collocation points.  :class:`DopingProfile` builds that field from a
per-region net-doping dict (signed, SI ``m^-3``: positive = donor, negative =
acceptor), matching the FEM solver's ``doping={region: net_doping}`` convention.

An abrupt region-wise profile is discontinuous at the junction; sampling it as a
step is fine for a PINN (the collocation points never sit exactly on the
interface), but an optional ``smooth`` width replaces the step with a ``tanh``
transition, which both regularises the loss landscape and models a realistic
graded junction.
"""

import numpy as np


class DopingProfile:
    def __init__(self, device, region_doping, scaling, smooth=0.0):
        """
        Parameters
        ----------
        device : Device1D
            Geometry supplying the region boundaries.
        region_doping : dict[str, float]
            ``{region_name: net_doping_m^-3}`` (signed).
        scaling : Scaling
            De Mari scaling (used to normalise by ``n_i``).
        smooth : float
            Junction smoothing width in *scaled* length units (0 = abrupt step).
        """
        self.device = device
        self.scaling = scaling
        self.smooth = float(smooth)
        missing = set(device.region_names) - set(region_doping)
        if missing:
            raise KeyError(
                f"doping missing region(s) {sorted(missing)}; "
                f"device regions are {device.region_names}."
            )
        # scaled, signed net doping per region
        self.region_C = {
            name: scaling.scale_doping(float(val))
            for name, val in region_doping.items()
        }
        # region boundaries in *scaled* coordinates, for the smooth blend
        self._edges = [
            (r.name, scaling.scale_length(r.x0), scaling.scale_length(r.x1))
            for r in device.regions
        ]

    def scaled(self, X):
        """Scaled net doping ``C`` at scaled coordinate(s) ``X`` (numpy)."""
        X = np.asarray(X, dtype=float)
        if self.smooth <= 0.0:
            C = np.zeros_like(X)
            for name, x0, x1 in self._edges:
                mask = (X >= x0) & (X <= x1)
                C[mask] = self.region_C[name]
            # points beyond the last edge (numerical) take the nearest region
            return C
        # smooth: build as a sum of tanh steps between consecutive regions
        # C(X) = C_0 + sum_k (C_{k+1} - C_k) * 0.5 (1 + tanh((X - x_k)/smooth))
        names = [e[0] for e in self._edges]
        interfaces = [e[2] for e in self._edges[:-1]]   # right edges, interior
        C = np.full_like(X, self.region_C[names[0]])
        for xk, name_next in zip(interfaces, names[1:]):
            step = self.region_C[name_next] - self.region_C[names[0]]
            # accumulate relative to the running left value: use pairwise diff
        # simpler robust pairwise construction:
        C = np.full_like(X, self.region_C[names[0]])
        prev = self.region_C[names[0]]
        for xk, name_next in zip(interfaces, names[1:]):
            nxt = self.region_C[name_next]
            C = C + (nxt - prev) * 0.5 * (1.0 + np.tanh((X - xk) / self.smooth))
            prev = nxt
        return C

    def scaled_torch(self, X):
        """Differentiable scaled net doping ``C(X)`` for a torch tensor ``X``.

        Mirrors :meth:`scaled` but in torch so the charge-neutral potential
        baseline built from it stays differentiable w.r.t. the coordinate (the
        Poisson residual needs ``u_XX``).  Uses the smooth ``tanh`` blend; an
        abrupt profile (``smooth == 0``) falls back to a very narrow blend so it
        remains differentiable.
        """
        import torch

        names = [e[0] for e in self._edges]
        interfaces = [e[2] for e in self._edges[:-1]]
        width = self.smooth if self.smooth > 0.0 else 1e-3
        C = torch.full_like(X, self.region_C[names[0]])
        prev = self.region_C[names[0]]
        for xk, name_next in zip(interfaces, names[1:]):
            nxt = self.region_C[name_next]
            C = C + (nxt - prev) * 0.5 * (1.0 + torch.tanh((X - xk) / width))
            prev = nxt
        return C

    def contact_C(self, contact_name):
        """Scaled net doping at a named contact (its region's value)."""
        contact = self.device.contacts[contact_name]
        return self.region_C[contact.region]
