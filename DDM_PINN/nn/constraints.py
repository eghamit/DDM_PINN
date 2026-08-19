"""Hard boundary-constraint output transform, with a physical baseline.

This wrapper builds the ohmic terminal values into the trial solution *exactly*
and starts every component from a physically sensible baseline, so the network
only has to learn a small interior correction.

For each output component ``f`` the trial solution is

    f(X) = base_f(X)  +  (X/L)(1 - X/L) * raw_f(X)

The parabolic bubble ``(X/L)(1 - X/L)`` vanishes at both terminals, so - as long
as ``base_f`` already hits the terminal targets - the boundary values are met to
machine precision whatever the network does.

Baselines:

* **potential ``u``** - a smooth ``tanh`` ramp between the two neutral-bulk
  potentials, centred on the junction and spread over the depletion width.  This
  is flat in each neutral region (charge-neutral) and transitions across the
  junction with a *physically bounded* curvature ``~ V_bi / W_dep^2`` - matching
  the real depletion space charge.  (An ``asinh(C/2)`` charge-neutral baseline,
  by contrast, is only valid in the neutral bulk and injects a huge spurious
  curvature spike at the junction, which the network cannot cancel.)
* **quasi-Fermi ``v, w``** - a linear interpolation of the terminal values (at
  equilibrium both are zero, so the baseline is exact; under bias the network
  learns the junction split).

The baseline is differentiable w.r.t. ``X``, so ``u_XX`` in the Poisson residual
flows through it correctly.
"""

import torch
import torch.nn as nn


class HardBC1D(nn.Module):
    def __init__(self, net, length_scaled, junctions, transition_width,
                 ramp_components=(0,)):
        super().__init__()
        self.net = net
        self.Ls = float(length_scaled)
        self.junctions = [float(j) for j in junctions]
        self.width = float(transition_width)
        # which output components use the smooth junction ramp baseline (vs a
        # plain linear interpolation).  Quasi-Fermi: only the potential (0).
        # Direct (u, ln n, ln p): all three, since ln n / ln p also step across
        # the junction with the full Boltzmann swing.
        self.ramp_components = tuple(ramp_components)
        self.register_buffer("aL", torch.zeros(3))
        self.register_buffer("aR", torch.zeros(3))
        # normalisation constants so the ramp hits the terminals exactly
        self._s0 = float(self._shape_np(0.0))
        self._sL = float(self._shape_np(self.Ls))

    # -- monotone 0->1 shape across the junction(s) -----------------------
    def _shape_np(self, x):
        import numpy as np
        s = 0.0
        for xj in self.junctions:
            s = s + 0.5 * (1.0 + np.tanh((x - xj) / self.width))
        return s / max(len(self.junctions), 1)

    def _shape(self, x):
        s = torch.zeros_like(x)
        for xj in self.junctions:
            s = s + 0.5 * (1.0 + torch.tanh((x - xj) / self.width))
        return s / max(len(self.junctions), 1)

    def set_targets(self, left, right):
        device = self.aL.device
        self.aL = torch.as_tensor(left, dtype=self.aL.dtype, device=device)
        self.aR = torch.as_tensor(right, dtype=self.aR.dtype, device=device)

    def forward(self, x):
        t = x / self.Ls                         # (M, 1) in [0, 1]
        # feed the network the NORMALISED coordinate (De Mari scaling makes the
        # raw scaled coordinate tiny); autograd threads d/dx through t.
        raw = self.net(t)                       # (M, 3)
        # normalised junction-ramp shape (exact 0 at X=0, 1 at X=L)
        s = (self._shape(x) - self._s0) / (self._sL - self._s0)   # (M, 1)
        # per-component interpolation variable: ramp for ramp_components, else t
        cols = []
        for i in range(3):
            shape_i = s if i in self.ramp_components else t
            cols.append(self.aL[i] + (self.aR[i] - self.aL[i]) * shape_i)
        baseline = torch.cat(cols, dim=1)
        bubble = t * (1.0 - t)
        return baseline + bubble * raw
