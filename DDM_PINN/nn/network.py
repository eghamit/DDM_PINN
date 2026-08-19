"""Neural field for the drift-diffusion unknowns.

``PINNField`` is an MLP ``R^d -> R^3`` mapping scaled coordinates ``X`` to the
three scaled unknowns ``(u, v, w)`` = (electrostatic potential, electron
quasi-Fermi potential, hole quasi-Fermi potential).  ``d`` is the spatial
dimension (1 here; the same class serves 2-D by construction).

Design choices that matter for a semiconductor PINN:

* **tanh activations** - smooth and infinitely differentiable, so the
  second-order autodiff the Poisson residual needs is well behaved.
* **plain tanh MLP by default** - because the hard-constraint wrapper already
  supplies a physical baseline (a smooth junction ramp), the network only has to
  learn a *small, smooth* interior correction.  A plain MLP does that cleanly;
  global Fourier features tend to inject ripple into the neutral bulk, where the
  exponential densities amplify any wiggle into a large charge-neutrality error.
* **optional Fourier features** - a fixed random projection
  ``X -> [sin(2 pi B X), cos(2 pi B X)]`` is available (``fourier=True``) for
  problems with genuinely high-frequency interior structure.
"""

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fixed random Fourier feature embedding ``X -> [sin, cos](2 pi B X)``."""

    def __init__(self, in_dim, num_features=32, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, num_features, generator=g) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINNField(nn.Module):
    def __init__(self, in_dim=1, out_dim=3, width=48, depth=4,
                 fourier=False, fourier_features=32, fourier_sigma=2.0,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim

        if fourier:
            self.embed = FourierFeatures(in_dim, fourier_features,
                                         fourier_sigma, seed=seed)
            first_in = self.embed.out_dim
        else:
            self.embed = None
            first_in = in_dim

        layers = [nn.Linear(first_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

        # Glorot init keeps early residuals O(1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.embed(x) if self.embed is not None else x
        return self.head(self.body(h))
