"""Collocation-point sampling.

The PINN residual loss is a Monte-Carlo estimate of the PDE residual over the
device, so the choice of collocation points matters.  :class:`Sampler` draws:

* **interior points** uniformly over the scaled domain, plus
* **junction-refined points** clustered around the metallurgical junction(s),
  where the depletion layer makes the fields vary fastest and a uniform sample
  would under-resolve them.

Boundary points in 1-D are just the two terminal coordinates.  Points are
resampled each call so the network sees fresh collocation locations over
training (a light, cheap form of importance sampling that improves coverage).
"""

import numpy as np
import torch


class Sampler:
    def __init__(self, device, scaling, junctions=None, junction_frac=0.4,
                 junction_width=None, dtype=torch.float64, seed=0):
        self.length = scaling.scale_length(device.length)
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.junction_frac = float(junction_frac)
        # scaled junction coordinates (interior region interfaces)
        if junctions is None:
            junctions = [scaling.scale_length(r.x1)
                         for r in device.regions[:-1]]
        self.junctions = [float(j) for j in junctions]
        # default clustering width: a fraction of the (scaled) device length,
        # so refinement concentrates points in the depletion region regardless
        # of how short the device is in De Mari-scaled units.
        self.junction_width = (float(junction_width)
                               if junction_width is not None
                               else 0.1 * self.length)
        # terminal coordinates
        self.boundary = np.array(
            sorted(scaling.scale_length(c.position)
                   for c in device.contacts.values()),
            dtype=float,
        )

    def interior(self, num):
        """Draw ``num`` interior collocation coordinates as an ``(M,1)`` tensor."""
        n_junc = int(self.junction_frac * num) if self.junctions else 0
        n_unif = num - n_junc
        xs = [self.rng.uniform(0.0, self.length, size=n_unif)]
        if n_junc and self.junctions:
            per = max(1, n_junc // len(self.junctions))
            for j in self.junctions:
                x = self.rng.normal(j, self.junction_width, size=per)
                xs.append(np.clip(x, 0.0, self.length))
        x = np.concatenate(xs)[:num] if num else np.empty(0)
        t = torch.tensor(x.reshape(-1, 1), dtype=self.dtype)
        t.requires_grad_(True)
        return t

    def boundary_points(self):
        """Return the terminal coordinates as an ``(B,1)`` tensor (no grad)."""
        return torch.tensor(self.boundary.reshape(-1, 1), dtype=self.dtype)
