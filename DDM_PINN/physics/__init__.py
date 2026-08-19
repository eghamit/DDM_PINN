"""Physics: carrier densities, autodiff operators, PDE residuals."""

from DDM_PINN.physics.densities import electron_density, hole_density
from DDM_PINN.physics.residuals import DriftDiffusion1D

__all__ = ["DriftDiffusion1D", "electron_density", "hole_density"]
