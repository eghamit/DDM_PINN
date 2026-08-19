"""DDM_PINN - a Physics-Informed Neural Network drift-diffusion device solver.

The PINN counterpart of the FEM reference solver ``DDM.SPC``: it solves the same
coupled Poisson + electron/hole-continuity system in the same De Mari scaled
quasi-Fermi variables ``(u, v, w)``, but represents the solution as a neural
field trained to minimise the PDE residual (autodiff, mesh-free) instead of
assembling a finite-element system and Newton-solving it.

Quick start
-----------
>>> import numpy as np
>>> from DDM_PINN import Device1D, MaterialLibrary, PINNSolver
>>> device = Device1D.pn_junction(length=2e-6, junction=1e-6)
>>> solver = PINNSolver(device, MaterialLibrary().load("silicon"),
...                     doping={"p-region": -1e22, "n-region": 1e22})
>>> solver.solve_equilibrium()
>>> V, I = solver.sweep("anode", np.arange(0.0, 0.71, 0.1))
"""

from DDM_PINN.core.constants import PhysicalConstants
from DDM_PINN.core.loader import MaterialLibrary
from DDM_PINN.core.material import Material
from DDM_PINN.core.scaling import Scaling
from DDM_PINN.device.doping import DopingProfile
from DDM_PINN.device.geometry import Contact, Device1D, Region
from DDM_PINN.nn.network import PINNField
from DDM_PINN.physics.residuals import DriftDiffusion1D
from DDM_PINN.postprocess.fields import Solution1D, evaluate
from DDM_PINN.solver.pinn_solver import PINNSolver
from DDM_PINN.training.losses import LossWeights

__version__ = "0.1.0"

__all__ = [
    "PhysicalConstants",
    "Material",
    "MaterialLibrary",
    "Scaling",
    "Device1D",
    "Region",
    "Contact",
    "DopingProfile",
    "PINNField",
    "DriftDiffusion1D",
    "PINNSolver",
    "LossWeights",
    "Solution1D",
    "evaluate",
]
