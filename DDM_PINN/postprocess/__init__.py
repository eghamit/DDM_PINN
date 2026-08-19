"""Post-processing: physical fields and terminal current."""

from DDM_PINN.postprocess.current import current_density
from DDM_PINN.postprocess.fields import Solution1D, evaluate

__all__ = ["evaluate", "Solution1D", "current_density"]
