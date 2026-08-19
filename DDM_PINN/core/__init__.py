"""Core physics inputs: constants, material model, De Mari scaling."""

from DDM_PINN.core.constants import ConstantsLibrary, PhysicalConstants
from DDM_PINN.core.loader import MaterialLibrary
from DDM_PINN.core.material import Material
from DDM_PINN.core.scaling import Scaling

__all__ = [
    "PhysicalConstants",
    "ConstantsLibrary",
    "Material",
    "MaterialLibrary",
    "Scaling",
]
