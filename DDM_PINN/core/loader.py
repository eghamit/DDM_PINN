"""Material-file loader.

Reads the plain-text material files in ``DDM_PINN/core/materials/`` and builds
:class:`~DDM_PINN.core.material.Material` objects, converting the human-friendly
units used in the files (cm^-3 for concentrations, cm^2/V/s for mobilities,
cm^6/s for Auger coefficients) into SI on load.
"""

import os

from DDM_PINN.core.material import Material

_MATERIALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "materials")

# keys whose file value is in cm^-3 -> multiply by 1e6 for m^-3
_CONC_KEYS = ("ni", "Nc", "Nv")
# keys whose file value is in cm^2/V/s -> multiply by 1e-4 for m^2/V/s
_MOBILITY_KEYS = ("mobility_n", "mobility_p")
# Auger coefficients cm^6/s -> m^6/s (x 1e-12); radiative B cm^3/s -> m^3/s (x 1e-6)
_AUGER_KEYS = ("C_n", "C_p")
_B_KEYS = ("B",)


def _parse_material_file(path):
    values = {}
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, val = (part.strip() for part in line.split("=", 1))
            values[key] = val
    return values


def _to_si(values):
    name = values.pop("name", "custom")
    si = {}
    for key, raw in values.items():
        try:
            v = float(raw)
        except ValueError:
            continue
        if key in _CONC_KEYS:
            v *= 1e6
        elif key in _MOBILITY_KEYS:
            v *= 1e-4
        elif key in _AUGER_KEYS:
            v *= 1e-12
        elif key in _B_KEYS:
            v *= 1e-6
        si[key] = v
    return name, si


class MaterialLibrary:
    """Loads bundled material files by name."""

    def __init__(self, directory=None):
        self.directory = directory or _MATERIALS_DIR

    def available(self):
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.directory)
            if f.endswith(".txt")
        )

    def load(self, name):
        path = os.path.join(self.directory, f"{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"material '{name}' not found in {self.directory} "
                f"(available: {self.available()})"
            )
        mat_name, si = _to_si(_parse_material_file(path))
        return Material.from_properties(mat_name, si)
