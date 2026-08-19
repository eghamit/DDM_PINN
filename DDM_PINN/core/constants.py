"""Fundamental physical constants, loaded from a data file.

Values live in the editable ``physical_constants.txt`` next to this module
(``symbol = value  # name [unit]``, all SI).  The module-level
``PhysicalConstants`` is a ready-loaded singleton, so attribute access works
everywhere::

    from DDM_PINN.core.constants import PhysicalConstants
    PhysicalConstants.q      # 1.602176634e-19
    PhysicalConstants.kB     # 1.380649e-23

This mirrors the constants model of the FEM reference solver (DDM.SPC) so the
two packages share identical physics inputs.
"""

import os

_CONSTANTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "physical_constants.txt"
)


def parse_constants_file(path):
    """Parse a ``symbol = value`` constants file into a dict of floats."""
    values = {}
    with open(path, "r") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected 'symbol = value', got {raw!r}"
                )
            symbol, value = (part.strip() for part in line.split("=", 1))
            try:
                values[symbol] = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: '{symbol}' value {value!r} is not numeric"
                ) from exc
    return values


class Constants:
    """A set of named physical constants, accessible as attributes."""

    def __init__(self, values):
        self._values = dict(values)
        for symbol, value in self._values.items():
            setattr(self, symbol, value)

    def get(self, symbol):
        return self._values[symbol]

    def __getitem__(self, symbol):
        return self._values[symbol]

    def as_dict(self):
        return dict(self._values)

    def __repr__(self):
        body = ", ".join(f"{k}={v:g}" for k, v in self._values.items())
        return f"Constants({body})"


class ConstantsLibrary:
    """Loads a physical-constants file into a :class:`Constants` object."""

    def __init__(self, path=None):
        self.path = path or _CONSTANTS_FILE

    def load(self):
        return Constants(parse_constants_file(self.path))


PhysicalConstants = ConstantsLibrary().load()
