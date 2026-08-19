"""Semiconductor material model.

A :class:`Material` bundles the parameters the drift-diffusion PINN needs and
exposes the derived quantities (thermal voltage, permittivity, intrinsic Debye
length and the De Mari current scale) as always-consistent properties.  All
stored values are SI (m, V, s, K, m^-3); energies are kept in eV.

This is intentionally identical in interface and numerics to the FEM reference
solver's ``Material`` so that both solvers consume the same physical inputs and
their results can be compared directly.
"""

import numpy as np

from DDM_PINN.core.constants import PhysicalConstants

_KNOWN_PROPERTIES = (
    "temperature", "epsilon_r", "electron_affinity", "band_gap",
    "ni", "Nc", "Nv", "mobility_n", "mobility_p",
    "m_eff_n", "m_eff_p",
    "tau_n", "tau_p", "C_n", "C_p", "B",
)


class Material:
    def __init__(self, name="custom", **properties):
        """Create a material from SI-valued keyword properties.

        Required for a drift-diffusion solve: ``ni``, ``epsilon_r``,
        ``mobility_n``, ``mobility_p``.  ``temperature`` defaults to 300 K.
        """
        self.name = name
        properties.setdefault("temperature", 300.0)
        self.properties = {k: float(v) for k, v in properties.items()}
        for key, value in self.properties.items():
            setattr(self, key, value)

    @classmethod
    def from_properties(cls, name, properties):
        return cls(name=name, **properties)

    @classmethod
    def silicon(cls, temperature=300.0):
        """Silicon at (default) 300 K, constant-mobility model."""
        return cls(
            name="silicon",
            ni=1.0e10 * 1e6,        # 1e10 cm^-3 -> m^-3
            epsilon_r=11.7,
            electron_affinity=4.05,
            band_gap=1.12,
            Nc=2.8e19 * 1e6,
            Nv=1.04e19 * 1e6,
            mobility_n=1350.0e-4,   # 1350 cm^2/V/s -> m^2/V/s
            mobility_p=480.0e-4,
            m_eff_n=0.26,
            m_eff_p=0.49,
            tau_n=1.0e-6,
            tau_p=1.0e-6,
            temperature=temperature,
        )

    # -- derived quantities ------------------------------------------------
    @property
    def thermal_voltage(self):
        """V_T = kB T / q  [V]."""
        return PhysicalConstants.kB * self.temperature / PhysicalConstants.q

    @property
    def permittivity(self):
        """eps = eps_r eps_0  [F/m]."""
        return self.epsilon_r * PhysicalConstants.eps0

    @property
    def debye_length(self):
        """Intrinsic Debye length L_D = sqrt(eps V_T / (q n_i))  [m]."""
        return np.sqrt(
            self.permittivity
            * self.thermal_voltage
            / (PhysicalConstants.q * self.ni)
        )

    @property
    def reference_mobility(self):
        """Mobility used to normalise transport (taken as the electron one)."""
        return self.mobility_n

    @property
    def current_scale(self):
        """De Mari current-density scale J_0 = q mu0 V_T n_i / L_D  [A/m^2]."""
        return (
            PhysicalConstants.q
            * self.reference_mobility
            * self.thermal_voltage
            * self.ni
            / self.debye_length
        )

    def __repr__(self):
        return (f"Material(name={self.name!r}, ni={self.ni:.3e}, "
                f"epsilon_r={self.epsilon_r})")
