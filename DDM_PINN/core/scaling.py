"""De Mari (intrinsic) scaling.

The drift-diffusion system is solved in dimensionless form:

    length      x   -> X = x / L_D        (L_D = intrinsic Debye length)
    potential   psi -> u = psi / V_T
    densities   n,p -> n / n_i, p / n_i
    mobility    mu  -> mu / mu_0
    current     J   -> J / J_0

Working in these variables keeps every primary unknown O(1) and removes the huge
dynamic range of the carrier densities from the optimisation - which is exactly
what makes the PINN residual loss well conditioned.  Numerically identical to the
FEM reference solver's scaling.
"""


class Scaling:
    def __init__(self, material):
        self.material = material

    @property
    def length(self):
        return self.material.debye_length

    @property
    def potential(self):
        return self.material.thermal_voltage

    @property
    def density(self):
        return self.material.ni

    @property
    def mobility(self):
        return self.material.reference_mobility

    @property
    def current(self):
        return self.material.current_scale

    # -- helpers -----------------------------------------------------------
    def scale_length(self, x):
        return x / self.length

    def unscale_length(self, X):
        return X * self.length

    def scale_doping(self, doping):
        return doping / self.density

    def scale_mobility(self, mobility):
        return mobility / self.mobility

    def scale_voltage(self, voltage):
        return voltage / self.potential

    def unscale_potential(self, u):
        return u * self.potential

    def unscale_density(self, value):
        return value * self.density

    def unscale_current(self, value):
        return value * self.current
