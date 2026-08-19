"""Ohmic-contact boundary conditions.

An ideal ohmic contact imposes local charge neutrality and thermal equilibrium
at the terminal, with the carrier quasi-Fermi levels tied to the applied
terminal voltage.  In scaled variables this has a closed form (identical to the
FEM solver's ohmic contact):

    n - p = C   and   n p = 1   at the contact
      =>  n = exp(u - V) , p = exp(V - u)   with   v = w = V
      =>  2 sinh(u - V) = C
      =>  u = V + asinh(C / 2)

so a contact at scaled applied voltage ``V`` with scaled net doping ``C`` pins

    u = V + asinh(C/2) ,   v = V ,   w = V .

The built-in potential of a PN junction follows immediately as the difference of
the two contacts' ``asinh(C/2)`` at ``V = 0``, which for a highly doped junction
tends to the textbook ``V_bi = V_T ln(N_A N_D / n_i^2)``.
"""

import numpy as np


class OhmicContact:
    def __init__(self, name, position_scaled, scaled_doping):
        self.name = name
        self.position = float(position_scaled)     # scaled coordinate
        self.scaled_doping = float(scaled_doping)  # C at the contact

    def equilibrium_potential(self):
        """Charge-neutral scaled potential ``asinh(C/2)`` at zero bias."""
        return float(np.arcsinh(0.5 * self.scaled_doping))

    def targets(self, scaled_voltage):
        """Return the ``(u, v, w)`` Dirichlet targets at applied scaled ``V``."""
        V = float(scaled_voltage)
        u = V + self.equilibrium_potential()
        return u, V, V

    def targets_direct(self, scaled_voltage):
        """Return the ``(u, ln n, ln p)`` Dirichlet targets at applied ``V``.

        The ohmic contact pins the carrier densities to their charge-neutral
        equilibrium values, which are independent of the applied bias:
        ``ln n = asinh(C/2)``, ``ln p = -asinh(C/2)`` (so ``n p = 1`` and
        ``n - p = C``).  Only the potential carries the applied voltage.
        """
        V = float(scaled_voltage)
        u_eq = self.equilibrium_potential()
        return V + u_eq, u_eq, -u_eq


def build_ohmic_contacts(device, doping, scaling):
    """Build the ohmic contacts for every terminal of ``device``.

    Returns ``{name: OhmicContact}`` with each contact carrying its scaled
    position and its region's scaled net doping.
    """
    contacts = {}
    for name, contact in device.contacts.items():
        contacts[name] = OhmicContact(
            name,
            scaling.scale_length(contact.position),
            doping.contact_C(name),
        )
    return contacts
