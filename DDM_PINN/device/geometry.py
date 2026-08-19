"""1-D device geometry.

A :class:`Device1D` describes a one-dimensional semiconductor device on the
physical interval ``[0, length]`` (metres), partitioned into named regions and
terminated by two ohmic contacts.  It is the PINN analogue of the FEM solver's
meshed ``Mesh2D`` + region/contact tags, but it stores the *continuous*
geometry (region boundaries as coordinates) rather than a discretisation - the
PINN samples collocation points from it on the fly.

The canonical example is an abrupt PN junction::

    device = Device1D.pn_junction(length=2e-6, junction=1e-6,
                                   p_region="p", n_region="n",
                                   anode="anode", cathode="cathode")

which places a ``p`` region on ``[0, junction]`` and an ``n`` region on
``[junction, length]``, with the anode contact at ``x = 0`` and the cathode at
``x = length``.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Region:
    """A named 1-D sub-interval ``[x0, x1]`` (metres)."""
    name: str
    x0: float
    x1: float

    def contains(self, x):
        return (x >= self.x0) & (x <= self.x1)

    @property
    def width(self):
        return self.x1 - self.x0


@dataclass(frozen=True)
class Contact:
    """A terminal at one end of the device (a boundary point in 1-D)."""
    name: str
    position: float          # metres
    region: str              # region the contact sits in (for its doping)


@dataclass
class Device1D:
    length: float
    regions: list = field(default_factory=list)
    contacts: dict = field(default_factory=dict)

    @classmethod
    def pn_junction(cls, length, junction, p_region="p-region",
                    n_region="n-region", anode="anode", cathode="cathode"):
        """Abrupt PN junction: p on ``[0, junction]``, n on ``[junction, L]``."""
        regions = [
            Region(p_region, 0.0, junction),
            Region(n_region, junction, length),
        ]
        contacts = {
            anode: Contact(anode, 0.0, p_region),
            cathode: Contact(cathode, length, n_region),
        }
        return cls(length=length, regions=regions, contacts=contacts)

    def region_of(self, x):
        """Name of the region containing scalar coordinate ``x``."""
        for region in self.regions:
            if region.contains(x):
                return region.name
        return None

    @property
    def region_names(self):
        return [r.name for r in self.regions]
