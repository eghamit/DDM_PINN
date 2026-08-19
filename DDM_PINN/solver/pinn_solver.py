"""Top-level PINN drift-diffusion solver.

``PINNSolver`` is the PINN counterpart of the FEM reference solver's
``DriftDiffusionSolver``: same conceptual inputs (a device, a material, a
per-region doping dict, contacts), same top-level verbs
(:meth:`solve_equilibrium`, :meth:`solve_bias`, :meth:`sweep`,
:meth:`terminal_current`), but the coupled Poisson + electron/hole-continuity
system is solved by training a neural field to minimise the PDE residual instead
of assembling an FEM system and Newton-solving it.

    from DDM_PINN import Device1D, MaterialLibrary, PINNSolver

    device = Device1D.pn_junction(length=2e-6, junction=1e-6)
    solver = PINNSolver(
        device=device,
        material=MaterialLibrary().load("silicon"),
        doping={"p-region": -1e22, "n-region": 1e22},   # net doping [m^-3]
    )
    solver.solve_equilibrium()
    V, I = solver.sweep("anode", np.arange(0, 0.71, 0.05))

The network is warm-started between bias points (continuation), exactly as the
FEM sweep warm-starts each Newton solve from the previous solution.
"""

import numpy as np
import torch

from DDM_PINN.boundary.contacts import build_ohmic_contacts
from DDM_PINN.core.scaling import Scaling
from DDM_PINN.device.doping import DopingProfile
from DDM_PINN.nn.constraints import HardBC1D
from DDM_PINN.nn.network import PINNField
from DDM_PINN.physics.residuals import DriftDiffusion1D, DriftDiffusionDirect1D
from DDM_PINN.postprocess.current import current_density
from DDM_PINN.postprocess.fields import evaluate
from DDM_PINN.training.losses import LossWeights
from DDM_PINN.training.sampler import Sampler
from DDM_PINN.training.trainer import Trainer


class PINNSolver:
    def __init__(self, device, material, doping, contacts=None,
                 recombination=None, area=1.0, smooth_junction=None,
                 network=None, weights=None, trainer=None, dtype=torch.float64,
                 formulation="quasi-fermi", seed=0, verbose=False):
        """
        Parameters
        ----------
        device : Device1D
            Continuous 1-D device geometry (regions + contacts).
        material : Material
            Semiconductor material (single-material Phase-1 solve).
        doping : dict[str, float]
            ``{region_name: net_doping_m^-3}`` (signed).
        contacts : dict, optional
            Reserved for contact-type overrides; ohmic contacts are built from
            the device terminals by default.
        area : float
            Cross-sectional area [m^2] used to turn the current *density* into a
            terminal current in amperes (analogous to ``device_depth`` in the
            FEM solver; default 1.0 makes the ampere value equal to the current
            density numerically).
        smooth_junction : float
            Junction smoothing width in scaled length units (regularises the
            abrupt doping step; set 0 for a hard step).
        recombination : callable, optional
            ``R(n, p) -> scaled net rate`` added to the continuity residuals
            (default ``None`` = recombination-free ideal-diffusion model).
        """
        torch.set_default_dtype(dtype)
        if formulation not in ("quasi-fermi", "direct"):
            raise ValueError(
                f"formulation must be 'quasi-fermi' or 'direct', got "
                f"{formulation!r}.")
        self.formulation = formulation
        self.device = device
        self.material = material
        self.scaling = Scaling(material)
        self.area = float(area)
        self.dtype = dtype
        self.verbose = verbose

        # junction smoothing width in *scaled* length units.  It is matched to
        # the estimated depletion width so the charge-neutral baseline's junction
        # step has about the right curvature (a much sharper step would put a
        # large spurious spike into the Poisson residual right at the junction).
        Ls = self.scaling.scale_length(device.length)
        if smooth_junction is None:
            smooth_junction = 0.5 * self._depletion_width(doping)
        self.doping = DopingProfile(device, doping, self.scaling,
                                    smooth=smooth_junction)
        self.contacts = build_ohmic_contacts(device, self.doping, self.scaling)

        # Poisson-residual normalisation: the largest scaled net doping in the
        # device (C = N / n_i), so the space-charge term is O(1) in the loss.
        poisson_scale = max(
            1.0, max(abs(c) for c in self.doping.region_C.values()))
        mun = self.scaling.scale_mobility(material.mobility_n)
        mup = self.scaling.scale_mobility(material.mobility_p)
        physics_cls = (DriftDiffusion1D if formulation == "quasi-fermi"
                       else DriftDiffusionDirect1D)
        self.physics = physics_cls(mun, mup, recombination=recombination,
                                   poisson_scale=poisson_scale,
                                   continuity_scale=poisson_scale)

        # neural field wrapped in the hard ohmic-BC transform, so the terminal
        # values (hence the built-in potential) are exact by construction and
        # the boundary loss term is redundant.  Quasi-fermi ramps only the
        # potential baseline; the direct (u, ln n, ln p) form ramps all three,
        # since the log densities also step across the junction.
        base_net = network or PINNField(in_dim=1, out_dim=3, seed=seed)
        junctions = [self.scaling.scale_length(r.x1)
                     for r in device.regions[:-1]]
        w_dep = self._depletion_width(doping)
        ramp = (0,) if formulation == "quasi-fermi" else (0, 1, 2)
        self.net = HardBC1D(base_net, Ls, junctions, 0.4 * w_dep,
                            ramp_components=ramp)
        self.net.to(dtype)
        # ordered (left, right) terminals by scaled position
        ordered = sorted(self.contacts.values(), key=lambda c: c.position)
        self._left, self._right = ordered[0], ordered[-1]
        # hard BC makes the boundary term exact -> weight it 0 (kept as a
        # diagnostic in the loss dict)
        self.weights = weights or LossWeights(poisson=1.0, electron=1.0,
                                              hole=1.0, boundary=0.0)
        self.sampler = Sampler(device, self.scaling, dtype=dtype, seed=seed)
        self.trainer = trainer or Trainer(
            self.physics, self.sampler, self.doping, self.contacts,
            dtype=dtype, formulation=formulation, verbose=verbose)

        self.applied_voltages = {name: 0.0 for name in self.contacts}
        self.state_ready = False

    def _depletion_width(self, doping):
        """Estimate the equilibrium depletion width in scaled length units.

        Abrupt-junction depletion approximation in De Mari variables:
        ``W = sqrt(2 u_bi (1/|C_p| + 1/C_n))`` with the scaled built-in
        potential ``u_bi = asinh(C_n/2) - asinh(C_p/2)``.  Used only to size the
        junction-smoothing width, so a rough estimate is sufficient.
        """
        Cs = [self.scaling.scale_doping(float(v)) for v in doping.values()]
        neg = [abs(c) for c in Cs if c < 0]
        pos = [c for c in Cs if c > 0]
        if not neg or not pos:
            return 0.02 * self.scaling.scale_length(self.device.length)
        Cp, Cn = max(neg), max(pos)
        u_bi = float(np.arcsinh(Cn / 2) - np.arcsinh(-Cp / 2))
        return float(np.sqrt(2.0 * u_bi * (1.0 / Cp + 1.0 / Cn)))

    # -- scaled-voltage helper --------------------------------------------
    def _scaled(self, voltages):
        return {name: self.scaling.scale_voltage(v)
                for name, v in voltages.items()}

    def _apply_bc(self, scaled_voltages):
        """Set the hard-BC terminal targets for the current applied bias."""
        vL = scaled_voltages.get(self._left.name, 0.0)
        vR = scaled_voltages.get(self._right.name, 0.0)
        if self.formulation == "direct":
            left, right = (self._left.targets_direct(vL),
                           self._right.targets_direct(vR))
        else:
            left, right = self._left.targets(vL), self._right.targets(vR)
        self.net.set_targets(left, right)

    # -- equilibrium -------------------------------------------------------
    def solve_equilibrium(self):
        """Train the field at thermal equilibrium (all contacts at 0 V)."""
        self._log("=== PINN thermal equilibrium (V = 0) ===")
        applied = {name: 0.0 for name in self.contacts}
        self._apply_bc(self._scaled(applied))
        self.trainer.solve_point(self.net, self._scaled(applied), self.weights)
        self.applied_voltages = dict(applied)
        self.state_ready = True
        return self.solution()

    # -- biased solve ------------------------------------------------------
    def solve_bias(self, voltages):
        """Train (warm-started) at the given applied terminal voltages [V]."""
        if not self.state_ready:
            self.solve_equilibrium()
        applied = {name: float(voltages.get(name, 0.0))
                   for name in self.contacts}
        bias = ", ".join(f"{k}={v:+.4g} V" for k, v in applied.items())
        self._log(f"=== PINN bias point: {bias} ===")
        self._apply_bc(self._scaled(applied))
        self.trainer.solve_point(self.net, self._scaled(applied), self.weights)
        self.applied_voltages = dict(applied)
        return self.solution()

    # -- continuation sweep ------------------------------------------------
    def sweep(self, contact, voltages, hold=None, ramp_step=0.1):
        """Sweep ``contact`` over ``voltages`` [V]; return ``(V, I)`` in amps.

        Warm-starts each point from the previous solution and ramps from
        equilibrium to the first point in ``ramp_step`` increments so the field
        never has to make a large jump (the PINN analogue of Newton
        continuation).
        """
        hold = dict(hold or {})
        voltages = np.asarray(voltages, dtype=float)
        self.solve_equilibrium()

        # continuation to the first target
        first = float(voltages[0])
        nramp = max(1, int(np.ceil(abs(first) / ramp_step)))
        for s in range(1, nramp + 1):
            f = s / nramp
            self.solve_bias({**hold, contact: f * first})

        currents = np.zeros(voltages.size)
        for k, V in enumerate(voltages):
            self.solve_bias({**hold, contact: float(V)})
            currents[k] = self.terminal_current()
            self._log(f"  V({contact})={V:+.4g} V   I={currents[k]:+.4e} A")
        return voltages, currents

    # -- observables -------------------------------------------------------
    def terminal_current(self):
        """Total terminal current [A] = current density [A/m^2] * area."""
        J, _ = current_density(self.physics, self.net, self.scaling,
                               self.device.length)
        return J * self.area

    def current_uniformity(self):
        """Spread of J(x) across the device [A/m^2] (convergence diagnostic)."""
        _, spread = current_density(self.physics, self.net, self.scaling,
                                    self.device.length)
        return spread

    def solution(self, num=401):
        """Physical fields (potential, quasi-Fermi potentials, n, p) on a grid."""
        return evaluate(self.net, self.scaling, self.device.length, num=num,
                        formulation=self.formulation)

    def _log(self, msg):
        if self.verbose:
            print(msg)
