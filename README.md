# DDM_PINN

**A Physics-Informed Neural Network (PINN) drift-diffusion semiconductor device
solver** — the neural-network counterpart of the finite-element solver
[`DDM.SPC`](https://github.com/eghamit/schrodinger).

`DDM.SPC` solves the coupled **Poisson + electron/hole-continuity** equations of
a semiconductor device with linear-triangle finite elements and a fully-coupled
Newton method. `DDM_PINN` solves *the same physics, in the same De Mari scaled
quasi-Fermi variables*, but represents the solution as a **neural field trained
to minimise the PDE residual** by automatic differentiation — mesh-free, with no
assembled Jacobian.

The goal of the project is to reproduce, step by step, what `DDM.SPC` does, using
PINNs. This repository is **Phase 1**: the 1-D PN-junction diode (equilibrium and
forward bias), the foundation on which the later phases (2-D, recombination,
heterojunctions, the Schrödinger–Poisson quantum correction) are built.

---

## Physics

The primary unknowns are the three **De Mari-scaled** fields (every unknown is
O(1)), identical to the FEM solver:

| symbol | meaning | scaling |
|--------|---------|---------|
| `u` | electrostatic potential ψ | ψ / V_T |
| `v` | electron quasi-Fermi potential φₙ | φₙ / V_T |
| `w` | hole quasi-Fermi potential φₚ | φₚ / V_T |

with carrier densities `n = nᵢ e^(u−v)`, `p = nᵢ e^(w−u)` and scaled net doping
`C = (N_D − N_A)/nᵢ`. Distances are scaled by the intrinsic Debye length `L_D`.

The PINN minimises the **strong form** of the three coupled equations (the FEM
solver assembles their weak form):

```
Poisson    r_u = u_XX − (n − p − C)
electrons  r_v = d/dX( μ_n n v_X ) + R
holes      r_w = d/dX( μ_p p w_X ) − R
```

`R` is the optional net recombination rate (`R = 0` is the recombination-free
ideal-diffusion model). All derivatives are taken by autodiff, so there is no
mesh and no analytic Jacobian — the network *is* the trial solution.

## Why a semiconductor PINN is hard, and how this one is made to work

In De Mari scaling a realistically doped device has scaled net doping
`C = N/nᵢ ~ 10⁶` and is only a small fraction of an intrinsic Debye length long.
Naïvely this makes the residual loss span ~12 orders of magnitude and the
depletion layer a sub-percent-wide feature — a PINN trained directly on it
either ignores Poisson or diverges. Four ingredients make it converge:

1. **Hard boundary constraints with a physical baseline.** The ohmic-contact
   `(u,v,w)` values are built into the trial solution *exactly* through an
   output transform `f = base(X) + (X/L)(1−X/L)·raw(X)`; the parabolic bubble
   vanishes at both terminals. The potential baseline is a **smooth `tanh` ramp
   between the two neutral-bulk potentials over the depletion width** — flat in
   the neutral regions and stepped at the junction, with *physically bounded*
   curvature. (A charge-neutral `asinh(C/2)` baseline injects a spurious
   `u_XX ~ 10⁹` spike at the junction; matching the depletion width brings it to
   the physical `~10⁶`.) The built-in potential is therefore exact by
   construction.
2. **Normalised input.** The network reads the coordinate normalised to `[0,1]`,
   not the tiny raw scaled coordinate, so its activations actually resolve the
   junction.
3. **Residual normalisation.** Each residual is divided by the doping scale, so
   Poisson and the two continuity equations are all O(1) in the loss — and in
   the neutral bulk the normalised Poisson residual *is* the (exponentially
   sensitive) relative charge-neutrality error.
4. **L-BFGS as the workhorse.** A short Adam warm-up on a fixed collocation grid
   is followed by L-BFGS, whose quasi-Newton steps drive the stiff Poisson
   residual far below what (scale-invariant, noisy) Adam reaches.

## Validation — equilibrium PN diode

`examples/pn_diode_equilibrium.py`, silicon, N_A = N_D = 10²² m⁻³, checked
against the same analytic physics the FEM test-suite uses:

| quantity | analytic | PINN | error |
|----------|----------|------|-------|
| built-in potential V_bi | 0.7143 V | 0.7175 V | **0.45 %** |
| mass action `max|np/nᵢ² − 1|` | 0 | 2 × 10⁻⁵ | — |
| deep-bulk charge neutrality `n−p=C` | 0 | ≤ 5 % | — |

The potential is flat in the neutral regions and drops across the junction; the
carrier densities span the full ~12 decades from majority to minority. Trains in
~2–3 minutes on CPU.

## Forward bias

`examples/pn_diode_iv.py` sweeps the anode with warm-started continuation (each
bias point refines the previous solution — the PINN analogue of Newton
warm-starting) and extracts the terminal current. The junction **rectifies**
(forward current orders of magnitude above reverse saturation). Quantitatively
accurate diode I–V is the demanding frontier of drift-diffusion PINNs — the
terminal current is a small net flux set by exponentially sensitive
minority-carrier injection — and needs a heavier per-point training budget than
the equilibrium solve; tightening it is active work (see the roadmap).

## Installation

```bash
pip install -e .            # numpy, scipy, torch
pip install -e ".[dev]"     # + pytest, matplotlib
```

Requires PyTorch (CPU is fine).

## Quick start

```python
import numpy as np
from DDM_PINN import Device1D, MaterialLibrary, PINNSolver

device = Device1D.pn_junction(length=2e-6, junction=1e-6)   # metres
solver = PINNSolver(
    device,
    MaterialLibrary().load("silicon"),
    doping={"p-region": -1e22, "n-region": 1e22},           # net doping [m^-3]
)

sol = solver.solve_equilibrium()
print(sol.built_in_potential)                                # ~0.71 V

V, I = solver.sweep("anode", np.linspace(0.0, 0.5, 6))       # forward I-V
```

The `PINNSolver` API deliberately mirrors `DDM.SPC`'s `DriftDiffusionSolver`
(`solve_equilibrium`, `solve_bias`, `sweep`, `terminal_current`), so the two
solvers are drop-in comparable.

## Package layout

```
DDM_PINN/
├── core/          physical constants, Material model, De Mari scaling, loader
├── device/        1-D device geometry (regions, contacts) + doping profile
├── nn/            neural field (MLP + optional Fourier features) + hard-BC transform
├── physics/       carrier densities, autodiff operators, PDE residuals
├── boundary/      ohmic-contact boundary conditions (closed-form scaled targets)
├── training/      collocation sampler, composite loss, Adam+L-BFGS trainer
├── postprocess/   physical-field evaluation, terminal-current extraction
└── solver/        top-level PINNSolver (mirrors DDM.SPC's DriftDiffusionSolver)
examples/          runnable demos (equilibrium, forward-bias I-V)
tests/             pytest suite (core, physics, network, equilibrium)
```

## Tests

```bash
pytest -m "not slow"     # fast unit tests (no training)
pytest                   # + the equilibrium integration test (~1 min)
```

## Roadmap — mapping the rest of DDM.SPC onto PINNs

Phase 1 (this repo) is the 1-D diode. The architecture is built so each later
capability enters the *same* residual/network spine:

- **Tight forward-bias I–V** — adaptive loss weighting and a flux-based current
  functional for quantitative ideality-factor accuracy.
- **Recombination** (SRH / Auger / radiative) — plug `R(n,p)` into the
  continuity residuals (the hook already exists) and validate the LED optical
  output.
- **2-D** — the network already takes an arbitrary input dimension; extend the
  residuals to `∇·` operators and sample collocation points over a 2-D domain.
- **Heterojunctions & Fermi–Dirac** — additive band-offset / degeneracy terms in
  the density exponents (as in the FEM solver).
- **Schrödinger–Poisson quantum correction** — couple the confinement
  eigenproblem, exploiting autodiff to avoid the FEM solver's outer Gummel loop.

## Relationship to DDM.SPC

`DDM.SPC` is the finite-element reference solver and the ground-truth oracle for
every validation here. `DDM_PINN` is not a replacement but a **peer** built on
the same physics and scaling, exploring what a mesh-free, differentiable,
parametric formulation buys (instant warm-started sweeps, trivial coupling of
additive physics, and — in later phases — parametric surrogates and inverse
parameter extraction that the FEM solver cannot provide).
