"""Equilibrium integration test: validate the PINN against analytic physics.

Uses a reduced training budget so it runs in about a minute; the tolerances are
correspondingly loose but still pin the physics (built-in potential within a few
percent, mass action tight, bulk charge neutrality within ~15%).  Marked ``slow``
so it can be deselected with ``pytest -m 'not slow'``.
"""

import numpy as np
import pytest

from DDM_PINN import Device1D, MaterialLibrary, PINNSolver


@pytest.mark.slow
def test_equilibrium_matches_analytic_physics():
    mat = MaterialLibrary().load("silicon")
    VT = mat.thermal_voltage
    NA = ND = 1e22
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)

    solver = PINNSolver(device, mat,
                        doping={"p-region": -NA, "n-region": ND},
                        seed=0)
    # reduced budget for CI speed
    solver.trainer.adam_steps = 400
    solver.trainer.lbfgs_steps = 400
    solver.trainer.num_collocation = 3000

    sol = solver.solve_equilibrium()

    # built-in potential
    Vbi_ana = VT * np.log(NA * ND / mat.ni ** 2)
    assert abs(sol.built_in_potential - Vbi_ana) / Vbi_ana < 0.05

    # mass action n p = ni^2
    mass_action = np.max(np.abs(sol.electron_density * sol.hole_density
                                / mat.ni ** 2 - 1))
    assert mass_action < 1e-2

    # deep-bulk charge neutrality n - p = C (outside the depletion region)
    x_um = sol.x * 1e6
    bulk = np.abs(x_um - 1.0) > 0.7
    C = np.where(x_um < 1.0, -NA, ND) / mat.ni
    rel = np.abs((sol.electron_density - sol.hole_density) / mat.ni - C) / np.abs(C)
    assert np.max(rel[bulk]) < 0.15
