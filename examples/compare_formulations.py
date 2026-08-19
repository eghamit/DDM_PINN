"""Compare the two PINN formulations against the DDM.SPC FEM oracle.

Solves the equilibrium PN diode three ways and compares them point for point:

* the quasi-Fermi ``(u, v, w)`` PINN,
* the direct ``(u, ln n, ln p)`` PINN (primary-variable psi, n, p form with a
  log-density parametrisation - the form used by classical simulators / DDNet),
* the finite-element reference solved by DDM.SPC (loaded from
  ``fem_reference.npz``; regenerate with ``examples/make_fem_reference.py``).

Both PINN formulations minimise the strong-form residual of the *same* three
drift-diffusion equations, so the comparison isolates the effect of the
formulation on conditioning and accuracy, with the FEM solve as ground truth.

Reported per formulation: built-in potential, mass action, deep-bulk charge
neutrality, wall-clock training time, and the relative L2 error of the potential
and the log-density fields against the FEM cut.

Run (after generating the FEM reference)::

    python examples/make_fem_reference.py       # needs DDM.SPC on the path
    python examples/compare_formulations.py --plot
"""

import argparse
import os
import time

import numpy as np

from DDM_PINN import Device1D, MaterialLibrary, PINNSolver

HERE = os.path.dirname(os.path.abspath(__file__))
FEM_REF = os.path.join(HERE, "..", "fem_reference.npz")


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(formulation, device, mat, NA, ND, budget, fem):
    solver = PINNSolver(device, mat, doping={"p-region": -NA, "n-region": ND},
                        formulation=formulation, seed=0)
    solver.trainer.adam_steps = budget["adam"]
    solver.trainer.lbfgs_steps = budget["lbfgs"]
    solver.trainer.num_collocation = budget["colloc"]
    t0 = time.time()
    sol = solver.solve_equilibrium()
    dt = time.time() - t0

    VT = mat.thermal_voltage
    Vbi_ana = VT * np.log(NA * ND / mat.ni ** 2)
    mass = np.max(np.abs(sol.electron_density * sol.hole_density
                         / mat.ni ** 2 - 1))
    x_um = sol.x * 1e6
    bulk = np.abs(x_um - 1.0) > 0.7
    C = np.where(x_um < 1.0, -NA, ND) / mat.ni
    rel = np.abs((sol.electron_density - sol.hole_density) / mat.ni - C) / np.abs(C)

    out = {"formulation": formulation, "Vbi": sol.built_in_potential,
           "Vbi_err": abs(sol.built_in_potential - Vbi_ana) / Vbi_ana,
           "mass_action": mass, "bulk_neutrality": np.max(rel[bulk]),
           "time_s": dt, "sol": sol}

    if fem is not None:
        # interpolate the PINN fields onto the FEM cut coordinates
        phi_p = np.interp(fem["x"], sol.x, sol.potential)
        ln_n_p = np.interp(fem["x"], sol.x, np.log(sol.electron_density))
        ln_p_p = np.interp(fem["x"], sol.x, np.log(sol.hole_density))
        out["phi_L2_vs_fem"] = _rel_l2(phi_p, fem["phi"])
        out["lnn_L2_vs_fem"] = _rel_l2(ln_n_p, np.log(fem["n"]))
        out["lnp_L2_vs_fem"] = _rel_l2(ln_p_p, np.log(fem["p"]))
    return out


def main(plot=False):
    mat = MaterialLibrary().load("silicon")
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)
    NA = ND = 1e22
    budget = {"adam": 800, "lbfgs": 800, "colloc": 4000}

    fem = None
    if os.path.exists(FEM_REF):
        fem = dict(np.load(FEM_REF))
        print(f"loaded FEM reference ({fem['x'].size} points)\n")
    else:
        print("no fem_reference.npz found - skipping FEM comparison\n")

    print("=== Formulation comparison (equilibrium Si PN diode) ===\n")
    rows = [run(f, device, mat, NA, ND, budget, fem)
            for f in ("quasi-fermi", "direct")]

    cols = f"{'formulation':<13}{'V_bi[V]':>9}{'V_bi err':>9}{'mass act':>11}" \
           f"{'bulk neut':>11}{'t[s]':>7}"
    if fem is not None:
        cols += f"{'phi L2':>9}{'lnn L2':>9}{'lnp L2':>9}"
    print(cols)
    print("-" * len(cols))
    for r in rows:
        line = f"{r['formulation']:<13}{r['Vbi']:>9.4f}{r['Vbi_err']:>9.2%}" \
               f"{r['mass_action']:>11.2e}{r['bulk_neutrality']:>11.2e}" \
               f"{r['time_s']:>7.0f}"
        if fem is not None:
            line += f"{r['phi_L2_vs_fem']:>9.2e}{r['lnn_L2_vs_fem']:>9.2e}" \
                    f"{r['lnp_L2_vs_fem']:>9.2e}"
        print(line)

    if plot and fem is not None:
        _plot(rows, fem)


def _plot(rows, fem):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    xf = fem["x"] * 1e6
    ax[0].plot(xf, fem["phi"], "k-", lw=3, alpha=0.4, label="FEM (DDM.SPC)")
    ax[1].semilogy(xf, fem["n"], "k-", lw=3, alpha=0.4, label="FEM n")
    ax[1].semilogy(xf, fem["p"], "k--", lw=3, alpha=0.4, label="FEM p")
    styles = {"quasi-fermi": "C0", "direct": "C1"}
    for r in rows:
        s = r["sol"]; x = s.x * 1e6; c = styles[r["formulation"]]
        ax[0].plot(x, s.potential, c, lw=1.5, label=r["formulation"])
        ax[1].semilogy(x, s.electron_density, c, lw=1.2, label=f"{r['formulation']} n")
    ax[0].set_xlabel("x [um]"); ax[0].set_ylabel("potential [V]")
    ax[0].set_title("Potential vs FEM"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("x [um]"); ax[1].set_ylabel("density [m^-3]")
    ax[1].set_title("Densities vs FEM"); ax[1].legend(fontsize=8)
    fig.suptitle("PINN formulations vs DDM.SPC FEM (equilibrium)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "compare_formulations.png"), dpi=120)
    print("\nsaved compare_formulations.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(plot=args.plot)
