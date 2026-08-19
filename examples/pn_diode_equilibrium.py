"""PN-junction diode at thermal equilibrium, solved with the PINN.

This is the PINN counterpart of the FEM reference solver's equilibrium solve.  A
neural field is trained to minimise the coupled Poisson + continuity residuals in
De Mari scaled variables, and the result is checked against the same analytic
semiconductor physics the FEM suite uses:

* the built-in potential vs. ``V_bi = V_T ln(N_A N_D / n_i^2)``,
* the law of mass action ``n p = n_i^2`` at equilibrium,
* charge neutrality ``n - p = C`` in the neutral bulk.

Run::

    python examples/pn_diode_equilibrium.py
    python examples/pn_diode_equilibrium.py --plot
"""

import argparse

import numpy as np

from DDM_PINN import Device1D, MaterialLibrary, PINNSolver


def main(plot=False, NA=1e22, ND=1e22):
    mat = MaterialLibrary().load("silicon")
    VT = mat.thermal_voltage
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)

    solver = PINNSolver(
        device, mat,
        doping={"p-region": -NA, "n-region": ND},
        verbose=True,
    )

    print("=== PINN 1-D PN diode drift-diffusion (equilibrium) ===\n")
    print(f"material : {mat.name}  (ni = {mat.ni:.3e} m^-3, eps_r = {mat.epsilon_r})")
    print(f"scaling  : V_T = {VT:.4f} V,  L_D = {mat.debye_length:.3e} m")

    sol = solver.solve_equilibrium()

    Vbi_num = sol.built_in_potential
    Vbi_ana = VT * np.log(NA * ND / mat.ni ** 2)
    mass_action = np.max(np.abs(sol.electron_density * sol.hole_density
                                / mat.ni ** 2 - 1))

    x_um = sol.x * 1e6
    junction_um = 1.0
    # deep-bulk points: well outside the depletion region
    bulk = (np.abs(x_um - junction_um) > 0.7)
    C = np.where(x_um < junction_um, -NA, ND) / mat.ni
    neutrality = np.abs((sol.electron_density - sol.hole_density) / mat.ni - C)
    rel_neutrality = np.max((neutrality / np.abs(C))[bulk])

    print("\n--- validation against analytic physics ---")
    print(f"  built-in potential : PINN {Vbi_num:.4f} V   analytic {Vbi_ana:.4f} V"
          f"   ({100 * abs(Vbi_num - Vbi_ana) / Vbi_ana:.2f}% error)")
    print(f"  max |n p / ni^2 - 1|          : {mass_action:.2e}")
    print(f"  max bulk charge-neutrality err: {rel_neutrality:.2e}")

    if plot:
        _plot(sol, NA, ND, mat)

    print("\nDone.")


def _plot(sol, NA, ND, mat):
    import matplotlib.pyplot as plt

    x = sol.x * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, sol.potential, lw=2)
    ax[0].set_xlabel("x [um]"); ax[0].set_ylabel("potential [V]")
    ax[0].set_title("Electrostatic potential")
    ax[1].semilogy(x, sol.electron_density, label="n")
    ax[1].semilogy(x, sol.hole_density, label="p")
    ax[1].set_xlabel("x [um]"); ax[1].set_ylabel("density [m^-3]")
    ax[1].set_title("Carrier densities"); ax[1].legend()
    fig.suptitle("PINN PN-diode equilibrium")
    fig.tight_layout()
    plt.savefig("pn_diode_equilibrium.png", dpi=120)
    print("saved pn_diode_equilibrium.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(plot=args.plot)
