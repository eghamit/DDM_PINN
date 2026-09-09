"""PN-junction diode forward-bias I-V curve, solved with the PINN.

Sweeps the anode voltage with warm-started continuation (each bias point refines
the previous solution, the PINN analogue of Newton warm-starting), extracts the
terminal current at each point, and checks the diode is rectifying.

Accurate diode I-V is the demanding part of a drift-diffusion PINN: the terminal
current is a small net flux set by exponentially sensitive minority-carrier
injection, so it needs a heavier training budget per bias point than the
equilibrium solve.  The per-point budget is exposed below.

Run::

    python examples/pn_diode_iv.py
    python examples/pn_diode_iv.py --plot
"""

import argparse

import numpy as np

from DDM_PINN import Device1D, MaterialLibrary, PINNSolver


def main(plot=False, NA=1e22, ND=1e22, vmax=0.5, npts=6):
    mat = MaterialLibrary().load("silicon")
    VT = mat.thermal_voltage
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)

    solver = PINNSolver(
        device, mat,
        doping={"p-region": -NA, "n-region": ND},
        area=1e-8,               # 100 um^2 cross-section -> amperes
        verbose=False,
    )
    # per-bias-point training budget (warm-started continuation)
    solver.trainer.adam_steps = 400
    solver.trainer.lbfgs_steps = 700
    solver.trainer.num_collocation = 4000
    # M1: gradient-norm balancing so the current-carrying continuity terms are
    # not starved in the bulk under forward bias (key to a clean exponential I-V)
    solver.trainer.adaptive_weights = True
    solver.trainer.reweight_every = 150

    V = np.linspace(0.0, vmax, npts)
    print(f"=== PINN PN-diode forward-bias I-V (0 -> {vmax} V) ===")
    Vout, I = solver.sweep("anode", V, ramp_step=0.05)

    print("\n  V [V]      I [A]")
    for v, i in zip(Vout, I):
        print(f"  {v:5.3f}   {i:+.4e}")

    # ideality factor from the mid-bias slope of ln I vs V (ideal diode -> ~1)
    VT = mat.thermal_voltage
    mask = (Vout >= 0.15) & (Vout <= 0.45) & (I > 0)
    if mask.sum() >= 2:
        slope = np.polyfit(Vout[mask], np.log(I[mask]), 1)[0]
        print(f"\n  ideality factor ~ {(1.0 / VT) / slope:.3f}  (ideal = 1.0)")
    mono = bool(np.all(np.diff(I) >= -1e-15))
    print(f"  monotonic I(V): {mono}")

    if plot:
        _plot(Vout, I)
    print("\nDone.")
    return Vout, I


def _plot(V, I):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(V, I, "o-")
    ax[0].set_xlabel("V [V]"); ax[0].set_ylabel("I [A]"); ax[0].set_title("I-V")
    ax[1].semilogy(V, np.abs(I), "o-")
    ax[1].set_xlabel("V [V]"); ax[1].set_ylabel("|I| [A]")
    ax[1].set_title("I-V (semilog)")
    fig.suptitle("PINN PN-diode forward-bias I-V")
    fig.tight_layout()
    plt.savefig("pn_diode_iv.png", dpi=120)
    print("saved pn_diode_iv.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(plot=args.plot)
