"""Generate the DDM.SPC finite-element reference for the formulation comparison.

Runs the FEM reference solver (DDM.SPC) on a thin rectangular PN diode that is
1-D in physics (fields vary only along x), solves thermal equilibrium, extracts a
horizontal cut through the mid-height row, and saves ``(x, phi, n, p)`` to
``fem_reference.npz`` for :mod:`examples.compare_formulations` to compare against.

DDM.SPC must be importable.  Point ``--ddmspc`` at its repository root (the
directory containing the ``DDM_SPC`` package)::

    python examples/make_fem_reference.py --ddmspc /path/to/schrodinger
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fem_reference.npz")


def main(ddmspc_root, NA=1e22, ND=1e22, L=2e-6, xj=1e-6):
    sys.path.insert(0, ddmspc_root)
    from DDM_SPC import (
        DriftDiffusionSolver, MaterialLibrary, RectangleMeshBuilder,
    )

    mesh = RectangleMeshBuilder(L, 0.2e-6, 200, 4).build(
        region_of=lambda x, y: "p-region" if x < xj else "n-region",
        region_names=["p-region", "n-region"])
    mat = MaterialLibrary().load("silicon")
    solver = DriftDiffusionSolver(
        mesh=mesh, material=mat,
        doping={"p-region": -NA, "n-region": ND},
        contacts={
            "anode": {"type": "ohmic", "region": "p-region",
                      "nodes": mesh.boundary_nodes["left"]},
            "cathode": {"type": "ohmic", "region": "n-region",
                        "nodes": mesh.boundary_nodes["right"]},
        })
    sol = solver.solve_equilibrium()

    nodes = solver.physical_mesh.nodes
    ys = np.unique(nodes[:, 1])
    yrow = ys[np.argmin(np.abs(ys - 0.1e-6))]
    mask = np.abs(nodes[:, 1] - yrow) < 1e-12
    order = np.argsort(nodes[mask, 0])
    np.savez(OUT,
             x=nodes[mask, 0][order],
             phi=sol.potential[mask][order],
             n=sol.electron_density[mask][order],
             p=sol.hole_density[mask][order],
             NA=NA, ND=ND)
    VT = mat.thermal_voltage
    Vbi = sol.potential[mask].max() - sol.potential[mask].min()
    print(f"saved {OUT}")
    print(f"FEM V_bi = {Vbi:.4f} V  (analytic {VT * np.log(NA * ND / mat.ni**2):.4f} V)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddmspc", default="/home/user/Schrodinger",
                        help="path to the DDM.SPC repository root")
    args = parser.parse_args()
    main(args.ddmspc)
