"""Core physics inputs: constants, material, scaling, doping."""

import numpy as np

from DDM_PINN import Device1D, DopingProfile, MaterialLibrary, Scaling
from DDM_PINN.core.constants import PhysicalConstants


def test_constants_loaded():
    assert abs(PhysicalConstants.q - 1.602176634e-19) < 1e-30
    assert abs(PhysicalConstants.kB - 1.380649e-23) < 1e-30


def test_silicon_derived_quantities():
    mat = MaterialLibrary().load("silicon")
    # unit conversions cm^-3 -> m^-3, cm^2/Vs -> m^2/Vs
    assert abs(mat.ni - 1e16) / 1e16 < 1e-9
    assert abs(mat.mobility_n - 0.135) < 1e-6
    # thermal voltage ~ 0.02585 V at 300 K
    assert abs(mat.thermal_voltage - 0.02585) < 5e-4
    # intrinsic Debye length is positive and ~ tens of microns for intrinsic Si
    assert mat.debye_length > 0
    assert 1e-6 < mat.debye_length < 1e-3


def test_scaling_roundtrip():
    mat = MaterialLibrary().load("silicon")
    s = Scaling(mat)
    x = np.array([0.0, 1e-6, 2e-6])
    assert np.allclose(s.unscale_length(s.scale_length(x)), x)
    assert abs(s.scale_voltage(mat.thermal_voltage) - 1.0) < 1e-12
    # scaled doping N/ni
    assert abs(s.scale_doping(1e22) - 1e22 / mat.ni) < 1e-3


def test_doping_regions_and_smoothing():
    mat = MaterialLibrary().load("silicon")
    s = Scaling(mat)
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)
    doping = DopingProfile(device, {"p-region": -1e22, "n-region": 1e22}, s,
                           smooth=0.0)
    # abrupt profile: p side negative, n side positive
    Xp = s.scale_length(0.2e-6)
    Xn = s.scale_length(1.8e-6)
    assert doping.scaled(Xp) < 0
    assert doping.scaled(Xn) > 0
    # contact doping matches the region
    assert doping.contact_C("anode") < 0
    assert doping.contact_C("cathode") > 0


def test_doping_torch_matches_numpy():
    import torch

    mat = MaterialLibrary().load("silicon")
    s = Scaling(mat)
    device = Device1D.pn_junction(length=2e-6, junction=1e-6)
    doping = DopingProfile(device, {"p-region": -1e22, "n-region": 1e22}, s,
                           smooth=1e-3)
    X = np.linspace(0, s.scale_length(2e-6), 50)
    a = doping.scaled(X)
    b = doping.scaled_torch(torch.tensor(X.reshape(-1, 1))).numpy().reshape(-1)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)
