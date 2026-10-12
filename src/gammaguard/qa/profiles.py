"""Detector profiles: everything that differs between a germanium and a scintillation spectrometer.

A profile fixes the default reference lines, how peaks are fitted (background shape, region of
interest, blending), which resolution law is used, and the matching tolerances for spectral
features. Profiles only provide *defaults*: they are copied into the `Baseline` when it is built,
so a saved baseline is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..isotopes import NAI_REFERENCE_ENERGIES, QA_REFERENCE_ENERGIES
from ..resolution import PowerLawResolution, Resolution, ResolutionModel
from .baseline import FitOptions, Tolerances


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    reference_energies: tuple[float, ...]
    initial_resolution: Resolution  # seeds the first-pass peak widths only
    resolution_kind: str  # "quadrature" (FWHM^2 = a + bE) or "power" (FWHM = k E^p)
    fit: FitOptions = field(default_factory=FitOptions)
    tolerances: Tolerances = field(default_factory=Tolerances)
    calibration_order: int = 1
    power_law_exponent: float = 0.5  # used when too few lines to fit the exponent


HPGE = DetectorProfile(
    name="hpge",
    reference_energies=QA_REFERENCE_ENERGIES,
    initial_resolution=ResolutionModel(a=1.5**2, b=0.0),  # constant 1.5 keV seed
    resolution_kind="quadrature",
    fit=FitOptions(step=False, roi_fwhm=4.0, search_fwhm=10.0, multiplet_fwhm=2.5),
    tolerances=Tolerances(),
)

# Scintillator: broad, partly blended peaks on a Compton continuum with a step under each line.
NAI = DetectorProfile(
    name="nai",
    reference_energies=NAI_REFERENCE_ENERGIES,
    initial_resolution=PowerLawResolution(k=1.6, p=0.55),  # ~8.6 % at 662 keV, ~6.3 % at 1332 keV
    resolution_kind="power",
    fit=FitOptions(step=True, roi_fwhm=2.0, search_fwhm=2.5, multiplet_fwhm=3.0),
    tolerances=Tolerances(peak_match_fwhm=1.0, min_search_keV=50.0),
    power_law_exponent=0.55,
)

PROFILES: dict[str, DetectorProfile] = {"hpge": HPGE, "nai": NAI}


def get_profile(name: str | DetectorProfile) -> DetectorProfile:
    if isinstance(name, DetectorProfile):
        return name
    try:
        return PROFILES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown detector profile {name!r}; choose from {sorted(PROFILES)}") from None
