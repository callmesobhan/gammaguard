"""Forward model for HPGe-like and NaI(Tl)-like pulse-height spectra.

The model is deliberately simple and transparent: Gaussian full-energy peaks integrated
exactly over each channel, a flat Compton shelf per source line, an exponential ambient
continuum, ambient background lines and Poisson sampling. It is meant for exercising and
benchmarking QA logic, not for detector physics. Known omissions: low-energy peak tails,
escape/sum peaks, X-rays, pile-up, differential non-linearity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import erfc, ndtr

from ..calibration import EnergyCalibration
from ..isotopes import BACKGROUND_LINES, SOURCES
from ..resolution import PowerLawResolution, Resolution, ResolutionModel
from ..spectrum import Spectrum

ME_C2_KEV = 510.999
CS137_KEV = 661.657


@dataclass(frozen=True)
class DetectorConfig:
    """Detector as configured (``calibration``) and as it physically is (``actual_calibration``)."""

    n_channels: int = 8192
    calibration: EnergyCalibration = EnergyCalibration((0.0, 0.4))
    resolution: Resolution = ResolutionModel(a=0.488, b=0.002636)  # ~0.9 keV @ 122, 2.0 @ 1332
    eff_ref: float = 0.01  # absolute full-energy-peak efficiency at 661.657 keV
    lld_keV: float = 15.0
    actual_calibration: EnergyCalibration | None = None
    # peak-to-total ratio (full-energy peak / all events from that line) vs energy: (keV, ratio) pairs
    peak_to_total: tuple[tuple[float, float], ...] = ((60.0, 0.9), (200.0, 0.5), (600.0, 0.25), (1500.0, 0.18))

    @property
    def true_calibration(self) -> EnergyCalibration:
        return self.actual_calibration or self.calibration

    def efficiency(self, energy):
        e = np.asarray(energy, dtype=float)
        return self.eff_ref * (e / CS137_KEV) ** -0.95 / (1.0 + (40.0 / e) ** 4)


def hpge_detector() -> DetectorConfig:
    """Coaxial HPGe: 8192 channels at 0.4 keV, FWHM ~2.0 keV at 1332 keV."""
    return DetectorConfig()


def nai_detector() -> DetectorConfig:
    """NaI(Tl) scintillator: 1024 channels at 3 keV, FWHM ~ 7 % at 662 keV (k E^0.55), larger
    peak-to-total than germanium and a higher low-energy threshold."""
    return DetectorConfig(
        n_channels=1024,
        calibration=EnergyCalibration((0.0, 3.0)),
        resolution=PowerLawResolution(k=1.3, p=0.55),
        eff_ref=0.01,
        lld_keV=30.0,
        peak_to_total=((60.0, 0.95), (200.0, 0.8), (600.0, 0.55), (1500.0, 0.35)),
    )


@dataclass(frozen=True)
class BackgroundModel:
    lines: tuple[tuple[float, str, float], ...] = BACKGROUND_LINES  # (keV, origin, net cps)
    continuum_rate: float = 1.2  # cps, exponential in energy
    continuum_scale_keV: float = 700.0
    noise_rate: float = 0.0  # cps of electronic noise, exponential below ~noise_scale
    noise_scale_keV: float = 30.0


def nai_background() -> BackgroundModel:
    """Ambient background of a large scintillator: ~8x the germanium line rates and continuum."""
    return BackgroundModel(
        lines=tuple((e, n, 8.0 * r) for e, n, r in BACKGROUND_LINES),
        continuum_rate=9.0, continuum_scale_keV=800.0, noise_scale_keV=60.0,
    )


@dataclass(frozen=True)
class Scenario:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    sources: Mapping[str, float] = field(default_factory=dict)  # nuclide -> activity (Bq)
    background: BackgroundModel = field(default_factory=BackgroundModel)
    live_time: float = 600.0
    dead_time_fraction: float = 0.0
    extra_lines: tuple[tuple[float, float], ...] = ()  # (keV, net cps) on the true energy scale
    channel_extra: tuple[tuple[int, float], ...] = ()  # (channel, extra cps)
    dead_channels: tuple[int, ...] = ()
    faults: tuple[str, ...] = ()  # labels of injected faults (ground truth)


QA_SOURCE: dict[str, float] = {"Ba-133": 2000.0, "Cs-137": 2000.0, "Co-60": 1000.0}
QA_SOURCE_NAI: dict[str, float] = {"Am-241": 500.0, "Cs-137": 2000.0, "Co-60": 1000.0}


def source_scenario(live_time: float = 600.0, detector: str = "hpge") -> Scenario:
    """Routine check-source measurement: Ba-133 + Cs-137 + Co-60 (HPGe) or Am-241 + Cs-137 + Co-60 (NaI)."""
    if detector == "nai":
        return Scenario(nai_detector(), dict(QA_SOURCE_NAI), nai_background(), live_time)
    if detector == "hpge":
        return Scenario(sources=dict(QA_SOURCE), live_time=live_time)
    raise ValueError(f"unknown detector {detector!r}")


def background_scenario(live_time: float = 3600.0, detector: str = "hpge") -> Scenario:
    """Background measurement, no sources in the cave."""
    sc = source_scenario(live_time, detector)
    return replace(sc, sources={})


def _peak_to_total(det: DetectorConfig, energy: float) -> float:
    e, r = zip(*det.peak_to_total, strict=True)
    return float(np.interp(energy, e, r))


def expected_counts(sc: Scenario) -> np.ndarray:
    """Expected counts per channel for the (possibly faulted) scenario."""
    det = sc.detector
    cal = det.true_calibration
    n = det.n_channels
    ch = np.arange(n, dtype=float)
    e_edges = cal(np.arange(n + 1) - 0.5)
    e_mid = cal(ch)
    de = np.diff(e_edges)
    rate = np.zeros(n)  # counts per second per channel

    def add_peak(energy: float, cps: float, shelf: bool) -> None:
        sig = float(det.resolution.sigma(energy))
        rate[:] += cps * np.diff(ndtr((e_edges - energy) / sig))
        if shelf:
            pt = _peak_to_total(det, energy)
            e_c = energy / (1.0 + ME_C2_KEV / (2.0 * energy))  # Compton edge
            density = cps * (1.0 / pt - 1.0) / e_c  # counts / s / keV
            rate[:] += density * 0.5 * erfc((e_mid - e_c) / (np.sqrt(2.0) * sig)) * de

    for nuclide, activity in sc.sources.items():
        for line in SOURCES[nuclide]:
            add_peak(line.energy, activity * line.intensity * float(det.efficiency(line.energy)), True)
    for energy, _, cps in sc.background.lines:
        add_peak(energy, cps, False)
    for energy, cps in sc.extra_lines:
        add_peak(energy, cps, False)

    bg = sc.background
    rate += bg.continuum_rate * np.exp(-e_mid / bg.continuum_scale_keV) / bg.continuum_scale_keV * de
    if bg.noise_rate > 0:
        rate += bg.noise_rate * np.exp(-e_mid / bg.noise_scale_keV) / bg.noise_scale_keV * de

    rate[e_mid < det.lld_keV] = 0.0
    for channel, extra in sc.channel_extra:
        rate[channel] += extra
    for channel in sc.dead_channels:
        rate[channel] = 0.0
    return rate * sc.live_time


def simulate(sc: Scenario, rng: np.random.Generator | int | None = None) -> Spectrum:
    """Draw one Poisson realisation of ``sc``. The spectrum carries the *nominal* calibration."""
    rng = np.random.default_rng(rng)
    counts = rng.poisson(expected_counts(sc))
    real_time = sc.live_time / (1.0 - sc.dead_time_fraction)
    return Spectrum(
        counts=counts,
        live_time=sc.live_time,
        real_time=real_time,
        calibration=sc.detector.calibration,
        meta={"simulated": True, "faults": list(sc.faults)},
    )
