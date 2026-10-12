"""Fault injection: each fault transforms a `Scenario`, leaving ground-truth labels behind."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import ClassVar

from ..calibration import EnergyCalibration
from ..isotopes import RADON_PROGENY
from .generator import Scenario


class Fault:
    name: ClassVar[str]

    def apply(self, sc: Scenario) -> Scenario:  # pragma: no cover - interface
        raise NotImplementedError

    def label(self) -> str:
        args = ",".join(f"{f.name}={getattr(self, f.name):g}" for f in fields(self))  # type: ignore[arg-type]
        return f"{self.name}({args})"


def _with_calibration(sc: Scenario, coeffs: tuple[float, ...]) -> Scenario:
    det = replace(sc.detector, actual_calibration=EnergyCalibration(coeffs))
    return replace(sc, detector=det)


@dataclass(frozen=True)
class GainDrift(Fault):
    """Peaks move to channel*(1+fraction); the nominal calibration reads energies too high."""

    name: ClassVar[str] = "gain_drift"
    fraction: float

    def apply(self, sc: Scenario) -> Scenario:
        cal = sc.detector.true_calibration
        k = 1.0 / (1.0 + self.fraction)
        return _with_calibration(sc, tuple(c * k**i for i, c in enumerate(cal.coeffs)))


@dataclass(frozen=True)
class ZeroShift(Fault):
    """Every line is read ``keV`` too high by the nominal calibration."""

    name: ClassVar[str] = "zero_shift"
    keV: float

    def apply(self, sc: Scenario) -> Scenario:
        c = list(sc.detector.true_calibration.coeffs)
        c[0] -= self.keV
        return _with_calibration(sc, tuple(c))


@dataclass(frozen=True)
class ResolutionLoss(Fault):
    """FWHM is ``factor`` times the nominal value at every energy."""

    name: ClassVar[str] = "resolution_loss"
    factor: float

    def apply(self, sc: Scenario) -> Scenario:
        det = replace(sc.detector, resolution=sc.detector.resolution.scaled(self.factor))
        return replace(sc, detector=det)


@dataclass(frozen=True)
class BackgroundScale(Fault):
    """All ambient lines and the continuum scale by ``factor``."""

    name: ClassVar[str] = "background_scale"
    factor: float

    def apply(self, sc: Scenario) -> Scenario:
        bg = sc.background
        lines = tuple((e, n, r * self.factor) for e, n, r in bg.lines)
        return replace(sc, background=replace(bg, lines=lines, continuum_rate=bg.continuum_rate * self.factor))


@dataclass(frozen=True)
class RadonSurge(Fault):
    """Only radon-progeny lines (Pb-214, Bi-214) scale by ``factor``."""

    name: ClassVar[str] = "radon_surge"
    factor: float

    def apply(self, sc: Scenario) -> Scenario:
        bg = sc.background
        lines = tuple((e, n, r * self.factor if n in RADON_PROGENY else r) for e, n, r in bg.lines)
        return replace(sc, background=replace(bg, lines=lines))


@dataclass(frozen=True)
class NoiseBurst(Fault):
    """Electronic noise: extra low-energy counts at ``rate_cps``."""

    name: ClassVar[str] = "noise_burst"
    rate_cps: float

    def apply(self, sc: Scenario) -> Scenario:
        return replace(sc, background=replace(sc.background, noise_rate=self.rate_cps))


@dataclass(frozen=True)
class SpuriousPeak(Fault):
    """An unexpected line (contamination, interference) at ``energy_keV`` with ``rate_cps``."""

    name: ClassVar[str] = "spurious_peak"
    energy_keV: float
    rate_cps: float

    def apply(self, sc: Scenario) -> Scenario:
        return replace(sc, extra_lines=(*sc.extra_lines, (self.energy_keV, self.rate_cps)))


@dataclass(frozen=True)
class HotChannel(Fault):
    """A single channel receives ``rate_cps`` extra counts (ADC / digitiser fault)."""

    name: ClassVar[str] = "hot_channel"
    channel: int
    rate_cps: float

    def apply(self, sc: Scenario) -> Scenario:
        return replace(sc, channel_extra=(*sc.channel_extra, (int(self.channel), self.rate_cps)))


@dataclass(frozen=True)
class DeadChannel(Fault):
    """A single channel never records counts."""

    name: ClassVar[str] = "dead_channel"
    channel: int

    def apply(self, sc: Scenario) -> Scenario:
        return replace(sc, dead_channels=(*sc.dead_channels, int(self.channel)))


@dataclass(frozen=True)
class DeadTime(Fault):
    """Fraction of real time the system is busy (live time is shorter than real time)."""

    name: ClassVar[str] = "dead_time"
    fraction: float

    def apply(self, sc: Scenario) -> Scenario:
        return replace(sc, dead_time_fraction=self.fraction)


FAULTS: dict[str, type[Fault]] = {
    cls.name: cls
    for cls in (
        GainDrift, ZeroShift, ResolutionLoss, BackgroundScale, RadonSurge,
        NoiseBurst, SpuriousPeak, HotChannel, DeadChannel, DeadTime,
    )
}


def inject(sc: Scenario, *faults: Fault) -> Scenario:
    """Apply faults in order and record their labels on the scenario."""
    for fault in faults:
        sc = replace(fault.apply(sc), faults=(*sc.faults, fault.label()))
    return sc


def parse_fault(spec: str) -> Fault:
    """Parse ``name:arg1:arg2`` (numeric args), e.g. ``spurious_peak:800:0.05``."""
    name, *args = spec.split(":")
    if name not in FAULTS:
        raise ValueError(f"unknown fault {name!r}; choose from {sorted(FAULTS)}")
    cls = FAULTS[name]
    n_expected = len(fields(cls))  # type: ignore[arg-type]
    if len(args) != n_expected:
        raise ValueError(f"fault {name!r} takes {n_expected} argument(s), got {len(args)}")
    return cls(*(float(a) for a in args))  # type: ignore[arg-type]
