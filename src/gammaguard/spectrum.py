"""The `Spectrum` container: histogram of counts per channel plus acquisition metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import EnergyCalibration


@dataclass
class Spectrum:
    """A pulse-height spectrum.

    ``calibration`` is the *nominal* calibration currently configured in the acquisition
    system. Whether it still describes the detector is exactly what GammaGuard checks.
    """

    counts: np.ndarray
    live_time: float
    real_time: float | None = None
    calibration: EnergyCalibration | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=np.int64)
        if self.counts.ndim != 1 or self.counts.size == 0:
            raise ValueError("counts must be a non-empty 1-D array")
        if (self.counts < 0).any():
            raise ValueError("counts must be non-negative")
        if self.live_time <= 0:
            raise ValueError("live_time must be positive")

    @property
    def n_channels(self) -> int:
        return int(self.counts.size)

    @property
    def total_counts(self) -> int:
        return int(self.counts.sum())

    @property
    def dead_time_fraction(self) -> float | None:
        if self.real_time is None or self.real_time <= 0:
            return None
        return float(1.0 - self.live_time / self.real_time)

    def channels(self) -> np.ndarray:
        return np.arange(self.n_channels, dtype=float)

    def energies(self, calibration: EnergyCalibration | None = None) -> np.ndarray:
        cal = calibration or self.calibration
        if cal is None:
            raise ValueError("spectrum has no energy calibration")
        return cal(self.channels())

    def band_counts(self, lo_keV: float, hi_keV: float, calibration=None) -> int:
        """Counts in channels whose centre energy lies in ``[lo, hi)``."""
        e = self.energies(calibration)
        return int(self.counts[(e >= lo_keV) & (e < hi_keV)].sum())

    def with_calibration(self, calibration: EnergyCalibration) -> Spectrum:
        return Spectrum(self.counts, self.live_time, self.real_time, calibration, dict(self.meta))
