"""Energy calibration: channel -> energy (keV) as a low-order polynomial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as P


@dataclass(frozen=True)
class EnergyCalibration:
    """``E(ch) = c0 + c1*ch + c2*ch^2 + ...`` with ``coeffs = (c0, c1, ...)`` in keV."""

    coeffs: tuple[float, ...]

    def __post_init__(self) -> None:
        coeffs = tuple(float(c) for c in self.coeffs)
        if len(coeffs) < 2:
            raise ValueError("a calibration needs at least offset and gain")
        if coeffs[1] <= 0:
            raise ValueError("gain (c1) must be positive")
        object.__setattr__(self, "coeffs", coeffs)

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def __call__(self, channel):
        return P.polyval(np.asarray(channel, dtype=float), self.coeffs)

    def slope(self, channel):
        """dE/dch at ``channel`` (keV per channel)."""
        return P.polyval(np.asarray(channel, dtype=float), P.polyder(self.coeffs))

    def channel(self, energy, iterations: int = 6):
        """Inverse mapping (Newton iteration from the linear estimate)."""
        e = np.asarray(energy, dtype=float)
        ch = (e - self.coeffs[0]) / self.coeffs[1]
        for _ in range(iterations if self.order > 1 else 0):
            ch = ch - (self(ch) - e) / self.slope(ch)
        return ch

    def corrected(self, offset_keV: float, gain_fraction: float) -> EnergyCalibration:
        """Calibration that undoes a measured reading error ``shift = offset + gain*E``.

        If this calibration reads ``E_read = E_true + offset + gain * E_true`` then the
        returned calibration reads ``E_true``.
        """
        scale = 1.0 / (1.0 + gain_fraction)
        coeffs = [c * scale for c in self.coeffs]
        coeffs[0] = (self.coeffs[0] - offset_keV) * scale
        return EnergyCalibration(tuple(coeffs))

    def to_dict(self) -> dict:
        return {"coeffs": list(self.coeffs)}

    @classmethod
    def from_dict(cls, d: dict) -> EnergyCalibration:
        return cls(tuple(d["coeffs"]))


def weighted_polyfit(x, y, sigma=None, order: int = 1):
    """Weighted least-squares polynomial fit. Returns ``(coeffs, covariance)``.

    ``coeffs`` is low-order first. ``sigma`` are 1-sigma errors on ``y``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.ones_like(y) if sigma is None else np.asarray(sigma, dtype=float)
    if len(x) < order + 1:
        raise ValueError(f"need at least {order + 1} points for an order-{order} fit")
    A = np.vander(x, order + 1, increasing=True) / s[:, None]
    b = y / s
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    return coeffs, cov


def fit_calibration(channels, energies, channel_sigma=None, order: int = 1) -> EnergyCalibration:
    """Fit ``E(ch)`` from peak centroids (channels) and their nominal energies (keV)."""
    channels = np.asarray(channels, dtype=float)
    energies = np.asarray(energies, dtype=float)
    sigma = None
    if channel_sigma is not None:
        first, _ = weighted_polyfit(channels, energies, None, order)
        slope = P.polyval(channels, P.polyder(first))
        sigma = np.maximum(slope * np.asarray(channel_sigma, dtype=float), 1e-9)
    coeffs, _ = weighted_polyfit(channels, energies, sigma, order)
    return EnergyCalibration(tuple(coeffs))
