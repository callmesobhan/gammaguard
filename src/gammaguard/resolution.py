"""Energy-resolution models: FWHM as a function of energy (keV).

* `ResolutionModel` — ``FWHM(E) = sqrt(a + b*E + c*E^2)``, the usual form for germanium.
* `PowerLawResolution` — ``FWHM(E) = k * E**p``, a common empirical form for scintillators
  (p ~ 0.5-0.6 for NaI(Tl)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548...


@dataclass(frozen=True)
class ResolutionModel:
    """``a``: electronic noise, ``b``: Fano / statistical term, ``c``: incomplete collection."""

    a: float
    b: float
    c: float = 0.0

    def fwhm(self, energy):
        e = np.asarray(energy, dtype=float)
        return np.sqrt(np.maximum(self.a + self.b * e + self.c * e * e, 1e-12))

    def sigma(self, energy):
        return self.fwhm(energy) / FWHM_PER_SIGMA

    def scaled(self, factor: float) -> ResolutionModel:
        """Model whose FWHM is ``factor`` times larger at every energy."""
        f2 = factor * factor
        return ResolutionModel(self.a * f2, self.b * f2, self.c * f2)

    def to_dict(self) -> dict:
        return {"kind": "quadrature", "a": self.a, "b": self.b, "c": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> ResolutionModel:
        return cls(d["a"], d["b"], d.get("c", 0.0))


@dataclass(frozen=True)
class PowerLawResolution:
    """``FWHM = k * E**p`` with ``E`` in keV."""

    k: float
    p: float = 0.5

    def fwhm(self, energy):
        e = np.maximum(np.asarray(energy, dtype=float), 1e-6)
        return self.k * e**self.p

    def sigma(self, energy):
        return self.fwhm(energy) / FWHM_PER_SIGMA

    def scaled(self, factor: float) -> PowerLawResolution:
        return PowerLawResolution(self.k * factor, self.p)

    def to_dict(self) -> dict:
        return {"kind": "power", "k": self.k, "p": self.p}

    @classmethod
    def from_dict(cls, d: dict) -> PowerLawResolution:
        return cls(d["k"], d["p"])


Resolution = ResolutionModel | PowerLawResolution


def resolution_from_dict(d: dict) -> Resolution:
    """Inverse of ``to_dict`` (files written before ``kind`` existed are quadrature models)."""
    return PowerLawResolution.from_dict(d) if d.get("kind") == "power" else ResolutionModel.from_dict(d)


def fit_resolution(energies, fwhm, fwhm_err=None, with_c: bool = False) -> ResolutionModel:
    """Fit FWHM^2 = a + bE (+ cE^2) with non-negative coefficients (weighted LSQ)."""
    e = np.asarray(energies, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    err = np.ones_like(f) if fwhm_err is None else np.asarray(fwhm_err, dtype=float)
    w = 1.0 / np.maximum(2.0 * f * err, 1e-12)  # sigma(FWHM^2) = 2 F sigma(F)
    cols = [np.ones_like(e), e] + ([e * e] if with_c else [])
    A = np.stack(cols, axis=1) * w[:, None]
    res = lsq_linear(A, f * f * w, bounds=(0.0, np.inf))
    a, b, *rest = res.x
    return ResolutionModel(float(a), float(b), float(rest[0]) if rest else 0.0)


def fit_power_law(energies, fwhm, fwhm_err=None, fixed_p: float | None = None) -> PowerLawResolution:
    """Weighted log-log fit of ``FWHM = k E^p``. With ``fixed_p`` only ``k`` is fitted
    (the only option with a single line)."""
    e = np.log(np.asarray(energies, dtype=float))
    f = np.asarray(fwhm, dtype=float)
    err = np.ones_like(f) if fwhm_err is None else np.asarray(fwhm_err, dtype=float)
    y, sig = np.log(f), np.maximum(err / f, 1e-9)  # relative error = error of the log
    if fixed_p is not None or len(e) < 2:
        p = 0.5 if fixed_p is None else fixed_p
        w = 1.0 / sig**2
        return PowerLawResolution(float(np.exp(np.sum(w * (y - p * e)) / np.sum(w))), p)
    A = np.stack([np.ones_like(e), e], axis=1) / sig[:, None]
    (ln_k, p), *_ = np.linalg.lstsq(A, y / sig, rcond=None)
    return PowerLawResolution(float(np.exp(ln_k)), float(p))
