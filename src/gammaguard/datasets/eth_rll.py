"""Loader for the ETH/PSI NaI(Tl) airborne-spectrometer laboratory data (Breitenmoser et al. 2022).

Files are ``RLL_data_<source>_<gr|bg>.csv`` (semicolon separated, 1024 rows): channel, then count
RATES in cps for the sum of the four crystals and for each crystal. The spectrometer linearizes
spectra on board, so the values are not integer counts divided by a live time. To feed count-based
statistics we scale by an *effective counting time* estimated from the channel-to-channel scatter
(``effective_time``), which reproduces the observed noise level; it is an estimate, not the actual
acquisition time.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ..calibration import EnergyCalibration
from ..spectrum import Spectrum
from .fetch import fetch

SOURCES = ("Co57", "Co60", "Y88", "Cd109", "Ba133", "Cs137", "Eu152", "Knat", "Thnat", "Unat")
RLL_NOMINAL = EnergyCalibration((0.0, 3.0))  # ~3 keV/channel (the instrument covers ~30-3000 keV in 1024 ch)


def effective_time(rate: np.ndarray, lo: int = 600, hi: int = 1000) -> float:
    """Counting time (s) at which Poisson noise would match the observed second-difference scatter of
    a smooth stretch of the spectrum. For Poisson data, var(c_k - (c_{k-1}+c_{k+1})/2) = 1.5 r / T."""
    x = np.asarray(rate, dtype=float)[lo:hi]
    d = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    return float(1.5 * x[1:-1].sum() / np.sum(d**2))


def read_rate_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return lines[0].split(";"), np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=";")


def load_rll(source: str, kind: str = "gr", crystal: int | str = "sum", root: Path | None = None,
             time_s: float | None = None) -> Spectrum:
    """One spectrum. ``kind``: ``gr`` (source present) or ``bg``. ``crystal``: 1-4 or ``"sum"``.
    ``time_s`` overrides the estimated effective counting time."""
    if source not in SOURCES or kind not in ("gr", "bg"):
        raise ValueError(f"source must be one of {SOURCES} and kind 'gr' or 'bg'")
    col = 1 if crystal == "sum" else 1 + int(crystal)
    data_dir = fetch("eth_rll", root, quiet=True)
    _, arr = read_rate_csv(data_dir / f"RLL_data_{source}_{kind}.csv")
    rate = arr[:, col]
    t = time_s or effective_time(rate)
    counts = np.rint(rate * t).astype(np.int64)
    return Spectrum(counts, live_time=t, real_time=t, calibration=RLL_NOMINAL,
                    meta={"dataset": "eth_rll", "source": source, "kind": kind, "crystal": str(crystal),
                          "effective_time_s": t})
