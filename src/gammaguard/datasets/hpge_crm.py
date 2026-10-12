"""Loader for the HPGe certified-reference-material spectra (Huynh & Truong, Zenodo 23093533).

Counts are on a common 0.5 keV grid (20-2900 keV) after the authors refitted every spectrum's
energy calibration on known lines. Counts are fractional (rebinned); we round them to integers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..calibration import EnergyCalibration
from ..spectrum import Spectrum
from .fetch import fetch

GRID = EnergyCalibration((20.25, 0.5))  # channel centres of the 0.5 keV grid starting at 20 keV


@dataclass
class HpgeCrm:
    counts: np.ndarray  # (n, 5760) float32
    meta: list[dict]
    bkg_counts: np.ndarray
    bkg_meta: list[dict]

    def indices(self, detector: str = "GMX35P4-70", material: str | None = None) -> list[int]:
        return [i for i, m in enumerate(self.meta)
                if detector in m["detector"] and (material is None or m["material"] == material)]

    def spectrum(self, i: int) -> Spectrum:
        m = self.meta[i]
        return Spectrum(np.rint(self.counts[i]).astype(np.int64), live_time=float(m["lt_s"]),
                        real_time=float(m["rt_s"]), calibration=GRID, meta=dict(m))

    def background(self, j: int) -> Spectrum:
        m = self.bkg_meta[j]
        lt = float(m["lt_s"])
        return Spectrum(np.rint(self.bkg_counts[j]).astype(np.int64), live_time=lt, real_time=lt,
                        calibration=GRID, meta=dict(m))


def load_crm(root: Path | None = None) -> HpgeCrm:
    d = fetch("hpge_crm", root, quiet=True) / "Pub6_dataset_v1.0"
    z = np.load(d / "pub6_hpge_crm_v1.npz", allow_pickle=False)  # no pickles: data only
    return HpgeCrm(z["counts"], json.loads(str(z["meta_json"])), z["bkg_counts"], json.loads(str(z["bkg_meta_json"])))
