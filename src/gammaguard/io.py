"""Spectrum file I/O.

CSV is the exchange format: ``# key: value`` header lines followed by ``channel,counts`` rows.
Recognised keys: ``live_time``, ``real_time``, ``calibration`` (comma-separated polynomial
coefficients, keV, low order first). Other keys are kept in ``Spectrum.meta``.

Adding readers for IAEA SPE, ANSI N42.42 and vendor formats is on the roadmap; each should
produce a `Spectrum` and nothing else, so the QA code never sees file formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import EnergyCalibration
from .spectrum import Spectrum


def write_spectrum(spectrum: Spectrum, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps({
            "counts": spectrum.counts.tolist(),
            "live_time": spectrum.live_time,
            "real_time": spectrum.real_time,
            "calibration": list(spectrum.calibration.coeffs) if spectrum.calibration else None,
            "meta": spectrum.meta,
        }))
        return
    lines = ["# gammaguard spectrum v1", f"# live_time: {spectrum.live_time!r}"]
    if spectrum.real_time is not None:
        lines.append(f"# real_time: {spectrum.real_time!r}")
    if spectrum.calibration is not None:
        lines.append("# calibration: " + ",".join(repr(c) for c in spectrum.calibration.coeffs))
    lines += [f"# {k}: {v}" for k, v in spectrum.meta.items() if k not in ("simulated", "faults")]
    if spectrum.meta.get("faults"):
        lines.append("# faults (ground truth): " + "; ".join(spectrum.meta["faults"]))
    lines.append("channel,counts")
    lines += [f"{i},{c}" for i, c in enumerate(spectrum.counts)]
    path.write_text("\n".join(lines) + "\n")


def read_spectrum(path: str | Path) -> Spectrum:
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        cal = EnergyCalibration(tuple(d["calibration"])) if d.get("calibration") else None
        return Spectrum(np.array(d["counts"]), d["live_time"], d.get("real_time"), cal, d.get("meta", {}))

    header: dict[str, str] = {}
    counts: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep:
                header[key.strip()] = value.strip()
            continue
        fields = line.split(",")
        if not fields[-1].lstrip("-").isdigit():
            continue  # column header
        counts.append(int(fields[-1]))
    if "live_time" not in header:
        raise ValueError(f"{path}: missing '# live_time: <seconds>' header line")
    cal = None
    if "calibration" in header:
        cal = EnergyCalibration(tuple(float(c) for c in header.pop("calibration").split(",")))
    live = float(header.pop("live_time"))
    real = float(header.pop("real_time")) if "real_time" in header else None
    return Spectrum(np.array(counts), live, real, cal, header)
