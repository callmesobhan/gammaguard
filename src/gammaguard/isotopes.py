"""Small gamma-line library used by the simulator and for labelling anomalies.

Energies (keV) and emission probabilities (photons per decay) are transcribed from memory of
standard evaluated data and are **illustrative**. Verify against NNDC / LNHB / DDEP before
using the library for anything beyond simulation and hinting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GammaLine:
    energy: float  # keV
    intensity: float  # photons per decay
    nuclide: str

    @property
    def label(self) -> str:
        return f"{self.nuclide} {self.energy:.1f}"


def _lines(nuclide: str, rows: list[tuple[float, float]]) -> tuple[GammaLine, ...]:
    return tuple(GammaLine(e, i, nuclide) for e, i in rows)


SOURCES: dict[str, tuple[GammaLine, ...]] = {
    "Am-241": _lines("Am-241", [(59.54, 0.359)]),
    "Ba-133": _lines(
        "Ba-133",
        [(80.998, 0.341), (276.399, 0.0716), (302.851, 0.183), (356.013, 0.6205), (383.849, 0.0894)],
    ),
    "Cs-137": _lines("Cs-137", [(661.657, 0.851)]),
    "Co-60": _lines("Co-60", [(1173.228, 0.9985), (1332.492, 0.9998)]),
    "Na-22": _lines("Na-22", [(1274.537, 0.9994)]),
    "Eu-152": _lines(
        "Eu-152",
        [
            (121.782, 0.2853), (244.697, 0.0755), (344.279, 0.2659), (778.904, 0.1293),
            (964.057, 0.1451), (1085.837, 0.1011), (1112.076, 0.1367), (1408.013, 0.2087),
        ],
    ),
}

# Ambient background lines: (energy keV, nuclide / origin, illustrative net rate in cps for a
# mid-size HPGe detector in an unshielded room). Rates are placeholders for simulation only.
BACKGROUND_LINES: tuple[tuple[float, str, float], ...] = (
    (295.22, "Pb-214", 0.010),
    (351.93, "Pb-214", 0.020),
    (511.00, "annihilation", 0.015),
    (609.31, "Bi-214", 0.020),
    (911.20, "Ac-228", 0.010),
    (1120.29, "Bi-214", 0.006),
    (1460.82, "K-40", 0.030),
    (1764.49, "Bi-214", 0.006),
    (2614.51, "Tl-208", 0.010),
)

RADON_PROGENY = ("Pb-214", "Bi-214")

# Natural-chain lines (energy keV, origin) seen in environmental samples and reference materials.
# Used for labelling and as reference lines when no check source is available.
NATURAL_LINES: tuple[tuple[float, str], ...] = (
    (186.21, "Ra-226"), (238.63, "Pb-212"), (295.22, "Pb-214"), (338.32, "Ac-228"), (351.93, "Pb-214"),
    (583.19, "Tl-208"), (609.31, "Bi-214"), (727.33, "Bi-212"), (768.36, "Bi-214"), (911.20, "Ac-228"),
    (934.06, "Bi-214"), (968.97, "Ac-228"), (1120.29, "Bi-214"), (1238.11, "Bi-214"), (1377.67, "Bi-214"),
    (1460.82, "K-40"), (1729.60, "Bi-214"), (1764.49, "Bi-214"), (2204.21, "Bi-214"), (2614.51, "Tl-208"),
)

# Default calibration lines for a mixed Ba-133 + Cs-137 + Co-60 check source. Isolated lines
# only (Ba-133 80.998 keV is excluded: it sits 1.4 keV from the 79.6 keV line).
QA_REFERENCE_ENERGIES: tuple[float, ...] = (302.851, 356.013, 661.657, 1173.228, 1332.492)


# Scintillator check source (Am-241 + Cs-137 + Co-60). 1173/1332 keV are blended in NaI(Tl) and are
# fitted together.
NAI_REFERENCE_ENERGIES: tuple[float, ...] = (59.54, 661.657, 1173.228, 1332.492)


def all_lines() -> list[GammaLine]:
    out: list[GammaLine] = [ln for lines in SOURCES.values() for ln in lines]
    out += [GammaLine(e, 0.0, n) for e, n, _ in BACKGROUND_LINES]
    out += [GammaLine(e, 0.0, n) for e, n in NATURAL_LINES]
    return out


def identify(energy: float, tolerance: float) -> list[GammaLine]:
    """Library lines within ``tolerance`` keV of ``energy``, nearest first."""
    hits = [ln for ln in all_lines() if abs(ln.energy - energy) <= tolerance]
    return sorted(hits, key=lambda ln: abs(ln.energy - energy))
