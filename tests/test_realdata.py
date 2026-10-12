"""Tests on the real open datasets. They need the data in the local cache (`gammaguard fetch-data all`,
or GAMMAGUARD_DATA pointing at it) and are skipped otherwise, so the suite runs offline."""

import numpy as np
import pytest

from gammaguard import Baseline, Status, run_qa
from gammaguard.datasets.fetch import cache_dir

pytestmark = pytest.mark.realdata


def _have(key):
    return (cache_dir() / key / "extracted" / ".complete").exists()


@pytest.fixture(scope="module")
def crm():
    if not _have("hpge_crm"):
        pytest.skip("HPGe dataset not in the local cache (run: gammaguard fetch-data hpge_crm)")
    from gammaguard.datasets import load_crm
    return load_crm()


@pytest.fixture(scope="module")
def rll():
    if not _have("eth_rll"):
        pytest.skip("NaI dataset not in the local cache (run: gammaguard fetch-data eth_rll)")
    from gammaguard.datasets import load_rll
    return load_rll


def _mixed(rll, crystal):
    from gammaguard.datasets.eth_rll import RLL_NOMINAL, effective_time
    from gammaguard.spectrum import Spectrum
    a, b = rll("Cs137", "gr", crystal), rll("Co60", "gr", crystal)
    ta, tb = a.live_time, b.live_time
    rate = a.counts / ta + b.counts / tb
    t = effective_time(rate)
    return Spectrum(np.rint(rate * t).astype(np.int64), t, t, RLL_NOMINAL)


def test_nai_all_reference_lines_found_and_resolution_is_plausible(rll):
    base = Baseline.build(_mixed(rll, 3), profile="nai", reference_energies=(661.657, 1173.228, 1332.492))
    assert len(base.lines) == 3
    cs = base.lines[0]
    assert 0.07 < cs.fwhm_keV / cs.energy < 0.10  # large airborne crystal: ~8 % at 662 keV
    assert base.calibration.coeffs[1] == pytest.approx(3.0, rel=0.03)


def test_nai_a_crystal_passes_against_itself_and_differences_are_reported(rll):
    base = Baseline.build(_mixed(rll, 3), profile="nai", reference_energies=(661.657, 1173.228, 1332.492))
    assert run_qa(base, _mixed(rll, 3)).status == Status.PASS
    other = run_qa(base, _mixed(rll, 2))
    assert other.check("resolution").metrics["fwhm_ratio_mean"] < 0.95  # crystal 2 is clearly narrower than crystal 3
    assert other.status >= Status.WARN


def test_hpge_natural_lines_are_found_and_reading_errors_are_small(crm):
    from gammaguard.datasets.hpge_crm import GRID
    from gammaguard.qa.measure import measure_lines
    shifts, found = [], 0
    for i in crm.indices(material="RGU-1")[:10]:
        spec = crm.spectrum(i)
        fits = measure_lines(spec.counts, GRID, [351.93, 609.31, 1120.29, 1764.49], lambda e: 2.0, search_fwhm=3.0)
        found += sum(f is not None for f in fits)
        shifts += [GRID(f.centroid) - e for f, e in zip(fits, [351.93, 609.31, 1120.29, 1764.49], strict=True) if f is not None]
    assert found == 40
    assert np.std(shifts) < 0.25 and np.max(np.abs(shifts)) < 0.6


def test_hpge_same_material_spectra_mostly_pass_and_other_material_is_caught(crm):
    u = [i for i in crm.indices(material="RGU-1") if crm.meta[i]["tier"] == "gold"]
    ref = max(u, key=lambda i: crm.counts[i].sum())
    base = Baseline.build(crm.spectrum(ref), profile="hpge", reference_energies=[351.93, 609.31, 1120.29, 1764.49])
    tests = [i for i in crm.indices(material="RGU-1") if i != ref and crm.counts[i].sum() <= crm.counts[ref].sum()][:20]
    passed = sum(run_qa(base, crm.spectrum(i)).status == Status.PASS for i in tests)
    assert passed >= 0.6 * len(tests)
    th = run_qa(base, crm.spectrum(crm.indices(material="RGTh-1")[0]))
    assert th.status == Status.FAIL and th.check("unexpected_peaks").status == Status.WARN
    assert th.check("channel_anomalies").metrics["n_anomalous"] <= 20  # was >3000 before the strength-aware peak mask
