import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from gammaguard.datasets import DATASETS, fetch, load_crm, load_rll
from gammaguard.datasets.eth_rll import effective_time
from gammaguard.datasets.fetch import cache_dir
from gammaguard.datasets.registry import Dataset


def test_registry_is_complete_and_does_not_claim_a_false_licence():
    assert set(DATASETS) == {"eth_rll", "hpge_crm"}
    for ds in DATASETS.values():
        assert len(ds.md5) == 32 and ds.size_bytes > 0 and ds.url.startswith("https://") and ds.citation
    assert "Non-Commercial" in DATASETS["eth_rll"].license  # must not be mistaken for a permissively licensed dataset
    assert DATASETS["hpge_crm"].license == "CC BY 4.0"


def test_cache_dir_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GAMMAGUARD_DATA", str(tmp_path))
    assert cache_dir() == tmp_path


def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake(tmp_path, members, md5=None):
    blob = _zip_bytes(members)
    src = tmp_path / "src.zip"
    src.write_bytes(blob)
    return Dataset("fake", "t", "d", src.as_uri(), "fake.zip", md5 or hashlib.md5(blob).hexdigest(), len(blob), "x", "c", "l")


def test_fetch_downloads_verifies_and_extracts(tmp_path):
    ds = _fake(tmp_path, {"a.txt": "hello"})
    out = fetch(ds, tmp_path / "cache", quiet=True)
    assert (out / "a.txt").read_text() == "hello"
    assert fetch(ds, tmp_path / "cache", quiet=True) == out  # second call is a no-op


def test_fetch_rejects_checksum_mismatch(tmp_path):
    ds = _fake(tmp_path, {"a.txt": "hello"}, md5="0" * 32)
    with pytest.raises(RuntimeError, match="checksum"):
        fetch(ds, tmp_path / "cache", quiet=True)
    assert not (tmp_path / "cache" / "fake" / "fake.zip").exists()  # a bad download is not kept


def test_fetch_rejects_path_traversal(tmp_path):
    ds = _fake(tmp_path, {"../evil.txt": "x"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        fetch(ds, tmp_path / "cache", quiet=True)
    assert not (tmp_path / "cache" / "evil.txt").exists()


def test_effective_time_recovers_poisson_scale():
    rng = np.random.default_rng(0)
    shape = 0.3 + 0.0001 * np.arange(1024)
    for t in (500.0, 20000.0):
        rate = rng.poisson(shape * t) / t
        assert effective_time(rate) == pytest.approx(t, rel=0.2)


def test_rll_loader_on_a_synthetic_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMMAGUARD_DATA", str(tmp_path))
    d = tmp_path / "eth_rll" / "extracted"
    d.mkdir(parents=True)
    (d / ".complete").touch()
    rng = np.random.default_rng(1)
    body = ["Channel [-];Sum Crystal Count Rate [CPS];Crystal 1 Count Rate [CPS];Crystal 2 Count Rate [CPS];"
            "Crystal 3 Count Rate [CPS];Crystal 4 Count Rate [CPS]"]
    for ch in range(1024):
        r = rng.poisson((5 + 0.01 * ch) * 4000) / 4000
        body.append(f"{ch};{4 * r};{r};{r};{r};{r}")
    (d / "RLL_data_Cs137_gr.csv").write_text("﻿" + "\n".join(body))
    spec = load_rll("Cs137", "gr", crystal=2)
    assert spec.n_channels == 1024 and spec.total_counts > 0 and spec.meta["crystal"] == "2"
    assert spec.live_time == pytest.approx(4000, rel=0.3)
    with pytest.raises(ValueError):
        load_rll("Pu239", "gr")


def test_crm_loader_on_a_synthetic_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("GAMMAGUARD_DATA", str(tmp_path))
    d = tmp_path / "hpge_crm" / "extracted" / "Pub6_dataset_v1.0"
    d.mkdir(parents=True)
    (tmp_path / "hpge_crm" / "extracted" / ".complete").touch()
    meta = [{"ID": "S1", "detector": "GMX35P4-70 (ORTEC, HPGe)", "material": "RGU-1", "lt_s": 100.0, "rt_s": 101.0, "start": "2016-01-01"},
            {"ID": "S2", "detector": "GC3520 (Canberra, HPGe)", "material": "RGK-1", "lt_s": 50.0, "rt_s": 50.0, "start": "2016-01-02"}]
    np.savez(d / "pub6_hpge_crm_v1.npz", counts=np.full((2, 5760), 3.4, dtype=np.float32), meta_json=json.dumps(meta),
             bkg_counts=np.full((1, 5760), 1.0, dtype=np.float32), bkg_meta_json=json.dumps([{"lt_s": 10.0, "file": "b"}]))
    crm = load_crm()
    assert crm.indices() == [0] and crm.indices(material="RGK-1", detector="GC3520") == [1]
    s = crm.spectrum(0)
    assert s.counts.dtype == np.int64 and s.counts[0] == 3 and s.calibration(0) == pytest.approx(20.25)
    assert s.dead_time_fraction == pytest.approx(1 - 100 / 101)
    assert crm.background(0).live_time == 10.0
