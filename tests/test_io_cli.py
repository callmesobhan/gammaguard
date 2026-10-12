import json

import pytest

from gammaguard.cli import main
from gammaguard.io import read_spectrum, write_spectrum
from gammaguard.simulate import GainDrift, inject, simulate, source_scenario


def test_csv_roundtrip_preserves_everything(tmp_path):
    spec = simulate(inject(source_scenario(), GainDrift(1e-4)), 1)
    path = tmp_path / "s.csv"
    write_spectrum(spec, path)
    back = read_spectrum(path)
    assert (back.counts == spec.counts).all()
    assert back.live_time == spec.live_time and back.real_time == spec.real_time
    assert back.calibration == spec.calibration
    assert "faults (ground truth)" in path.read_text()


def test_json_roundtrip(tmp_path):
    spec = simulate(source_scenario(), 1)
    path = tmp_path / "s.json"
    write_spectrum(spec, path)
    back = read_spectrum(path)
    assert (back.counts == spec.counts).all() and back.calibration == spec.calibration


def test_csv_without_header_rows_or_live_time(tmp_path):
    path = tmp_path / "bare.csv"
    path.write_text("0,1\n1,2\n")
    with pytest.raises(ValueError, match="live_time"):
        read_spectrum(path)
    path.write_text("# live_time: 10\n5\n6\n7\n")  # single-column data is accepted
    assert read_spectrum(path).counts.tolist() == [5, 6, 7]


def test_cli_workflow_and_exit_codes(tmp_path, capsys):
    p = lambda name: str(tmp_path / name)  # noqa: E731
    assert main(["simulate", "--kind", "source", "--live-time", "1800", "--seed", "1", "--out", p("src.csv")]) == 0
    assert main(["simulate", "--kind", "background", "--live-time", "14400", "--seed", "2", "--out", p("bkg.csv")]) == 0
    assert main(["baseline", "--source", p("src.csv"), "--background", p("bkg.csv"), "--out", p("base.json")]) == 0

    assert main(["simulate", "--seed", "3", "--out", p("ok.csv")]) == 0
    assert main(["check", "--baseline", p("base.json"), p("ok.csv")]) == 0

    assert main(["simulate", "--seed", "4", "--fault", "gain_drift:0.0003", "--out", p("warn.csv")]) == 0
    assert main(["check", "--baseline", p("base.json"), p("warn.csv")]) == 1

    assert main(["simulate", "--seed", "5", "--fault", "gain_drift:0.001", "--out", p("fail.csv")]) == 0
    args = ["check", "--baseline", p("base.json"), "--format", "json", "--output", p("r.json"), p("fail.csv")]
    assert main(args) == 2
    assert json.loads((tmp_path / "r.json").read_text())["status"] == "FAIL"

    assert main(["simulate", "--kind", "background", "--seed", "6", "--fault", "background_scale:2.5",
                 "--out", p("bkg_bad.csv")]) == 0
    assert main(["check", "--baseline", p("base.json"), "--mode", "background", p("bkg_bad.csv")]) == 2
    capsys.readouterr()


def test_cli_reports_bad_input_with_exit_3(tmp_path, capsys):
    assert main(["check", "--baseline", str(tmp_path / "nope.json"), str(tmp_path / "nope.csv")]) == 3
    assert "error" in capsys.readouterr().err
    assert main(["simulate", "--fault", "bogus:1", "--out", str(tmp_path / "x.csv")]) == 3


def test_cli_demo_runs(capsys):
    assert main(["demo", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "healthy detector" in out and "FAIL" in out
