"""Spectrum file I/O: CSV, JSON, IAEA/ORTEC SPE, ORTEC binary CHN, ANSI N42.42 XML, and Canberra CNF.

Every reader returns a plain `Spectrum` object so QA checks, hardware diagnostics, and trend
monitoring remain completely format-agnostic.
"""

from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .calibration import EnergyCalibration
from .spectrum import Spectrum

SUPPORTED_FORMATS = (".csv", ".json", ".spe", ".chn", ".n42", ".xml", ".cnf")


# --------------------------------------------------------------------------- IAEA / ORTEC .SPE
def read_spe(path: str | Path) -> Spectrum:
    """Read an IAEA / ORTEC Maestro ASCII ``.spe`` spectrum file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in lines:
        s = raw.strip()
        if s.startswith("$") and s.endswith(":"):
            current = s[1:-1].upper()
            sections[current] = []
        elif current is not None:
            sections[current].append(s)

    if "DATA" not in sections:
        raise ValueError(f"{path}: missing $DATA: section in SPE file")
    data_lines = [ln for ln in sections["DATA"] if ln]
    header_parts = data_lines[0].split()
    if len(header_parts) < 2:
        raise ValueError(f"{path}: malformed $DATA: channel range line")
    lo_ch, hi_ch = int(header_parts[0]), int(header_parts[1])
    n_expected = hi_ch - lo_ch + 1
    counts: list[int] = []
    for ln in data_lines[1:]:
        for tok in ln.split():
            if tok.lstrip("-").isdigit():
                counts.append(int(tok))
    if len(counts) < n_expected:
        raise ValueError(f"{path}: expected {n_expected} channels, found {len(counts)}")
    counts = counts[:n_expected]

    live_time = 0.0
    real_time: float | None = None
    if "MEAS_TIM" in sections:
        tim_tokens = " ".join(sections["MEAS_TIM"]).split()
        if len(tim_tokens) >= 1:
            live_time = float(tim_tokens[0])
        if len(tim_tokens) >= 2:
            real_time = float(tim_tokens[1])
    if live_time <= 0:
        raise ValueError(f"{path}: missing or non-positive $MEAS_TIM: live time")

    cal: EnergyCalibration | None = None
    if "ENER_FIT" in sections:
        fit_tokens = " ".join(sections["ENER_FIT"]).split()
        coeffs = [float(t) for t in fit_tokens[:3]]
        if len(coeffs) >= 2 and abs(coeffs[1]) > 0:
            cal = EnergyCalibration(tuple(coeffs))

    meta: dict[str, str] = {"format": "spe"}
    if "SPEC_ID" in sections and sections["SPEC_ID"]:
        meta["spec_id"] = " ".join(sections["SPEC_ID"]).strip()
    if "DATE_MEA" in sections and sections["DATE_MEA"]:
        meta["date_mea"] = " ".join(sections["DATE_MEA"]).strip()
    return Spectrum(np.array(counts, dtype=int), live_time, real_time, cal, meta)


def write_spe(spectrum: Spectrum, path: str | Path) -> None:
    """Write a Spectrum as an IAEA / ORTEC Maestro ASCII ``.spe`` file."""
    n = spectrum.n_channels
    rt = spectrum.real_time if spectrum.real_time is not None else spectrum.live_time
    out = [
        "$SPEC_ID:",
        str(spectrum.meta.get("spec_id", "GammaGuard spectrum")),
        "$MEAS_TIM:",
        f"{spectrum.live_time:.3f} {rt:.3f}",
        "$DATA:",
        f"0 {n - 1}",
    ]
    out.extend(str(int(c)) for c in spectrum.counts)
    if spectrum.calibration is not None:
        out.append("$ENER_FIT:")
        out.append(" ".join(f"{c:.8g}" for c in spectrum.calibration.coeffs))
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- ORTEC binary .CHN
def read_chn(path: str | Path) -> Spectrum:
    """Read an ORTEC binary integer ``.chn`` MCA spectrum file (32-byte header + uint32 channels)."""
    raw = Path(path).read_bytes()
    if len(raw) < 32:
        raise ValueError(f"{path}: file too small to be an ORTEC CHN spectrum")
    magic, _mca, _seg, _sec, real_ticks, live_ticks = struct.unpack_from("<hhh2sII", raw, 0)
    if magic != -1:
        raise ValueError(f"{path}: invalid ORTEC CHN magic {magic} (expected -1)")
    start_ch, n_ch = struct.unpack_from("<HH", raw, 28)
    data_end = 32 + 4 * n_ch
    if len(raw) < data_end:
        raise ValueError(f"{path}: truncated CHN channel payload ({len(raw)} < {data_end})")
    counts = np.frombuffer(raw, dtype="<u4", count=n_ch, offset=32).astype(int)
    live_time = live_ticks * 0.02  # ORTEC stores 20 ms ticks
    real_time = real_ticks * 0.02 if real_ticks > 0 else None

    cal: EnergyCalibration | None = None
    if len(raw) >= data_end + 16:
        footer_magic = struct.unpack_from("<h", raw, data_end)[0]
        if footer_magic in (-101, -102):
            c0, c1, c2 = struct.unpack_from("<fff", raw, data_end + 4)
            if np.isfinite([c0, c1, c2]).all() and abs(c1) > 0:
                coeffs = (float(c0), float(c1), float(c2)) if abs(c2) > 0 else (float(c0), float(c1))
                cal = EnergyCalibration(coeffs)
    return Spectrum(counts, live_time, real_time, cal, {"format": "chn", "start_channel": start_ch})


def write_chn(spectrum: Spectrum, path: str | Path) -> None:
    """Write a Spectrum as a standard ORTEC binary ``.chn`` file."""
    n = spectrum.n_channels
    rt = spectrum.real_time if spectrum.real_time is not None else spectrum.live_time
    header = struct.pack(
        "<hhh2sII8s4sHH",
        -1,
        0,
        0,
        b"00",
        int(round(rt / 0.02)),
        int(round(spectrum.live_time / 0.02)),
        b"01Jan26 ",
        b"0000",
        0,
        n,
    )
    payload = np.asarray(spectrum.counts, dtype="<u4").tobytes()
    footer = b""
    if spectrum.calibration is not None:
        coeffs = list(spectrum.calibration.coeffs) + [0.0, 0.0, 0.0]
        footer = struct.pack("<hhfff", -102, 0, float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    Path(path).write_bytes(header + payload + footer)


# --------------------------------------------------------------------------- ANSI N42.42 XML
def _parse_iso_duration(text: str) -> float:
    """Parse ISO-8601 duration (e.g. ``PT600S``, ``PT10M5.5S``, ``PT1H``) into seconds."""
    s = text.strip().upper()
    m = re.fullmatch(r"P(?:(\d+(?:\.\d+)?)D)?T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", s)
    if not m:
        return float(s)
    days, hours, mins, secs = (float(g) if g is not None else 0.0 for g in m.groups())
    return days * 86400.0 + hours * 3600.0 + mins * 60.0 + secs


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_n42(path: str | Path) -> Spectrum:
    """Read an ANSI/IEEE N42.42 XML gamma spectrum file (2006 / 2012 / 2020 schemas)."""
    root = ET.fromstring(Path(path).read_text(encoding="utf-8", errors="replace"))
    live_time: float | None = None
    real_time: float | None = None
    counts: list[int] | None = None
    coeffs: tuple[float, ...] | None = None

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        txt = (elem.text or "").strip()
        if not txt:
            continue
        if tag in ("LiveTimeDuration", "LiveTime") and live_time is None:
            live_time = _parse_iso_duration(txt)
        elif tag in ("RealTimeDuration", "RealTime") and real_time is None:
            real_time = _parse_iso_duration(txt)
        elif tag == "ChannelData" and counts is None:
            counts = [int(round(float(t))) for t in txt.split()]
        elif tag == "CoefficientValues" and coeffs is None:
            parsed = tuple(float(t) for t in txt.split())
            if len(parsed) >= 2 and abs(parsed[1]) > 0:
                coeffs = parsed

    if counts is None or len(counts) == 0:
        raise ValueError(f"{path}: no <ChannelData> found in ANSI N42.42 file")
    if live_time is None or live_time <= 0:
        raise ValueError(f"{path}: missing or non-positive <LiveTimeDuration> in ANSI N42.42 file")
    cal = EnergyCalibration(coeffs) if coeffs is not None else None
    return Spectrum(np.array(counts, dtype=int), live_time, real_time, cal, {"format": "n42"})


def write_n42(spectrum: Spectrum, path: str | Path) -> None:
    """Write a Spectrum as an ANSI/IEEE N42.42-2012 XML file."""
    ns = "http://physics.nist.gov/N42/2011/N42"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}RadInstrumentData")
    if spectrum.calibration is not None:
        ecal = ET.SubElement(root, f"{{{ns}}}EnergyCalibration", {"id": "EC1"})
        cval = ET.SubElement(ecal, f"{{{ns}}}CoefficientValues")
        cval.text = " ".join(f"{c:.8g}" for c in spectrum.calibration.coeffs)
    meas = ET.SubElement(root, f"{{{ns}}}RadMeasurement", {"id": "M1"})
    rt = spectrum.real_time if spectrum.real_time is not None else spectrum.live_time
    ET.SubElement(meas, f"{{{ns}}}RealTimeDuration").text = f"PT{rt:.3f}S"
    spec_el = ET.SubElement(
        meas,
        f"{{{ns}}}Spectrum",
        {"id": "S1", **({"energyCalibrationReference": "EC1"} if spectrum.calibration else {})},
    )
    ET.SubElement(spec_el, f"{{{ns}}}LiveTimeDuration").text = f"PT{spectrum.live_time:.3f}S"
    ch_el = ET.SubElement(spec_el, f"{{{ns}}}ChannelData")
    ch_el.text = " ".join(str(int(c)) for c in spectrum.counts)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- Canberra / Mirion .CNF
_CNF_SEC_ACQ = 0x00012000
_CNF_SEC_CAL = 0x00012001
_CNF_SEC_DATA = 0x00012005


def _pdp11_to_float(raw4: bytes) -> float:
    """Decode a 32-bit VAX/PDP-11 F-float used in Canberra Genie CNF headers."""
    w0, w1 = struct.unpack("<HH", raw4)
    if w0 == 0 and w1 == 0:
        return 0.0
    swapped = struct.pack("<HH", w1, w0)
    val = struct.unpack("<f", swapped)[0]
    return float(val / 4.0)


def _float_to_pdp11(val: float) -> bytes:
    if val == 0.0:
        return b"\x00\x00\x00\x00"
    raw = struct.pack("<f", float(val * 4.0))
    w1, w0 = struct.unpack("<HH", raw)
    return struct.pack("<HH", w0, w1)


def read_cnf(path: str | Path) -> Spectrum:
    """Read a Canberra/Mirion Genie 2000 binary ``.cnf`` spectrum file."""
    raw = Path(path).read_bytes()
    if len(raw) < 160:
        raise ValueError(f"{path}: file too small to be a Canberra CNF file")

    offsets: dict[int, int] = {}
    pos = 112
    while pos + 48 <= len(raw):
        sec_id = struct.unpack_from("<I", raw, pos)[0]
        if sec_id == 0:
            break
        sec_off = struct.unpack_from("<I", raw, pos + 10)[0]
        if sec_off == 0 or sec_off >= len(raw):
            break
        offsets[sec_id] = sec_off
        pos += 48
        if len(offsets) >= 32:
            break

    if _CNF_SEC_DATA not in offsets:
        raise ValueError(f"{path}: missing channel-data section (0x00012005) in CNF file")

    live_time = 0.0
    real_time: float | None = None
    n_channels = 0
    cal: EnergyCalibration | None = None

    if _CNF_SEC_ACQ in offsets:
        acq = offsets[_CNF_SEC_ACQ]
        if acq + 0x48 <= len(raw):
            # Standard CAM acquisition block: real_time & live_time in 100 ns ticks (negative uint64)
            t_off = acq + 48 + struct.unpack_from("<H", raw, acq + 34)[0]
            if t_off + 24 <= len(raw):
                rt_ticks, lt_ticks = struct.unpack_from("<qq", raw, t_off + 8)
                real_time = abs(rt_ticks) * 1e-7 if rt_ticks != 0 else None
                live_time = abs(lt_ticks) * 1e-7
            n_ch_cand = struct.unpack_from("<H", raw, acq + 0xBA)[0] if acq + 0xBC <= len(raw) else 0
            if n_ch_cand > 0:
                n_channels = int(n_ch_cand)

    if _CNF_SEC_CAL in offsets:
        cal_off = offsets[_CNF_SEC_CAL]
        if cal_off + 0x5C <= len(raw):
            sub_off = cal_off + 48 + struct.unpack_from("<H", raw, cal_off + 34)[0]
            if sub_off + 0x54 <= len(raw):
                c0 = _pdp11_to_float(raw[sub_off + 0x44 : sub_off + 0x48])
                c1 = _pdp11_to_float(raw[sub_off + 0x48 : sub_off + 0x4C])
                c2 = _pdp11_to_float(raw[sub_off + 0x4C : sub_off + 0x50])
                if np.isfinite([c0, c1, c2]).all() and abs(c1) > 1e-9:
                    coeffs = (c0, c1, c2) if abs(c2) > 1e-12 else (c0, c1)
                    cal = EnergyCalibration(coeffs)

    data_off = offsets[_CNF_SEC_DATA] + 512
    if data_off >= len(raw):
        raise ValueError(f"{path}: invalid CNF channel data offset")
    max_avail = (len(raw) - data_off) // 4
    n_read = n_channels if 0 < n_channels <= max_avail else max_avail
    counts = np.frombuffer(raw, dtype="<u4", count=n_read, offset=data_off).astype(int)
    if live_time <= 0:
        raise ValueError(f"{path}: could not parse positive live time from CNF file")
    return Spectrum(counts, live_time, real_time, cal, {"format": "cnf"})


def write_cnf(spectrum: Spectrum, path: str | Path) -> None:
    """Write a minimal valid Canberra/Mirion CAM ``.cnf`` binary spectrum file."""
    n = spectrum.n_channels
    rt = spectrum.real_time if spectrum.real_time is not None else spectrum.live_time

    # Layout: 112-byte file header + 4 * 48-byte section directory = 304 bytes -> align to 512
    acq_off = 512
    cal_off = 1024
    data_sec_off = 1536  # channel payload begins at data_sec_off + 512 = 2048

    buf = bytearray(data_sec_off + 512 + 4 * n)
    # Section table at offset 112
    sections = ((_CNF_SEC_ACQ, acq_off), (_CNF_SEC_CAL, cal_off), (_CNF_SEC_DATA, data_sec_off))
    for idx, (sec_id, off) in enumerate(sections):
        entry = 112 + idx * 48
        struct.pack_into("<I", buf, entry, sec_id)
        struct.pack_into("<I", buf, entry + 10, off)

    # Acquisition section (offset 512): time sub-header offset = 0 -> ticks at acq_off + 48 + 8
    struct.pack_into("<H", buf, acq_off + 34, 0)
    struct.pack_into("<qq", buf, acq_off + 48 + 8, -int(round(rt * 1e7)), -int(round(spectrum.live_time * 1e7)))
    struct.pack_into("<H", buf, acq_off + 0xBA, n)

    # Calibration section (offset 1024)
    struct.pack_into("<H", buf, cal_off + 34, 0)
    if spectrum.calibration is not None:
        coeffs = list(spectrum.calibration.coeffs) + [0.0, 0.0, 0.0]
        sub = cal_off + 48
        buf[sub + 0x44 : sub + 0x48] = _float_to_pdp11(coeffs[0])
        buf[sub + 0x48 : sub + 0x4C] = _float_to_pdp11(coeffs[1])
        buf[sub + 0x4C : sub + 0x50] = _float_to_pdp11(coeffs[2])

    # Channel payload (uint32 little-endian)
    payload = np.asarray(spectrum.counts, dtype="<u4").tobytes()
    buf[data_sec_off + 512 : data_sec_off + 512 + len(payload)] = payload
    Path(path).write_bytes(bytes(buf))


# --------------------------------------------------------------------------- unified entry points
def write_spectrum(spectrum: Spectrum, path: str | Path) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".spe":
        write_spe(spectrum, path)
        return
    if ext == ".chn":
        write_chn(spectrum, path)
        return
    if ext in (".n42", ".xml"):
        write_n42(spectrum, path)
        return
    if ext == ".cnf":
        write_cnf(spectrum, path)
        return
    if ext == ".json":
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
    ext = path.suffix.lower()
    if ext == ".spe":
        return read_spe(path)
    if ext == ".chn":
        return read_chn(path)
    if ext in (".n42", ".xml"):
        return read_n42(path)
    if ext == ".cnf":
        return read_cnf(path)
    if ext == ".json":
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
