"""Download and unpack a registered dataset into a local cache, verifying its checksum."""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

from .registry import DATASETS, Dataset

_UA = "gammaguard-validation (+https://github.com/; research use)"


def cache_dir() -> Path:
    """``$GAMMAGUARD_DATA`` or ``~/.cache/gammaguard``."""
    return Path(os.environ.get("GAMMAGUARD_DATA", Path.home() / ".cache" / "gammaguard"))


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(key: str | Dataset, root: Path | None = None, quiet: bool = False) -> Path:
    """Return the directory holding the extracted dataset, downloading it first if needed."""
    ds = DATASETS[key] if isinstance(key, str) else key
    base = (root or cache_dir()) / ds.key
    done = base / "extracted" / ".complete"
    if done.exists():
        return done.parent
    base.mkdir(parents=True, exist_ok=True)
    archive = base / ds.filename
    if not archive.exists() or _md5(archive) != ds.md5:
        if not quiet:
            print(f"downloading {ds.filename} ({ds.size_bytes / 1e6:.1f} MB) from {ds.landing_page}\n"
                  f"  licence: {ds.license}")
        tmp = archive.with_suffix(archive.suffix + ".part")
        req = urllib.request.Request(ds.url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310 - https only
            while block := resp.read(1 << 20):
                out.write(block)
        if _md5(tmp) != ds.md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{ds.filename}: checksum mismatch, refusing to use the download")
        tmp.replace(archive)
    out_dir = base / "extracted"
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():  # reject path traversal before extracting anything
            target = (out_dir / name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"unsafe path in archive: {name!r}")
        zf.extractall(out_dir)
    done.touch()
    if not quiet:
        print(f"cite: {ds.citation}")
    return out_dir
