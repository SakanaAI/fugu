"""
Low-level GRIB2 retrieval helpers.

Two strategies are used:
  * idx byte-range subsetting (GFS on AWS, GDPS on MSC Datamart) -- we read the
    sidecar `.idx` inventory, select the records we need and pull only those
    byte ranges, then decode the concatenated messages with cfgrib.
  * for whole single-variable files (GDPS, ICON) we just download the file.

All decoding is done with cfgrib/eccodes on the provider's NATIVE grid; no field
is ever synthesized.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
import xarray as xr

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "japan-typhoon-pipeline/1.0 (research)"})


@dataclass
class RecordSel:
    """One selected GRIB record: byte range + human label."""
    start: int
    end: int | None      # inclusive end byte, or None for "to EOF"
    label: str


def fetch_idx(idx_url: str, timeout: int = 60) -> list[tuple[int, str]]:
    """Return list of (start_byte, raw_idx_line) for a GRIB .idx file."""
    r = SESSION.get(idx_url, timeout=timeout)
    r.raise_for_status()
    out = []
    for line in r.text.splitlines():
        # format: <num>:<startbyte>:d=...:VAR:level:fcst:
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            start = int(parts[1])
        except ValueError:
            continue
        out.append((start, line))
    return out


def select_ranges(idx_lines: list[tuple[int, str]], patterns: list[str]) -> list[RecordSel]:
    """
    Given idx lines and a list of regexes, return byte ranges for the FIRST
    line matching each pattern. The end byte is the start of the next record.
    """
    sels: list[RecordSel] = []
    for pat in patterns:
        rx = re.compile(pat)
        for i, (start, line) in enumerate(idx_lines):
            if rx.search(line):
                end = idx_lines[i + 1][0] - 1 if i + 1 < len(idx_lines) else None
                sels.append(RecordSel(start, end, line))
                break
        else:
            raise KeyError(f"pattern {pat!r} not found in idx ({len(idx_lines)} records)")
    return sels


def download_ranges(grib_url: str, sels: list[RecordSel], dest: Path, timeout: int = 120) -> int:
    """Download the selected byte ranges and concatenate into `dest`. Returns bytes."""
    total = 0
    with open(dest, "wb") as fh:
        for sel in sels:
            rng = f"bytes={sel.start}-" + ("" if sel.end is None else str(sel.end))
            r = SESSION.get(grib_url, headers={"Range": rng}, timeout=timeout)
            if r.status_code not in (200, 206):
                r.raise_for_status()
            fh.write(r.content)
            total += len(r.content)
    return total


def download_file(url: str, dest: Path, timeout: int = 180) -> int:
    r = SESSION.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    n = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
            n += len(chunk)
    return n


def _valid_grib(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 256:
            return False
        with open(path, "rb") as fh:
            if fh.read(4) != b"GRIB":
                return False
    except OSError:
        return False
    return True


def ensure_file(url: str, dest: Path, tries: int = 3, timeout: int = 180) -> int:
    """
    Download `url` to `dest` and validate it is a real GRIB message; retry on
    truncation/transient failure. If a valid file already exists, reuse it.
    Returns the byte size.
    """
    if _valid_grib(dest):
        return dest.stat().st_size
    last = None
    for _ in range(tries):
        try:
            download_file(url, dest, timeout=timeout)
            if _valid_grib(dest):
                return dest.stat().st_size
        except requests.RequestException as e:
            last = e
    raise RuntimeError(f"could not retrieve valid GRIB from {url}: {last}")


def head_ok(url: str, timeout: int = 30) -> bool:
    """True if a tiny range request returns 200/206 (object exists)."""
    try:
        r = SESSION.get(url, headers={"Range": "bytes=0-1"}, timeout=timeout)
        return r.status_code in (200, 206)
    except requests.RequestException:
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_grib(path: Path, filter_keys: dict | None = None) -> xr.Dataset:
    backend_kwargs = {"indexpath": ""}
    if filter_keys:
        backend_kwargs["filter_by_keys"] = filter_keys
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)


def thread_map(fn, items, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))
