"""
Real ENSO (NINO3.4) conditioning, causally fenced.

Source: NOAA CPC monthly NINO indices (ERSSTv5, 1991-2020 base):
    https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii

Columns: YEAR MON  NINO1+2_SST NINO1+2_ANOM  NINO3_SST NINO3_ANOM
         NINO4_SST NINO4_ANOM  NINO3.4_SST NINO3.4_ANOM

We use the NINO3.4 monthly anomaly (last column).  CPC publishes a month's
value in the first days of the *following* month, so the only value legitimately
known at an initialization time `init` is the most recent calendar month that
ended strictly before `init`'s own month -- i.e. we never use the month that
contains `init`, nor any later month.  This respects the real publication lag.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

import config as cfg

NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"


@dataclass
class EnsoResult:
    nino34_anom: float
    valid_month: str          # "YYYY-MM" of the SST month used
    source: str
    url: str
    retrieved_utc: str
    sha256: str
    n_months: int


def _download(dest: Path) -> bytes:
    r = requests.get(NINO_URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return r.content


def _parse(raw: bytes) -> list[tuple[int, int, float]]:
    """Return list of (year, month, nino34_anom)."""
    out = []
    for line in raw.decode("ascii", "replace").splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            nino34_anom = float(parts[9])
        except ValueError:
            continue  # header row
        out.append((year, month, nino34_anom))
    return out


def get_nino34(target_init: dt.datetime) -> EnsoResult:
    """
    Causally-fenced NINO3.4 anomaly for a given initialization datetime.

    The latest usable month is the calendar month immediately preceding the
    month that contains `target_init` (published with real lag, never future).
    """
    dest = cfg.ENSO_DIR / "ersst5.nino.mth.91-20.ascii"
    raw = _download(dest)
    series = _parse(raw)
    if not series:
        raise RuntimeError("could not parse any NINO3.4 rows")

    # latest month strictly before target_init's month
    ty, tm = target_init.year, target_init.month
    # the cutoff: (year, month) must be < (ty, tm)
    usable = [(y, m, v) for (y, m, v) in series if (y, m) < (ty, tm)]
    if not usable:
        raise RuntimeError("no causally-valid NINO3.4 month available")
    usable.sort(key=lambda t: (t[0], t[1]))
    y, m, v = usable[-1]

    return EnsoResult(
        nino34_anom=float(v),
        valid_month=f"{y:04d}-{m:02d}",
        source="NOAA CPC ERSSTv5 monthly NINO3.4 (1991-2020 base)",
        url=NINO_URL,
        retrieved_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sha256=hashlib.sha256(raw).hexdigest(),
        n_months=len(series),
    )


def nino34_for_date(series: list[tuple[int, int, float]], date: dt.date) -> float | None:
    """
    For training: the NINO3.4 anomaly that would have been known on `date`,
    i.e. the most recent month strictly before `date`'s month.
    """
    usable = [(y, m, v) for (y, m, v) in series if (y, m) < (date.year, date.month)]
    if not usable:
        return None
    usable.sort(key=lambda t: (t[0], t[1]))
    return float(usable[-1][2])


def load_series_cached() -> list[tuple[int, int, float]]:
    dest = cfg.ENSO_DIR / "ersst5.nino.mth.91-20.ascii"
    if dest.exists():
        raw = dest.read_bytes()
    else:
        raw = _download(dest)
    return _parse(raw)


if __name__ == "__main__":
    res = get_nino34(cfg.get_target_init())
    print(asdict(res))
