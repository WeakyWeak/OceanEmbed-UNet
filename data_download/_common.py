"""
Shared download machinery: the resume contract.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
A file counts as done only when it has been OPENED AND VALIDATED. The presence
of a .nc never gates anything.

That is not paranoia, it is the specific failure this expansion invites. Every
download script in this repo used to skip on `os.path.exists(...)`, and the
domain change keeps every filename identical while changing every file's
contents. Re-running the old scripts after widening the box would have skipped
all 2005-2022 files and left the raw directory silently mixed-domain, with no
error anywhere downstream.

Three mechanisms, each closing a different hole:

1. SPEC HASH. Every expected file carries a hash of the parameters that
   produced it - dataset id, variables, bounding box, dates, depth. The hash
   goes in the sidecar. Change the domain and every hash changes, so every
   sidecar goes stale and every file is re-fetched. There is no way to
   "forget" to invalidate.

2. ATOMIC WRITES. Downloads land on <name>.part and are only os.replace()d
   into place after they open and validate. Ctrl-C or a dropped connection can
   leave a .part, which is deleted and retried; it can never leave a truncated
   .nc that the next run treats as complete.

3. VALIDATION BEFORE MARKING. The sidecar is written after the file opens, its
   day count matches what the product should supply for that year, and its
   grid covers the whole target domain. A file that downloads "successfully"
   but contains half a year does not get marked done.

verify_downloads.py deliberately ignores the sidecars and re-checks everything
from scratch. A verifier that trusts the downloader's own bookkeeping verifies
nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data_cleaning"))
import grid as G                                                # noqa: E402

# Versioned on purpose - see EXPANSION_PLAN.txt section 5.0. Writing beside the
# old data rather than over it is what makes every resume gate correct by
# construction, and it keeps the old artefacts for the "did this help?" A/B.
RAW_ROOT = PROJECT_ROOT / "data" / "raw2"

RETRIES = 3
BACKOFF = 20          # seconds, multiplied by the attempt number


# ── expected-file description ────────────────────────────────────────────────

@dataclass
class FileSpec:
    """One file we expect to exist, and everything needed to check it."""
    path: Path
    product: str                      # key into grid.PRODUCT_END
    year: int
    label: str                        # human-readable, for progress lines
    variables: list[str]
    params: dict = field(default_factory=dict)   # goes into the spec hash
    min_days: int | None = None       # override; defaults to grid.expected_days
    steps_per_day: int = 1            # 24 for ERA5, which is hourly

    @property
    def expected_days(self) -> int:
        return (self.min_days if self.min_days is not None
                else G.expected_days(self.year, self.product))

    @property
    def expected_steps(self) -> int:
        """Time steps the file should hold. Daily products give one per day;
        ERA5 is hourly, so a year is days*24 and checking against days alone
        would reject every good file."""
        return self.expected_days * self.steps_per_day

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".ok")

    @property
    def partial(self) -> Path:
        """Temp path for an in-flight download.

        Note it is <stem>.part.nc, not <name>.nc.part: copernicusmarine
        appends '.nc' to any output filename that does not already end in it,
        so a '.part' suffix silently produced 'sss_1993.nc.part.nc' and the
        os.replace() then moved nothing.
        """
        return self.path.with_name(self.path.stem + ".part" + self.path.suffix)

    def spec_hash(self) -> str:
        """Hash of everything that determines the file's contents.

        The bounding box is in here, so widening the domain invalidates every
        sidecar automatically. That is the whole point.
        """
        payload = {
            "product": self.product, "year": self.year,
            "variables": sorted(self.variables),
            "box": [G.DL_LAT_MIN, G.DL_LAT_MAX, G.DL_LON_MIN, G.DL_LON_MAX],
            "target": [G.LAT_MIN, G.LAT_MAX, G.LON_MIN, G.LON_MAX],
            "expected_days": self.expected_days,
            "steps_per_day": self.steps_per_day,
            "params": {k: str(v) for k, v in sorted(self.params.items())},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ── inspection ───────────────────────────────────────────────────────────────

def coord_names(ds):
    """(lat, lon, time) coordinate names. Products disagree: CMEMS uses
    latitude/longitude, OISST uses lat/lon, ERA5 uses valid_time."""
    lat = next((c for c in ("latitude", "lat", "nav_lat") if c in ds.coords), None)
    lon = next((c for c in ("longitude", "lon", "nav_lon") if c in ds.coords), None)
    tim = next((c for c in ("time", "valid_time") if c in ds.dims), None)
    return lat, lon, tim


def inspect(path: Path) -> dict:
    """Open a NetCDF and pull out the facts we check against. Raises on
    anything unreadable."""
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        lat, lon, tim = coord_names(ds)
        if lat is None or lon is None:
            raise ValueError(f"no lat/lon coordinates; found {list(ds.coords)}")
        la, lo = np.asarray(ds[lat].values), np.asarray(ds[lon].values)
        return {
            "n_time": int(ds.sizes.get(tim, 0)) if tim else 0,
            "time_dim": tim,
            "lat_min": float(la.min()), "lat_max": float(la.max()),
            "lon_min": float(lo.min()), "lon_max": float(lo.max()),
            "n_lat": int(la.size), "n_lon": int(lo.size),
            "variables": sorted([str(v) for v in ds.data_vars]),
            "bytes": path.stat().st_size,
        }


def check(found: dict, spec: FileSpec) -> list[str]:
    """Compare an inspection against a spec. Returns a list of problems."""
    bad = []
    if spec.expected_steps and found["n_time"] != spec.expected_steps:
        bad.append(f"time={found['n_time']} expected {spec.expected_steps}")

    # The grid must COVER the target domain, not equal the padded request:
    # every product snaps to its own native points inside the box we asked
    # for. Covering the target is what guarantees the regridding step never
    # has to extrapolate - which it would do silently, by nearest-neighbour
    # fill, smearing one edge column across the whole Arabian Sea.
    if found["lat_min"] > G.LAT_MIN + 1e-6:
        bad.append(f"lat starts at {found['lat_min']:.4f} > {G.LAT_MIN}")
    if found["lat_max"] < G.LAT_MAX - 1e-6:
        bad.append(f"lat ends at {found['lat_max']:.4f} < {G.LAT_MAX}")
    if found["lon_min"] > G.LON_MIN + 1e-6:
        bad.append(f"lon starts at {found['lon_min']:.4f} > {G.LON_MIN}")
    if found["lon_max"] < G.LON_MAX - 1e-6:
        bad.append(f"lon ends at {found['lon_max']:.4f} < {G.LON_MAX}")

    missing = [v for v in spec.variables if v not in found["variables"]]
    if missing:
        bad.append(f"missing variables {missing} (has {found['variables']})")
    if found["bytes"] < 1024:
        bad.append(f"suspiciously small: {found['bytes']} bytes")
    return bad


# ── the resume gate ──────────────────────────────────────────────────────────

def is_done(spec: FileSpec) -> bool:
    """True only if the sidecar exists, matches the current spec hash, and the
    file it describes is still there with the same size."""
    if not spec.sidecar.exists() or not spec.path.exists():
        return False
    try:
        rec = json.loads(spec.sidecar.read_text())
    except Exception:                                       # noqa: BLE001
        return False
    if rec.get("spec_hash") != spec.spec_hash():
        return False                                        # domain/date changed
    return rec.get("bytes") == spec.path.stat().st_size


def mark_done(spec: FileSpec, found: dict) -> None:
    spec.sidecar.write_text(json.dumps(
        {"spec_hash": spec.spec_hash(), "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime()),
         **found}, indent=2))


def _clear_partials(spec: FileSpec) -> None:
    """Remove our .part file AND any scratch files the client left behind.

    copernicusmarine writes its own randomly-suffixed temporaries next to the
    output (e.g. sss_1993.part.nc.ro59z08d) and does not clean them up when
    interrupted. Globbing the prefix catches those; otherwise they accumulate
    silently across a long run and quietly eat disk.
    """
    for p in spec.path.parent.glob(spec.partial.name + "*"):
        try:
            p.unlink()
        except OSError:
            pass
    if spec.partial.exists():
        try:
            spec.partial.unlink()
        except OSError:
            pass


def fetch_one(spec: FileSpec, fetch_fn, dry_run: bool = False) -> str:
    """Download, validate and mark one file.

    `fetch_fn(dest: Path)` must write the file to exactly `dest`.
    Returns one of: "skip", "ok", "fail".
    """
    if is_done(spec):
        return "skip"
    if dry_run:
        return "would-fetch"

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        _clear_partials(spec)
        try:
            fetch_fn(spec.partial)
            found = inspect(spec.partial)
            problems = check(found, spec)
            if problems:
                raise ValueError("; ".join(problems))
            os.replace(spec.partial, spec.path)
            mark_done(spec, found)
            return "ok"
        except KeyboardInterrupt:
            _clear_partials(spec)
            raise
        except Exception as e:                              # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt == RETRIES:
                print(f"    FAILED after {RETRIES} attempts - {msg}", flush=True)
                _clear_partials(spec)
                return "fail"
            wait = BACKOFF * attempt
            print(f"    attempt {attempt} failed ({msg}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    return "fail"


# ── progress ─────────────────────────────────────────────────────────────────

class Progress:
    """Per-file progress with a running tally, so an overnight run is legible
    in the morning and a failure is obvious without reading everything."""

    def __init__(self, total: int, title: str):
        self.total, self.title = total, title
        self.n = self.ok = self.skip = self.fail = 0
        self.t0 = time.time()
        print(f"\n{'=' * 78}\n  {title}  -  {total} files\n{'=' * 78}", flush=True)

    def record(self, spec: FileSpec, status: str) -> None:
        self.n += 1
        if status == "ok":
            self.ok += 1
        elif status == "skip":
            self.skip += 1
        elif status == "fail":
            self.fail += 1
        if status == "skip" and self.n % 25 and self.n != self.total:
            return                                    # keep skip runs quiet
        el = time.time() - self.t0
        rate = (self.ok / el * 3600) if self.ok and el > 0 else 0
        eta = ((self.total - self.n) / rate * 3600) if rate > 0 else 0
        print(f"  [{self.n:4d}/{self.total}] {status:5s} {spec.label:<44} "
              f"ok={self.ok} skip={self.skip} fail={self.fail}"
              + (f"  eta {eta / 3600:.1f}h" if eta else ""), flush=True)

    def finish(self) -> int:
        el = time.time() - self.t0
        print(f"\n  {self.title}: {self.ok} fetched, {self.skip} already done, "
              f"{self.fail} FAILED   ({el / 60:.1f} min)", flush=True)
        return self.fail


def run_specs(specs: list[FileSpec], fetch_fn_for, title: str,
              dry_run: bool = False) -> int:
    """Drive a list of specs. Returns the failure count."""
    todo = [s for s in specs if not is_done(s)]
    if dry_run:
        print(f"\n{title}: {len(todo)} of {len(specs)} files would be fetched")
        for s in todo[:15]:
            print(f"    {s.label}")
        if len(todo) > 15:
            print(f"    ... and {len(todo) - 15} more")
        return 0

    p = Progress(len(specs), title)
    for spec in specs:
        p.record(spec, fetch_one(spec, fetch_fn_for(spec)))
    return p.finish()
