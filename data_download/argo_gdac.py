#!/usr/bin/env python3
"""
Download Argo profile observations for the Bay of Bengal domain.

Source
------
Argo Global Data Assembly Centre (GDAC), Ifremer mirror, "geo" tree:

    https://data-argo.ifremer.fr/geo/indian_ocean/YYYY/MM/YYYYMMDD_prof.nc

One file per day holding every profile in the Indian Ocean basin for that day.
No credentials are required, which is why this route was chosen over the
INCOIS LAS server named in the problem statement: the LAS has no stable
documented URL pattern and would need hand-holding.

Why raw profiles and not a gridded Argo product
-----------------------------------------------
The problem statement asks for a "Validation framework using independent ARGO
observations". Raw profiles are the strongest reading of that: each one is a
single float's measurement at a point in space and time, compared against the
model's nearest grid cell on the same day. Gridded Argo products are objective
analyses - they have already been smoothed and interpolated by somebody else's
model, and are typically monthly at 1 degree, which would force us to average
away most of the daily 0.25 degree signal we are trying to demonstrate.

Honest caveat to state in the report: Argo profiles are ASSIMILATED INTO
GLORYS, which is our training target. So Argo is independent of our model's
INPUTS, but not fully independent of the data our target was built from. It is
still the right validation set - it is what the problem statement asks for, and
it is the only in-situ truth available - but do not describe it as fully
independent.

What this script does
---------------------
Downloading every daily file for 2005-2022 would be roughly 13 GB of mostly
irrelevant ocean. Instead each day is fetched to a temporary file, filtered to
the domain immediately, and discarded. Only profiles inside
5-25 N / 75-100 E survive, and they are accumulated into one compact NetCDF
per year.

Output: data/raw/ARGO/argo_bob_<year>.nc

Stored as a FLAT OBSERVATION TABLE, one row per measured level, because
profiles have wildly different level counts and padding them into a rectangle
wastes space and invites silent fill-value bugs:

    dim obs      : time, lat, lon, pres, depth_m, temp, profile_id, platform,
                   cycle, data_mode, temp_qc, pres_qc, adjusted
    dim profile  : profile metadata, indexed by profile_id

Quality control applied
-----------------------
* Position and JULD QC must be good (flag 1) - a profile with a bad position
  cannot be co-located with a grid cell.
* Delayed-mode / adjusted values are preferred wherever DATA_MODE is 'D' or
  'A' and the adjusted value exists; the `adjusted` column records which was
  used per level so you can report the split.
* Level kept only if the QC flag of the value actually used is 1 or 2
  (good / probably good). Flags 3, 4, 8, 9 are dropped.
* Levels with non-finite pressure or temperature are dropped.
* Pressure is converted to depth with the UNESCO/Fofonoff formula (latitude
  dependent). Do not compare Argo pressure directly against GLORYS depth: at
  1000 dbar near 15 N the difference is about 5 m.

Bandwidth
---------
A daily Indian Ocean file is about 3.3 MB and only a handful of its profiles
fall inside the Bay of Bengal, so this trades bandwidth for simplicity: roughly
1.2 GB downloaded per year, of which a few MB is kept. Nothing is left on disk
except the yearly output.

BUDGET 2-3 HOURS PER YEAR, not minutes. The GDAC serves these files slowly -
measured at roughly 20-30 s per daily file end to end, and there are 365 of
them per year with no parallelism here. Two years is most of a working day.
Run it overnight, or narrow the range:

    --years 2021          just the validation year, ~2-3 h
    --start 2021-06-01 --end 2021-08-31    one season, ~45 min

A single season is enough to demonstrate the validation framework works; the
full two years is only needed for a headline in-situ skill number. Progress
prints every 30 days, so you can watch it and stop early with Ctrl-C - but
note that a year is written only after ALL its days are fetched, so an
interrupted year produces no file.

Usage
-----
    .venv/bin/python data_download/argo_gdac.py            # 2021-2022, default
    .venv/bin/python data_download/argo_gdac.py --dry-run  # check first, 5 days
    .venv/bin/python data_download/argo_gdac.py --years 2021
    .venv/bin/python data_download/argo_gdac.py --start 2005-01-01 \
                                                --end   2022-12-31

The default is 2021-2022 because that is val + test - the only period where an
independent check means anything. Training years would cost ~22 GB of download
to validate data the model already fit, so ask for them deliberately.

Resumable: a year whose output file already exists is skipped unless
--overwrite is passed. Missing daily files (404) are normal - the GDAC has
gaps - and are counted, not fatal.
"""
import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

# ── domain: must match data_cleaning/config.py ───────────────────────────────
LAT_MIN, LAT_MAX = 5.0, 25.0
LON_MIN, LON_MAX = 75.0, 100.0

BASE_URL = "https://data-argo.ifremer.fr/geo/indian_ocean"
# US mirror, same layout, if Ifremer is down:
#   https://usgodae.org/pub/outgoing/argo/geo/indian_ocean

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/raw/ARGO"))

GOOD_QC = {b"1", b"2"}          # good, probably good
STRICT_QC = {b"1"}              # position / time must be unambiguous
TIMEOUT = 60
RETRIES = 3


# ── helpers ──────────────────────────────────────────────────────────────────

def depth_from_pressure(pres_dbar, lat_deg):
    """UNESCO (Fofonoff & Millard 1983) depth in metres from pressure in dbar.

    Pure arithmetic, no gsw dependency. Accurate to a few centimetres, which is
    far below the ~20 m vertical spacing that matters here. The latitude term
    is not optional: ignoring it costs about 5 m at 1000 dbar at this latitude,
    and our depth levels near the thermocline are only 25 m apart.
    """
    p = np.asarray(pres_dbar, dtype=np.float64)
    x = np.sin(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)) ) ** 2
    g = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    num = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return num / g


def _as_bytes_array(da):
    """Argo char variables come back as bytes, str, or arrays of either.

    xarray hands these back as dtype=object holding real bytes objects. Do NOT
    reach for str(x) on those: str(b"1") is "b\'1\'", so every QC comparison
    silently fails and the whole domain filters down to nothing.
    """
    v = np.asarray(da.values)
    if v.dtype.kind == "S":
        return v
    if v.dtype.kind == "U":
        return np.char.encode(v, "ascii")
    flat = [x if isinstance(x, bytes)
            else x.encode("ascii", "replace") if isinstance(x, str)
            else str(x).encode("ascii", "replace")
            for x in v.ravel()]
    return np.array(flat).reshape(v.shape)


def _platform_strings(ds):
    if "PLATFORM_NUMBER" not in ds:
        return None
    v = np.asarray(ds["PLATFORM_NUMBER"].values)
    if v.ndim == 2:                       # (N_PROF, STRING8) char matrix
        v = np.array([b"".join(row).decode("ascii", "replace").strip()
                      for row in _as_bytes_array(ds["PLATFORM_NUMBER"])])
    else:
        v = np.array([(x.decode("ascii", "replace") if isinstance(x, bytes)
                       else str(x)).strip() for x in v])
    return v


def fetch_day(day: date, tmpdir: str):
    """Download one daily profile file. Returns a path, or None if absent."""
    url = f"{BASE_URL}/{day:%Y}/{day:%m}/{day:%Y%m%d}_prof.nc"
    dest = os.path.join(tmpdir, f"{day:%Y%m%d}_prof.nc")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r, \
                 open(dest, "wb") as f:
                f.write(r.read())
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # genuine gap in the GDAC, not an error
            if attempt == RETRIES:
                print(f"    HTTP {e.code} on {url}", file=sys.stderr)
                return None
        except Exception as e:                       # noqa: BLE001
            if attempt == RETRIES:
                print(f"    failed {url}: {e}", file=sys.stderr)
                return None
    return None


def extract(path: str, day: date):
    """One daily file -> tidy DataFrame of in-domain, QC-passed levels."""
    try:
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=True)
    except Exception as e:                            # noqa: BLE001
        print(f"    unreadable {os.path.basename(path)}: {e}", file=sys.stderr)
        return None

    with ds:
        if "LATITUDE" not in ds or "TEMP" not in ds:
            return None

        lat = np.asarray(ds["LATITUDE"].values, dtype=np.float64)
        lon = np.asarray(ds["LONGITUDE"].values, dtype=np.float64)
        lon = np.where(lon > 180.0, lon - 360.0, lon)

        keep = ((lat >= LAT_MIN) & (lat <= LAT_MAX) &
                (lon >= LON_MIN) & (lon <= LON_MAX) &
                np.isfinite(lat) & np.isfinite(lon))

        # A profile whose position or time is flagged cannot be co-located.
        for qc_name, allowed in (("POSITION_QC", STRICT_QC),
                                 ("JULD_QC", STRICT_QC)):
            if qc_name in ds:
                q = _as_bytes_array(ds[qc_name]).ravel()
                if q.shape[0] == keep.shape[0]:
                    keep &= np.isin(q, list(allowed))

        if not keep.any():
            return None

        pidx = np.flatnonzero(keep)
        juld = np.asarray(ds["JULD"].values, dtype=np.float64)[pidx]
        time = (pd.Timestamp("1950-01-01") +
                pd.to_timedelta(juld, unit="D"))

        platform = _platform_strings(ds)
        platform = (platform[pidx] if platform is not None
                    else np.array([""] * len(pidx)))
        cycle = (np.asarray(ds["CYCLE_NUMBER"].values)[pidx]
                 if "CYCLE_NUMBER" in ds else np.full(len(pidx), -1))
        dmode = (_as_bytes_array(ds["DATA_MODE"]).ravel()[pidx]
                 if "DATA_MODE" in ds else np.full(len(pidx), b"R"))

        def field(base):
            """(raw, raw_qc, adj, adj_qc) for PRES or TEMP, sliced to pidx."""
            raw = np.asarray(ds[base].values, dtype=np.float64)[pidx]
            rqc = (_as_bytes_array(ds[f"{base}_QC"])[pidx]
                   if f"{base}_QC" in ds else None)
            adj = (np.asarray(ds[f"{base}_ADJUSTED"].values,
                              dtype=np.float64)[pidx]
                   if f"{base}_ADJUSTED" in ds else None)
            aqc = (_as_bytes_array(ds[f"{base}_ADJUSTED_QC"])[pidx]
                   if f"{base}_ADJUSTED_QC" in ds else None)
            return raw, rqc, adj, aqc

        p_raw, p_rqc, p_adj, p_aqc = field("PRES")
        t_raw, t_rqc, t_adj, t_aqc = field("TEMP")

        n_prof, n_lev = t_raw.shape
        # Adjusted values are only meaningful in delayed / adjusted mode.
        use_adj = np.isin(dmode, [b"D", b"A"])[:, None] & np.ones(
            (1, n_lev), dtype=bool)
        if p_adj is None or t_adj is None:
            use_adj = np.zeros((n_prof, n_lev), dtype=bool)
        else:
            use_adj &= np.isfinite(p_adj) & np.isfinite(t_adj)

        pres = np.where(use_adj, p_adj, p_raw) if p_adj is not None else p_raw
        temp = np.where(use_adj, t_adj, t_raw) if t_adj is not None else t_raw

        def qc_of(rqc, aqc):
            if rqc is None:
                return np.full((n_prof, n_lev), b"1")
            if aqc is None:
                return rqc
            return np.where(use_adj, aqc, rqc)

        pqc, tqc = qc_of(p_rqc, p_aqc), qc_of(t_rqc, t_aqc)

        good = (np.isin(pqc, list(GOOD_QC)) & np.isin(tqc, list(GOOD_QC)) &
                np.isfinite(pres) & np.isfinite(temp))
        if not good.any():
            return None

        r, c = np.nonzero(good)
        lat_o = lat[pidx][r]
        return pd.DataFrame({
            "time":       time.values[r],
            "lat":        lat_o,
            "lon":        lon[pidx][r],
            "pres":       pres[r, c],
            "depth_m":    depth_from_pressure(pres[r, c], lat_o),
            "temp":       temp[r, c],
            "platform":   platform[r],
            "cycle":      np.asarray(cycle, dtype=np.int32)[r],
            "data_mode":  np.array([m.decode() for m in dmode])[r],
            "adjusted":   use_adj[r, c],
            "prof_key":   np.array([f"{platform[i]}_{int(cycle[i])}"
                                    for i in range(len(pidx))])[r],
            "file_day":   np.full(len(r), f"{day:%Y-%m-%d}"),
        })


def write_year(frames, year: int, out_path: str):
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["time", "prof_key", "pres"]).reset_index(drop=True)

    keys, prof_id = np.unique(df["prof_key"].values, return_inverse=True)
    first = {k: i for i, k in enumerate(keys)}
    order = df.drop_duplicates("prof_key").set_index("prof_key")

    ds = xr.Dataset(
        data_vars=dict(
            time=("obs", df["time"].values),
            lat=("obs", df["lat"].to_numpy(np.float32)),
            lon=("obs", df["lon"].to_numpy(np.float32)),
            pres=("obs", df["pres"].to_numpy(np.float32)),
            depth_m=("obs", df["depth_m"].to_numpy(np.float32)),
            temp=("obs", df["temp"].to_numpy(np.float32)),
            profile_id=("obs", prof_id.astype(np.int32)),
            adjusted=("obs", df["adjusted"].to_numpy(np.int8)),
            profile_key=("profile", keys.astype(object)),
            profile_platform=("profile",
                              order.loc[keys, "platform"].to_numpy(object)),
            profile_cycle=("profile",
                           order.loc[keys, "cycle"].to_numpy(np.int32)),
            profile_data_mode=("profile",
                               order.loc[keys, "data_mode"].to_numpy(object)),
        ),
        attrs=dict(
            title=f"Argo profiles, Bay of Bengal domain, {year}",
            source=f"{BASE_URL} (Argo GDAC, Ifremer mirror)",
            domain=f"lat {LAT_MIN}-{LAT_MAX} N, lon {LON_MIN}-{LON_MAX} E",
            qc="value QC flag in {1,2}; POSITION_QC and JULD_QC == 1",
            adjusted_policy="ADJUSTED used where DATA_MODE in {D,A} and finite",
            depth_convention="depth_m from pres via UNESCO/Fofonoff, "
                             "latitude-dependent; positive down",
            note="Argo is assimilated into GLORYS, the training target. "
                 "Independent of model INPUTS, not of the target.",
            created_by="data_download/argo_gdac.py",
        ),
    )
    enc = {v: {"zlib": True, "complevel": 4}
           for v in ("lat", "lon", "pres", "depth_m", "temp",
                     "profile_id", "adjusted")}
    ds.to_netcdf(out_path, encoding=enc)
    return len(df), len(keys)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--years", nargs="+", type=int,
                   help="explicit years; overrides --start/--end")
    p.add_argument("--out-dir", default=RAW_DIR)
    p.add_argument("--overwrite", action="store_true",
                   help="re-download years whose output file already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch only the first 5 days of each year")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.years:
        spans = [(date(y, 1, 1), date(y, 12, 31)) for y in sorted(args.years)]
    else:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end)
        spans = [(max(s, date(y, 1, 1)), min(e, date(y, 12, 31)))
                 for y in range(s.year, e.year + 1)]

    print(f"Argo GDAC -> {args.out_dir}")
    print(f"domain: {LAT_MIN}-{LAT_MAX} N, {LON_MIN}-{LON_MAX} E")
    if args.dry_run:
        print("DRY RUN: first 5 days of each year only, nothing written\n")

    grand = 0
    for lo, hi in spans:
        year = lo.year
        out = os.path.join(args.out_dir, f"argo_bob_{year}.nc")
        if os.path.exists(out) and not args.overwrite and not args.dry_run:
            print(f"{year}: exists, skipping ({out})")
            continue

        days = [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
        if args.dry_run:
            days = days[:5]

        frames, missing, n_lev = [], 0, 0
        with tempfile.TemporaryDirectory() as tmp:
            for i, d in enumerate(days, 1):
                path = fetch_day(d, tmp)
                if path is None:
                    missing += 1
                else:
                    df = extract(path, d)
                    if df is not None and len(df):
                        frames.append(df)
                        n_lev += len(df)
                    os.remove(path)
                if i % 30 == 0 or i == len(days):
                    print(f"  {year}: {i:3d}/{len(days)} days  "
                          f"levels={n_lev:7d}  missing={missing}", flush=True)

        if not frames:
            print(f"{year}: no in-domain profiles found\n")
            continue
        if args.dry_run:
            print(f"{year}: DRY RUN would write {n_lev} levels\n")
            continue

        n_obs, n_prof = write_year(frames, year, out)
        grand += n_obs
        print(f"{year}: {n_prof} profiles, {n_obs} levels -> {out}\n")

    print(f"done. {grand} levels written.")
    if not args.dry_run:
        print("\nNext: validate a model against these with")
        print("  .venv/bin/python models/evaluate.py --run <tag> --argo")


if __name__ == "__main__":
    main()
