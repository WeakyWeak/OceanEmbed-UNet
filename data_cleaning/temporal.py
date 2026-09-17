import xarray as xr
import pandas as pd
import numpy as np

import grid as G


def align_daily_times(*arrays: xr.DataArray, expect: int | None = None):
    """Ensure all DataArrays share one daily time coordinate.

    `expect` is the number of days the intersection SHOULD contain. Pass it -
    the whole point of this function is that a partial source silently shrinks
    the record, and only the caller knows how long the year ought to be.
    """
    print("[5] Aligning daily time coordinates …")

    start  = np.datetime64(str(G.RECORD_START))
    cutoff = np.datetime64(str(G.RECORD_END))

    aligned = []
    for da in arrays:
        # Floor to day
        da = da.copy()
        da["time"] = da.time.dt.floor("D")
        # Slice to valid range
        da = da.sel(time=slice(start, cutoff))
        aligned.append(da)

    # Find common time intersection
    common_times = aligned[0].time.values
    for da in aligned[1:]:
        common_times = np.intersect1d(common_times, da.time.values)

    if len(common_times) == 0:
        spans = ", ".join(
            f"{str(da.time.values[0])[:10]}..{str(da.time.values[-1])[:10]}"
            if da.time.size else "EMPTY" for da in aligned)
        raise ValueError(
            f"no common dates across the {len(aligned)} sources after "
            f"clamping to {start}..{cutoff}. Source spans: {spans}")

    print(f"     Common timeline: {str(common_times[0])[:10]} → "
          f"{str(common_times[-1])[:10]}, {len(common_times)} days")

    # A gap is invisible in a day COUNT - 365 steps can still skip 3 May and
    # repeat 4 May - so check regularity, not just length.
    diffs = np.diff(common_times).astype("timedelta64[D]").astype(int)
    if diffs.size and not (diffs == 1).all():
        bad = np.flatnonzero(diffs != 1)[:5]
        raise ValueError(
            f"common timeline is not strictly daily: "
            f"{[(str(common_times[i])[:10], int(diffs[i])) for i in bad]}")

    if expect is not None and len(common_times) != expect:
        raise ValueError(
            f"expected {expect} common days, got {len(common_times)} "
            f"({str(common_times[0])[:10]}..{str(common_times[-1])[:10]}). "
            f"A source is short - not to proceed with a truncated year.")

    result = [da.sel(time=common_times) for da in aligned]
    return result
