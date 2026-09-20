import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import splits as S


# ── static, shared, loaded once ──────────────────────────────────────────────

def load_masks():
    """(target_mask (81,101,15) bool, pair_mask (81,101,14) bool,
        land (81,101) float32)"""
    return (np.load(C.TARGET_MASK_NPY),
            np.load(C.PAIR_MASK_NPY),
            np.load(C.LAND_MASK_NPY))


def load_norm_arrays():
    """Per-depth (mu, sigma) of the ABSOLUTE target, in TARGET_VARS order."""
    meta = json.load(open(C.META_JSON))
    mus = np.array([meta["mus"][v] for v in C.TARGET_VARS], dtype=np.float32)
    sig = np.array([meta["sigmas"][v] for v in C.TARGET_VARS], dtype=np.float32)
    return mus, sig


def load_anom_stats():
    """Anomaly normalisation constants (built by build_climatology.py).

    Returns a dict with numpy arrays:
      target_anom_mean/std  (15,)  normalisation of the anomaly target
      phys_scale            (15,)  degC per unit of normalised anomaly
      phys_offset           (15,)  degC offset (anomaly mean carried through)
      input_anom_mean/std   dict keyed by input variable name

    Absolute degC = clim_degC + phys_scale * anom_norm + phys_offset.
    phys_offset is tiny (< 2e-3 degC) because the anomaly mean is ~0 by
    construction, but it is kept so the round trip is exact.
    """
    if not C.ANOM_STATS_JSON.exists():
        raise FileNotFoundError(
            f"{C.ANOM_STATS_JSON} not found - run training/build_climatology.py first")
    s = json.load(open(C.ANOM_STATS_JSON))
    sig_abs = np.array(s["target_sigma_abs"], dtype=np.float32)
    t_mean  = np.array(s["target_anom_mean"], dtype=np.float32)
    return {
        "target_anom_mean": t_mean,
        "target_anom_std":  np.array(s["target_anom_std"], dtype=np.float32),
        "sigma_abs":        sig_abs,
        "phys_scale":       np.array(s["target_phys_scale_degC"], dtype=np.float32),
        "phys_offset":      (t_mean * sig_abs).astype(np.float32),
        "input_anom_mean":  s["input_anom_mean"],
        "input_anom_std":   s["input_anom_std"],
        # Running-mean constants, absent from stats files built before the
        # feature existed. Keys are the window as a STRING (JSON has no int
        # keys); the caller casts. Nested by window so adding one is a dict
        # entry rather than five flat keys.
        "input_rm_anom_mean": s.get("input_rm_anom_mean", {}),
        "input_rm_anom_std":  s.get("input_rm_anom_std", {}),
        "running_mean_windows": s.get("running_mean_windows", []),
    }


def to_absolute_degC(anom_norm, clim_degC, phys_scale, phys_offset):
    """Normalised anomaly -> absolute temperature in degC.

    The single definition of this transform. losses.py and evaluate.py both
    call it (or mirror it in backend ops) so the formula cannot drift apart
    between training and evaluation.
    """
    return clim_degC + anom_norm * phys_scale + phys_offset


def _coord_planes():
    """lat/lon linearly mapped to [-1, +1] over the domain."""
    lat = (C.TARGET_LAT - C.TARGET_LAT.mean()) / (np.ptp(C.TARGET_LAT) / 2)
    lon = (C.TARGET_LON - C.TARGET_LON.mean()) / (np.ptp(C.TARGET_LON) / 2)
    lat_p = np.repeat(lat[:, None], C.N_LON, axis=1).astype(np.float32)
    lon_p = np.repeat(lon[None, :], C.N_LAT, axis=0).astype(np.float32)
    return lat_p, lon_p


# ── dataset ──────────────────────────────────────────────────────────────────

class OceanDataset(keras.utils.PyDataset):
    """Batches of (X, Y) for one split.

    X : (B, 81, 101, n_channels) float32
    Y : (B, 81, 101, 30) float32  in anomaly mode  -> [anomaly | clim_degC]
        (B, 81, 101, 15) float32  in legacy absolute mode

    Input channel order is fixed:
        [vars @ t] [vars @ t-LAGS[1]] ... [vars rm W0] [vars rm W1] ...
        [land] [lat] [lon] [sin doy] [cos doy]

    The running-mean blocks sit AFTER the lag blocks and BEFORE land, so every
    5-wide per-variable block stays contiguous - SubsetOceanDataset depends on
    that, it subsets each block with the same [..., var_idx] expression.
    """

    def __init__(self, split, lags=None, batch_size=None, shuffle=None,
                 seed=C.SEED, use_land=None, use_coord=None, use_doy=None,
                 anomalise_target=None, anomalise_inputs=None,
                 running_mean_windows=None, **kwargs):
        super().__init__(**kwargs)
        # "buffer" and "any" are membership names splits.py can return, and
        # serving needs both: index_to_split() reports "buffer" for a day
        # deliberately withheld near a held-out boundary, and predict_dates()
        # groups by that name and calls predict() with it. Rejecting them here
        # meant any buffered split broke the backend on a perfectly servable
        # date. Harmless under buffer_days=0, which is why it stayed hidden.
        if split not in ("train", "val", "test", "buffer", "any"):
            raise ValueError(
                f"unknown split {split!r}; expected one of "
                f"train/val/test/buffer/any")

        self.split      = split
        self.lags       = list(C.LAGS if lags is None else lags)
        self.batch_size = C.BATCH_SIZE if batch_size is None else batch_size
        self.shuffle    = (split == "train") if shuffle is None else shuffle
        self.use_land   = C.USE_LAND_CHANNEL   if use_land  is None else use_land
        self.use_coord  = C.USE_COORD_CHANNELS if use_coord is None else use_coord
        self.use_doy    = C.USE_DOY_CHANNELS   if use_doy   is None else use_doy
        self.anom_target = (C.ANOMALISE_TARGET if anomalise_target is None
                            else anomalise_target)
        self.anom_inputs = dict(C.ANOMALISE_INPUTS if anomalise_inputs is None
                                else anomalise_inputs)

        self.windows = [int(w) for w in (C.RUNNING_MEAN_WINDOWS
                                         if running_mean_windows is None
                                         else running_mean_windows)]

        if self.lags[0] != 0:
            raise ValueError("LAGS[0] must be 0 (the target day)")
        if any(w < 2 for w in self.windows):
            raise ValueError(f"running-mean windows must be >= 2, got {self.windows}")
        self.max_lag = max(self.lags)          # kept for external readers
        # A W-day trailing mean reaches W-1 days back, which can exceed max_lag.
        self.history = C.required_history(self.lags, self.windows)

        self.X = np.load(C.X_NPY, mmap_mode="r")
        self.Y = np.load(C.Y_NPY, mmap_mode="r")

        # Sample indices: every target index belonging to this split that has
        # `max_lag` days of record in front of it.
        #
        # From splits.py, which resolves membership by calendar YEAR against
        # the real date axis. The old form took a contiguous (lo, hi) range,
        # which silently assumes each split is one unbroken block - true for
        # 2005/2021/2022, false for any configurable split.
        #
        # Reading lag context across a split boundary stays legitimate: X.npy
        # is one contiguous record and lags are absolute indices, so a val
        # sample on 1 January reads its t-7 input from a training year. Those
        # are PAST SURFACE OBSERVATIONS, never targets.
        self.indices = S.split_indices(split, self.history)
        if self.indices.size == 0:
            raise ValueError(
                f"split {split!r} is empty under SPLIT_MODE={C.SPLIT_MODE!r}"
                + ("  (buffer_days=0, so no buffer days exist)"
                   if split == "buffer" else ""))

        _, _, land = load_masks()
        self.land = land
        self.lat_p, self.lon_p = _coord_planes()

        dates = np.load(C.DATES_NPY)
        self.doy = pd.DatetimeIndex(dates).dayofyear.values.astype(np.int32)
        ang = 2.0 * np.pi * self.doy.astype(np.float32) / 365.25
        self.doy_sin = np.sin(ang).astype(np.float32)
        self.doy_cos = np.cos(ang).astype(np.float32)

        # ── anomaly machinery ────────────────────────────────────────────────
        self.any_input_anom = any(self.anom_inputs.get(v, False)
                                  for v in C.INPUT_VARS)
        if self.anom_target or self.any_input_anom:
            a = load_anom_stats()
            self.clim_X = np.load(C.CLIM_X_NPY, mmap_mode="r")
            self.clim_Y = np.load(C.CLIM_Y_NPY, mmap_mode="r")
            self.t_mean = a["target_anom_mean"]
            self.t_std  = a["target_anom_std"]
            self.phys_scale = a["phys_scale"]
            self.x_mean = np.array([a["input_anom_mean"][v] for v in C.INPUT_VARS],
                                   dtype=np.float32)
            self.x_std = np.array([a["input_anom_std"][v] for v in C.INPUT_VARS],
                                  dtype=np.float32)
            self.in_anom_flags = np.array(
                [self.anom_inputs.get(v, False) for v in C.INPUT_VARS], bool)
            _, sig = load_norm_arrays()
            self.abs_mu, self.abs_sig = load_norm_arrays()

        # ── running-mean machinery ───────────────────────────────────────────
        # rm{W}.npy holds the climatology-subtracted trailing mean, NOT yet
        # standardised, so the statistics below and the arrays can be
        # regenerated independently. Running means are always of the anomaly,
        # whatever ANOMALISE_INPUTS says - a running mean of raw z is dominated
        # by the seasonal cycle and is not a useful feature under any setting.
        self.rm = {}
        self.rm_mean = {}
        self.rm_std = {}
        if self.windows:
            a = load_anom_stats()
            if not a["input_rm_anom_std"]:
                raise RuntimeError(
                    f"{C.ANOM_STATS_JSON} has no running-mean statistics. "
                    f"Run build_climatology.py to (re)build them.")
            have = [int(w) for w in a.get("running_mean_windows", [])]
            missing = [w for w in self.windows if w not in have]
            if missing:
                raise RuntimeError(
                    f"running-mean windows {missing} requested but "
                    f"{C.ANOM_STATS_JSON} only has {have}. Rebuild with "
                    f"build_climatology.py.")
            for w in self.windows:
                p = C.rm_npy(w)
                if not p.exists():
                    raise FileNotFoundError(
                        f"{p} is missing. Run build_climatology.py.")
                self.rm[w] = np.load(p, mmap_mode="r")
                # JSON forces string keys; cast on the way in.
                self.rm_mean[w] = np.array(
                    [a["input_rm_anom_mean"][str(w)][v] for v in C.INPUT_VARS],
                    dtype=np.float32)
                self.rm_std[w] = np.array(
                    [a["input_rm_anom_std"][str(w)][v] for v in C.INPUT_VARS],
                    dtype=np.float32)

        self.n_channels = C.n_input_channels(
            self.lags, self.use_land, self.use_coord, self.use_doy,
            self.windows)
        self.n_true_channels = C.N_DEPTHS * (2 if self.anom_target else 1)

        self._rng = np.random.default_rng(seed)
        if self.shuffle:
            self._rng.shuffle(self.indices)

    # ── internals ────────────────────────────────────────────────────────────

    def _inputs_at(self, rows):
        """(b,81,101,7) with the flagged variables converted to normalised
        anomalies and the rest left as stored."""
        x = np.asarray(self.X[rows], dtype=np.float32)
        if not self.any_input_anom:
            return x
        cl = np.asarray(self.clim_X[self.doy[rows]], dtype=np.float32)
        f = self.in_anom_flags
        x[..., f] = ((x[..., f] - cl[..., f]) - self.x_mean[f]) / self.x_std[f]
        return x

    def _running_means_at(self, rows, w):
        """(b,81,101,N_INPUT_VARS) standardised trailing w-day mean anomaly.

        The stored array is already climatology-subtracted and averaged, so
        this only has to standardise. Mirrors _inputs_at, but with per-window
        statistics: a 90-day mean of these fields retains 53-79% of the
        instantaneous spread (they are strongly autocorrelated, so it is
        nothing like sigma/sqrt(N)), and reusing input_anom_std would leave
        channels with std 0.53-0.90 - wrong, but quietly so.
        """
        m = np.asarray(self.rm[w][rows], dtype=np.float32)
        return (m - self.rm_mean[w]) / self.rm_std[w]

    # ── Keras API ────────────────────────────────────────────────────────────

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, k):
        idx = self.indices[k * self.batch_size:(k + 1) * self.batch_size]
        b = len(idx)

        X = np.empty((b, C.N_LAT, C.N_LON, self.n_channels), dtype=np.float32)
        c = 0
        for lag in self.lags:
            X[..., c:c + C.N_INPUT_VARS] = self._inputs_at(idx - lag)
            c += C.N_INPUT_VARS
        for w in self.windows:
            X[..., c:c + C.N_INPUT_VARS] = self._running_means_at(idx, w)
            c += C.N_INPUT_VARS
        if self.use_land:
            X[..., c] = self.land; c += 1
        if self.use_coord:
            X[..., c] = self.lat_p; c += 1
            X[..., c] = self.lon_p; c += 1
        if self.use_doy:
            X[..., c] = self.doy_sin[idx][:, None, None]; c += 1
            X[..., c] = self.doy_cos[idx][:, None, None]; c += 1
        assert c == self.n_channels

        y_abs_z = np.asarray(self.Y[idx], dtype=np.float32)
        if not self.anom_target:
            return X, y_abs_z

        clim_z = np.asarray(self.clim_Y[self.doy[idx]], dtype=np.float32)
        anom = ((y_abs_z - clim_z) - self.t_mean) / self.t_std
        clim_degC = clim_z * self.abs_sig + self.abs_mu
        return X, np.concatenate([anom, clim_degC], axis=-1)

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self.indices)

    # ── helpers ──────────────────────────────────────────────────────────────

    def target_dates(self):
        """Target dates in the current index order (evaluation / plotting)."""
        return np.load(C.DATES_NPY)[self.indices]

    def describe(self) -> str:
        d = np.load(C.DATES_NPY)
        on = [v for v in C.INPUT_VARS if self.anom_inputs.get(v, False)]
        return (f"{self.split:5s}  samples={len(self.indices):5d}  "
                f"batches={len(self):4d}  X_ch={self.n_channels}  "
                f"Y_ch={self.n_true_channels}  "
                f"targets {d[self.indices.min()]} .. {d[self.indices.max()]}  "
                f"lags={self.lags}  rm={self.windows or 'none'}  "
                f"history={self.history}  "
                f"target={'anomaly' if self.anom_target else 'absolute'}  "
                f"anom_inputs={on or 'none'}")


def make_datasets(lags=None, batch_size=None, **kw):
    """(train, val, test) datasets sharing one configuration."""
    return tuple(OceanDataset(s, lags=lags, batch_size=batch_size, **kw)
                 for s in ("train", "val", "test"))
