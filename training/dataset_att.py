"""
Input-subsetting dataset. Shared by the attention model and models/.

Why this exists
---------------
OceanDataset always stacks all seven surface variables. The attention model is
deliberately SATELLITE-ONLY - it drops uo/vo (GLORYS reanalysis) and u10/v10
(ERA5 reanalysis) so it can carry the problem statement's actual objective,
"estimate the three-dimensional ocean temperature using only surface satellite
observations". The three that remain all come from satellite products:

    sst   NOAA OISST v2.1        (AVHRR, in-situ bias correction)
    sla   Copernicus DUACS L4    (multi-mission satellite altimetry)
    sss   Copernicus Multi-Obs   (SMOS / SMAP based)

An input ablation on anom_v1_best measured what that costs. Zeroing each
variable and re-scoring 2021:

    removed          val RMSE     cost
    sla               1.0372     +74.7%      <- the whole thermocline signal
    sst               0.6812     +14.7%
    sss               0.6262      +5.4%
    uo + vo           0.6183      +4.1%      GLORYS  - dropped here
    u10 + v10         0.6035      +1.6%      ERA5    - dropped here
    all four          0.6243      +5.1%

So the reanalysis inputs are worth about 5% RMSE on a model that was TRAINED
with them; a model trained without them should recover much of that. That is a
cheap price for a compliant headline claim, and it removes a real awkwardness:
uo/vo come from the same GLORYS run that produces the target, so surface
currents and subsurface temperature are dynamically consistent through the
ocean model's own equations in a way no true satellite pipeline would be.

SubsetOceanDataset(input_vars=all seven) is exactly equivalent to
OceanDataset - asserted in smoke_test_att.py - so models/ can use this one
class for every member of the ensemble regardless of how it was trained.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import splits as S
from dataset import OceanDataset

# The satellite-only input set.
SATELLITE_VARS = ["sst", "sla", "sss"]


class SubsetOceanDataset(OceanDataset):
    """OceanDataset restricted to a chosen subset of the seven input variables.

    Extra arguments
    ---------------
    input_vars : list[str]   subset of C.INPUT_VARS, in C.INPUT_VARS order.
    indices    : array-like  explicit target indices, overriding the split's
                             default range. Used for serving single dates and
                             for evaluation on an arbitrary window; the split
                             argument still decides shuffling and bounds
                             checking, so pass indices that belong to it.

    Channel order is unchanged apart from the dropped variables:
        [vars @ t] [vars @ t-lag1] ... [vars rm W0] [vars rm W1] ...
        [land] [lat] [lon] [sin doy] [cos doy]
    """

    def __init__(self, split, input_vars=None, indices=None, **kw):
        iv = list(C.INPUT_VARS if input_vars is None else input_vars)
        unknown = [v for v in iv if v not in C.INPUT_VARS]
        if unknown:
            raise ValueError(
                f"unknown input variables {unknown}; choose from {C.INPUT_VARS}"
            )
        if not iv:
            raise ValueError("input_vars must not be empty")
        # Keep canonical order so a saved model's channel axis is reproducible
        # from the variable list alone.
        self.input_vars = [v for v in C.INPUT_VARS if v in iv]
        self.var_idx = np.array([C.INPUT_VARS.index(v) for v in self.input_vars])
        self.n_vars = len(self.input_vars)

        super().__init__(split, **kw)

        # Recompute: the parent assumed all seven. Running-mean blocks are the
        # same width as lag blocks, so they enter the formula the same way.
        self.n_channels = (
            (len(self.lags) + len(self.windows)) * self.n_vars
            + int(self.use_land)
            + 2 * int(self.use_coord)
            + 2 * int(self.use_doy)
        )

        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            # SET MEMBERSHIP, not a min/max range test.
            #
            # This was `idx.min() >= lo and idx.max() < hi`, which is only
            # equivalent to membership while a split is one contiguous block.
            # The moment val/test are non-contiguous, every training day
            # sitting in the gap between two held-out years passes a range
            # test - silent contamination with no error raised. Measured on a
            # two-year probe config, 1095 training days sit inside the val
            # min/max span. splits.py is the authority on membership.
            allowed = S.split_indices(self.split, self.history)
            bad = idx[~np.isin(idx, allowed)]
            if bad.size:
                raise ValueError(
                    f"{bad.size} index(es) do not belong to split "
                    f"{self.split!r} (e.g. {bad[:5].tolist()}); "
                    f"split has {allowed.size} usable days"
                )
            self.indices = idx

    def __getitem__(self, k):
        idx = self.indices[k * self.batch_size : (k + 1) * self.batch_size]
        b = len(idx)

        X = np.empty((b, C.N_LAT, C.N_LON, self.n_channels), dtype=np.float32)
        c = 0
        for lag in self.lags:
            # _inputs_at applies the per-variable anomaly transform on the full
            # seven before the subset is taken, so the transform is identical
            # to what the full-input models see.
            X[..., c : c + self.n_vars] = self._inputs_at(idx - lag)[..., self.var_idx]
            c += self.n_vars
        for w in self.windows:
            # Same subsetting expression as the lag blocks - which is exactly
            # why the running means go here and not after the land channel.
            X[..., c : c + self.n_vars] = self._running_means_at(idx, w)[
                ..., self.var_idx
            ]
            c += self.n_vars
        if self.use_land:
            X[..., c] = self.land
            c += 1
        if self.use_coord:
            X[..., c] = self.lat_p
            c += 1
            X[..., c] = self.lon_p
            c += 1
        if self.use_doy:
            X[..., c] = self.doy_sin[idx][:, None, None]
            c += 1
            X[..., c] = self.doy_cos[idx][:, None, None]
            c += 1
        assert c == self.n_channels

        y_abs_z = np.asarray(self.Y[idx], dtype=np.float32)
        if not self.anom_target:
            return X, y_abs_z

        clim_z = np.asarray(self.clim_Y[self.doy[idx]], dtype=np.float32)
        anom = ((y_abs_z - clim_z) - self.t_mean) / self.t_std
        clim_degC = clim_z * self.abs_sig + self.abs_mu
        return X, np.concatenate([anom, clim_degC], axis=-1)

    def describe(self) -> str:
        on = [
            v
            for v in C.INPUT_VARS
            if self.anom_inputs.get(v, False) and v in self.input_vars
        ]
        d = np.load(C.DATES_NPY)
        return (
            f"{self.split:5s}  samples={len(self.indices):5d}  "
            f"batches={len(self):4d}  X_ch={self.n_channels}  "
            f"Y_ch={self.n_true_channels}  "
            f"vars={'+'.join(self.input_vars)}  "
            f"targets {d[self.indices.min()]} .. {d[self.indices.max()]}  "
            f"lags={self.lags}  "
            f"target={'anomaly' if self.anom_target else 'absolute'}  "
            f"anom_inputs={on or 'none'}"
        )
