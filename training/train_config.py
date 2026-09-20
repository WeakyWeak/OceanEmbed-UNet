"""
OceanEmbed training pipeline - constants and switches.
This module contains constants only. No logic, no I/O.
"""
import os
from pathlib import Path
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
# ── Dataset version ──────────────────────────────────────────────────────────
# Hey there! We have a few dataset versions:
# v1: 2005-2022, Bay of Bengal (81x101), 7 inputs (including ERA5 winds and GLORYS currents).
#     Superseded - kept only so the old runs in training/runs/ still load.
# v2: 1993-2024, Bay of Bengal + Arabian Sea (81x201), 5 SATELLITE-ONLY inputs.
# v3: 1993-2024, Bay of Bengal ONLY (81x101), the same 5 satellite inputs.
#     THE DEFAULT, and what the backend serves. Just a heads up, this isn't a
#     new run—build_arrays just crops v2's record.nc.
#
# So, why do we have v3? Well, it turns out that sharing filters across two basins
# actually hurt performance more than the extra 14 years of data helped! v3 keeps
# those extra years but drops the second basin to keep things running smoothly.
#
# The default is v3, so a fresh clone needs NO environment variable anywhere:
# downloading, cleaning, training and serving all agree out of the box. Set it
# explicitly ONLY to reach a superseded version, e.g. to re-score an old run:
#
#     OCEAN_DATASET_VERSION=v1 .venv/bin/python training/evaluate.py --run primary_best
DATASET_VERSION = os.environ.get("OCEAN_DATASET_VERSION", "v3")
if DATASET_VERSION not in ("v1", "v2", "v3"):
    raise ValueError(f"OCEAN_DATASET_VERSION must be v1, v2 or v3, "
                     f"got {DATASET_VERSION!r}")
_V3 = DATASET_VERSION == "v3"
# When you see _V2, it means we're using the 1993-2024 satellite-only record. 
# Since v3 is just a cropped version of it, most of the _V2 settings (like inputs and splits) 
# work perfectly for v3 too! The only things that change are the grid and a few specifics.
_V2 = DATASET_VERSION in ("v2", "v3")

PROC_DIR     = PROJECT_ROOT / "data" / ("processed_v2" if _V2 else "processed")
ARRAYS_DIR   = TRAINING_DIR / {"v1": "arrays", "v2": "arrays_v2",
                               "v3": "arrays_v3"}[DATASET_VERSION]
RUNS_DIR     = TRAINING_DIR / "runs"

# With v2, we ship a single record and handle splits later on. But don't worry, 
# v1 keeps its original three split files so older models can still load without a hitch!
RECORD_NC  = PROC_DIR / "record.nc"
TRAIN_NC   = PROC_DIR / "train.nc"
VAL_NC     = PROC_DIR / "val.nc"
TEST_NC    = PROC_DIR / "test.nc"
MASK_NC    = PROC_DIR / "land_mask.nc"
STATS_JSON = PROC_DIR / "normalization_stats.json"

# ── Variable order (Please keep this FIXED! Our channels depend on it 😅)
# A quick tip: avoid iterating over ds.data_vars, as NetCDF sometimes sneaks in 
# extra variables that can mess things up!
# v2 is satellite-only: ERA5 u10/v10 are atmospheric reanalysis and GLORYS
# uo/vo are ocean-model output. ugos/vgos are DUACS geostrophic velocity,
# derived from the altimeter's own SSH field, and replace uo/vo.
INPUT_VARS = (["sst", "sla", "sss", "ugos", "vgos"] if _V2 else
              ["sst", "sla", "sss", "uo", "vo", "u10", "v10"])

DEPTH_LABELS = ["0m", "5m", "10m", "20m", "30m", "50m", "75m", "100m",
                "125m", "150m", "200m", "300m", "500m", "700m", "1000m"]
TARGET_VARS  = [f"thetao_{lbl}" for lbl in DEPTH_LABELS]

N_INPUT_VARS = len(INPUT_VARS)    # 7
N_DEPTHS     = len(DEPTH_LABELS)  # 15
N_PAIRS      = N_DEPTHS - 1       # 14

# Nominal label -> actual GLORYS12V1 native z-level (metres).
# Use these for any comparison against in-situ / ARGO observations.
ACTUAL_DEPTH_M = [0.4940, 5.0782, 9.5730, 18.4956, 29.4447, 47.3737, 77.8539,
                  92.3261, 130.6660, 155.8507, 186.1256, 318.1274, 541.0889,
                  643.5668, 1062.4399]

# ── Grid ─────────────────────────────────────────────────────────────────────
if _V2:
    # 5-25 N, 50-100 E - adds the Arabian Sea. Single-sourced from
    # data_cleaning/grid.py so the grid cannot be declared in two places and
    # silently disagree.
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "data_cleaning"))
    import grid as _G
    TARGET_LAT, TARGET_LON = _G.TARGET_LAT, _G.TARGET_LON
    if _V3:
        # Bay of Bengal: the v2 grid from 75.0 E eastward. Sliced from the
        # grid.py axis rather than re-declared, so the values are bit-identical
        # to the record's and build_arrays' coordinate assert stays exact.
        TARGET_LON = TARGET_LON[TARGET_LON >= 75.0]
    N_LAT, N_LON = len(TARGET_LAT), len(TARGET_LON)      # 81, 201 (v3: 81, 101)
    PAD_LAT = (8, 7)
    # Pad to multiples of 16 so depth 3 and 4 both work:
    #   v2 81x201 -> 96x208      v3 81x101 -> 96x112 (the same padding as v1)
    PAD_LON = (6, 5) if _V3 else (4, 3)
else:
    N_LAT, N_LON = 81, 101
    TARGET_LAT = np.arange(5.0, 25.25, 0.25)
    TARGET_LON = np.arange(75.0, 100.25, 0.25)
    # Pad 81x101 -> 96x112 so both depth 3 (/8) and depth 4 (/16) divide evenly.
    PAD_LAT = (8, 7)
    PAD_LON = (6, 5)

PADDED_SHAPE = (N_LAT + sum(PAD_LAT), N_LON + sum(PAD_LON))
MAX_UNET_DEPTH = 4

# ── Temporal record & splits (by TARGET index into the contiguous array) ─────
N_TRAIN_DAYS, N_VAL_DAYS, N_TEST_DAYS = 5844, 365, 365
N_DAYS = N_TRAIN_DAYS + N_VAL_DAYS + N_TEST_DAYS               # 6574
if _V2:
    # Derived from the date axis, never declared. 1993-01-01 .. 2024-12-15 with
    # the SSS end date binding = 11,672 days. build_arrays asserts the record
    # it actually opens matches this, so a short assembly cannot slip through.
    import datetime as _dt
    N_DAYS = (_dt.date(2024, 12, 15) - _dt.date(1993, 1, 1)).days + 1
    N_TRAIN_DAYS = N_VAL_DAYS = N_TEST_DAYS = None   # meaningless under v2

def record_chunks():
    """(name, path, expected_days) chunks build_arrays concatenates, in order.

    v2 is a single record; v1 is the three legacy split files.
    `None` for expected_days means "take whatever the file has".
    """
    if _V2:
        return [("record", RECORD_NC, N_DAYS)]
    return [("train", TRAIN_NC, N_TRAIN_DAYS),
            ("val",   VAL_NC,   N_VAL_DAYS),
            ("test",  TEST_NC,  N_TEST_DAYS)]

if not _V2:
    SPLIT_RANGES = {                  # [start, stop) target indices
        "train": (0, N_TRAIN_DAYS),                            # 0    .. 5843
        "val":   (N_TRAIN_DAYS, N_TRAIN_DAYS + N_VAL_DAYS),    # 5844 .. 6208
        "test":  (N_TRAIN_DAYS + N_VAL_DAYS, N_DAYS),          # 6209 .. 6573
    }
# Under v2 the name is simply ABSENT. That is the deletion the migration note
# below describes, scoped to the version that no longer needs it: any v2 call
# site still reaching for a contiguous index range raises AttributeError
# instead of silently reinterpreting v1's tuples against a 32-year record.
# DEPRECATED. Superseded by SPLIT_MODE / SPLIT_CONFIGS below, which express
# membership as calendar YEARS and derive indices from the actual date array.
# Kept only so the currently-deployed models and backend keep working until the
# expanded record lands; it is deleted (not redefined) at that point, so any
# call site still reading it fails loudly instead of silently reinterpreting a
# tuple. Everything new must go through training/splits.py.

# ── Split configuration (training/splits.py) ─────────────────────────────────
# Membership by calendar year. Held-out years are whole years so every split
# sees a complete seasonal cycle - half-years risk never evaluating on the
# southwest monsoon, which is the dominant season in this basin.
# v1's only meaningful split is the 2021/2022 one its models were trained on.
# v2 defaults to chronological: held-out years at the END of the record, which
# is the harder test and the one a reviewer expects. Override per run with
# --split-mode.
SPLIT_MODE = "chronological" if _V2 else "legacy"

SPLIT_CONFIGS = {
    # Reproduces SPLIT_RANGES exactly on the 2005-2022 record. Asserted in
    # smoke tests; do not edit.
    "legacy": {
        "val_years":  [2021],
        "test_years": [2022],
        "buffer_days": 0,
    },
    # DEFAULT once the 1993-2024 record exists. Held-out period at the END,
    # as a reviewer expects, and as "can this model be used going forward"
    # actually means. Training on 28 years instead of 16 also defuses most of
    # the trend-extrapolation penalty that motivated interleaving.
    "chronological": {
        "val_years":  [2021, 2022],
        "test_years": [2023, 2024],     # 2024 is partial: 350 days
        "buffer_days": 0,               # no interior boundaries to buffer
    },
    # EXPERIMENT. The warming trend is interpolated rather than extrapolated.
    # Leakier by construction - a held-out year between two training years
    # shares low-frequency ocean state with them. Years chosen so each is
    # bracketed by training years, 1993 stays in train (the only year without
    # lag history), and the extreme IOD years (1994, 1997, 2006, 2019) plus the
    # 2015-16 super El Nino stay in train so val/test are never scored on
    # regimes the model has not seen. Check against ENSO/IOD indices before
    # trusting the specific years.
    "interleaved": {
        "val_years":  [2000, 2013],
        "test_years": [2004, 2017],
        "buffer_days": 45,
    },
}

# ── Input construction ───────────────────────────────────────────────────────
# LAGS[0] must be 0 (the target day). Lag L reads X[t - L].
#
# Channel count is (len(LAGS) + len(RUNNING_MEAN_WINDOWS)) * N_INPUT_VARS
# + 1 land + 2 coord + 2 day-of-year, i.e. n_input_channels() below - do not
# trust a hardcoded number here, the helper is the authority. For the record,
# under the current switches:
#
#                              v2/v3 (5 satellite vars)   v1 (7 vars, superseded)
#   [0]                              10 channels                  12
#   [0,1,2]                          20 channels                  26
#   [0,3,7]                          20 channels                  26
#   [0,3,7] + windows [30,90]        30 channels                   -   <- PRIMARY
#
# All three-lag options cost exactly the same width; [0,3,7] is chosen because
# it spans more mixed-layer / Ekman memory for the same price. [0] is the
# no-history control experiment.
LAGS = [0, 3, 7]

# Trailing running-mean windows, in days, appended as extra per-variable blocks.
#
# WHY: [0,3,7] spans one week. Satellite altimetry only constrains ocean
# variability at periods longer than ~20-30 days (TS-Cast, Ocean Science 2026,
# uses a 31-day window; the ESSD attention 3D-U-Net++ uses 26), so a one-week
# span sits BELOW the timescale at which SLA carries information about the deep
# ocean - which is exactly where our skill collapses (4% at 1000 m).
#
# Measured correlation against the lag-0 anomaly on train days:
#
#                     sst    sla    sss   ugos   vgos
#   lag 7 (existing) 0.513  0.845  0.707  0.778  0.767
#   rm30             0.622  0.741  0.675  0.638  0.575
#   rm90             0.530  0.534  0.483  0.353  0.298
#
# rm90 is far more decorrelated from lag 0 than the lag-7 channel we already
# pay for. If RAM ever forces a cut, drop 30 and KEEP 90 - rm30 is 0.62-0.80
# correlated with rm90, so it is the redundant one.
#
# Setting this to [] restores the historical 20-channel pipeline exactly, and
# required_history() falls back to max(LAGS). That is the escape hatch.
RUNNING_MEAN_WINDOWS = [30, 90]

USE_LAND_CHANNEL  = True     # binary 0/1, never normalised
USE_COORD_CHANNELS = True    # lat_norm, lon_norm  (CoordConv)
USE_DOY_CHANNELS   = True    # sin/cos of day-of-year


def n_input_channels(lags=None, land=None, coord=None, doy=None,
                     windows=None) -> int:
    lags    = LAGS if lags is None else lags
    land    = USE_LAND_CHANNEL if land is None else land
    coord   = USE_COORD_CHANNELS if coord is None else coord
    doy     = USE_DOY_CHANNELS if doy is None else doy
    windows = RUNNING_MEAN_WINDOWS if windows is None else windows
    return ((len(lags) + len(windows)) * N_INPUT_VARS
            + int(land) + 2 * int(coord) + 2 * int(doy))


def required_history(lags=None, windows=None) -> int:
    """Days of record a sample needs in front of it: lag reach or window reach.

    A lag of L reads X[t-L], so it needs L days. A trailing W-day mean covers
    [t-W+1, t], so it needs W-1. With windows=[] this returns max(LAGS) = 7,
    byte-identical to the pre-running-mean pipeline.

    This is PER MEMBER, not global: the shipped 20-channel models need only 7
    days and must not lose servable dates because a 90-day model exists.
    """
    lags    = LAGS if lags is None else lags
    windows = RUNNING_MEAN_WINDOWS if windows is None else windows
    return int(max(max(lags), max(windows, default=1) - 1))


# ── Anomaly formulation ──────────────────────────────────────────────────────
# We always convert our targets to anomalies! If we didn't do this, our poor model
# would spend all its energy just trying to relearn the normal seasonal cycles. 
# This simple trick helped us predict tricky anomalies much better!
ANOMALISE_TARGET = True

# Per-input switches. Measured on the train split, predicting the target
# anomaly with a linear model and scoring on 2021:
#
#   variable   seasonal share of variance   delta 2021 R^2 if left RAW (100m)
#   sla                42.5%                       -0.4920   <- decisive
#   sst                76.8%                       +0.0015
#   sss                92.0%                       +0.0062
#   uo                 41.5%                       -0.0044
#   vo                 26.4%                       -0.0002
#   u10                72.1%                       -0.0012
#   v10                71.7%                       -0.0001
#
# SLA is the whole story. Although DUACS SLA is already an "anomaly" product,
# it is an anomaly against a fixed 1993-2012 reference and still carries its
# full seasonal cycle; removing the day-of-year climatology isolates the
# interannual/trend component that actually tracks the thermocline.
# Anomalising everything EXCEPT sla scores -0.321 on 2021 at 100m - identical
# to doing nothing (-0.319). Anomalising sla as well gives +0.171.
#
# The other six are neutral to a LINEAR probe. They may still help a CNN, which
# unlike a linear model can waste capacity memorising seasonal variance - flip
# them on to test. Watch sss: it is 92% seasonal, so its anomaly is only 8% of
# the original variance and may be largely product noise (see
# analysis/claude.md section 1).
ANOMALISE_INPUTS = {
    "sst": True,
    "sla": True,      # keep on - this is the one with direct evidence, for the anom_v1 model. 
    "sss": True,
    "uo":  True,
    "vo":  True,
    "u10": True,
    "v10": True,
    # v2. ugos/vgos are geostrophic velocity derived from the same SSH field as
    # sla, so the argument that made anomalising sla decisive applies to them by
    # construction: they inherit its fixed 1993-2012 reference and its seasonal
    # cycle. Starting them ON for that reason, but this is INHERITED REASONING,
    # not a measurement - re-run the linear probe on the v2 arrays before
    # quoting it as evidence.
    "ugos": True,
    "vgos": True,
}

# Our climatology is smoothed out nicely to reduce any sampling noise. 
# We use a 15-day window, which also magically smooths over those pesky leap-year bumps!
CLIM_SMOOTH_HALF_WINDOW = 7

# y_true carries 30 channels when ANOMALISE_TARGET is on:
#   [...,  :15] normalised anomaly target (what the model predicts)
#   [..., 15:30] that sample's climatology in physical degC, so the physics
#                loss can rebuild absolute temperature. Vertical ordering is a
#                statement about ABSOLUTE temperature - comparing anomalies
#                between depths is meaningless.
# The MODEL still outputs 15 channels. Only the label tensor is wider.
N_TRUE_CHANNELS = N_DEPTHS * 2 if ANOMALISE_TARGET else N_DEPTHS


# ── Physics loss gate ────────────────────────────────────────────────────────
# One weight per adjacent depth pair, derived from how often the GLORYS target
# ITSELF violates T_deep <= T_shallow (stride-3 sample of the train split,
# 6.8-8.9M valid cell-days per pair, measured in physical degrees C):
#
#   pair          mean|dT|  inverts   >0.2C    >0.5C   weight
#   0m  -> 5m      0.078 C  13.68%    2.20%    1.00%     0.0
#   5m  -> 10m     0.056 C  27.45%    4.79%    1.82%     0.0
#   10m -> 20m     0.130 C  40.06%   10.20%    3.88%     0.0
#   20m -> 30m     0.256 C  36.69%   10.58%    3.67%     0.0
#   30m -> 50m     0.767 C  21.97%    8.97%    3.18%     0.0
#   50m -> 75m     2.668 C   2.37%    1.22%    0.45%     0.3
#   75m -> 100m    1.852 C   0.39%    0.13%    0.03%     0.3
#   100m-> 125m    4.804 C   0.019%   0.010%   0.005%    1.0
#   125m-> 150m    2.279 C   0.021%   0.004%   0.002%    1.0
#   150m-> 200m    1.818 C   0.012%   0.002%   0.0005%   1.0
#   200m-> 300m    3.149 C   0.000%   0%       0%        1.0   (1 cell / 6.78M)
#   300m-> 500m    1.843 C   0.002%   0.000%   0%        1.0
#   500m-> 700m    0.765 C   0.022%   0.004%   0.001%    1.0
#   700m-> 1000m   2.589 C   0.000%   0%       0%        1.0   (6 cells / 6.78M)
#
# 0-10 m is a near-isothermal mixed layer (the sign of a ~0.06 C gradient is
# noise). 10-50 m carries genuine Bay of Bengal barrier-layer inversions -
# real physics we want the model to reproduce, not penalise.
PHYSICS_GATE = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0,      # 0->5, 5->10, 10->20, 20->30, 30->50
     0.3, 0.3,                     # 50->75, 75->100
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # 100->125 ... 700->1000
    dtype=np.float32)
assert len(PHYSICS_GATE) == N_PAIRS

# margin = 0. A positive margin would have to be depth-gap aware: the pairs
# span 4.5 m (0m->5m) to 419 m (700m->1000m).
PHYSICS_MARGIN = 0.0

# ── Model defaults (overridden by the tuner) ─────────────────────────────────
DEFAULT_HP = {
    "base_filters":   32,
    "depth":           4,
    "kernel_size":     3,      # fixed, not searched
    "dropout":       0.1,      # spatial, bottleneck + deepest encoder only
    "learning_rate": 3e-4,
    "lambda_physics": 1e-3,
    "weight_decay":   1e-4,
}
GROUP_NORM_GROUPS = 8
ACTIVATION = "swish"

# Ablation: index of the shallowest level held under a hard monotonic
# constraint. 7 == the 100m level (violations there are <= 0.022%).
# None disables it. Off by default; see spec section 2.3.
HARD_MONOTONIC_BELOW = None

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16      # fixed, not searched
EPOCHS_FINAL    = 100
EPOCHS_TUNING   = 40
PATIENCE_FINAL  = 12
PATIENCE_TUNING = 8
SEED            = 42

# ── Tuner ────────────────────────────────────────────────────────────────────
TUNER_TRIALS         = 25
TUNER_INITIAL_POINTS = 6

if _V2:
    # Sized from MEASURED v2 throughput, not from the v1 numbers:
    #   639 batches/epoch at batch_size 16
    #   2.2 min/epoch at base 32 depth 3 ... 2.9 min/epoch at base 64 depth 4
    # so a trial that early-stops around epoch 6 with patience 6 costs roughly
    # 26-35 min, and a 6-hour night fits about 12-16 trials.
    #
    # TUNER_TRIALS is the CAP, not the plan. --max-hours is the real bound:
    # it stops cleanly on a deadline instead of being killed mid-trial, and
    # the search is resumable, so an unfinished budget can be continued.
    EPOCHS_TUNING        = 30
    PATIENCE_TUNING      = 6
    TUNER_TRIALS         = 20
    TUNER_INITIAL_POINTS = 7    # random seeding before the GP takes over;
                                # ~40% of the realistic budget, which is about
                                # right for 5 dimensions
# Must be the DATA term alone. lambda_physics varies across trials, so
# val_loss is not comparable between them.
TUNER_OBJECTIVE = "val_masked_mse"

# Revised after the first search. The previous space produced 22 of 23 trials
# inside a 7% band (0.312-0.335) and converged on base_filters=64 with
# dropout=0.1 - because every config overfitted by epoch 2, so the tuner was
# effectively ranking configs on "who fits fastest in one epoch", which rewards
# maximum capacity and minimum regularisation. Capacity is now shifted down,
# dropout up, weight decay added, and lambda_physics pinned (it was already
# doing its job at 1e-3) to free a dimension.
SEARCH_SPACE_V1 = {
    "base_filters":  [16, 24, 32, 48],
    "depth":         [3, 4],
    "learning_rate": (1e-4, 3e-3),           # log-uniform
    "dropout":       [0.1, 0.2, 0.3, 0.4],
    "weight_decay":  (1e-6, 1e-2),           # log-uniform, decoupled (AdamW)
}

# v2 space. Every change below is a response to something the v1 searches
# actually did, not a general "try bigger".
#
# base_filters: the tuner picked the MAXIMUM offered value (48) in BOTH v1
#   searches. That is the signature of a truncated space - the optimum was
#   somewhere outside it and the search was pressed against the wall. It was
#   deliberately shifted DOWN for 5,837 training days on one basin; there are
#   now 10,227 days over 2.41x the ocean cells, so it is shifted back up.
#   Capacity was never the binding constraint before (every config overfit
#   within 1-3 epochs), but that is exactly what more data changes.
#   Measured cost at the ceiling: 80/depth4 is ~48M params, ~5 GB peak RSS
#   against ~10 GB available, so the top of this range is affordable.
#
# depth: still [3, 4]. The padded grid is (96, 208); depth 4 divides it by 16
#   to (6, 13). Depth 5 would need /32 and 208/32 = 6.5, so it is not
#   representable without re-padding - a real constraint, not a choice.
#
# dropout: 0.0 added, 0.4 dropped. Dropout is a response to overfitting, and
#   the whole point of the expansion was to attack overfitting at the source.
#   The search should be allowed to conclude it needs none.
#
# learning_rate: ceiling raised 3e-3 -> 5e-3. The two best v1 runs landed at
#   3.0e-3 and 3.65e-3 - i.e. at or above the old ceiling, so the old range
#   was clipping the optimum here too.
#
# NOT searched, on purpose:
#   kernel_size stays 3. A wider kernel is one way to attack the receptive
#   field over a domain that is now 2x wider - but bottleneck attention is
#   the NEXT experiment and attacks the same thing. Searching both would
#   confound the A/B that experiment exists to answer.
#   batch_size stays 16: it trades off against learning_rate, and a search
#   this size cannot afford to spend trials untangling the two.
SEARCH_SPACE_V2 = {
    "base_filters":  [32, 48, 64, 80],
    "depth":         [3, 4],
    "learning_rate": (2e-4, 5e-3),           # log-uniform
    "dropout":       [0.0, 0.1, 0.2, 0.3],
    "weight_decay":  (1e-6, 1e-2),           # log-uniform, decoupled (AdamW)
}

# v3 space. The two previous searches pulled in opposite directions, and v3
# sits between the problems they solved:
#   v1 (BoB, 16 train years) picked base_filters 48 = the MAXIMUM offered, and
#      learning rates at the ceiling.
#   v2 (two basins, 28 train years) picked base_filters 32 = the MINIMUM
#      offered, with weight_decay 3.3e-3 near the top of its range.
# v3 is v1's area with v2's years, so the space is centred on 32-48 with a
# step either side, and weight decay starts higher than v1's floor because
# both winners wanted real regularisation.
# Cost: half v2's pixels, so roughly half v2's 2.2-2.9 min/epoch.
SEARCH_SPACE_V3 = {
    "base_filters":  [24, 32, 48, 64],
    "depth":         [3, 4],
    "learning_rate": (3e-4, 5e-3),           # log-uniform
    "dropout":       [0.1, 0.2, 0.3],
    "weight_decay":  (1e-5, 1e-2),           # log-uniform, decoupled (AdamW)
}
if _V3:
    # Half the pixels -> about twice the trials in the same night. The cap is
    # raised so --max-hours, not the cap, is what ends the search.
    TUNER_TRIALS = 30

SEARCH_SPACE = (SEARCH_SPACE_V3 if _V3 else
                SEARCH_SPACE_V2 if _V2 else SEARCH_SPACE_V1)
# Pinned, not searched.
FIXED_LAMBDA_PHYSICS = 1e-3

# ── Generated artefact paths ─────────────────────────────────────────────────
X_NPY           = ARRAYS_DIR / "X.npy"
Y_NPY           = ARRAYS_DIR / "Y.npy"
TARGET_MASK_NPY = ARRAYS_DIR / "target_mask.npy"
PAIR_MASK_NPY   = ARRAYS_DIR / "pair_mask.npy"
LAND_MASK_NPY   = ARRAYS_DIR / "land_mask.npy"
DATES_NPY       = ARRAYS_DIR / "dates.npy"
META_JSON       = ARRAYS_DIR / "meta.json"
CLIM_X_NPY      = ARRAYS_DIR / "clim_X.npy"      # (367,81,101,7)  z-space
CLIM_Y_NPY      = ARRAYS_DIR / "clim_Y.npy"      # (367,81,101,15) z-space
ANOM_STATS_JSON = ARRAYS_DIR / "anom_stats.json"


def rm_npy(window: int):
    """Path to the trailing `window`-day running-mean array.

    Shape (N_DAYS, N_LAT, N_LON, N_INPUT_VARS), float16, holding the
    CLIMATOLOGY-SUBTRACTED mean, pre-standardisation. Written by
    build_climatology.py; see its docstring for why the order matters.
    """
    return ARRAYS_DIR / f"rm{int(window)}.npy"

# Expected valid ocean cells per depth. 
# If you're building v2 for the first time, you won't know these yet! Just leave 
# it as None, let the script print them out, and then paste them back here to lock it in. 
# No need to guess and confuse our tests! 😊
EXPECTED_VALID_CELLS_V2 = [10974, 10974, 10919, 10771, 10609, 10419,
                           10184, 10047, 9868, 9822, 9773, 9671,
                           9537, 9481, 9273]   # measured 2026-09-11

# v3 = arrays_v2's target_mask[:, 100:, :] summed, measured 2026-09-13.
# 1-4 cells per depth more than v1: all on the 75.0 E column, which v1 built
# from a 2-cell edge stencil and v2's wider box builds from a full one.
EXPECTED_VALID_CELLS_V3 = [4547, 4547, 4503, 4407, 4320, 4193, 4052, 3984,
                           3875, 3842, 3812, 3751, 3654, 3612, 3482]

EXPECTED_VALID_CELLS_V1 = [4545, 4545, 4500, 4403, 4319, 4192, 4050, 3983,
                        3874, 3841, 3811, 3750, 3653, 3611, 3482]

EXPECTED_VALID_CELLS = {"v1": EXPECTED_VALID_CELLS_V1,
                        "v2": EXPECTED_VALID_CELLS_V2,
                        "v3": EXPECTED_VALID_CELLS_V3}[DATASET_VERSION]
