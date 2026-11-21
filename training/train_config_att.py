"""
Configuration for the satellite-only attention model.

Everything not named here is inherited from train_config.py by importing it:
grid, splits, lags, anomaly formulation, physics gate, climatology, paths.
This module only records what DIFFERS, so the two runs cannot silently drift
apart on the things that must stay identical (the target, the masks, the
normalisation, the split boundaries).

Two changes from the primary model, both deliberate.

1. SATELLITE-ONLY INPUTS
   sst + sla + sss. See dataset_att.py for the ablation that priced this at
   about 5% RMSE. It buys the problem statement's stated objective - "using
   only surface satellite observations" - which no other model in the lineup
   can claim, and it removes uo/vo, which come from the same GLORYS run as the
   target.

2. ATTENTION
   Listed in the problem statement under the architectures to use
   ("Attention-based hybrid architectures"), and independently motivated by a
   measured deficiency. The effective receptive field of the primary U-Net is
   50% within 7x10 px and only 1% out at 60x65 px, so the network is close to
   local. But the thermocline responds to BASIN-SCALE sea level: a one
   parameter linear fit on domain-mean SLA recovered about 60% of the 2021
   deep warm anomaly, while the U-Net recovered about 30%. The network is
   under-using the single input that matters most. Self-attention at the
   bottleneck gives every location access to the entire basin in one hop,
   which is the specific thing a stack of 3x3 convolutions cannot do cheaply.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_config import *          # noqa: F401,F403  - shared constants
import train_config as _base

# ── inputs ───────────────────────────────────────────────────────────────────
# v1: the attention run was satellite-only when the main model was not, which
#     is exactly what made att_v1's result uninterpretable - it changed the
#     architecture AND dropped 7 inputs -> 3 at the same time.
# v2: the whole dataset is satellite-only, so the attention model takes the
#     SAME five inputs as v2_primary_best. That is the point: with inputs,
#     data and split held constant, a difference in score is attributable to
#     the architecture and nothing else.
# v3 inherits the v2 behaviour: same satellite-only record, cropped.
INPUT_VARS_ATT = (list(_base.INPUT_VARS) if _base.DATASET_VERSION != "v1"
                  else ["sst", "sla", "sss"])
N_INPUT_VARS_ATT = len(INPUT_VARS_ATT)

# Anomalise all three. anom_v2 (all seven anomalised) beat anom_v1 (sla only)
# in the top 30 m - 0 m RMSE 0.333 vs 0.360, ACC 0.640 vs 0.570 - and lost
# only at the thermocline, where the loss was traced to the currents and winds
# being anomalised, not sst/sss. With those gone, the surface gain is worth
# taking. sla must be on: leaving it raw costs -0.49 R^2 at 100 m.
# Under v2 this must MATCH the base config exactly, or the A/B is measuring
# the anomaly formulation instead of the architecture.
ANOMALISE_INPUTS_ATT = (dict(_base.ANOMALISE_INPUTS)
                        if _base.DATASET_VERSION != "v1" else
                        {"sst": True, "sla": True, "sss": True,
                         "uo": False, "vo": False, "u10": False, "v10": False})


def n_input_channels_att(lags=None, land=None, coord=None, doy=None,
                         windows=None) -> int:
    lags    = _base.LAGS if lags is None else lags
    land    = _base.USE_LAND_CHANNEL   if land  is None else land
    coord   = _base.USE_COORD_CHANNELS if coord is None else coord
    doy     = _base.USE_DOY_CHANNELS   if doy   is None else doy
    windows = _base.RUNNING_MEAN_WINDOWS if windows is None else windows
    return ((len(lags) + len(windows)) * N_INPUT_VARS_ATT + int(land)
            + 2 * int(coord) + 2 * int(doy))


# Under v2/v3 this is 5 vars, so lags [0,3,7] with no running means gives
# 3*5 + 1 land + 2 coord + 2 doy = 20 channels, and adding windows [30,90]
# gives 30. (The old comment here claimed 14, which was the v1 three-variable
# count and has been wrong since v2.) The helper is the authority.
N_CHANNELS_ATT = n_input_channels_att()

# ── attention ────────────────────────────────────────────────────────────────
# Bottleneck spatial size is 96/2^d x 112/2^d: 12x14 = 168 tokens at depth 3,
# 6x7 = 42 at depth 4. Both are tiny, so full self-attention is affordable -
# no windowing, no linear-attention approximation needed.
USE_BOTTLENECK_ATTENTION = True
USE_SKIP_ATTENTION       = True    # attention gates on skips (Oktay et al.)
N_ATTENTION_BLOCKS       = 1
ATTENTION_MLP_RATIO      = 2.0
ATTENTION_DROPOUT        = 0.1

# The layer whose output is the "satellite embedding" deliverable. Named so
# models/ can build an extractor without knowing the architecture.
EMBEDDING_LAYER_NAME = "embedding"

# ── search space ─────────────────────────────────────────────────────────────
# Revised again on evidence from the two completed searches.
#
# base_filters: the tuner picked the MAXIMUM offered value both times - 64 of
#   64 in the original space (8 of 8 top trials), 48 of 48 in the reduced one
#   (7 of 8). Capacity reduction was imposed, never chosen, and it did not
#   improve skill. The ceiling goes back up; regularisation does the work.
#   This model also starts smaller: 14 input channels instead of 26.
#
# learning_rate: the ceiling was binding. Scores improved monotonically toward
#   it - 3e-3 -> 0.6151, 2.5e-3 -> 0.6196, 1.6e-3 -> 0.6277, 4.1e-4 -> 0.6335.
#   That is not a healthy signal, it is what severe overfitting pressure looks
#   like: with ~200 effectively independent samples, training fast and stopping
#   early generalises better than fitting slowly. Ceiling raised so the optimum
#   is inside the space rather than on its edge.
#
# num_heads: the only attention-specific dimension worth spending a slot on.
SEARCH_SPACE_ATT = {
    "base_filters":  [24, 32, 48, 64],
    "depth":         [3, 4],
    "learning_rate": (3e-4, 5e-3),           # log-uniform
    "dropout":       [0.1, 0.2, 0.3, 0.4],
    "weight_decay":  (1e-6, 1e-2),           # log-uniform, decoupled (AdamW)
    "num_heads":     [2, 4],
}

if _base.DATASET_VERSION != "v1":
    # Applies to v3 as well (the bottleneck there is 12x14 = 168 tokens at
    # depth 3, back to v1's size; heads 8 stays available but is no longer
    # expected to win).
    #
    # The v2 base search returned base_filters=32 - the MINIMUM of a space
    # that ran up to 80 - paired with weight_decay 3.3e-3, near the top of its
    # range. Read together that is the search asking for LESS capacity and
    # MORE regularisation, i.e. still overfitting-limited even with 1.75x the
    # days. So this space is NOT widened upward; it is centred where the base
    # model actually landed and given more room to regularise.
    #
    # num_heads gains 8: the bottleneck is now 12x26 = 312 tokens rather than
    # 12x14 = 168, so there is more structure for multiple heads to divide.
    SEARCH_SPACE_ATT = {
        "base_filters":  [16, 24, 32, 48],
        "depth":         [3, 4],
        "learning_rate": (3e-4, 5e-3),
        "dropout":       [0.1, 0.2, 0.3, 0.4],
        "weight_decay":  (1e-4, 3e-2),
        "num_heads":     [2, 4, 8],
    }

DEFAULT_HP_ATT = {
    "base_filters":   32,
    "depth":           3,
    "kernel_size":     3,
    "dropout":       0.2,
    "learning_rate": 1e-3,
    "lambda_physics": 1e-3,
    "weight_decay":   1e-4,
    "num_heads":       4,
}
