#!/usr/bin/env python3
"""
Final evaluation: per-depth metrics in physical units, baselines, physics stats.

Only run this on test.nc once, with the single model you have already selected
using val.nc. The test set must stay untouched during tuning, architecture
selection, hyperparameter selection, and early stopping.

Usage:
    .venv/bin/python training/evaluate.py --run primary_best --split val
    .venv/bin/python training/evaluate.py --run primary_best --split test   # ONCE

Reports per depth, in degrees C:
    RMSE, MAE, BIAS, Pearson r, ANOMALY correlation, skill vs climatology

Anomaly and legacy runs
-----------------------
The formulation is read from each run's run_config.json, so pre-anomaly runs
(absolute targets, plain Adam) still evaluate correctly and remain directly
comparable. Predictions are always converted back to absolute degC before
metrics are computed, whichever formulation produced them.

Why anomaly correlation matters
-------------------------------
Raw Pearson r comes out around 0.9+ at every depth purely from the seasonal
cycle and the mean vertical profile. It looks excellent and means nothing. ACC
removes the per-cell, per-day-of-year climatology (TRAIN split only) and
correlates the residuals - that is the number that shows real skill.

Baseline
--------
The day-of-year climatology. skill = 1 - RMSE_model / RMSE_climatology;
0 means no better than climatology, negative means worse.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import dataset as D
import losses as L  # noqa: F401  (registers custom objects for model loading)


def evaluate(run_tag: str, split: str, batch_size: int) -> pd.DataFrame:
    run_dir = C.RUNS_DIR / run_tag
    cfg = json.load(open(run_dir / "run_config.json"))
    lags = [int(v) for v in cfg["lags"]]
    # Runs predating the running-mean channels have no windows key; [] rebuilds
    # their input exactly. Without this the dataset would fall back to the
    # module-level config and feed a 20-channel model 30 channels.
    windows = [int(w) for w in cfg.get("running_mean_windows", [])]
    # Legacy runs predate these keys and were trained on absolute targets.
    anom = bool(cfg.get("anomalise_target", False))
    anom_inputs = cfg.get("anomalise_inputs", {v: False for v in C.INPUT_VARS})
    print(
        f"run {run_tag}: target={'anomaly' if anom else 'absolute'}  "
        f"lags={lags}  rm={windows or 'none'}  anom_inputs="
        f"{[v for v, on in anom_inputs.items() if on] or 'none'}"
    )

    tm, _, _ = D.load_masks()
    mus, sig = D.load_norm_arrays()
    ds = D.OceanDataset(
        split,
        lags=lags,
        running_mean_windows=windows,
        batch_size=batch_size,
        shuffle=False,
        anomalise_target=anom,
        anomalise_inputs=anom_inputs,
    )
    print(ds.describe())

    model = keras.models.load_model(run_dir / "best.keras", compile=False)
    print(f"loaded {run_dir / 'best.keras'} ({model.count_params():,} params)")

    pred = model.predict(ds, verbose=1)  # (N,81,101,15)
    Ym = np.load(C.Y_NPY, mmap_mode="r")
    true_abs = np.asarray(Ym[ds.indices], dtype=np.float32) * sig + mus

    # ── climatology, always in absolute degC, always from the TRAIN split ────
    dates = np.load(C.DATES_NPY)
    doy = pd.DatetimeIndex(dates).dayofyear.values[ds.indices]
    clim_z = np.asarray(np.load(C.CLIM_Y_NPY, mmap_mode="r")[doy], dtype=np.float32)
    clim_abs = clim_z * sig + mus

    # ── predictions -> absolute degC ────────────────────────────────────────
    if anom:
        a = D.load_anom_stats()
        pred_abs = D.to_absolute_degC(pred, clim_abs, a["phys_scale"], a["phys_offset"])
    else:
        pred_abs = pred * sig + mus

    # ── per-depth metrics ───────────────────────────────────────────────────
    rows = []
    for k, lbl in enumerate(C.DEPTH_LABELS):
        m = tm[:, :, k]
        t = true_abs[:, :, :, k][:, m].ravel()
        p = pred_abs[:, :, :, k][:, m].ravel()
        c = clim_abs[:, :, :, k][:, m].ravel()

        err = p - t
        rmse = float(np.sqrt(np.mean(err**2)))
        rmse_clim = float(np.sqrt(np.mean((c - t) ** 2)))
        ta, pa = t - c, p - c
        acc = (
            float(np.corrcoef(ta, pa)[0, 1])
            if ta.std() > 0 and pa.std() > 0
            else np.nan
        )

        rows.append(
            {
                "depth": lbl,
                "actual_m": C.ACTUAL_DEPTH_M[k],
                "n_cells": int(m.sum()),
                "sigma_degC": float(sig[k]),
                "rmse_degC": rmse,
                "mae_degC": float(np.mean(np.abs(err))),
                "bias_degC": float(np.mean(err)),
                "pearson_r": float(np.corrcoef(t, p)[0, 1]),
                "anomaly_corr": acc,
                "anom_sd_ratio": float(pa.std() / ta.std()) if ta.std() > 0 else np.nan,
                "rmse_clim_degC": rmse_clim,
                "skill_vs_clim": float(1.0 - rmse / rmse_clim)
                if rmse_clim > 0
                else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    w = df.n_cells.to_numpy(float)
    overall = {
        "rmse_degC": float(np.sqrt(np.average(df.rmse_degC**2, weights=w))),
        "mae_degC": float(np.average(df.mae_degC, weights=w)),
        "bias_degC": float(np.average(df.bias_degC, weights=w)),
        "anomaly_corr": float(np.average(df.anomaly_corr, weights=w)),
        "skill_vs_clim": float(np.average(df.skill_vs_clim, weights=w)),
    }

    # ── physics violation rates, on absolute temperature ────────────────────
    pmask = np.load(C.PAIR_MASK_NPY)

    def violation_rate(field, gated):
        d = field[..., 1:] - field[..., :-1]
        sel = pmask & ((C.PHYSICS_GATE > 0) if gated else np.ones(C.N_PAIRS, bool))
        return float((d[:, sel] > 0).mean())

    phys = {
        "pred_gated_pairs": violation_rate(pred_abs, True),
        "truth_gated_pairs": violation_rate(true_abs, True),
        "pred_all_pairs": violation_rate(pred_abs, False),
        "truth_all_pairs": violation_rate(true_abs, False),
    }

    # ── report ──────────────────────────────────────────────────────────────
    pd.set_option("display.width", 220, "display.max_columns", 24)
    print(
        f"\n{'=' * 112}\n  {run_tag} / {split} - per-depth metrics (physical degrees C)\n{'=' * 112}"
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(
        "\noverall (valid-cell weighted): "
        + "  ".join(f"{k}={v:.4f}" for k, v in overall.items())
    )
    print("\nvertical ordering violations (absolute temperature):")
    print(
        f"  gated pairs (50m and deeper)  model {phys['pred_gated_pairs'] * 100:7.4f}%   "
        f"truth {phys['truth_gated_pairs'] * 100:7.4f}%"
    )
    print(
        f"  all 14 pairs                  model {phys['pred_all_pairs'] * 100:7.4f}%   "
        f"truth {phys['truth_all_pairs'] * 100:7.4f}%"
    )
    print("\n  Reading the table:")
    print(
        "   * skill_vs_clim <= 0 means the day-of-year climatology is as good or better."
    )
    print(
        "   * anom_sd_ratio < 1 means the model is under-predicting anomaly amplitude"
    )
    print("     (regression to the mean); ~1 with high anomaly_corr is what you want.")
    print("   * a model violation rate above the truth in the UNGATED shallow pairs is")
    print("     expected: 10-50 m genuinely inverts 22-40% of the time in the Bay of")
    print("     Bengal (barrier layer). Judge the gated row.")

    out = run_dir / f"metrics_{split}.csv"
    df.to_csv(out, index=False)
    json.dump(
        {"overall": overall, "physics": phys, "anomalise_target": anom, "split": split},
        open(run_dir / f"metrics_{split}.json", "w"),
        indent=2,
    )
    print(f"\nsaved -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="run tag under training/runs/")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    a = p.parse_args()
    if a.split == "test":
        print("\n*** TEST SET: run this once, on your final selected model only ***\n")
    evaluate(a.run, a.split, a.batch_size)
