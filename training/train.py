#!/usr/bin/env python3
"""
Single training run from an explicit configuration.

Usage:
    .venv/bin/python training/train.py --tag primary
    .venv/bin/python training/train.py --tag control --lags 0
    .venv/bin/python training/train.py --tag no_physics --lambda-physics 0
    .venv/bin/python training/train.py --tag hard_mono --hard-monotonic-below 7

Writes to training/runs/<tag>/:
    best.keras       best weights by val_masked_mse
    history.csv      per-epoch metrics
    run_config.json  everything needed to rebuild this run for evaluation
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import dataset as D
import dataset_att as DA
import model as M
import losses as L


def build_callbacks(run_dir: Path, patience: int):
    return [
        keras.callbacks.EarlyStopping(
            monitor=C.TUNER_OBJECTIVE, mode="min", patience=patience,
            min_delta=1e-5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor=C.TUNER_OBJECTIVE, mode="min", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"), monitor=C.TUNER_OBJECTIVE,
            mode="min", save_best_only=True, verbose=0),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        # Overnight runs must survive an interruption.
        keras.callbacks.BackupAndRestore(backup_dir=str(run_dir / "backup")),
    ]


def run(args) -> keras.Model:
    keras.utils.set_random_seed(args.seed)

    run_dir = C.RUNS_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    lags = [int(v) for v in args.lags]
    windows = [int(w) for w in args.running_mean_windows]
    tm, pm, _ = D.load_masks()
    mus, sig = D.load_norm_arrays()
    anom = C.ANOMALISE_TARGET

    # --input-vars trains the PLAIN U-Net on a subset of the seven surface
    # inputs. This is what makes a satellite-only run possible without a second
    # architecture: SubsetOceanDataset is documented as exactly equivalent to
    # OceanDataset when handed all seven, and it applies each variable's anomaly
    # transform on the full set BEFORE subsetting, so the surviving channels are
    # numerically identical to what a full-input model sees.
    #
    # It also gives the clean A/B that att_v1 never provided: that run changed
    # architecture AND dropped 7 inputs -> 3 at the same time, so its loss could
    # not be attributed to either.
    if args.input_vars:
        train_ds = DA.SubsetOceanDataset("train", input_vars=args.input_vars,
                                         lags=lags, running_mean_windows=windows,
                                         batch_size=args.batch_size)
        val_ds   = DA.SubsetOceanDataset("val", input_vars=args.input_vars,
                                         lags=lags, running_mean_windows=windows,
                                         batch_size=args.batch_size)
    else:
        train_ds = D.OceanDataset("train", lags=lags,
                                  running_mean_windows=windows,
                                  batch_size=args.batch_size)
        val_ds   = D.OceanDataset("val",   lags=lags,
                                  running_mean_windows=windows,
                                  batch_size=args.batch_size)
    print(train_ds.describe())
    print(val_ds.describe())

    n_channels = train_ds.n_channels
    model = M.build_unet(
        n_channels,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        hard_monotonic_below=args.hard_monotonic_below,
        mus=mus, sigmas=sig,
    )
    print(f"\nmodel: {model.count_params():,} parameters, "
          f"{n_channels} input channels\n")

    if anom:
        a = D.load_anom_stats()
        loss_kw = dict(phys_scale=a["phys_scale"], phys_offset=a["phys_offset"])
        met_args = (a["phys_scale"], a["phys_offset"], None, None, True)
    else:
        loss_kw = dict(mus=mus, sigmas=sig)
        met_args = (sig, None, mus, sig, False)

    model.compile(
        # AdamW: decoupled weight decay. The first run used plain Adam with no
        # decay, and a 7.7M-parameter model reached its best val score after a
        # single epoch. Regularisation is the point of this revision.
        optimizer=keras.optimizers.AdamW(learning_rate=args.learning_rate,
                                         weight_decay=args.weight_decay,
                                         clipnorm=1.0),
        loss=L.CombinedLoss(tm, pm, lambda_physics=args.lambda_physics,
                            anomaly_mode=anom, **loss_kw),
        metrics=L.default_metrics(tm, pm, *met_args),
    )

    # run_config.json records the data formulation so evaluate.py can rebuild
    # this exact configuration - including for legacy absolute-target runs.
    # dataset_version / n_lat / n_lon are PROVENANCE, not settings. Without
    # them a run cannot be told apart from one trained on another dataset
    # version, and models/rank.py would score it against the wrong arrays -
    # which for a fully convolutional net can return plausible numbers rather
    # than raising. Never drop these keys.
    cfg = vars(args) | {"n_channels": n_channels,
                        "running_mean_windows": windows,
                        "input_vars": list(getattr(train_ds, "input_vars",
                                                   C.INPUT_VARS)),
                        "params": int(model.count_params()),
                        "anomalise_target": anom,
                        "anomalise_inputs": dict(C.ANOMALISE_INPUTS),
                        "dataset_version": C.DATASET_VERSION,
                        "n_lat": C.N_LAT,
                        "n_lon": C.N_LON,
                        "optimizer": "AdamW"}
    json.dump(cfg, open(run_dir / "run_config.json", "w"), indent=2, default=str)

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=build_callbacks(run_dir, args.patience),
              verbose=1)

    model.save(run_dir / "final.keras")
    print(f"\nrun complete -> {run_dir}")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    hp = C.DEFAULT_HP
    p.add_argument("--tag", default="primary", help="run name under training/runs/")
    p.add_argument("--input-vars", nargs="+", default=None, choices=C.INPUT_VARS,
                   metavar="VAR",
                   help="train on a subset of the surface inputs, e.g. "
                        "--input-vars sst sla sss for a satellite-only run. "
                        "Default: all seven.")
    p.add_argument("--lags", nargs="+", default=C.LAGS,
                   help="input lags in days; first must be 0 (default: 0 3 7)")
    p.add_argument("--running-mean-windows", nargs="*", type=int,
                   default=C.RUNNING_MEAN_WINDOWS,
                   help="trailing running-mean windows in days, e.g. 30 90; "
                        "pass the flag with no values for none")
    p.add_argument("--base-filters", type=int, default=hp["base_filters"])
    p.add_argument("--depth", type=int, default=hp["depth"])
    p.add_argument("--kernel-size", type=int, default=hp["kernel_size"])
    p.add_argument("--dropout", type=float, default=hp["dropout"])
    p.add_argument("--learning-rate", type=float, default=hp["learning_rate"])
    p.add_argument("--lambda-physics", type=float, default=hp["lambda_physics"])
    p.add_argument("--weight-decay", type=float, default=hp["weight_decay"],
                   help="decoupled weight decay for AdamW")
    p.add_argument("--hard-monotonic-below", type=int,
                   default=C.HARD_MONOTONIC_BELOW,
                   help="depth index below which monotonicity is enforced "
                        "architecturally (7 == the 100m level); omit to disable")
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=C.EPOCHS_FINAL)
    p.add_argument("--patience", type=int, default=C.PATIENCE_FINAL)
    p.add_argument("--seed", type=int, default=C.SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
