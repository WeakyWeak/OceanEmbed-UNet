#!/usr/bin/env python3
"""
KerasTuner Bayesian search for the satellite-only attention U-Net.

Run training/smoke_test_att.py first. It is a hard gate.

Usage:
    .venv/bin/python training/tune_att.py --project att_v1

Objective is val_masked_mse - the DATA term alone, never val_loss.

Scale warning: this model is trained on a DIFFERENT INPUT SET (3 satellite
variables, 14 channels) from the primary/anom projects. Its val_masked_mse is
comparable with those - same target, same normalisation, same mask - but do
not resume an existing project directory, because the hyperparameter space
itself has changed (num_heads was added, and the base_filters and
learning_rate ranges moved).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import keras
import keras_tuner as kt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config_att as C
from tune import (DeadlineStopper, _DeadlineReached,          # noqa: E402
                  _deadline_tuner_class, _fingerprint, _guard_resume)
import dataset as D
import dataset_att as DA
import model_att as M
import losses as L


class AttentionHyperModel(kt.HyperModel):
    """Six dimensions. kernel_size (3) and batch_size (16) stay fixed for the
    same reason as before: 25 trials cannot explore more than this, and
    batch_size confounds with the learning rate."""

    def __init__(self, n_channels, target_mask, pair_mask, mus, sigmas):
        self.n_channels = n_channels
        self.tm, self.pm = target_mask, pair_mask
        self.anom = C.ANOMALISE_TARGET
        if self.anom:
            a = D.load_anom_stats()
            self.loss_kw = dict(phys_scale=a["phys_scale"],
                                phys_offset=a["phys_offset"])
            self.met_args = (a["phys_scale"], a["phys_offset"], None, None, True)
        else:
            self.loss_kw = dict(mus=mus, sigmas=sigmas)
            self.met_args = (sigmas, None, mus, sigmas, False)

    def build(self, hp):
        s = C.SEARCH_SPACE_ATT
        base    = hp.Choice("base_filters", s["base_filters"])
        depth   = hp.Choice("depth", s["depth"])
        dropout = hp.Choice("dropout", s["dropout"])
        heads   = hp.Choice("num_heads", s["num_heads"])
        lr      = hp.Float("learning_rate", *s["learning_rate"], sampling="log")
        wd      = hp.Float("weight_decay", *s["weight_decay"], sampling="log")

        model = M.build_attention_unet(
            self.n_channels, base_filters=base, depth=depth,
            kernel_size=C.DEFAULT_HP_ATT["kernel_size"],
            dropout=dropout, num_heads=heads)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd,
                                             clipnorm=1.0),
            loss=L.CombinedLoss(self.tm, self.pm,
                                lambda_physics=C.FIXED_LAMBDA_PHYSICS,
                                anomaly_mode=self.anom, **self.loss_kw),
            metrics=L.default_metrics(self.tm, self.pm, *self.met_args),
        )
        return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lags", nargs="+", default=C.LAGS)
    p.add_argument("--input-vars", nargs="+", default=C.INPUT_VARS_ATT)
    p.add_argument("--project", default=None,
                   help="tuner project (default: <dataset_version>_att)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; no new trial starts past it and "
                        "the in-flight trial is cut at its next epoch. "
                        "Resumable.")
    p.add_argument("--trials", type=int, default=C.TUNER_TRIALS)
    p.add_argument("--epochs", type=int, default=C.EPOCHS_TUNING)
    p.add_argument("--patience", type=int, default=C.PATIENCE_TUNING)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=C.SEED)
    args = p.parse_args(argv)

    if args.project is None:
        args.project = f"{C.DATASET_VERSION}_att"
    if args.max_hours is not None and args.max_hours < 0:
        raise SystemExit("--max-hours cannot be negative")
    deadline = (time.time() + args.max_hours * 3600
                if args.max_hours is not None else None)

    keras.utils.set_random_seed(args.seed)
    lags = [int(v) for v in args.lags]

    tm, pm, _ = D.load_masks()
    mus, sig = D.load_norm_arrays()
    ds_kw = dict(input_vars=list(args.input_vars), lags=lags,
                 batch_size=args.batch_size,
                 anomalise_target=C.ANOMALISE_TARGET,
                 anomalise_inputs=C.ANOMALISE_INPUTS_ATT)
    train_ds = DA.SubsetOceanDataset("train", **ds_kw)
    val_ds   = DA.SubsetOceanDataset("val",   **ds_kw)
    print(train_ds.describe())
    print(val_ds.describe())

    project_dir = C.RUNS_DIR / "tuner" / args.project
    _guard_resume(project_dir, _fingerprint(train_ds))

    tuner = _deadline_tuner_class(deadline)(
        AttentionHyperModel(train_ds.n_channels, tm, pm, mus, sig),
        objective=kt.Objective(C.TUNER_OBJECTIVE, direction="min"),
        max_trials=args.trials,
        num_initial_points=C.TUNER_INITIAL_POINTS,
        directory=str(C.RUNS_DIR / "tuner"),
        project_name=args.project,
        seed=args.seed,
        overwrite=False,
    )
    tuner.search_space_summary()
    print(f"\nlambda_physics pinned at {C.FIXED_LAMBDA_PHYSICS} (not searched).")
    print(f"inputs: {'+'.join(train_ds.input_vars)} "
          f"({train_ds.n_channels} channels) - SATELLITE ONLY")

    t_start = time.time()
    try:
        tuner.search(
            train_ds, validation_data=val_ds, epochs=args.epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor=C.TUNER_OBJECTIVE,
                                              mode="min",
                                              patience=args.patience,
                                              min_delta=1e-5, verbose=1),
                keras.callbacks.TerminateOnNaN(),
                DeadlineStopper(deadline),
            ],
            verbose=1,
        )
    except _DeadlineReached:
        print(f"\n{'=' * 66}\n  Wall-clock budget spent - stopping cleanly.\n"
              f"{'=' * 66}")
    except KeyboardInterrupt:
        print(f"\n{'=' * 66}\n  Interrupted - reporting best completed trial.\n"
              f"  Resume with the same --project.\n{'=' * 66}")

    n_done = len(tuner.oracle.trials)
    print(f"\n{n_done} trial(s) completed in "
          f"{(time.time() - t_start) / 3600:.2f} h.")
    if n_done < C.TUNER_INITIAL_POINTS:
        print(f"  WARNING: fewer than TUNER_INITIAL_POINTS "
              f"({C.TUNER_INITIAL_POINTS}) trials ran - the Bayesian model "
              f"never took over from random sampling.")

    best_list = tuner.get_best_hyperparameters(1)
    if not best_list:
        raise SystemExit(
            "\nNo trial completed, so there is no best configuration.\n"
            "The project is resumable - re-run with the same --project.")
    best = best_list[0]
    print("\n" + "=" * 66)
    print("  Best hyperparameters")
    print("=" * 66)
    for k, v in best.values.items():
        print(f"  {k:16s} {v}")

    out = C.RUNS_DIR / "tuner" / args.project / "best_hp.json"
    json.dump({"lags": lags, "input_vars": list(args.input_vars), **best.values},
              open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")
    print("\nNow retrain the winner for the full schedule:")
    # v3 is the default now, so no prefix is needed for it. For v1/v2 the
    # prefix is NOT optional: without it train_att.py falls back to the default
    # - different grid, different inputs, different record - and produces a
    # model that looks like a v2 result and is not one. That happened once
    # already: a search tuned on v2 was retrained on v1 because this line
    # omitted the prefix. Hence: print it whenever it is not the default.
    skip = {"input_vars"}
    prefix = ("" if C.DATASET_VERSION == "v3"
              else f"OCEAN_DATASET_VERSION={C.DATASET_VERSION} ")
    print(f"  {prefix}"
          f".venv/bin/python training/train_att.py --tag {args.project}_best "
          + " ".join(f"--{k.replace('_','-')} {v}"
                     for k, v in best.values.items() if k not in skip))
    return tuner


if __name__ == "__main__":
    main()
