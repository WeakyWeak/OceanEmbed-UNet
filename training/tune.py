#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import keras
import keras_tuner as kt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import dataset as D
import model as M
import losses as L


class _DeadlineReached(Exception):
    """Raised between trials once the wall-clock budget is spent."""


class DeadlineStopper(keras.callbacks.Callback):
    """Cut the CURRENT trial short at the next epoch boundary past the deadline.

    Without this, a deadline checked only between trials could overrun by a
    whole trial - up to ~35 minutes here. Stopping at an epoch boundary means
    the trial still reports the best score it reached, so the budget ends with
    a usable result rather than a discarded one.
    """

    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def on_epoch_end(self, epoch, logs=None):
        if self.deadline is not None and time.time() > self.deadline:
            print(
                f"\n  [deadline] wall-clock budget spent - ending this "
                f"trial at epoch {epoch}"
            )
            self.model.stop_training = True


def _deadline_tuner_class(deadline):
    """BayesianOptimization that refuses to START a trial past the deadline.

    `pre_create_trial` is the first thing BaseTuner.search() calls each time
    round its loop, so it is the one place a new trial can be declined without
    corrupting the oracle's state.
    """

    class _DeadlineBayesian(kt.BayesianOptimization):
        def pre_create_trial(self):
            if deadline is not None and time.time() > deadline:
                raise _DeadlineReached()
            super().pre_create_trial()

    return _DeadlineBayesian


def _fingerprint(train_ds) -> dict:
    """What the oracle's scores are only meaningful against."""
    return {
        "dataset_version": C.DATASET_VERSION,
        "n_channels": int(train_ds.n_channels),
        "n_train_samples": int(train_ds.indices.size),
        "grid": [int(C.N_LAT), int(C.N_LON)],
        "input_vars": list(getattr(train_ds, "input_vars", C.INPUT_VARS)),
        "split_mode": C.SPLIT_MODE,
        "objective": C.TUNER_OBJECTIVE,
    }


def _guard_resume(project_dir: Path, fp: dict) -> None:
    """Refuse to resume an oracle that was fitted on DIFFERENT data.

    overwrite=False makes the search resumable, which is what allows a 6-hour
    night to be continued. The same flag will just as happily resume a project
    whose scores came from another dataset - and a Bayesian oracle fed scores
    from two incomparable datasets does not error, it just proposes nonsense
    while looking like it is converging. So the fingerprint travels with the
    project and is checked before anything resumes.
    """
    marker = project_dir / "dataset_fingerprint.json"
    if not marker.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        json.dump(fp, open(marker, "w"), indent=2)
        return
    old = json.load(open(marker))
    diff = {k: (old.get(k), fp[k]) for k in fp if old.get(k) != fp[k]}
    if diff:
        lines = "\n".join(
            f"    {k}: project has {a!r}, this run has {b!r}"
            for k, (a, b) in diff.items()
        )
        raise SystemExit(
            f"\nRefusing to resume {project_dir.name!r}: it was built on "
            f"different data.\n{lines}\n\n"
            f"Use a fresh --project name. Scores from two datasets are not "
            f"comparable,\nand the oracle cannot tell that they are not.\n"
        )


class OceanHyperModel(kt.HyperModel):
    """Five search dimensions. See spec section 6 for why kernel_size (3) and
    batch_size (16) are fixed rather than searched: 25 trials cannot
    meaningfully explore seven dimensions, and batch_size confounds with the
    learning rate."""

    def __init__(self, n_channels, target_mask, pair_mask, mus, sigmas):
        self.n_channels = n_channels
        self.tm, self.pm = target_mask, pair_mask
        self.mus, self.sigmas = mus, sigmas
        self.anom = C.ANOMALISE_TARGET
        if self.anom:
            a = D.load_anom_stats()
            self.loss_kw = dict(
                phys_scale=a["phys_scale"], phys_offset=a["phys_offset"]
            )
            self.met_args = (a["phys_scale"], a["phys_offset"], None, None, True)
        else:
            self.loss_kw = dict(mus=mus, sigmas=sigmas)
            self.met_args = (sigmas, None, mus, sigmas, False)

    def build(self, hp):
        s = C.SEARCH_SPACE
        base = hp.Choice("base_filters", s["base_filters"])
        depth = hp.Choice("depth", s["depth"])
        dropout = hp.Choice("dropout", s["dropout"])
        lr = hp.Float(
            "learning_rate",
            s["learning_rate"][0],
            s["learning_rate"][1],
            sampling="log",
        )
        wd = hp.Float(
            "weight_decay", s["weight_decay"][0], s["weight_decay"][1], sampling="log"
        )

        model = M.build_unet(
            self.n_channels,
            base_filters=base,
            depth=depth,
            kernel_size=C.DEFAULT_HP["kernel_size"],
            dropout=dropout,
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=lr, weight_decay=wd, clipnorm=1.0
            ),
            loss=L.CombinedLoss(
                self.tm,
                self.pm,
                lambda_physics=C.FIXED_LAMBDA_PHYSICS,
                anomaly_mode=self.anom,
                **self.loss_kw,
            ),
            metrics=L.default_metrics(self.tm, self.pm, *self.met_args),
        )
        return model


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lags", nargs="+", default=C.LAGS)
    p.add_argument(
        "--project",
        default=None,
        help="tuner project name (default: <dataset_version>_primary). "
        "Scores are only comparable within a project.",
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="wall-clock budget. No new trial starts past it and "
        "the in-flight trial is cut at its next epoch. "
        "Resumable: re-run with another --max-hours to continue.",
    )
    p.add_argument("--trials", type=int, default=C.TUNER_TRIALS)
    p.add_argument("--epochs", type=int, default=C.EPOCHS_TUNING)
    p.add_argument("--patience", type=int, default=C.PATIENCE_TUNING)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=C.SEED)
    args = p.parse_args(argv)

    if args.project is None:
        args.project = f"{C.DATASET_VERSION}_primary"

    # `is not None`, not truthiness: --max-hours 0 is a MEANINGFUL request
    # ("start nothing new"), and a falsy test silently turned it into "no
    # budget at all" - which launched an unbounded overnight search.
    if args.max_hours is not None and args.max_hours < 0:
        raise SystemExit("--max-hours cannot be negative")
    deadline = (
        time.time() + args.max_hours * 3600 if args.max_hours is not None else None
    )

    keras.utils.set_random_seed(args.seed)
    lags = [int(v) for v in args.lags]

    tm, pm, _ = D.load_masks()
    mus, sig = D.load_norm_arrays()
    train_ds = D.OceanDataset("train", lags=lags, batch_size=args.batch_size)
    val_ds = D.OceanDataset("val", lags=lags, batch_size=args.batch_size)
    print(train_ds.describe())
    print(val_ds.describe())

    project_dir = C.RUNS_DIR / "tuner" / args.project
    fp = _fingerprint(train_ds)
    _guard_resume(project_dir, fp)

    if deadline is not None:
        # Measured on this machine: ~0.20-0.27 s/step, 639 steps/epoch.
        # Deliberately a RANGE - trial cost varies with base_filters and depth
        # and with where early stopping bites, so a single number would be
        # false precision.
        lo, hi = 26, 35
        print(
            f"\nWall-clock budget: {args.max_hours:.1f} h. "
            f"At ~{lo}-{hi} min/trial that is roughly "
            f"{int(args.max_hours * 60 / hi)}-{int(args.max_hours * 60 / lo)} trials "
            f"(cap {args.trials})."
        )
        print(
            "No new trial starts past the deadline; the in-flight one is cut "
            "at its next epoch."
        )

    tuner = _deadline_tuner_class(deadline)(
        OceanHyperModel(train_ds.n_channels, tm, pm, mus, sig),
        objective=kt.Objective(C.TUNER_OBJECTIVE, direction="min"),
        max_trials=args.trials,
        num_initial_points=C.TUNER_INITIAL_POINTS,
        directory=str(C.RUNS_DIR / "tuner"),
        project_name=args.project,
        seed=args.seed,
        overwrite=False,  # resumable
    )
    tuner.search_space_summary()
    print(f"\nlambda_physics pinned at {C.FIXED_LAMBDA_PHYSICS} (not searched).")
    print(
        "NOTE: val_masked_mse is on a different scale under the anomaly "
        "formulation.\n      Scores are NOT comparable with pre-anomaly tuner "
        "projects - use a fresh\n      --project name rather than resuming an "
        "old one (overwrite=False resumes)."
    )

    # The test set is never touched here.
    t_start = time.time()
    try:
        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor=C.TUNER_OBJECTIVE,
                    mode="min",
                    patience=args.patience,
                    min_delta=1e-5,
                    verbose=1,
                ),
                keras.callbacks.TerminateOnNaN(),
                DeadlineStopper(deadline),
            ],
            verbose=1,
        )
    except _DeadlineReached:
        print(
            f"\n{'=' * 66}\n  Wall-clock budget spent - stopping cleanly.\n{'=' * 66}"
        )
    except KeyboardInterrupt:
        # An overnight search is exactly the thing someone interrupts. Trials
        # already finished are on disk and the project resumes, so report the
        # best so far rather than losing hours of work to a traceback.
        print(
            f"\n{'=' * 66}\n  Interrupted - reporting best completed trial.\n"
            f"  Resume with the same --project to continue.\n{'=' * 66}"
        )

    n_done = len(tuner.oracle.trials)
    print(f"\n{n_done} trial(s) completed in {(time.time() - t_start) / 3600:.2f} h.")
    if n_done < C.TUNER_INITIAL_POINTS:
        print(
            f"  WARNING: fewer than TUNER_INITIAL_POINTS "
            f"({C.TUNER_INITIAL_POINTS}) trials ran, so the Bayesian model "
            f"never took over from random sampling.\n"
            f"  Treat this as a random search and give it more time."
        )

    best_list = tuner.get_best_hyperparameters(1)
    if not best_list:
        raise SystemExit(
            "\nNo trial completed, so there is no best configuration to "
            "report.\nGive the search more time: a single trial needs roughly "
            "26-35 minutes here.\n"
            "The project is resumable - re-run with the same --project."
        )
    best = best_list[0]
    print("\n" + "=" * 66)
    print("  Best hyperparameters")
    print("=" * 66)
    for k, v in best.values.items():
        print(f"  {k:16s} {v}")

    out = C.RUNS_DIR / "tuner" / args.project / "best_hp.json"
    json.dump({"lags": lags, **best.values}, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")
    print("\nNow retrain the winner for the full schedule, e.g.:")
    # v3 is the default, so the prefix is noise there. It is still printed for
    # v1/v2, where running without it would silently retrain on the wrong data.
    prefix = (
        ""
        if C.DATASET_VERSION == "v3"
        else f"OCEAN_DATASET_VERSION={C.DATASET_VERSION} "
    )
    print(
        f"  {prefix}"
        f".venv/bin/python training/train.py --tag {args.project}_best "
        f"--lags {' '.join(map(str, lags))} "
        + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in best.values.items())
    )
    return tuner


if __name__ == "__main__":
    main()
