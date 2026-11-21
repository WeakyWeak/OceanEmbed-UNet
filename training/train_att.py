#!/usr/bin/env python3
"""
Single training run for the satellite-only attention U-Net.

Usage:
    .venv/bin/python training/train_att.py --tag att_best \
        --base-filters 48 --depth 3 --dropout 0.2 \
        --learning-rate 1e-3 --weight-decay 1e-4 --num-heads 4

    # ablations, if you want to show attention is doing something
    .venv/bin/python training/train_att.py --tag att_no_attn --no-attention
    .venv/bin/python training/train_att.py --tag att_all_inputs \
        --input-vars sst sla sss uo vo u10 v10

Writes training/runs/<tag>/ with the same layout as train.py, plus
`input_vars`, `num_heads` and the attention switches in run_config.json so
models/ can rebuild the exact input pipeline for evaluation and serving.
"""
import argparse
import json
import sys
from pathlib import Path

import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config_att as C
import dataset as D
import dataset_att as DA
import model_att as M
import losses as L
from train import build_callbacks


def run(args) -> keras.Model:
    keras.utils.set_random_seed(args.seed)

    run_dir = C.RUNS_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    lags = [int(v) for v in args.lags]
    windows = [int(w) for w in args.running_mean_windows]
    tm, pm, _ = D.load_masks()
    mus, sig = D.load_norm_arrays()
    anom = C.ANOMALISE_TARGET
    input_vars = list(args.input_vars)

    ds_kw = dict(input_vars=input_vars, lags=lags,
                 running_mean_windows=windows, batch_size=args.batch_size,
                 anomalise_target=anom,
                 anomalise_inputs=C.ANOMALISE_INPUTS_ATT)
    train_ds = DA.SubsetOceanDataset("train", **ds_kw)
    val_ds   = DA.SubsetOceanDataset("val",   **ds_kw)
    print(train_ds.describe())
    print(val_ds.describe())

    model = M.build_attention_unet(
        train_ds.n_channels,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_bottleneck_attention=not args.no_attention,
        use_skip_attention=not (args.no_attention or args.no_skip_attention),
    )
    print(f"\nmodel: {model.count_params():,} parameters, "
          f"{train_ds.n_channels} input channels, "
          f"attention={'off' if args.no_attention else 'on'}\n")

    if anom:
        a = D.load_anom_stats()
        loss_kw = dict(phys_scale=a["phys_scale"], phys_offset=a["phys_offset"])
        met_args = (a["phys_scale"], a["phys_offset"], None, None, True)
    else:
        loss_kw = dict(mus=mus, sigmas=sig)
        met_args = (sig, None, mus, sig, False)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.learning_rate,
                                         weight_decay=args.weight_decay,
                                         clipnorm=1.0),
        loss=L.CombinedLoss(tm, pm, lambda_physics=args.lambda_physics,
                            anomaly_mode=anom, **loss_kw),
        metrics=L.default_metrics(tm, pm, *met_args),
    )

    # dataset_version / n_lat / n_lon are PROVENANCE, not settings. Without
    # them a run cannot be told apart from one trained on another dataset
    # version, and models/rank.py would score it against the wrong arrays -
    # which for a fully convolutional net can return plausible numbers rather
    # than raising. Never drop these keys.
    cfg = vars(args) | {
        "n_channels": train_ds.n_channels,
        "running_mean_windows": windows,
        "params": int(model.count_params()),
        "anomalise_target": anom,
        "anomalise_inputs": dict(C.ANOMALISE_INPUTS_ATT),
        "input_vars": input_vars,
        "dataset_version": C.DATASET_VERSION,
        "n_lat": C.N_LAT,
        "n_lon": C.N_LON,
        "architecture": "attention_unet",
        "bottleneck_attention": not args.no_attention,
        "skip_attention": not (args.no_attention or args.no_skip_attention),
        "optimizer": "AdamW",
    }
    json.dump(cfg, open(run_dir / "run_config.json", "w"), indent=2, default=str)

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs,
              callbacks=build_callbacks(run_dir, args.patience), verbose=1)

    model.save(run_dir / "final.keras")
    print(f"\nrun complete -> {run_dir}")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    hp = C.DEFAULT_HP_ATT
    p.add_argument("--tag", default="att", help="run name under training/runs/")
    p.add_argument("--lags", nargs="+", default=C.LAGS)
    p.add_argument("--running-mean-windows", nargs="*", type=int,
                   default=C.RUNNING_MEAN_WINDOWS,
                   help="trailing running-mean windows in days, e.g. 30 90; "
                        "pass the flag with no values for none")
    p.add_argument("--input-vars", nargs="+", default=C.INPUT_VARS_ATT,
                   help="surface inputs to use (default: satellite only)")
    p.add_argument("--base-filters", type=int, default=hp["base_filters"])
    p.add_argument("--depth", type=int, default=hp["depth"])
    p.add_argument("--kernel-size", type=int, default=hp["kernel_size"])
    p.add_argument("--dropout", type=float, default=hp["dropout"])
    p.add_argument("--num-heads", type=int, default=hp["num_heads"])
    p.add_argument("--learning-rate", type=float, default=hp["learning_rate"])
    p.add_argument("--lambda-physics", type=float, default=hp["lambda_physics"])
    p.add_argument("--weight-decay", type=float, default=hp["weight_decay"])
    p.add_argument("--no-attention", action="store_true",
                   help="ablation: plain U-Net on the same satellite inputs")
    p.add_argument("--no-skip-attention", action="store_true",
                   help="ablation: bottleneck attention only, no skip gates")
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=C.EPOCHS_FINAL)
    p.add_argument("--patience", type=int, default=C.PATIENCE_FINAL)
    p.add_argument("--seed", type=int, default=C.SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
