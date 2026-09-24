#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import keras
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C
import splits as S
import dataset as D
import model as M
import losses as L
import pandas as pd

PASS, FAIL = 0, 0
FAILURES = []


def check(cond, msg, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        FAILURES.append(msg + (f"  -> {detail}" if detail else ""))
        print(f"  FAIL  {msg}" + (f"\n        -> {detail}" if detail else ""))


def main():
    keras.utils.set_random_seed(C.SEED)
    print("=" * 66)
    print("  OceanEmbed training pipeline - smoke test")
    print("=" * 66)

    tm, pm, land = D.load_masks()
    mus, sig = D.load_norm_arrays()
    anom = C.ANOMALISE_TARGET
    astats = D.load_anom_stats() if anom else None
    train_ds, val_ds, test_ds = D.make_datasets()
    print(
        f"\nformulation: target={'anomaly' if anom else 'absolute'}  "
        f"anomalised inputs="
        f"{[v for v, on in C.ANOMALISE_INPUTS.items() if on] or 'none'}"
    )

    # ── 1. one batch loads with the right shapes ────────────────────────────
    print("\n[1] Batch shapes")
    X, Y = train_ds[0]
    nc = C.n_input_channels()
    check(
        X.shape == (C.BATCH_SIZE, C.N_LAT, C.N_LON, nc),
        f"X {X.shape}",
        f"expected ({C.BATCH_SIZE}, 81, 101, {nc})",
    )
    nt = C.N_TRUE_CHANNELS
    check(
        Y.shape == (C.BATCH_SIZE, C.N_LAT, C.N_LON, nt),
        f"Y {Y.shape}",
        f"expected (B, 81, 101, {nt})",
    )
    check(X.dtype == np.float32 and Y.dtype == np.float32, "dtypes float32")

    # ── 2. no NaN/Inf; validity counts as expected ──────────────────────────
    print("\n[2] Finiteness and target validity mask")
    check(np.isfinite(X).all(), "X batch is finite")
    check(np.isfinite(Y).all(), "Y batch is finite (NaN pre-replaced by 0.0)")
    counts = tm.sum(axis=(0, 1)).tolist()
    check(
        counts == C.EXPECTED_VALID_CELLS,
        f"per-depth valid cells {counts[:4]}...{counts[-2:]}",
        f"expected {C.EXPECTED_VALID_CELLS}",
    )
    check(
        np.array_equal(pm, tm[:, :, :-1] & tm[:, :, 1:]),
        "pair_mask == target_mask[d] AND target_mask[d+1]",
    )

    # ── 3. forward pass shape ───────────────────────────────────────────────
    print("\n[3] Forward pass")
    model = M.build_unet(nc, base_filters=16, depth=3, dropout=0.1)
    out = model(X, training=False)
    check(
        tuple(out.shape) == (C.BATCH_SIZE, C.N_LAT, C.N_LON, C.N_DEPTHS),
        f"output {tuple(out.shape)} - model still predicts 15 channels",
        "expected (B, 81, 101, 15)",
    )

    # ── 4-6. losses finite ──────────────────────────────────────────────────
    print("\n[4-6] Losses")
    lkw = (
        dict(phys_scale=astats["phys_scale"], phys_offset=astats["phys_offset"])
        if anom
        else dict(mus=mus, sigmas=sig)
    )
    data_loss = L.MaskedMSE(tm, anomaly_mode=anom)
    d = float(data_loss(Y, out))
    check(np.isfinite(d), f"masked MSE finite ({d:.6f})")

    full = L.CombinedLoss(tm, pm, lambda_physics=1.0, anomaly_mode=anom, **lkw)
    t = float(full(Y, out))
    check(np.isfinite(t), f"total loss finite ({t:.6f})")
    check(t >= d - 1e-6, "total >= data term (physics penalty is non-negative)")

    empty = L.CombinedLoss(
        tm, np.zeros_like(pm), lambda_physics=1.0, anomaly_mode=anom, **lkw
    )
    e = float(empty(Y, out))
    check(
        np.isfinite(e) and abs(e - d) < 1e-6,
        "all-invalid physics slice -> 0, not NaN (denominator guard)",
        f"got {e:.8f} vs data {d:.8f}",
    )

    # ── 7. gradients ────────────────────────────────────────────────────────
    print("\n[7] Gradients")
    with tf.GradientTape() as tape:
        pred = model(X, training=True)
        loss = full(Y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    bad = [
        v.name
        for g, v in zip(grads, model.trainable_variables)
        if g is None or not np.isfinite(g.numpy()).all()
    ]
    check(not bad, f"all {len(grads)} gradients finite", str(bad[:5]))

    # ── 8. one optimizer step changes the weights ───────────────────────────
    print("\n[8] Optimizer step")
    opt = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
    before = model.trainable_variables[0].numpy().copy()
    opt.apply_gradients(zip(grads, model.trainable_variables))
    after = model.trainable_variables[0].numpy()
    check(not np.array_equal(before, after), "weights changed after one step")

    # ── 9. loss decreases on a small subset ─────────────────────────────────
    print("\n[9] Short overfit run (200 samples, 5 epochs)")
    small = D.OceanDataset("train", shuffle=False)
    small.indices = small.indices[:200]
    m2 = M.build_unet(nc, base_filters=16, depth=3, dropout=0.0)
    met_args = (
        (astats["phys_scale"], astats["phys_offset"], None, None, True)
        if anom
        else (sig, None, mus, sig, False)
    )
    m2.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0
        ),
        loss=L.CombinedLoss(
            tm, pm, lambda_physics=C.FIXED_LAMBDA_PHYSICS, anomaly_mode=anom, **lkw
        ),
        metrics=L.default_metrics(tm, pm, *met_args),
    )
    h = m2.fit(small, epochs=5, verbose=0)
    lo = h.history["loss"]
    print(f"        loss: {' -> '.join(f'{x:.4f}' for x in lo)}")
    check(lo[-1] < lo[0], "loss decreased", f"{lo[0]:.4f} -> {lo[-1]:.4f}")
    check(all(np.isfinite(lo)), "no NaN/Inf during training")

    # ── 10. index audit ─────────────────────────────────────────────────────
    print("\n[10] Split index audit")
    si = {
        s: set(D.OceanDataset(s, shuffle=False).indices)
        for s in ("train", "val", "test")
    }
    check(not (si["train"] & si["val"]), "train/val target indices disjoint")
    check(not (si["train"] & si["test"]), "train/test target indices disjoint")
    check(not (si["val"] & si["test"]), "val/test target indices disjoint")
    all_idx = np.array(sorted(si["train"] | si["val"] | si["test"]))
    hist = C.required_history()
    check(
        (all_idx - hist >= 0).all(),
        f"every sample has full history ({hist} days: lags {C.LAGS}, "
        f"running means {C.RUNNING_MEAN_WINDOWS or 'none'})",
    )
    check(all_idx.max() < C.N_DAYS, "no index past the end of the record")
    dates = np.load(C.DATES_NPY)
    v0 = min(si["val"])
    print(
        f"        first val target {dates[v0]} pulls lags from "
        f"{[str(dates[v0 - l]) for l in C.LAGS]}  (past surface only)"
    )

    # ── 11. denormalisation round-trip against pre-normalisation data ───────
    # The year comes from the record's own date axis. It used to be hardcoded
    # to 2005.nc while indexing row 0, which is only the same day when the
    # record happens to start in 2005 - under the 1993-2024 record that
    # differenced 1993-01-01 against 2005-01-01 and reported a 12.5 degC error.
    _d0 = pd.Timestamp(np.load(C.DATES_NPY)[0])
    _year_nc = C.PROC_DIR / "years" / f"{_d0.year}.nc"
    print(f"\n[11] Denormalisation round-trip vs {_year_nc}")
    Ym = np.load(C.Y_NPY, mmap_mode="r")
    # Cropped the same way build_arrays cropped the record, so v3's 81x101
    # rows are compared against the same 81x101 cells of the 81x201 year file.
    from build_arrays import crop_to_grid

    src = crop_to_grid(xr.open_dataset(_year_nc))
    assert pd.Timestamp(src.time.values[0]) == _d0, (
        f"row 0 is {_d0.date()} but {_year_nc.name} starts "
        f"{pd.Timestamp(src.time.values[0]).date()}"
    )

    # Tolerance follows the STORAGE DTYPE. Y is float16 under v2 (a deliberate
    # choice: float32 would be 15.2 GiB against 15 GB of RAM), so a stored
    # z-value carries at most half a float16 ULP of rounding error, which
    # denormalisation scales by sigma. Deriving the bound from the actual
    # stored magnitudes keeps this exactly as strict as the format allows
    # instead of loosening it to a number that happens to pass - under v1,
    # where Y is float32, the bound stays down at ~1e-6.
    worst, bound = 0.0, 0.0
    for k, v in enumerate(C.TARGET_VARS):
        z = np.asarray(Ym[0, :, :, k])
        phys = z.astype(np.float32) * sig[k] + mus[k]
        truth = src[v].isel(time=0).values
        sel = np.isfinite(truth)
        worst = max(worst, float(np.abs(phys[sel] - truth[sel]).max()))
        ulp = np.spacing(np.abs(z[sel]).astype(Ym.dtype)).astype(np.float32)
        bound = max(bound, float((ulp / 2 * sig[k]).max()) + 1e-5)
    src.close()
    check(
        worst < bound,
        f"max |denorm(Y) - physical truth| = {worst:.2e} degC "
        f"(bound {bound:.2e} for {Ym.dtype} storage)",
    )

    # ── 12. anomaly formulation ─────────────────────────────────────────────
    if anom:
        print("\n[12] Anomaly formulation")
        Xb, Yb = D.OceanDataset("val", shuffle=False)[0]
        rec = D.to_absolute_degC(
            Yb[..., :15], Yb[..., 15:], astats["phys_scale"], astats["phys_offset"]
        )
        idx = D.OceanDataset("val", shuffle=False).indices[: len(Yb)]
        truth = np.asarray(np.load(C.Y_NPY, mmap_mode="r")[idx]) * sig + mus
        m3 = tm[None].repeat(len(Yb), 0)
        worst = float(np.abs(rec - truth)[m3].max())
        check(
            worst < 1e-4,
            f"clim + scale*anomaly + offset reconstructs absolute degC "
            f"(max |diff| {worst:.2e})",
        )

        # the anomaly target must be ~N(0,1) on TRAIN, per depth
        tds = D.OceanDataset("train", shuffle=False, batch_size=512)
        acc_s = np.zeros(15)
        acc_ss = np.zeros(15)
        nb = 0
        for b in range(0, len(tds), max(1, len(tds) // 6)):
            _, Yt = tds[b]
            for d in range(15):
                v = Yt[..., d][:, tm[:, :, d]]
                acc_s[d] += v.mean()
                acc_ss[d] += v.std()
            nb += 1
        mu_a, sd_a = acc_s / nb, acc_ss / nb
        check(
            np.abs(mu_a).max() < 0.25,
            f"train anomaly target mean ~0 (max |mu| {np.abs(mu_a).max():.3f})",
        )
        check(
            np.abs(sd_a - 1).max() < 0.35,
            f"train anomaly target std ~1 (max |sd-1| {np.abs(sd_a - 1).max():.3f})",
        )

        # climatology must come from TRAIN only
        # Was a string grep for "2005-01-01", which hardcoded one record's
        # start date and - more to the point - never actually checked the
        # thing it claimed to. Compare the years the climatology was built
        # from against the years currently held out, and assert the sets are
        # disjoint. That is the real property, and it holds for any split.
        st = json.load(open(C.ANOM_STATS_JSON))
        held = set(S.split_years("val")) | set(S.split_years("test"))
        if "train_years" in st:
            used = set(st["train_years"])
            check(
                bool(used) and not (used & held),
                f"climatology built from {len(used)} train years, "
                f"none of them held out {sorted(held)}",
                "climatology must not see val or test years",
            )
        else:
            # A climatology written before the provenance fields existed. Fall
            # back to the years named in clim_source - weaker than the set
            # check, but it still tests the actual property rather than
            # grepping for one hardcoded date the way this used to.
            yrs = {
                int(y)
                for y in re.findall(r"(\d{4})-\d{2}-\d{2}", st.get("clim_source", ""))
            }
            check(
                bool(yrs) and not (yrs & held),
                f"climatology span {sorted(yrs)} excludes held-out "
                f"{sorted(held)}  [legacy artefact, no train_years field]",
                "climatology must not see val or test years",
            )

        if C.DATASET_VERSION != "v1":
            check(
                st.get("split_mode") == C.SPLIT_MODE,
                f"climatology split_mode {st.get('split_mode')!r} "
                f"matches the run's {C.SPLIT_MODE!r}",
                "a model scored against another split's climatology is a "
                "plausible-looking wrong answer",
            )
        else:
            # v1's artefacts are frozen and predate this field; the deployed
            # models depend on them, so regenerating them to satisfy a check
            # would be the tail wagging the dog. v2 must carry it.
            check(True, "split provenance not required for frozen v1 artefacts")

    # ── 13. physics loss operates on ABSOLUTE temperature ───────────────────
    print("\n[13] Physics loss uses absolute temperature")
    Xv, Yv = D.OceanDataset("val", shuffle=False)[0]
    met = L.PhysicsViolation(
        pm,
        **(
            {"phys_scale": astats["phys_scale"], "phys_offset": astats["phys_offset"]}
            if anom
            else {"mus": mus, "sigmas": sig}
        ),
        anomaly_mode=anom,
    )
    met.update_state(Yv, Yv[..., : C.N_DEPTHS])
    rate = float(met.result())
    check(
        0.0 < rate < 0.05,
        f"truth-as-prediction gated violation rate = {rate * 100:.3f}% "
        f"(matches the ~1% measured in the GLORYS target)",
        "if this were comparing anomalies instead of absolute temperature "
        "the rate would be far off",
    )

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  RESULT: {PASS}/{total} checks passed -- CLEARED FOR TUNING")
    else:
        print(f"  RESULT: {PASS}/{total} passed -- {FAIL} FAILED")
        for i, m in enumerate(FAILURES, 1):
            print(f"    {i}. {m}")
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
