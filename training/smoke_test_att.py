#!/usr/bin/env python3
"""
Hard gate before tune_att.py. Do not start 25 trials until this is clean.

Checks the things that would waste a whole search if they were wrong: channel
layout, the satellite-only subset, attention actually doing something,
serialisation of the custom positional-embedding layer, gradient health, and
that no future information reaches a training sample.

    .venv/bin/python training/smoke_test_att.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as CB
import splits as S
import train_config_att as C
import dataset as D
import dataset_att as DA
import model as MB
import model_att as M
import losses as L

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))
    return cond


print("=" * 78)
print("  smoke_test_att.py - satellite-only attention U-Net")
print("=" * 78)

# ── 1. configuration ────────────────────────────────────────────────────────
print("\n1. configuration")
# Under v2 the whole dataset is satellite-only, so the attention model must
# take the SAME inputs as the base model. That identity is the precondition
# for the A/B being about the architecture: att_v1 was uninterpretable
# precisely because it changed inputs and architecture together.
# Under v1 the base model still had reanalysis inputs, so the attention run
# carried its own three-variable satellite set.
if C.DATASET_VERSION != "v1":
    check("inputs identical to the base model (clean A/B)",
          list(C.INPUT_VARS_ATT) == list(CB.INPUT_VARS),
          f"{C.INPUT_VARS_ATT}")
    check("anomaly flags identical to the base model",
          {k: C.ANOMALISE_INPUTS_ATT[k] for k in C.INPUT_VARS_ATT}
          == {k: CB.ANOMALISE_INPUTS[k] for k in CB.INPUT_VARS},
          "otherwise the A/B measures the formulation, not the architecture")
else:
    check("satellite-only input set",
          C.INPUT_VARS_ATT == ["sst", "sla", "sss"], f"{C.INPUT_VARS_ATT}")
check("no reanalysis inputs",
      not any(v in C.INPUT_VARS_ATT for v in ("uo", "vo", "u10", "v10")),
      "uo/vo (GLORYS) and u10/v10 (ERA5) excluded")
check(f"channel count = {C.N_CHANNELS_ATT}",
      C.N_CHANNELS_ATT == C.n_input_channels_att(),
      f"3 vars x 3 lags + land + 2 coord + 2 doy = {C.N_CHANNELS_ATT}")
check("shares the target formulation with the other runs",
      C.ANOMALISE_TARGET == CB.ANOMALISE_TARGET and C.N_DEPTHS == 15)
check("sla anomalised", C.ANOMALISE_INPUTS_ATT["sla"] is True)

# ── 2. dataset equivalence ──────────────────────────────────────────────────
print("\n2. dataset")
a = D.OceanDataset("val", batch_size=4, shuffle=False)
b = DA.SubsetOceanDataset("val", input_vars=CB.INPUT_VARS, batch_size=4,
                          shuffle=False)
same = (a.n_channels == b.n_channels and
        all(np.array_equal(a[i][0], b[i][0]) and np.array_equal(a[i][1], b[i][1])
            for i in range(5)))
check("SubsetOceanDataset(all 7) == OceanDataset", same,
      "one class can serve every ensemble member")

s = DA.SubsetOceanDataset("val", input_vars=C.INPUT_VARS_ATT, batch_size=4,
                          shuffle=False,
                          anomalise_inputs=C.ANOMALISE_INPUTS_ATT)
X, Y = s[0]
check("satellite X shape",
      X.shape == (4, CB.N_LAT, CB.N_LON, C.N_CHANNELS_ATT), f"{X.shape}")
check("y_true carries 30 channels", Y.shape[-1] == 30, f"{Y.shape}")
check("no NaN/Inf in a batch", np.isfinite(X).all() and np.isfinite(Y).all())

full = a[0][0]
check("channel order preserved across the subset",
      np.array_equal(X[..., 0], full[..., 0]) and      # lag0 sst
      np.array_equal(X[..., C.N_INPUT_VARS_ATT],
                     full[..., CB.N_INPUT_VARS]),       # lag1 sst
      f"lag blocks are contiguous, {C.N_INPUT_VARS_ATT} vars wide")
# Running-mean blocks sit between the lag blocks and land, so land is no
# longer at len(LAGS)*n_vars. Getting this wrong is the whole point of the
# companion check below: if land drifted, the slot before it would be binary
# too and a lone <= {0,1} assertion would pass by coincidence.
land_ch = (len(CB.LAGS) + len(CB.RUNNING_MEAN_WINDOWS)) * C.N_INPUT_VARS_ATT
check("land channel lands in the right slot",
      set(np.unique(X[..., land_ch])) <= {0.0, 1.0}, f"channel {land_ch}")
check("the slot before land is NOT binary",
      not set(np.unique(X[..., land_ch - 1])) <= {0.0, 1.0},
      f"channel {land_ch - 1} is a geophysical field, not a mask")

# ── 3. anomalisation ────────────────────────────────────────────────────────
print("\n3. anomaly formulation")
raw = np.asarray(np.load(CB.X_NPY, mmap_mode="r")[s.indices[:4]], np.float32)
check("sst IS anomalised", not np.allclose(X[..., 0], raw[..., 0]))
check("sla IS anomalised", not np.allclose(X[..., 1], raw[..., 1]))
check("sss IS anomalised", not np.allclose(X[..., 2], raw[..., 2]))

an = D.load_anom_stats()
mus, sig = D.load_norm_arrays()
tm, pm, _ = D.load_masks()
import pandas as pd
doy = pd.DatetimeIndex(np.load(CB.DATES_NPY)).dayofyear.values[s.indices[:4]]
clim_abs = np.asarray(np.load(CB.CLIM_Y_NPY, mmap_mode="r")[doy], np.float32) * sig + mus
recon = D.to_absolute_degC(Y[..., :15], clim_abs, an["phys_scale"], an["phys_offset"])
truth = np.asarray(np.load(CB.Y_NPY, mmap_mode="r")[s.indices[:4]], np.float32) * sig + mus
err = np.abs(recon - truth)[tm[None].repeat(4, 0)].max()
check("anomaly -> absolute degC round trip exact", err < 1e-3,
      f"max |diff| = {err:.3e} degC")

# ── 4. architecture ─────────────────────────────────────────────────────────
print("\n4. model")
m3 = M.build_attention_unet(C.N_CHANNELS_ATT, base_filters=16, depth=3,
                            num_heads=2)
m4 = M.build_attention_unet(C.N_CHANNELS_ATT, base_filters=16, depth=4,
                            num_heads=2)
check(f"depth 3 output is {CB.N_DEPTHS} channels",
      m3.output_shape == (None, CB.N_LAT, CB.N_LON, CB.N_DEPTHS),
      f"{m3.output_shape}")
check(f"depth 4 output is {CB.N_DEPTHS} channels",
      m4.output_shape == (None, CB.N_LAT, CB.N_LON, CB.N_DEPTHS),
      f"{m4.output_shape}")
check("model output width unchanged by attention",
      m3.output_shape == MB.build_unet(C.N_CHANNELS_ATT, base_filters=16,
                                       depth=3).output_shape,
      "attention changes HOW the field is computed, never its shape")

emb = M.build_embedding_extractor(m3)
Z = emb.predict(X, verbose=0)
_bh = CB.PADDED_SHAPE[0] // 8      # depth 3 -> /2^3
_bw = CB.PADDED_SHAPE[1] // 8
check("embedding extractor works",
      Z.shape == (4, _bh, _bw, 16 * 8),
      f"{Z.shape}  bottleneck {_bh}x{_bw} = {_bh*_bw} tokens")
check("embedding is finite and non-degenerate",
      np.isfinite(Z).all() and Z.std() > 0, f"std={Z.std():.4f}")

na = M.build_attention_unet(C.N_CHANNELS_ATT, base_filters=16, depth=3,
                            num_heads=2,
                            use_bottleneck_attention=False,
                            use_skip_attention=False)
check("attention adds parameters", m3.count_params() > na.count_params(),
      f"{m3.count_params():,} vs {na.count_params():,} without")
names = [l.name for l in m3.layers]
check("bottleneck self-attention present", any("mhsa" in n for n in names))
check("skip attention gates present", any("gate" in n for n in names))

# ── 5. serialisation ────────────────────────────────────────────────────────
print("\n5. save / reload (custom PositionalEmbedding must survive)")
p0 = m3.predict(X, verbose=0)
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "t.keras"
    m3.save(f)
    r = keras.models.load_model(f, compile=False)
    p1 = r.predict(X, verbose=0)
    check("reloads and reproduces predictions", np.allclose(p0, p1, atol=1e-5),
          f"max |diff| = {np.abs(p0 - p1).max():.2e}")
    check("embedding extractable from a RELOADED model",
          M.build_embedding_extractor(r).output_shape == emb.output_shape,
          "models/ depends on this")

# ── 6. loss and gradients ───────────────────────────────────────────────────
print("\n6. loss / gradients")
loss = L.CombinedLoss(tm, pm, lambda_physics=CB.FIXED_LAMBDA_PHYSICS,
                      anomaly_mode=True, phys_scale=an["phys_scale"],
                      phys_offset=an["phys_offset"])
m3.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4,
                                            clipnorm=1.0),
           loss=loss,
           metrics=L.default_metrics(tm, pm, an["phys_scale"],
                                     an["phys_offset"], None, None, True))
l0 = float(loss(keras.ops.convert_to_tensor(Y),
                keras.ops.convert_to_tensor(p0)).numpy().mean())
check("loss is finite on an untrained model", np.isfinite(l0), f"{l0:.4f}")

h = m3.fit(s, epochs=1, steps_per_epoch=3, verbose=0)
vals = {k: v[-1] for k, v in h.history.items()}
check("all metrics finite after a training step",
      all(np.isfinite(v) for v in vals.values()),
      ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))

with keras.src.backend.tensorflow.core.tf.GradientTape() as tape:
    pr = m3(X, training=True)
    lv = loss(keras.ops.convert_to_tensor(Y), pr)
grads = tape.gradient(lv, m3.trainable_weights)
nz = [g for g in grads if g is not None]
check("every trainable weight receives a gradient", len(nz) == len(grads),
      f"{len(nz)}/{len(grads)}")
check("no NaN gradients (the ops.where trap)",
      all(bool(np.isfinite(g.numpy()).all()) for g in nz))

# ── 7. physics loss on absolute temperature ────────────────────────────────
print("\n7. physics")
pv = L.PhysicsViolation(pm, phys_scale=an["phys_scale"],
                        phys_offset=an["phys_offset"], anomaly_mode=True)
pv.update_state(keras.ops.convert_to_tensor(Y),
                keras.ops.convert_to_tensor(Y[..., :15]))
rate = float(pv.result().numpy())
check("truth-as-prediction matches the target's own violation rate",
      0.0 < rate < 0.05, f"{rate*100:.3f}% (GLORYS itself is ~1%)")
check("physics gate silences the barrier layer",
      CB.PHYSICS_GATE[:5].sum() == 0 and CB.PHYSICS_GATE[7:].min() == 1.0)

# ── 8. no leakage ───────────────────────────────────────────────────────────
print("\n8. split hygiene")
tr = DA.SubsetOceanDataset("train", input_vars=C.INPUT_VARS_ATT, batch_size=4)
va = DA.SubsetOceanDataset("val", input_vars=C.INPUT_VARS_ATT, batch_size=4)
te = DA.SubsetOceanDataset("test", input_vars=C.INPUT_VARS_ATT, batch_size=4)
check("train / val / test targets are disjoint",
      not (set(tr.indices) & set(va.indices)) and
      not (set(va.indices) & set(te.indices)) and
      not (set(tr.indices) & set(te.indices)))
check("no training target reaches past its split",
      tr.indices.max() < S.split_indices("val", 0).min(),
      f"max train target index {tr.indices.max()}")
check("lagged inputs only ever look BACKWARD",
      min(C.LAGS) == 0 and all(l >= 0 for l in C.LAGS), f"lags={C.LAGS}")

# ── summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f_ in FAIL:
        print(f"    FAILED: {f_}")
    print("=" * 78)
    sys.exit(1)
print("  CLEARED FOR TUNING:")
print("    .venv/bin/python training/tune_att.py --project att_v1")
print("=" * 78)
