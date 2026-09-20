import sys
from pathlib import Path

import keras
from keras import layers, ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C


def _groups_for(filters: int) -> int:
    g = C.GROUP_NORM_GROUPS
    while g > 1 and filters % g != 0:
        g //= 2
    return g


def _conv_block(x, filters: int, kernel: int, name: str):
    """Conv - GroupNorm - swish, twice.

    GroupNormalization rather than BatchNormalization: at batch 8-16, with
    ~44% of every field being land zeros, batch statistics are noisy and
    land-contaminated.
    """
    for i in (1, 2):
        x = layers.Conv2D(filters, kernel, padding="same", use_bias=False,
                          name=f"{name}_conv{i}")(x)
        x = layers.GroupNormalization(groups=_groups_for(filters),
                                      name=f"{name}_gn{i}")(x)
        x = layers.Activation(C.ACTIVATION, name=f"{name}_act{i}")(x)
    return x


@keras.saving.register_keras_serializable(package="oceanembed")
class MonotonicDepthHead(keras.layers.Layer):
    """Hard monotonic constraint below a chosen depth index (ablation).

    For k <= d0 the network output passes through untouched. For k > d0 the
    prediction is reparameterised in PHYSICAL units as a strictly positive
    decrement from the level above:

        T_k = T_{k-1} - softplus(raw_k)

    then mapped back to normalised space. This makes vertical inversion below
    level d0 structurally impossible.

    Off by default. Justified only below 100 m, where the GLORYS target itself
    inverts in <= 0.022% of valid cells (spec section 3.2). Note that it chains
    the deep predictions, so an error at level d0 propagates downward.
    """

    def __init__(self, mus, sigmas, d0, **kwargs):
        super().__init__(**kwargs)
        self.mus = [float(m) for m in mus]
        self.sigmas = [float(s) for s in sigmas]
        self.d0 = int(d0)

    def call(self, raw):
        outs = [raw[..., k] for k in range(self.d0 + 1)]
        t_prev = raw[..., self.d0] * self.sigmas[self.d0] + self.mus[self.d0]
        for k in range(self.d0 + 1, len(self.mus)):
            t_k = t_prev - ops.softplus(raw[..., k])
            outs.append((t_k - self.mus[k]) / self.sigmas[k])
            t_prev = t_k
        return ops.stack(outs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(mus=self.mus, sigmas=self.sigmas, d0=self.d0)
        return cfg


def build_unet(input_channels: int,
               base_filters: int = 32,
               depth: int = 4,
               kernel_size: int = 3,
               dropout: float = 0.1,
               hard_monotonic_below=None,
               mus=None, sigmas=None,
               name: str = "oceanembed_unet") -> keras.Model:
    """Return an uncompiled U-Net mapping (81,101,C) -> (81,101,15)."""
    if depth > C.MAX_UNET_DEPTH:
        raise ValueError(
            f"depth {depth} needs padding to a multiple of {2**depth}; "
            f"{C.PADDED_SHAPE} supports depth <= {C.MAX_UNET_DEPTH}")

    inp = layers.Input(shape=(C.N_LAT, C.N_LON, input_channels), name="inputs")
    x = layers.ZeroPadding2D((C.PAD_LAT, C.PAD_LON), name="pad")(inp)

    # ── encoder ─────────────────────────────────────────────────────────────
    skips = []
    for i in range(depth):
        f = base_filters * (2 ** i)
        x = _conv_block(x, f, kernel_size, name=f"enc{i}")
        if dropout > 0 and i == depth - 1:      # deepest encoder block only
            x = layers.SpatialDropout2D(dropout, name=f"enc{i}_drop")(x)
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"enc{i}_pool")(x)

    # ── bottleneck ──────────────────────────────────────────────────────────
    x = _conv_block(x, base_filters * (2 ** depth), kernel_size, name="bottleneck")
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name="bottleneck_drop")(x)

    # ── decoder ─────────────────────────────────────────────────────────────
    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same",
                                   name=f"dec{i}_up")(x)
        x = layers.Concatenate(name=f"dec{i}_cat")([x, skips[i]])
        x = _conv_block(x, f, kernel_size, name=f"dec{i}")

    # Linear head: this is continuous regression. No sigmoid, no softmax.
    x = layers.Conv2D(C.N_DEPTHS, 1, activation=None, name="head")(x)
    x = layers.Cropping2D((C.PAD_LAT, C.PAD_LON), name="crop")(x)

    if hard_monotonic_below is not None:
        if C.ANOMALISE_TARGET:
            # The head reparameterises PHYSICAL temperature as positive
            # decrements, which requires the absolute field. In anomaly mode
            # the model outputs a departure from a per-sample climatology it
            # cannot see, so the constraint cannot be applied inside the
            # network. Use the soft gated physics loss instead.
            raise ValueError(
                "hard_monotonic_below is incompatible with ANOMALISE_TARGET=True: "
                "the model predicts anomalies and has no access to the per-sample "
                "climatology needed to enforce ordering on absolute temperature. "
                "Set ANOMALISE_TARGET=False in train_config.py to use it.")
        if mus is None or sigmas is None:
            raise ValueError("hard_monotonic_below requires mus and sigmas")
        x = MonotonicDepthHead(mus, sigmas, hard_monotonic_below,
                               name="monotonic_head")(x)

    model = keras.Model(inp, x, name=name)
    out = model.output_shape
    assert out[1:] == (C.N_LAT, C.N_LON, C.N_DEPTHS), \
        f"output shape {out} != (None, {C.N_LAT}, {C.N_LON}, {C.N_DEPTHS})"
    return model
