"""
Satellite-only attention U-Net, and the embedding extractor.

Architecture
------------
The same padded U-Net as model.py (81x101 -> 96x112 -> crop back), with two
attention mechanisms added:

1. BOTTLENECK SELF-ATTENTION - the one that matters.
   The bottleneck feature map is flattened to a token sequence, given a learned
   positional embedding, and passed through a pre-norm transformer block
   (multi-head self-attention + MLP, both residual). Every bottleneck location
   can then read every other location in ONE hop.

   This is aimed at a measured failure. The primary U-Net's effective receptive
   field holds 50% of its input gradient within 7x10 px and only reaches 1% at
   60x65 px, so it is close to a local filter. But thermocline depth responds
   to basin-scale sea level, and a single-parameter linear fit on domain-mean
   SLA recovered ~60% of the 2021 deep warm anomaly where the U-Net recovered
   ~30%. Stacked 3x3 convolutions can only grow their receptive field
   linearly; attention makes it global immediately.

   Cost is negligible here: the bottleneck is 12x14 = 168 tokens at depth 3,
   6x7 = 42 at depth 4. This is not a case where attention needs windowing.

2. SKIP ATTENTION GATES (Oktay et al., "Attention U-Net").
   Each skip connection is multiplied by a spatial gate computed from the
   decoder features at that level, so the decoder can suppress skip features
   that are irrelevant at that depth. Applied AFTER upsampling, where the
   gating signal and the skip already share a spatial size - the original
   paper resamples instead, which buys nothing and adds shape fragility.

The satellite embedding deliverable
-----------------------------------
The problem statement asks for a "Satellite embedding engine capable of
learning latent ocean representations from surface observations". The
post-attention bottleneck IS that latent representation: surface fields
compressed to a compact spatial grid of learned features, from which the
decoder reconstructs the full 15-level water column. It is named
`embedding` so build_embedding_extractor() can pull it out of a saved model
without knowing anything about the architecture.
"""
import sys
from pathlib import Path

import keras
from keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config_att as C
from model import _conv_block, _groups_for      # noqa: F401  (shared blocks)


@keras.saving.register_keras_serializable(package="oceanembed")
class PositionalEmbedding(keras.layers.Layer):
    """Learned additive positional embedding over a flattened feature map.

    Self-attention is permutation invariant, so without this the bottleneck
    would be a bag of locations and the model could not tell the northern Bay
    of Bengal from the southern. Learned rather than sinusoidal because the
    domain is a fixed, small, non-square grid - there is nothing to extrapolate
    to, so there is no reason to pay for a hand-designed basis.
    """

    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.embed_dim = int(embed_dim)

    def build(self, input_shape):
        self.pos = self.add_weight(
            shape=(1, self.sequence_length, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True, name="pos_embedding")
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sequence_length=self.sequence_length,
                   embed_dim=self.embed_dim)
        return cfg


def _transformer_block(x, h, w, filters, num_heads, dropout, name):
    """Pre-norm MHSA + MLP over the flattened (h*w) bottleneck tokens."""
    seq = h * w
    key_dim = max(filters // num_heads, 8)

    t = layers.Reshape((seq, filters), name=f"{name}_flat")(x)
    t = PositionalEmbedding(seq, filters, name=f"{name}_pos")(t)

    # Pre-norm: normalise before the sublayer, add the residual after. More
    # stable than post-norm at this depth and needs no warmup schedule.
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(t)
    y = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                  dropout=dropout,
                                  name=f"{name}_mhsa")(y, y)
    t = layers.Add(name=f"{name}_add1")([t, y])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(t)
    y = layers.Dense(int(filters * C.ATTENTION_MLP_RATIO),
                     activation=C.ACTIVATION, name=f"{name}_mlp1")(y)
    y = layers.Dropout(dropout, name=f"{name}_mlp_drop")(y)
    y = layers.Dense(filters, name=f"{name}_mlp2")(y)
    t = layers.Add(name=f"{name}_add2")([t, y])

    return layers.Reshape((h, w, filters), name=f"{name}_unflat")(t)


def _attention_gate(skip, gating, filters, name):
    """Multiply a skip connection by a learned spatial gate in [0,1]."""
    inter = max(filters // 2, 8)
    theta = layers.Conv2D(inter, 1, use_bias=False, name=f"{name}_theta")(skip)
    phi = layers.Conv2D(inter, 1, use_bias=True, name=f"{name}_phi")(gating)
    f = layers.Activation("relu", name=f"{name}_relu")(
        layers.Add(name=f"{name}_add")([theta, phi]))
    alpha = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name}_psi")(f)
    return layers.Multiply(name=f"{name}_mul")([skip, alpha])


def build_attention_unet(input_channels: int,
                         base_filters: int = 32,
                         depth: int = 3,
                         kernel_size: int = 3,
                         dropout: float = 0.2,
                         num_heads: int = 4,
                         attention_dropout: float = None,
                         use_bottleneck_attention: bool = None,
                         use_skip_attention: bool = None,
                         n_blocks: int = None,
                         name: str = "oceanembed_attention_unet") -> keras.Model:
    """Uncompiled attention U-Net mapping (81,101,C) -> (81,101,15).

    Output is 15 channels - the normalised anomaly at each depth - exactly like
    build_unet. Attention changes how the field is computed, never its shape.
    """
    if depth > C.MAX_UNET_DEPTH:
        raise ValueError(
            f"depth {depth} needs padding to a multiple of {2**depth}; "
            f"{C.PADDED_SHAPE} supports depth <= {C.MAX_UNET_DEPTH}")

    use_bn_attn = (C.USE_BOTTLENECK_ATTENTION if use_bottleneck_attention is None
                   else use_bottleneck_attention)
    use_sk_attn = (C.USE_SKIP_ATTENTION if use_skip_attention is None
                   else use_skip_attention)
    n_blocks = C.N_ATTENTION_BLOCKS if n_blocks is None else n_blocks
    attn_drop = (C.ATTENTION_DROPOUT if attention_dropout is None
                 else attention_dropout)

    inp = layers.Input(shape=(C.N_LAT, C.N_LON, input_channels), name="inputs")
    x = layers.ZeroPadding2D((C.PAD_LAT, C.PAD_LON), name="pad")(inp)

    # ── encoder ─────────────────────────────────────────────────────────────
    skips = []
    for i in range(depth):
        f = base_filters * (2 ** i)
        x = _conv_block(x, f, kernel_size, name=f"enc{i}")
        if dropout > 0 and i == depth - 1:
            x = layers.SpatialDropout2D(dropout, name=f"enc{i}_drop")(x)
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"enc{i}_pool")(x)

    # ── bottleneck ──────────────────────────────────────────────────────────
    bf = base_filters * (2 ** depth)
    x = _conv_block(x, bf, kernel_size, name="bottleneck")

    bh = C.PADDED_SHAPE[0] // (2 ** depth)
    bw = C.PADDED_SHAPE[1] // (2 ** depth)
    if use_bn_attn:
        for b in range(n_blocks):
            x = _transformer_block(x, bh, bw, bf, num_heads, attn_drop,
                                   name=f"attn{b}")

    # The satellite embedding: surface observations compressed to a compact
    # latent grid. Everything downstream is reconstruction from this tensor.
    x = layers.Activation("linear", name=C.EMBEDDING_LAYER_NAME)(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name="bottleneck_drop")(x)

    # ── decoder ─────────────────────────────────────────────────────────────
    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same",
                                   name=f"dec{i}_up")(x)
        s = (_attention_gate(skips[i], x, f, name=f"dec{i}_gate")
             if use_sk_attn else skips[i])
        x = layers.Concatenate(name=f"dec{i}_cat")([x, s])
        x = _conv_block(x, f, kernel_size, name=f"dec{i}")

    x = layers.Conv2D(C.N_DEPTHS, 1, activation=None, name="head")(x)
    x = layers.Cropping2D((C.PAD_LAT, C.PAD_LON), name="crop")(x)

    model = keras.Model(inp, x, name=name)
    out = model.output_shape
    assert out[1:] == (C.N_LAT, C.N_LON, C.N_DEPTHS), \
        f"output shape {out} != (None, {C.N_LAT}, {C.N_LON}, {C.N_DEPTHS})"
    return model


def build_embedding_extractor(model: keras.Model) -> keras.Model:
    """Sub-model returning the satellite embedding instead of the temperature.

    Usage:
        m = keras.models.load_model("runs/att_best/best.keras", compile=False)
        emb = build_embedding_extractor(m)
        Z = emb.predict(ds)          # (N, H, W, bottleneck_filters)

    Serves the problem statement's "satellite embedding engine" deliverable,
    and is what you would visualise or cluster in the frontend to show that the
    latent space has learned something oceanographic.
    """
    try:
        layer = model.get_layer(C.EMBEDDING_LAYER_NAME)
    except ValueError as e:
        raise ValueError(
            f"no layer named {C.EMBEDDING_LAYER_NAME!r} in {model.name!r}. "
            "Embedding extraction only works on models built by "
            "build_attention_unet.") from e
    return keras.Model(model.inputs, layer.output,
                       name=f"{model.name}_embedding")
