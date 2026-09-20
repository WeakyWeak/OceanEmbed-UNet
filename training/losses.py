import sys
from pathlib import Path

import numpy as np
import keras
from keras import ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_config as C


def _f32(a):
    return ops.convert_to_tensor(np.asarray(a, dtype=np.float32))


def _split_true(y_true, anomaly_mode):
    """-> (anomaly-or-absolute target, climatology in degC or None)"""
    if anomaly_mode:
        return y_true[..., :C.N_DEPTHS], y_true[..., C.N_DEPTHS:]
    return y_true, None


def _absolute_degC(y_pred, clim_degC, scale, offset, mus, sigmas, anomaly_mode):
    """Model output -> absolute temperature in degC. Mirrors
    dataset.to_absolute_degC; keep the two in step."""
    if anomaly_mode:
        return clim_degC + y_pred * scale + offset
    return y_pred * sigmas + mus


# ── losses ───────────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="oceanembed")
class MaskedMSE(keras.losses.Loss):
    """Mean squared error in normalised space over valid target cells only.

    Kept in normalised space deliberately: since physical_MSE_d scales as
    (phys_scale_d)^2 * normalised_MSE_d, minimising the mean normalised MSE is
    equivalent to maximising the MEAN VARIANCE EXPLAINED across depths. The
    per-depth degC-per-unit scale spans 0.23 (1000 m) to 1.76 (100 m), so
    optimising in physical units would let the thermocline dominate and leave
    the deep ocean untrained. Optimise normalised; report in degC.
    """

    def __init__(self, target_mask, anomaly_mode=True, name="masked_mse", **kw):
        super().__init__(name=name, **kw)
        self._tm = np.asarray(target_mask, dtype=np.float32)
        self.anomaly_mode = bool(anomaly_mode)
        self.mask = _f32(self._tm)
        self.denom = float(max(self._tm.sum(), 1.0))

    def call(self, y_true, y_pred):
        t, _ = _split_true(y_true, self.anomaly_mode)
        se = self.mask * ops.square(t - y_pred)
        return ops.sum(se, axis=[1, 2, 3]) / self.denom      # per sample

    def get_config(self):
        cfg = super().get_config()
        cfg.update(target_mask=self._tm.tolist(), anomaly_mode=self.anomaly_mode)
        return cfg


@keras.saving.register_keras_serializable(package="oceanembed")
class CombinedLoss(keras.losses.Loss):
    """L_total = masked_MSE + lambda_physics * gated ordering penalty.

    The ordering penalty is a squared hinge on adjacent depth pairs, computed
    in PHYSICAL degC, weighted by a per-pair gate derived from how often the
    GLORYS target itself violates T_deep <= T_shallow (train_config.PHYSICS_GATE).
    Pairs above 50 m carry weight 0: the truth inverts there 22-40% of the time,
    so penalising them would train the model to contradict its own target and
    erase the Bay of Bengal barrier layer.

    Because the gated term is near-zero on well-behaved predictions it fires
    only on real violations, so lambda_physics can safely reach 1.0.
    """

    def __init__(self, target_mask, pair_mask, lambda_physics=0.0,
                 phys_scale=None, phys_offset=None, mus=None, sigmas=None,
                 anomaly_mode=True, gate=None, margin=C.PHYSICS_MARGIN,
                 name="combined_loss", **kw):
        super().__init__(name=name, **kw)
        self.anomaly_mode = bool(anomaly_mode)
        self.lambda_physics = float(lambda_physics)
        self.margin = float(margin)

        self._tm = np.asarray(target_mask, dtype=np.float32)
        self._pm = np.asarray(pair_mask, dtype=np.float32)
        self._gate = np.asarray(C.PHYSICS_GATE if gate is None else gate,
                                dtype=np.float32)
        z = np.zeros(C.N_DEPTHS, np.float32)
        self._scale = np.asarray(phys_scale if phys_scale is not None else z,
                                 dtype=np.float32)
        self._offset = np.asarray(phys_offset if phys_offset is not None else z,
                                  dtype=np.float32)
        self._mus = np.asarray(mus if mus is not None else z, dtype=np.float32)
        self._sig = np.asarray(sigmas if sigmas is not None else z, dtype=np.float32)

        self.mask = _f32(self._tm)
        self.scale, self.offset = _f32(self._scale), _f32(self._offset)
        self.mus_t, self.sig_t = _f32(self._mus), _f32(self._sig)
        self.data_denom = float(max(self._tm.sum(), 1.0))

        w = self._pm * self._gate                      # (81, 101, 14)
        self.w = _f32(w)
        self.phys_denom = float(max(w.sum(), 1.0))

    def call(self, y_true, y_pred):
        t, clim = _split_true(y_true, self.anomaly_mode)
        se = self.mask * ops.square(t - y_pred)
        data = ops.sum(se, axis=[1, 2, 3]) / self.data_denom

        if self.lambda_physics == 0.0:
            return data

        T = _absolute_degC(y_pred, clim, self.scale, self.offset,
                           self.mus_t, self.sig_t, self.anomaly_mode)
        diff = T[..., 1:] - T[..., :-1]                # >0 means deeper warmer
        viol = ops.relu(diff + self.margin)
        phys = ops.sum(self.w * ops.square(viol), axis=[1, 2, 3]) / self.phys_denom
        return data + self.lambda_physics * phys

    def get_config(self):
        cfg = super().get_config()
        cfg.update(target_mask=self._tm.tolist(), pair_mask=self._pm.tolist(),
                   phys_scale=self._scale.tolist(), phys_offset=self._offset.tolist(),
                   mus=self._mus.tolist(), sigmas=self._sig.tolist(),
                   anomaly_mode=self.anomaly_mode, gate=self._gate.tolist(),
                   lambda_physics=self.lambda_physics, margin=self.margin)
        return cfg


# ── metrics ──────────────────────────────────────────────────────────────────

class _AccumMetric(keras.metrics.Metric):
    """Population-weighted accumulator: exact across variable batch sizes."""

    def __init__(self, name, **kw):
        super().__init__(name=name, **kw)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def result(self):
        return self.total / ops.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable(package="oceanembed")
class MaskedMSEMetric(_AccumMetric):
    """The KerasTuner objective. Must be the DATA term alone: lambda_physics
    varies across trials, so val_loss is not comparable between them."""

    def __init__(self, target_mask, anomaly_mode=True, name="masked_mse", **kw):
        super().__init__(name=name, **kw)
        self._tm = np.asarray(target_mask, dtype=np.float32)
        self.anomaly_mode = bool(anomaly_mode)
        self.mask = _f32(self._tm)
        self.n_valid = float(self._tm.sum())

    def update_state(self, y_true, y_pred, sample_weight=None):
        t, _ = _split_true(y_true, self.anomaly_mode)
        b = ops.cast(ops.shape(y_pred)[0], "float32")
        self.total.assign_add(ops.sum(self.mask * ops.square(t - y_pred)))
        self.count.assign_add(self.n_valid * b)


@keras.saving.register_keras_serializable(package="oceanembed")
class MaskedMAEMetric(_AccumMetric):
    def __init__(self, target_mask, anomaly_mode=True, name="masked_mae", **kw):
        super().__init__(name=name, **kw)
        self._tm = np.asarray(target_mask, dtype=np.float32)
        self.anomaly_mode = bool(anomaly_mode)
        self.mask = _f32(self._tm)
        self.n_valid = float(self._tm.sum())

    def update_state(self, y_true, y_pred, sample_weight=None):
        t, _ = _split_true(y_true, self.anomaly_mode)
        b = ops.cast(ops.shape(y_pred)[0], "float32")
        self.total.assign_add(ops.sum(self.mask * ops.abs(t - y_pred)))
        self.count.assign_add(self.n_valid * b)


@keras.saving.register_keras_serializable(package="oceanembed")
class MaskedRMSEDegC(_AccumMetric):
    """Physical RMSE in degC. The climatology cancels in the difference, so the
    error scales by phys_scale (anomaly mode) or sigma (legacy). Reported,
    never optimised - see MaskedMSE for why the loss stays normalised."""

    def __init__(self, target_mask, scale, anomaly_mode=True,
                 name="masked_rmse_degC", **kw):
        super().__init__(name=name, **kw)
        self._tm = np.asarray(target_mask, dtype=np.float32)
        self._scale = np.asarray(scale, dtype=np.float32)
        self.anomaly_mode = bool(anomaly_mode)
        self.wsq = _f32(self._tm * self._scale ** 2)
        self.n_valid = float(self._tm.sum())

    def update_state(self, y_true, y_pred, sample_weight=None):
        t, _ = _split_true(y_true, self.anomaly_mode)
        b = ops.cast(ops.shape(y_pred)[0], "float32")
        self.total.assign_add(ops.sum(self.wsq * ops.square(t - y_pred)))
        self.count.assign_add(self.n_valid * b)

    def result(self):
        return ops.sqrt(self.total / ops.maximum(self.count, 1.0))


@keras.saving.register_keras_serializable(package="oceanembed")
class PhysicsViolation(_AccumMetric):
    """Fraction of GATED valid adjacent pairs where T_deep > T_shallow, on
    absolute temperature. The gate is used as a binary selector, not a weight,
    so the number reads as a plain violation rate over the pairs the constraint
    is claimed to apply to."""

    def __init__(self, pair_mask, phys_scale=None, phys_offset=None,
                 mus=None, sigmas=None, anomaly_mode=True, gate=None,
                 name="physics_violation", **kw):
        super().__init__(name=name, **kw)
        self.anomaly_mode = bool(anomaly_mode)
        self._pm = np.asarray(pair_mask, dtype=np.float32)
        self._gate = np.asarray(C.PHYSICS_GATE if gate is None else gate,
                                dtype=np.float32)
        z = np.zeros(C.N_DEPTHS, np.float32)
        self._scale = np.asarray(phys_scale if phys_scale is not None else z, np.float32)
        self._offset = np.asarray(phys_offset if phys_offset is not None else z, np.float32)
        self._mus = np.asarray(mus if mus is not None else z, np.float32)
        self._sig = np.asarray(sigmas if sigmas is not None else z, np.float32)

        sel = self._pm * (self._gate > 0).astype(np.float32)
        self.sel = _f32(sel)
        self.n_sel = float(max(sel.sum(), 1.0))
        self.scale, self.offset = _f32(self._scale), _f32(self._offset)
        self.mus_t, self.sig_t = _f32(self._mus), _f32(self._sig)

    def update_state(self, y_true, y_pred, sample_weight=None):
        _, clim = _split_true(y_true, self.anomaly_mode)
        b = ops.cast(ops.shape(y_pred)[0], "float32")
        T = _absolute_degC(y_pred, clim, self.scale, self.offset,
                           self.mus_t, self.sig_t, self.anomaly_mode)
        d = T[..., 1:] - T[..., :-1]
        self.total.assign_add(ops.sum(self.sel * ops.cast(d > 0.0, "float32")))
        self.count.assign_add(self.n_sel * b)


def default_metrics(target_mask, pair_mask, scale, offset=None,
                    mus=None, sigmas=None, anomaly_mode=True):
    """The four metrics every run reports. masked_mse is the tuner objective.

    anomaly mode: pass scale=phys_scale, offset=phys_offset.
    legacy mode : pass scale=sigmas, and mus/sigmas for the physics metric.
    """
    return [
        MaskedMSEMetric(target_mask, anomaly_mode),
        MaskedMAEMetric(target_mask, anomaly_mode),
        MaskedRMSEDegC(target_mask, scale, anomaly_mode),
        PhysicsViolation(pair_mask, phys_scale=scale, phys_offset=offset,
                         mus=mus, sigmas=sigmas, anomaly_mode=anomaly_mode),
    ]
