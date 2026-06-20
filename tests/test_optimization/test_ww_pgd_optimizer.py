"""Tests for the WW-PGD spectral tail-projection callback.

Covers the Step-1 module ``dl_techniques.optimization.ww_pgd_optimizer``:

* SC1 -- OFF / warmup / cadence no-op (byte-identity).
* SC2 -- alpha-toward-2 convergence on a synthetic heavy-tailed matrix (the
  Pre-Mortem falsification gate: the projection must move the tail TOWARD the
  q=1 template, not diverge), with a differential OFF control.
* SC3 -- trace-log invariant (the Cayley + retract step preserves
  ``sum(log(lam_tail))``).
* SC4 -- fail-soft on a degenerate (all-zeros) layer while a healthy sibling
  is still projected.
* SC5 -- Conv2D reshape round-trip (kernel keeps ``(kH,kW,Cin,Cout)``).
* SC6 -- serialization round-trip of the config and the callback (no live model).

Class-based, mirroring ``tests/test_optimization/test_sgld_optimizer.py`` /
``test_gefen_optimizer.py``. Each test fixes a numpy seed for determinism and the
whole suite is scoped to stay well under a minute.
"""

import numpy as np
import pytest
import keras

from dl_techniques.optimization.ww_pgd_optimizer import (
    WWTailConfig,
    ww_pgd_project,
    WWPGDProjectionCallback,
)
from dl_techniques.analyzer.spectral_metrics import (
    fit_powerlaw,
    compute_detX_constraint,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _heavy_tail_matrix(n: int = 256, m: int = 256, exponent: float = 2.0,
                       seed: int = 0) -> np.ndarray:
    """Build an ``(n, m)`` matrix with a steep power-law singular spectrum.

    The singular values follow ``S_i = i**(-exponent)`` (i from 1), so the
    squared-singular-value spectrum has a heavy tail far from the q=1 template
    (its fitted power-law alpha is well below 2). Returned as float32.
    """
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(n, n))
    V, _ = np.linalg.qr(rng.randn(m, m))
    k = min(n, m)
    s = (np.arange(1, k + 1).astype(np.float64)) ** (-exponent)
    W = (U[:, :k] * s) @ V[:k, :]
    return W.astype(np.float32)


def _dense_with_kernel(W: np.ndarray) -> keras.Sequential:
    """Bias-free Dense model whose kernel equals ``W`` (kernel shape == W.shape)."""
    in_dim, out_dim = W.shape
    model = keras.Sequential([
        keras.Input((in_dim,)),
        keras.layers.Dense(out_dim, use_bias=False),
    ])
    model.layers[0].kernel.assign(W)
    return model


def _kernel_np(layer: keras.layers.Layer) -> np.ndarray:
    return np.array(keras.ops.convert_to_numpy(layer.kernel))


def _fit_alpha(kernel: np.ndarray) -> float:
    """Fitted power-law alpha of a 2D kernel's squared singular values."""
    s = np.linalg.svd(kernel, full_matrices=False, compute_uv=False)
    lam = np.maximum(s, 1e-8) ** 2
    alpha, _xmin, _D, _sigma, _npl, _status, _w = fit_powerlaw(lam)
    return float(alpha)


# ---------------------------------------------------------------------
# SC2 -- alpha-toward-2 convergence (falsification gate)
# ---------------------------------------------------------------------

class TestWWPGDAlphaConvergence:
    """SC2: the projection moves the heavy tail TOWARD the q=1 template."""

    def test_alpha_moves_toward_two_not_diverge(self):
        """Iterated projection drives fitted alpha monotonically toward ~2.

        STOP-IF gate (Pre-Mortem): if alpha moves AWAY from 2 / the log-target
        distance grows, the Cayley/template math is mis-ported -- this assertion
        must NOT be weakened to pass.
        """
        np.random.seed(0)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=0)
        model = _dense_with_kernel(W)
        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=5, q=1.0,
        )

        alpha0 = _fit_alpha(_kernel_np(model.layers[0]))
        assert alpha0 < 1.7, "synthetic matrix should start far below alpha=2"

        alphas = [alpha0]
        for _ in range(10):
            summary = ww_pgd_project(model, cfg, epoch=2, num_epochs=3)
            assert summary["layers_projected"] == 1
            alphas.append(_fit_alpha(_kernel_np(model.layers[0])))

        alpha_final = alphas[-1]
        # Converges toward 2 (never diverges away from it).
        assert alpha_final > alpha0, "alpha must increase toward ~2"
        assert abs(alpha_final - 2.0) < abs(alpha0 - 2.0), \
            "final alpha must be closer to 2 than the initial alpha"
        assert alpha_final <= 2.6, "alpha must not overshoot/diverge upward"
        # Monotone-ish: distance to 2.0 should not blow up at any step.
        dists = [abs(a - 2.0) for a in alphas]
        assert dists[-1] < dists[0], "distance to alpha=2 must shrink overall"
        # No single step may increase the distance by a large amount (divergence).
        for prev, cur in zip(dists, dists[1:]):
            assert cur <= prev + 1e-3, "distance to alpha=2 grew (divergence)"

    def test_log_target_distance_shrinks(self):
        """The L2 distance ``||log(lam_tail) - log(lam_target)||`` shrinks."""
        np.random.seed(1)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=1)
        model = _dense_with_kernel(W)
        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=5, q=1.0,
            use_detx=True,
        )

        def _log_target_distance(kernel: np.ndarray) -> float:
            s = np.linalg.svd(kernel, full_matrices=False, compute_uv=False)
            lam = np.maximum(s, 1e-8) ** 2
            alpha, xmin, _D, _s, _n, status, _w = fit_powerlaw(lam)
            assert status == "success"
            lam_thr = float(xmin)
            detx = int(compute_detX_constraint(lam))
            k_pl = int((lam >= xmin).sum())
            if detx > 0 and k_pl > 0:
                k_star = max(1, int(0.5 * (k_pl + detx)))
                lam_thr = max(lam_thr, float(lam[k_star - 1]))
            mask = lam >= lam_thr
            lam_tail = lam[mask]
            ts = int(mask.sum())
            r = np.arange(1, ts + 1).astype(np.float64)
            mu = r ** (-cfg.q)
            t_target = float(np.log(lam_tail).sum())
            A = np.exp((t_target - float(np.log(mu).sum())) / ts)
            lam_target = A * mu
            return float(np.linalg.norm(np.log(lam_tail) - np.log(lam_target)))

        d0 = _log_target_distance(_kernel_np(model.layers[0]))
        for _ in range(8):
            ww_pgd_project(model, cfg, epoch=2, num_epochs=3)
        d1 = _log_target_distance(_kernel_np(model.layers[0]))
        assert d1 < d0, f"log-target distance must shrink ({d0:.4f} -> {d1:.4f})"

    def test_off_control_is_byte_identical_under_same_loop(self):
        """Differential control: enable=False leaves the matrix byte-identical.

        Proves the convergence above is attributable to the projection, not to
        the SVD/reshape round-trip or RNG.
        """
        np.random.seed(0)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=0)
        model = _dense_with_kernel(W)
        cfg_off = WWTailConfig(
            enable=False, warmup_epochs=0, ramp_epochs=1, min_tail=5, q=1.0,
        )
        before = _kernel_np(model.layers[0])
        for _ in range(10):
            summary = ww_pgd_project(model, cfg_off, epoch=2, num_epochs=3)
            assert summary["layers_projected"] == 0
        after = _kernel_np(model.layers[0])
        assert np.array_equal(before, after)


# ---------------------------------------------------------------------
# SC3 -- trace-log invariant
# ---------------------------------------------------------------------

class TestWWPGDTraceLogInvariant:
    """SC3: the Cayley + retract step preserves the tail ``sum(log(lam))``."""

    def test_tracelog_preserved_blend_eta_one(self):
        """At ``blend_eta=1`` the projected kernel == the fully-shaped matrix,
        so the tail trace-log is preserved within float32-SVD tolerance.
        """
        np.random.seed(2)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=2)
        model = _dense_with_kernel(W)
        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=5, q=1.0,
            blend_eta=1.0, cayley_eta=0.25, use_detx=True,
        )

        # Replicate the module's tail selection to identify the pre-projection tail.
        k0 = _kernel_np(model.layers[0])
        s0 = np.linalg.svd(k0, full_matrices=False, compute_uv=False)
        lam0 = np.maximum(s0, 1e-8) ** 2
        alpha, xmin, _D, _s, _n, status, _w = fit_powerlaw(lam0)
        assert status == "success"
        lam_thr = float(xmin)
        detx = int(compute_detX_constraint(lam0))
        k_pl = int((lam0 >= xmin).sum())
        if detx > 0 and k_pl > 0:
            k_star = max(1, int(0.5 * (k_pl + detx)))
            lam_thr = max(lam_thr, float(lam0[k_star - 1]))
        mask = lam0 >= lam_thr
        tail_size = int(mask.sum())
        assert tail_size >= cfg.min_tail
        pre_tracelog = float(np.log(lam0[mask]).sum())

        ww_pgd_project(model, cfg, epoch=2, num_epochs=3)

        k1 = _kernel_np(model.layers[0])
        s1 = np.linalg.svd(k1, full_matrices=False, compute_uv=False)
        lam1 = np.maximum(s1, 1e-8) ** 2
        # The top ``tail_size`` eigenvalues are the projected tail.
        post_tracelog = float(np.log(np.sort(lam1)[::-1][:tail_size]).sum())
        assert abs(post_tracelog - pre_tracelog) < 1e-4, \
            f"tail trace-log not preserved: {pre_tracelog} -> {post_tracelog}"

    def test_tracelog_invariant_eigenvalue_path_exact(self):
        """The eigenvalue-path math (float64) preserves trace-log to atol=1e-4.

        Reproduces the module's Cayley + retract update in float64 and checks the
        retraction restores ``sum(log(lam_new)) == T_target``, isolating the
        invariant from float32 SVD round-trip noise. The residual (~1e-5) is the
        ``+ _EPS`` guard the module applies inside every ``log`` for the
        retraction shift but not the bare ``T_target``; it is not exact zero by
        design, so the tolerance matches the SC3 ``atol<=1e-4`` budget.
        """
        np.random.seed(3)
        # A heavy tail of eigenvalues directly.
        lam_tail = np.sort(
            (np.arange(1, 21).astype(np.float64)) ** (-2.0)
        )[::-1].copy()
        tail_size = lam_tail.size
        q = 1.0
        eta = 0.25  # hardness=1 * cayley_eta
        eps = 1e-8

        r = np.arange(1, tail_size + 1).astype(np.float64)
        mu = r ** (-q)
        t_target = float(np.log(lam_tail).sum())
        A = np.exp((t_target - float(np.log(mu).sum())) / tail_size)
        lam_target = A * mu

        g = np.log(lam_tail + eps) - np.log(lam_target + eps)
        ratio = np.clip((1.0 - eta * g) / (1.0 + eta * g), 0.1, 10.0)
        lam_new = lam_tail * ratio
        shift = (t_target - float(np.log(lam_new + eps).sum())) / tail_size
        lam_new = lam_new * np.exp(shift)

        assert abs(float(np.log(lam_new).sum()) - t_target) < 1e-4


# ---------------------------------------------------------------------
# SC1 -- no-op / warmup / cadence
# ---------------------------------------------------------------------

class TestWWPGDNoop:
    """SC1: OFF, warmup, and off-cadence epochs are strict no-ops."""

    def test_noop_when_disabled(self):
        """enable=False leaves every kernel byte-identical."""
        np.random.seed(4)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=4)
        model = _dense_with_kernel(W)
        cfg = WWTailConfig(enable=False, warmup_epochs=0, ramp_epochs=1, min_tail=5)
        before = _kernel_np(model.layers[0])
        summary = ww_pgd_project(model, cfg, epoch=5, num_epochs=10)
        after = _kernel_np(model.layers[0])
        assert np.array_equal(before, after)
        assert summary["layers_projected"] == 0

    def test_noop_during_warmup(self):
        """enable=True but epoch < warmup_epochs (hardness 0) is a no-op."""
        np.random.seed(5)
        W = _heavy_tail_matrix(256, 256, exponent=2.0, seed=5)
        model = _dense_with_kernel(W)
        cfg = WWTailConfig(
            enable=True, warmup_epochs=5, ramp_epochs=3, min_tail=5,
        )
        before = _kernel_np(model.layers[0])
        summary = ww_pgd_project(model, cfg, epoch=0, num_epochs=10)
        after = _kernel_np(model.layers[0])
        assert np.array_equal(before, after)
        assert summary["hardness"] == 0.0
        assert summary["layers_projected"] == 0

    def test_callback_apply_every_epochs_warmup_cadence(self):
        """The callback no-ops on non-trigger epochs and projects on the trigger.

        With ``apply_every_epochs=3`` the projection fires when
        ``(epoch + 1) % 3 == 0`` i.e. epoch == 2, and is a no-op on epochs 0, 1.
        """
        np.random.seed(6)
        # Default Dense init has a qualifying tail; no manual heavy-tail needed.
        model = keras.Sequential([
            keras.Input((256,)),
            keras.layers.Dense(256, use_bias=False),
        ])
        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=5,
            apply_every_epochs=3,
        )
        cb = WWPGDProjectionCallback(config=cfg, num_epochs=10, model=model)

        k_start = _kernel_np(model.layers[0])
        cb.on_epoch_end(0)
        cb.on_epoch_end(1)
        k_pre_trigger = _kernel_np(model.layers[0])
        assert np.array_equal(k_start, k_pre_trigger), \
            "callback must be a no-op on non-trigger epochs"

        cb.on_epoch_end(2)
        k_post_trigger = _kernel_np(model.layers[0])
        assert not np.array_equal(k_pre_trigger, k_post_trigger), \
            "callback must project on the trigger epoch"


# ---------------------------------------------------------------------
# SC4 -- fail-soft
# ---------------------------------------------------------------------

class TestWWPGDFailSoft:
    """SC4: a degenerate layer is skipped; healthy siblings still project."""

    def test_degenerate_layer_skipped_sibling_projected(self):
        """An all-zeros kernel (failed power-law fit) does not raise or change;
        a healthy heavy-tailed sibling is still projected and ``layers_skipped >= 1``.
        """
        np.random.seed(7)
        Wh = _heavy_tail_matrix(256, 256, exponent=2.0, seed=7)
        model = keras.Sequential([
            keras.Input((256,)),
            keras.layers.Dense(256, use_bias=False, name="degenerate"),
            keras.layers.Dense(256, use_bias=False, name="healthy"),
        ])
        model.layers[0].kernel.assign(np.zeros((256, 256), np.float32))
        model.layers[1].kernel.assign(Wh.T)

        deg_before = _kernel_np(model.layers[0])
        healthy_before = _kernel_np(model.layers[1])

        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=5, q=1.0,
        )
        # Must not raise.
        summary = ww_pgd_project(model, cfg, epoch=2, num_epochs=3)

        deg_after = _kernel_np(model.layers[0])
        healthy_after = _kernel_np(model.layers[1])

        assert np.array_equal(deg_before, deg_after), \
            "degenerate layer must be left unchanged"
        assert not np.array_equal(healthy_before, healthy_after), \
            "healthy sibling must still be projected"
        assert summary["layers_skipped"] >= 1
        assert summary["layers_projected"] >= 1


# ---------------------------------------------------------------------
# SC5 -- conv reshape round-trip
# ---------------------------------------------------------------------

class TestWWPGDConvReshape:
    """SC5: a Conv2D kernel keeps its 4D shape after projection."""

    def test_conv2d_kernel_shape_preserved_and_finite(self):
        """Conv2D ``(3,3,32,64)`` -> reshape (288,64) -> project -> reshape back."""
        np.random.seed(8)
        model = keras.Sequential([
            keras.Input((16, 16, 32)),
            keras.layers.Conv2D(64, 3, use_bias=False),
        ])
        kernel = model.layers[0].kernel
        assert tuple(kernel.shape) == (3, 3, 32, 64)

        cfg = WWTailConfig(
            enable=True, warmup_epochs=0, ramp_epochs=1, min_tail=3, q=1.0,
            use_detx=True,
        )
        summary = ww_pgd_project(model, cfg, epoch=2, num_epochs=3)

        new_kernel = _kernel_np(model.layers[0])
        assert new_kernel.shape == (3, 3, 32, 64), "conv kernel shape must round-trip"
        assert np.all(np.isfinite(new_kernel)), "conv kernel must stay finite"
        assert summary["layers_projected"] == 1


# ---------------------------------------------------------------------
# SC6 -- serialization round-trip
# ---------------------------------------------------------------------

class TestWWPGDSerialization:
    """SC6: config + callback serialize/deserialize without a live model."""

    def test_registered_names_resolve(self):
        assert keras.saving.get_registered_name(WWTailConfig) == "Custom>WWTailConfig"
        assert (
            keras.saving.get_registered_name(WWPGDProjectionCallback)
            == "Custom>WWPGDProjectionCallback"
        )
        assert keras.saving.get_registered_object("Custom>WWTailConfig") is not None
        assert (
            keras.saving.get_registered_object("Custom>WWPGDProjectionCallback")
            is not None
        )

    def test_config_roundtrip(self):
        cfg = WWTailConfig(
            enable=True, min_tail=7, q=1.5, blend_eta=0.3, cayley_eta=0.4,
            use_detx=False, warmup_epochs=2, ramp_epochs=3, apply_every_epochs=4,
            verbose=True,
        )
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(cfg)
        )
        assert isinstance(restored, WWTailConfig)
        assert restored.enable is True
        assert restored.min_tail == 7
        assert restored.q == pytest.approx(1.5)
        assert restored.blend_eta == pytest.approx(0.3)
        assert restored.cayley_eta == pytest.approx(0.4)
        assert restored.use_detx is False
        assert restored.warmup_epochs == 2
        assert restored.ramp_epochs == 3
        assert restored.apply_every_epochs == 4
        assert restored.verbose is True

    def test_callback_roundtrip_without_live_model(self):
        """The deserialized callback reconstructs config + num_epochs and has no
        live model (``_explicit_model`` is None), and does not crash.
        """
        cfg = WWTailConfig(enable=True, min_tail=9, q=2.0, warmup_epochs=1)
        cb = WWPGDProjectionCallback(config=cfg, num_epochs=33)
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(cb)
        )
        assert isinstance(restored, WWPGDProjectionCallback)
        assert restored._num_epochs == 33
        assert restored._config.min_tail == 9
        assert restored._config.q == pytest.approx(2.0)
        assert restored._config.warmup_epochs == 1
        assert restored._explicit_model is None
        # No live model + Keras self.model None -> on_epoch_end is a safe no-op.
        restored.on_epoch_end(5)
