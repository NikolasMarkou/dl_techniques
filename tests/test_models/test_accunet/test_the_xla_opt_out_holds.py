"""Pin the XLA opt-out on every AccUNet compile path.

`HANCLayer` resizes its pooled summaries back to full resolution with nearest-neighbour
interpolation. The backward pass of that op, `ResizeNearestNeighborGrad`, has no
registered XLA-GPU kernel in TF 2.18, so Keras' default `jit_compile="auto"` -- which
resolves to XLA on a GPU -- makes `fit()` die with
`InvalidArgumentError: Detected unsupported operations ... on XLA_GPU_JIT`.
Reproduced 2026-09-01 on an RTX 4070 with `create_acc_unet_binary(...)`; the same model
compiled with `jit_compile=False` trains.

What this file pins:

* `jit_compile is False` after a plain `.compile()`, for a direct `AccUNet` instance
  and for a factory-built model.
* The opt-out wins even when the caller explicitly passes `jit_compile=True`. That is
  deliberate (plan invariant I4, following the VAE D-009 precedent): a caller asking
  for XLA here is asking for a crash, so the safe answer overrides the request.
* The opt-out survives `save()` / `keras.saving.load_model()`. Overriding `compile()`
  alone is not enough -- Keras routes the reload's recompile through
  `compile_from_config`, so a reloaded model would otherwise come back with a stale
  `jit_compile="auto"` and crash on the next GPU `fit()`.

**Why the plain-`.compile()` cases force the "auto" resolution.** `Trainer.compile`
resolves `jit_compile="auto"` through `Trainer._resolve_auto_jit_compile`, which returns
`False` outright on a CPU-only machine (`keras/src/trainers/trainer.py:220-238`). On the
CPU runner every plain-`.compile()` assertion here would therefore read `False` with or
without the fix -- a guard that cannot fail. The `_force_xla_auto_to_resolve_true` fixture
makes that resolver return `True`, which is exactly what it returns on the GPU where the
crash was measured; the assertion then discriminates. The explicit-`jit_compile=True`
cases need no such help and hold on any device.

What this file CANNOT catch: it never runs a training step, so it does not prove the op
is still XLA-incompatible or that the workaround is still needed. That is the GPU `fit()`
proof, run separately; this file only pins the setting the GPU proof depends on.

Every model here is deliberately tiny (32x32x1, `base_filters=4`,
`mlfc_iterations=1`) -- the assertions are about the compile configuration, not about
capacity, and a full-size model would make the save/load round trip slow.
"""

import os
import shutil
import tempfile

import keras
import numpy as np
import pytest
from keras.src.trainers.trainer import Trainer

from dl_techniques.models.vision.accunet.model import (
    AccUNet,
    create_acc_unet_binary,
)

# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

TINY = dict(input_channels=1, base_filters=4, mlfc_iterations=1)
INPUT_SHAPE = (32, 32)


@pytest.fixture
def _force_xla_auto_to_resolve_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `jit_compile="auto"` resolve to `True`, as it does on a GPU.

    Without this the CPU-only test machine resolves "auto" to `False` on its own and
    the plain-`.compile()` assertions pass for a reason that has nothing to do with
    the fix.
    """
    monkeypatch.setattr(Trainer, "_resolve_auto_jit_compile", lambda self: True)


def _direct_model() -> AccUNet:
    """Build a tiny `AccUNet` instance directly (no functional wrapper)."""
    model = AccUNet(num_classes=1, **TINY)
    # Build by invocation: AccUNet has no explicit `build()`, so calling `build()`
    # would only mark it built without creating its state.
    model(np.zeros((1,) + INPUT_SHAPE + (TINY["input_channels"],), dtype="float32"))
    return model


def _factory_model() -> keras.Model:
    """Build a tiny model through the documented factory path."""
    return create_acc_unet_binary(input_shape=INPUT_SHAPE, **TINY)


# ---------------------------------------------------------------------
# Direct AccUNet instance
# ---------------------------------------------------------------------


def test_a_direct_instance_compiles_without_xla(
    _force_xla_auto_to_resolve_true: None,
) -> None:
    model = _direct_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    assert model.jit_compile is False, (
        f"AccUNet.compile() left jit_compile={model.jit_compile!r}; XLA must be "
        "force-disabled because ResizeNearestNeighborGrad has no XLA-GPU kernel."
    )


def test_a_direct_instance_overrides_an_explicit_jit_compile_true() -> None:
    model = _direct_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", jit_compile=True)
    assert model.jit_compile is False, (
        f"An explicit jit_compile=True survived as {model.jit_compile!r}; the opt-out "
        "must win over the caller (invariant I4), since XLA here is a guaranteed crash."
    )


# ---------------------------------------------------------------------
# Factory-built model
# ---------------------------------------------------------------------


def test_a_factory_model_compiles_without_xla(
    _force_xla_auto_to_resolve_true: None,
) -> None:
    model = _factory_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    assert model.jit_compile is False, (
        f"create_acc_unet_binary(...).compile() left jit_compile={model.jit_compile!r}; "
        "the factory wrapper is the path the GPU reproduction used, so a class-level "
        "override on AccUNet alone does not reach it."
    )


def test_a_factory_model_overrides_an_explicit_jit_compile_true() -> None:
    model = _factory_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", jit_compile=True)
    assert model.jit_compile is False, (
        f"An explicit jit_compile=True survived as {model.jit_compile!r} on the factory "
        "model; the opt-out must win over the caller (invariant I4)."
    )


# ---------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------


def test_the_opt_out_survives_a_save_load_round_trip(
    _force_xla_auto_to_resolve_true: None,
) -> None:
    """A reloaded factory model must come back as its own type, still XLA-free.

    Keras routes `load_model()`'s recompile through `compile_from_config`, so a
    `compile()` override alone leaves the reloaded model with `jit_compile="auto"`.

    The model is fitted for one step before saving, for two reasons. It makes the
    archive carry real optimizer state, so the assertion on the reloaded optimizer's
    variable count pins the `optimizer.build(self.trainable_variables)` tail of
    `compile_from_config` -- dropping that tail is a measured regression (the VAE
    override records 122 saved optimizer variables restored as 2). And an
    unfitted-then-saved model provokes Keras' "Skipping variable loading for
    optimizer" `UserWarning`, which this repo escalates to an error tree-wide.
    """
    from dl_techniques.models.vision.accunet.model import AccUNetFunctional

    model = _factory_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")

    rng = np.random.default_rng(0)
    x = rng.random((2,) + INPUT_SHAPE + (1,)).astype("float32")
    y = rng.integers(0, 2, size=(2,) + INPUT_SHAPE + (1,)).astype("float32")
    model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    n_optimizer_variables = len(model.optimizer.variables)
    assert n_optimizer_variables > 2, (
        "the one-step fit did not build the optimizer, so the round trip below "
        "would not exercise the compile_from_config optimizer tail"
    )

    before = keras.ops.convert_to_numpy(model(x, training=False))

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "accunet.keras")
        model.save(path)
        reloaded = keras.saving.load_model(path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert isinstance(reloaded, AccUNetFunctional), (
        f"reloaded as {type(reloaded).__name__}, not AccUNetFunctional; the override "
        "that disables XLA lives on that class, so a plain keras.Model reload has lost it."
    )
    assert reloaded.jit_compile is False, (
        f"reloaded model has jit_compile={reloaded.jit_compile!r}; compile_from_config "
        "must funnel the reload's recompile through the overridden compile()."
    )
    assert len(reloaded.optimizer.variables) == n_optimizer_variables, (
        f"the reloaded optimizer holds {len(reloaded.optimizer.variables)} variables "
        f"against {n_optimizer_variables} in the archive; compile_from_config must "
        "reproduce Keras' optimizer.build(self.trainable_variables) tail or all saved "
        "optimizer state is silently dropped."
    )

    after = keras.ops.convert_to_numpy(reloaded(x, training=False))
    np.testing.assert_allclose(
        before,
        after,
        atol=1e-6,
        err_msg="the round trip changed the forward output -- weights were lost",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
