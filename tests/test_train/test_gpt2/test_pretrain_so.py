"""``pretrain_so.py``'s SO penalty must actually reach the gradient update.

MEASURED pre-fix defect (plan-2026-09-13T073704-245ab5d5, iter-1, step-3):
``wrap_model_with_so``'s ``train_step_with_so`` called
``original_train_step(data)`` (``CausalLanguageModel.train_step``), which
ALREADY computes gradients inside its own ``tf.GradientTape`` and calls
``self.optimizer.apply_gradients(...)`` before ``train_step_with_so`` ever
sees the returned metrics dict. ``so_loss`` was then computed and added to
``result["loss"]`` only AFTER that optimizer step had already happened --
so the orthonormality penalty was reported in the logged loss but never
backpropagated: it could not affect a single trained weight.

Direct measurement (two independently-built, identically-seeded tiny GPT-2
models, a fresh un-stepped ``AdamW`` optimizer each, the SAME kernel
deliberately perturbed far from orthonormal in both): running one bare
``train_step`` on model A and one SO-wrapped ``train_step`` on model B from
the identical starting point produced trainable-variable updates differing
by at most ``4.017e-6`` -- IDENTICAL, to the last count, to the GPU
nondeterminism noise floor measured by running the SAME bare ``train_step``
twice on two more identically-seeded models with no SO wrapper at all. That
equality is the proof: the SO wrapper contributed zero additional gradient
signal beyond ordinary floating-point noise.

This module is a MUTATION/REVERT-SENSITIVE test: reverting the Step-3 fix
(restoring the old ``train_step_with_so`` that discards the SO gradient)
turns ``test_so_wrapped_step_moves_weights_differently_than_bare_step`` RED.
It was proven RED against the pre-fix code before the fix landed (see
decisions.md D-010 for the recorded RED-before/GREEN-after evidence).
"""

import numpy as np
import keras
import pytest

from dl_techniques.regularizers import SoftOrthonormalConstraintRegularizer
from train.gpt2.pretrain import create_gpt2_model
from train.gpt2.pretrain_so import (
    SOTrainingConfig,
    wrap_model_with_so,
    collect_kernel_weights,
)

VOCAB_SIZE = 300
SEQ_LEN = 16
BATCH = 4
SEED = 42


def _build_model(cfg: SOTrainingConfig) -> keras.Model:
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    model = create_gpt2_model(cfg)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0))
    return model


def _cfg() -> SOTrainingConfig:
    return SOTrainingConfig(
        model_variant="tiny",
        vocab_size=VOCAB_SIZE,
        max_seq_length=SEQ_LEN + 1,
        so_lambda=1e-3,
    )


def _perturb_first_kernel_far_from_orthonormal(model: keras.Model, cfg: SOTrainingConfig):
    """Scale the first collected kernel to be deliberately non-orthonormal."""
    kernels = collect_kernel_weights(model, cfg.so_skip_embeddings)
    perturbed = kernels[0].numpy() * 50.0 + 5.0
    kernels[0].assign(perturbed)
    return kernels


def _synthetic_batch():
    x = np.random.RandomState(0).randint(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN)).astype("int32")
    y = np.random.RandomState(1).randint(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN)).astype("int32")
    return x, y


class TestSoLossPresenceAndCorrectness:
    """``so_loss`` must be present, finite, and match a hand-computed value."""

    def test_so_loss_is_present_finite_and_nonzero_for_non_orthonormal_kernels(self):
        cfg = _cfg()
        model = _build_model(cfg)
        kernels = _perturb_first_kernel_far_from_orthonormal(model, cfg)

        regularizer = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=cfg.so_lambda,
            l1_coefficient=cfg.so_l1,
            l2_coefficient=cfg.so_l2,
            use_matrix_scaling=cfg.so_matrix_scaling,
        )
        hand_computed = float(sum(regularizer(w) for w in kernels))
        assert np.isfinite(hand_computed)
        assert hand_computed > 0.0, "hand-computed SO penalty must be nonzero for a perturbed kernel"

        wrap_model_with_so(model, cfg)
        x, y = _synthetic_batch()
        result = model.train_step((x, y))
        so_loss = float(result["so_loss"])

        assert np.isfinite(so_loss)
        assert so_loss > 0.0
        # The reported so_loss is computed AFTER original_train_step's
        # optimizer.apply_gradients already ran, so the kernel has moved
        # slightly from the hand-computed snapshot -- allow a generous
        # relative tolerance, this is a sanity/order-of-magnitude check,
        # not a bit-exact one.
        assert so_loss == pytest.approx(hand_computed, rel=0.1)


class TestSoPenaltyReachesGradients:
    """The falsification bar: SO-wrapped training must differ from bare training."""

    def test_so_wrapped_step_moves_weights_differently_than_bare_step(self):
        cfg = _cfg()
        model_bare = _build_model(cfg)
        model_so = _build_model(cfg)

        _perturb_first_kernel_far_from_orthonormal(model_bare, cfg)
        _perturb_first_kernel_far_from_orthonormal(model_so, cfg)

        # Sanity: both models identical after perturbation.
        for va, vb in zip(model_bare.trainable_variables, model_so.trainable_variables):
            np.testing.assert_array_equal(va.numpy(), vb.numpy())

        x, y = _synthetic_batch()

        # Bare train_step (no SO) on model_bare.
        model_bare.train_step((x, y))

        # SO-wrapped train_step on model_so, from the identical starting point.
        wrap_model_with_so(model_so, cfg)
        result_so = model_so.train_step((x, y))
        assert "so_loss" in result_so
        assert float(result_so["so_loss"]) > 0.0

        max_diff = max(
            float(np.abs(va.numpy() - vb.numpy()).max())
            for va, vb in zip(model_bare.trainable_variables, model_so.trainable_variables)
        )
        # A GPU-nondeterminism noise floor exists even between two bare,
        # unwrapped runs (measured ~4e-6 for this tiny model/batch). The SO
        # penalty must move weights MEASURABLY beyond that floor, or the
        # penalty is not reaching the gradient computation at all.
        assert max_diff > 1e-4, (
            f"SO-wrapped step differs from bare step by only {max_diff}, "
            "indistinguishable from GPU nondeterminism noise -- the SO "
            "penalty is not reaching the gradient update."
        )
