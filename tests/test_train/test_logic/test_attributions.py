"""Unit tests for train.logic.attributions.

Plan: plan_2026-05-13_798d3a60.

We do NOT actually run LIME/SHAP in these tests — those are slow and have
randomness that's hard to seed. We only signature-smoke them
(LimeTabularExplainer / KernelExplainer constructors imported successfully)
and exercise the deterministic helpers (sparsity, suff_comp_aucs,
stability) on a tiny toy classifier.

The headline assertion (SC6) is that ``circuit_attributions`` on a
``parity_k6``-trained ``build_circuit`` model gives roughly-uniform
attribution mass across the 6 input bits — the symmetry oracle.
"""

import os
import numpy as np
import pytest
import keras

from train.logic.attributions import (
    circuit_attributions,
    sparsity,
    stability,
    suff_comp_aucs,
)
from train.logic.train_benchmark import build_circuit, gen_parity


# ---------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------


class TestSparsity:
    def test_uniform_is_one(self):
        v = np.ones(8, dtype=np.float32)
        assert sparsity(v) == pytest.approx(1.0, abs=1e-6)

    def test_one_hot_is_zero(self):
        v = np.zeros(8, dtype=np.float32)
        v[3] = 1.0
        assert sparsity(v) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_is_zero(self):
        assert sparsity(np.zeros(8, dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------
# Toy 2-input classifier for the metric helpers
# ---------------------------------------------------------------------


def _make_toy_classifier(num_bits: int = 6) -> keras.Model:
    """Tiny deterministic sigmoid network of the right input width.

    Used only to exercise suff_comp_aucs / stability without training —
    we set weights so the model implements ``y = sigmoid(sum(x) - num_bits/2)``,
    which makes the prediction depend on every bit symmetrically.
    """
    inputs = keras.Input(shape=(num_bits,), name="bits")
    head = keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="head")
    out = head(inputs)
    m = keras.Model(inputs, out)
    head.set_weights(
        [
            np.ones((num_bits, 1), dtype=np.float32),
            np.array([-num_bits / 2.0], dtype=np.float32),
        ]
    )
    return m


class TestSuffCompAUCs:
    def test_runs_and_returns_finite_aucs(self):
        m = _make_toy_classifier(num_bits=4)
        X = np.array(
            [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]], dtype=np.float32
        )

        def attr_fn(model, x):
            # Constant uniform attribution — sanity check the helper itself.
            return np.ones(4, dtype=np.float32)

        result = suff_comp_aucs(m, X, attr_fn, k_range=[0, 1, 2, 3, 4])
        assert "suff_auc" in result and "comp_auc" in result
        assert np.isfinite(result["suff_auc"])
        assert np.isfinite(result["comp_auc"])
        # k_curves has matching length.
        assert len(result["k_curves"]["k"]) == 5
        assert len(result["k_curves"]["suff"]) == 5
        assert len(result["k_curves"]["comp"]) == 5


class TestStability:
    def test_runs_without_error(self):
        m = _make_toy_classifier(num_bits=6)
        x = np.array([1, 0, 1, 1, 0, 1], dtype=np.float32)

        def attr_fn(model, xv):
            # Use the model's gradient × input — well-defined, not constant.
            return circuit_attributions(model, xv)

        s = stability(m, x, attr_fn, num_perturbations=4, seed=42)
        assert np.isfinite(s)
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


# ---------------------------------------------------------------------
# Symmetry oracle (SC6) — circuit_attributions on a parity_k6 circuit
# ---------------------------------------------------------------------


@pytest.mark.slow
class TestParitySymmetryOracle:
    """SC6: max/min bit-attribution ratio < 2x on a trained parity_k6 circuit.

    We train a small circuit on parity for a handful of epochs so the
    gradient×input attribution is meaningful (an untrained circuit has
    random gradients which would FAIL the ratio check by accident).
    """

    def test_parity_k6_attributions_within_2x_ratio(self):
        """SC6: each input bit has similar marginal influence on the
        parity_k6 prediction.

        Implementation: train a circuit on parity for enough epochs to
        learn the function, then estimate per-bit influence as the mean
        absolute gradient × input across many uniform-random inputs. This
        marginalizes out the per-bit embedding-weight asymmetry — the
        TASK is symmetric in bits, so the marginal influence over the
        input distribution must be too.
        """
        keras.utils.set_random_seed(7)
        rng = np.random.default_rng(7)
        X, Y = gen_parity(4096, 6, rng)
        Xv, Yv = gen_parity(512, 6, rng)
        model = build_circuit(num_bits=6, num_outputs=1)
        model.fit(
            X, Y, validation_data=(Xv, Yv),
            epochs=60, batch_size=64, verbose=0,
        )
        # Sanity: training reached some convergence (otherwise the test
        # is checking gradients on an unlearned model).
        val_acc = float(model.evaluate(Xv, Yv, verbose=0)[1])
        if val_acc < 0.85:
            pytest.skip(
                f"parity_k6 circuit failed to converge in test (val_acc={val_acc:.3f}); "
                "symmetry oracle is undefined without convergence."
            )
        # Aggregate |circuit_attributions| over many inputs. The function
        # uses integrated gradients which preserves task symmetry once the
        # model has converged on parity_k6.
        n_samples = 128
        sample_idxs = rng.choice(Xv.shape[0], size=n_samples, replace=False)
        mass = np.zeros(6, dtype=np.float64)
        for i in sample_idxs:
            a = circuit_attributions(model, Xv[i])
            mass += np.abs(a)
        mass /= mass.sum() + 1e-12
        ratio = float(mass.max() / max(mass.min(), 1e-12))
        # SC6 spec: < 2x at full training.
        assert ratio < 2.0, (
            f"parity_k6 symmetry oracle failed: "
            f"per-bit normalized mass = {mass.round(3).tolist()}, "
            f"max/min ratio = {ratio:.3f} (expect < 2.0)"
        )


# ---------------------------------------------------------------------
# Signature smoke tests for LIME/SHAP (no runtime calls)
# ---------------------------------------------------------------------


class TestAttributionImports:
    def test_lime_importable(self):
        import lime.lime_tabular  # noqa: F401

    def test_shap_importable(self):
        import shap  # noqa: F401
        assert hasattr(shap, "KernelExplainer")
