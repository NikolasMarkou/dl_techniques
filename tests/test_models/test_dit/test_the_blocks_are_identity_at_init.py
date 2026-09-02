"""Quirk guard: at initialisation every ``DiTBlock`` is the EXACT identity and the
whole ``DiT`` emits EXACTLY ``0.0``.

**The lines this file pins.**
``src/dl_techniques/models/vision_language/dit/blocks.py`` -- ``DiTBlock``'s
6-way modulation ``Dense`` (owned by ``sd3_adaln.AdaLayerNormZero``),
``DiTFinalLayer``'s 2-way modulation ``Dense``, and ``DiTFinalLayer.linear``::

    self.linear = keras.layers.Dense(
        ...,
        kernel_initializer=keras.initializers.Zeros(),
        bias_initializer=keras.initializers.Zeros(),
    )

reproducing upstream's ``initialize_weights``: ``nn.init.constant_(block.adaLN_
modulation[-1].weight, 0)`` per block and ``constant_(self.final_layer.linear.
weight, 0)`` (``reference/models.py``). With all six modulation chunks at zero the
gates ``gate_msa`` and ``gate_mlp`` are ``0``, so ``x + 0 * f(x) == x`` bit for
bit; with the read-out kernel and bias at zero the model's output is the zero
tensor. This is adaLN-Zero, not a defect: it is what makes a 28-block stack
trainable, and any test asserting "the untrained model changes its input" is
wrong here.

**The plausible WRONG alternative this file is RED against.** Dropping any one
of the three zero-inits -- for example initializing the block modulation
``Dense`` with the Keras ``glorot_uniform`` default, which is what a reader who
does not know adaLN-Zero would "restore".

**THE BLIND SPOT THIS FILE EXISTS FOR, measured.** The model-level claim "the
output is exactly ``0.0``" is caused SOLELY by the final layer's zero kernel. It
survives a completely broken block stack: with the blocks' modulation ``Dense``
randomized, the model output is STILL exactly ``0.0``, because the zero read-out
annihilates whatever the stack produced. ``TestTheModelOutputCannotSeeTheBlocks``
pins that measurement, and ``TestTheBlockStackIsAnExactIdentity`` supplies the
arm that CAN see it -- it compares the token stream before and after the stack,
inside the model, at ``atol=0``.

Step 5's ``test_dit_blocks.py::TestTheIdentityAtInitPremise`` owns the
LAYER-level version of this claim (one block, one final layer, and the
zero-in-kernel-and-bias check). This file is the MODEL-level version and does not
repeat it: what is new here is the whole-stack identity, the whole-model zero,
and the blind spot between them.

**RED proof (step 10).** Two injections, and they fire on DISJOINT arms:

* ``sd3_adaln.AdaLayerNormZero``'s modulation ``Dense`` set to
  ``kernel_initializer="glorot_uniform"`` -- **4 failed / 15 passed**:
  ``test_each_block_is_the_exact_identity`` and all three
  ``test_the_identity_holds_at_every_depth`` cases. **Every arm of
  ``TestTheModelEmitsExactlyZero`` stayed GREEN**, which is the blind spot above,
  measured.
* ``DiTFinalLayer.linear``'s ``kernel_initializer`` set to ``GlorotUniform()`` --
  **11 failed / 8 passed**: the whole of ``TestTheModelEmitsExactlyZero`` plus
  ``test_randomizing_the_block_modulation_leaves_the_output_at_zero``.
"""

from typing import Any, List

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.model import DiT

from ._dit_helpers import TINY, built_model, np_, tiny_inputs


def token_stream(model: DiT, x: Any, t: Any, y: Any) -> List[np.ndarray]:
    """Re-run ``DiT.call``'s token path, capturing it after every block.

    Interface contract: this walks the model's OWN sub-layers in ``call``'s
    order, so it measures the shipped stack rather than a paraphrase of it. The
    returned list is ``[after_patchify_and_pos] + [after_block_i for i]``; index
    ``0`` is the stack's input and index ``-1`` its output.
    """
    tokens = model.x_embedder(x, training=False)
    tokens = tokens + keras.ops.cast(model.pos_embed, tokens.dtype)
    c = model.t_embedder(t, training=False) + model.y_embedder(y, training=False)

    stream = [np_(tokens)]
    for block in model.blocks:
        tokens = block([tokens, c], training=False)
        stream.append(np_(tokens))
    return stream


# ---------------------------------------------------------------------
# The stack
# ---------------------------------------------------------------------


class TestTheBlockStackIsAnExactIdentity:
    """Every block returns its input unchanged, bit for bit."""

    def test_each_block_is_the_exact_identity(self) -> None:
        model = built_model(seed=0)
        x, t, y = tiny_inputs(seed=1)
        stream = token_stream(model, x, t, y)

        assert len(stream) == TINY["depth"] + 1
        for index in range(1, len(stream)):
            np.testing.assert_allclose(
                stream[index], stream[index - 1], rtol=0, atol=0.0,
                err_msg=f"block {index - 1} changed its input at init",
            )

    def test_the_stack_input_is_not_itself_zero(self) -> None:
        """Anti-vacuity: an identity on the zero tensor proves nothing."""
        model = built_model(seed=0)
        x, t, y = tiny_inputs(seed=1)
        stream = token_stream(model, x, t, y)
        assert float(np.max(np.abs(stream[0]))) > 0.0

    def test_the_conditioning_vector_is_not_zero_either(self) -> None:
        """The gates are zero because the MODULATION is, not because ``c`` is."""
        model = built_model(seed=0)
        x, t, y = tiny_inputs(seed=1)
        c = model.t_embedder(t, training=False) + model.y_embedder(y, training=False)
        assert float(np.max(np.abs(np_(c)))) > 0.0

    @pytest.mark.parametrize("depth", [1, 2, 4])
    def test_the_identity_holds_at_every_depth(self, depth: int) -> None:
        model = built_model(seed=0, depth=depth)
        x, t, y = tiny_inputs(seed=1)
        stream = token_stream(model, x, t, y)
        np.testing.assert_allclose(stream[-1], stream[0], rtol=0, atol=0.0)


# ---------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------


class TestTheModelEmitsExactlyZero:
    """The output is the zero tensor -- ``atol=0.0``, not "small"."""

    def test_the_output_is_exactly_zero(self) -> None:
        model = built_model(seed=0)
        x, t, y = tiny_inputs(seed=2)
        out = np_(model([x, t, y], training=False))
        assert out.shape == (
            x.shape[0], TINY["input_size"], TINY["input_size"], 2 * TINY["in_channels"]
        )
        np.testing.assert_array_equal(out, np.zeros_like(out))

    @pytest.mark.parametrize("training", [False, True])
    def test_it_is_zero_in_both_modes(self, training: bool) -> None:
        model = built_model(seed=0)
        x, t, y = tiny_inputs(seed=3)
        out = np_(model([x, t, y], training=training))
        np.testing.assert_array_equal(out, np.zeros_like(out))

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"depth": 1},
            {"learn_sigma": False},
            {"input_size": 2, "patch_size": 2},
            {"num_heads": 1, "hidden_size": 8},
            {"class_dropout_rate": 0.0},
        ],
        ids=["tiny", "depth1", "no_sigma", "single_token", "one_head", "no_dropout"],
    )
    def test_every_configuration_starts_at_zero(self, overrides: dict) -> None:
        model = built_model(seed=0, **overrides)
        config = dict(TINY, **overrides)
        rng = np.random.default_rng(4)
        batch = 2
        x = rng.normal(
            size=(batch, config["input_size"], config["input_size"],
                  config["in_channels"])
        ).astype("float32")
        t = rng.integers(0, 1000, size=(batch,)).astype("float32")
        y = rng.integers(0, config["num_classes"], size=(batch,)).astype("int32")
        out = np_(model([x, t, y], training=False))
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_a_variant_model_also_starts_at_zero(self) -> None:
        """Not an artefact of the tiny geometry."""
        keras.utils.set_random_seed(0)
        model = DiT.from_variant("DiT-S/8", input_size=16, num_classes=10)
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 16, 16, 4)).astype("float32")
        t = rng.integers(0, 1000, size=(2,)).astype("float32")
        y = np.array([0, 3], dtype="int32")
        out = np_(model([x, t, y], training=False))
        np.testing.assert_array_equal(out, np.zeros_like(out))


# ---------------------------------------------------------------------
# ... and why one arm is not enough
# ---------------------------------------------------------------------


class TestTheModelOutputCannotSeeTheBlocks:
    """MEASURED: a broken block stack still emits exactly ``0.0``.

    This is the reason ``TestTheBlockStackIsAnExactIdentity`` exists as a
    separate claim rather than being folded into "the output is zero". The
    injection here is performed in-test on a throwaway model, so the blind spot
    is pinned by an executable arm instead of a comment.
    """

    @staticmethod
    def _randomize_block_modulation(model: DiT, seed: int = 9) -> int:
        """Replace every zero weight inside the blocks' modulation Dense.

        Returns the number of tensors replaced, so a silent no-op is impossible.
        """
        rng = np.random.default_rng(seed)
        touched = 0
        for block in model.blocks:
            for weight in block.weights:
                value = np_(weight)
                if weight.trainable and not np.any(value):
                    weight.assign(
                        rng.normal(scale=0.5, size=value.shape).astype(value.dtype)
                    )
                    touched += 1
        return touched

    def test_randomizing_the_block_modulation_leaves_the_output_at_zero(self) -> None:
        model = built_model(seed=0)
        assert self._randomize_block_modulation(model) > 0

        x, t, y = tiny_inputs(seed=6)
        out = np_(model([x, t, y], training=False))
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_but_the_stack_arm_convicts_it(self) -> None:
        model = built_model(seed=0)
        self._randomize_block_modulation(model)

        x, t, y = tiny_inputs(seed=6)
        stream = token_stream(model, x, t, y)
        assert float(np.max(np.abs(stream[-1] - stream[0]))) > 0.0

    def test_randomizing_the_read_out_breaks_the_zero_output(self) -> None:
        """The complementary injection: the final layer IS what zeroes the output."""
        model = built_model(seed=0)
        rng = np.random.default_rng(10)
        touched = 0
        for weight in model.final_layer.weights:
            value = np_(weight)
            if weight.trainable and not np.any(value):
                weight.assign(rng.normal(scale=0.5, size=value.shape).astype(value.dtype))
                touched += 1
        assert touched > 0

        x, t, y = tiny_inputs(seed=6)
        out = np_(model([x, t, y], training=False))
        assert float(np.max(np.abs(out))) > 0.0
