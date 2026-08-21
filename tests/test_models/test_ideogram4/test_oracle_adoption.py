"""
Oracle adoption for ``models/ideogram4`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), ``Ideogram4Transformer`` at the ``tiny`` preset
(``emb_dim=128, num_layers=2, in_channels=32, llm_features_dim=64``) on a packed
batch of 3 text tokens + 4 image tokens, after one real Adam step: **42**
trainable weights, **0** dead, **0** disconnected.

Why a per-weight gradient assertion is the right instrument for THIS model.
Ideogram 4 is a packed-stream masked-add DiT: text tokens carry
``llm_features``, image tokens carry the noise ``x``, and each stream's
projection is applied only where its own ``indicator`` selects it. A masked-add
that selected the wrong stream -- or that added a projection whose mask is
empty -- produces a correctly shaped, finite velocity and would pass any
shape/finiteness smoke test, while leaving that projection's weights with an
identically-zero gradient. The two stream projections
(``input_proj``, ``llm_cond_proj``) and the conditioning path
(``t_embedding`` / ``adaln_proj`` / ``embed_image_indicator``) are therefore
named explicitly in the assertion below rather than left to the count.

This config carries no dropout and no stochastic depth (its dataclass has no
such field at all), so the reading is not a draw; that premise is asserted.
"""

import dataclasses

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.ideogram4.config import (
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.models.ideogram4.transformer import Ideogram4Transformer

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

TEXT_LEN = 3
IMAGE_LEN = 4
SEQ_LEN = TEXT_LEN + IMAGE_LEN
BATCH = 2

#: Measured 2026-08-21 at the ``tiny`` preset.
GF_N_WEIGHTS = 42

#: Weight-path fragments that must be PRESENT. Each names a component the
#: packed-stream design can silently disconnect -- see the module docstring.
STREAM_PATH_FRAGMENTS = (
    "input_proj",
    "llm_cond_proj",
    "adaln_proj",
    "embed_image_indicator",
)


def _tiny_config() -> Ideogram4Config:
    config, _ = get_ideogram4_config("tiny")
    return config


def _batch(config: Ideogram4Config, seed: int = 0) -> dict:
    """A valid packed batch: TEXT_LEN text tokens, then IMAGE_LEN image tokens."""
    rng = np.random.default_rng(seed)
    indicator = np.empty((BATCH, SEQ_LEN), dtype="int32")
    indicator[:, :TEXT_LEN] = LLM_TOKEN_INDICATOR
    indicator[:, TEXT_LEN:] = OUTPUT_IMAGE_INDICATOR
    position_ids = np.zeros((BATCH, SEQ_LEN, 3), dtype="int32")
    for b in range(BATCH):
        for l in range(SEQ_LEN):
            position_ids[b, l] = (l, l % 2, l % 3)
    return {
        "llm_features": rng.standard_normal(
            (BATCH, SEQ_LEN, config.llm_features_dim)).astype("float32"),
        "x": rng.standard_normal(
            (BATCH, SEQ_LEN, config.in_channels)).astype("float32"),
        "t": rng.uniform(0.0, 1.0, size=(BATCH,)).astype("float32"),
        "position_ids": position_ids,
        "segment_ids": np.zeros((BATCH, SEQ_LEN), dtype="int32"),
        "indicator": indicator,
    }


def _model(config: Ideogram4Config = None) -> Ideogram4Transformer:
    config = _tiny_config() if config is None else config
    model = Ideogram4Transformer(config=config)
    model(_batch(config), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestIdeogram4GradientFlow:

    def test_no_layer_is_stochastic(self):
        config = _tiny_config()
        assert not [f for f in dataclasses.asdict(config) if "drop" in f], (
            "the config grew a dropout field; pin it to 0.0 before trusting "
            "any gradient reading in this file"
        )
        model = _model(config)
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}"
        )

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        config = _tiny_config()
        model = _model(config)
        x = _batch(config)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        for fragment in STREAM_PATH_FRAGMENTS:
            assert any(fragment in path for path in report), (
                f"no weight under {fragment!r} -- a packed-stream component "
                f"the count above rests on is not in the trainable set"
            )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        config = _tiny_config()
        model = _model(config)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _batch(config))


class TestIdeogram4KnobSensitivity:

    def test_num_layers_changes_the_parameterisation(self):
        base = _tiny_config()
        builders = {
            n: (lambda n=n: _model(dataclasses.replace(base, num_layers=n)))
            for n in (1, 2, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="num_layers")

    def test_intermediate_size_changes_the_block_ffn(self):
        """A knob that reaches ONLY the per-block FFN.

        ``num_layers`` above would still pass if every block were an identity
        with a parameter attached. This one would not.
        """
        base = _tiny_config()
        builders = {
            s: (lambda s=s: _model(
                dataclasses.replace(base, intermediate_size=s)))
            for s in (128, 256, 512)
        }
        assert_structural_knob_changes_weights(builders, knob="intermediate_size")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_layers")


class TestIdeogram4SmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        config = _tiny_config()
        model = _model(config)
        x = _batch(config)

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"Ideogram4Transformer returns one velocity tensor, got "
                f"{type(out)}"
            )
            expected = (BATCH, SEQ_LEN, config.in_channels)
            assert tuple(out.shape) == expected, (
                f"expected {expected}, got {tuple(out.shape)}"
            )
            # The reference returns `.float()`; a velocity that came back in
            # the compute dtype would be a silent precision regression.
            assert keras.backend.standardize_dtype(out.dtype) == "float32", (
                f"velocity must be float32, got {out.dtype}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
