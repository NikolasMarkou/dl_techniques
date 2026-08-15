"""The equal-sequence-length precondition, and who is allowed to ignore it.

``_call_tensor_fusion`` was given a named guard for this on 2026-08-14. The
guard was right and its siblings were not: ``_call_concatenation`` concatenates
on the same feature axis under the same precondition, and
``addition``/``multiplication``/``gated``/``bilinear`` broadcast on the sequence
axis, all with no guard at all — they died inside a backend ``ConcatOp`` /
broadcast error naming neither the strategy nor the requirement. Only
``cross_attention`` and ``attention_pooling`` genuinely tolerate unequal lengths.

The other half of this module pins the two ``compute_output_shape``
implementations against each other over every registered strategy, which is the
assertion that stops them drifting apart again.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.layers.fusion.multimodal_fusion import (
    MultiModalFusion,
    FusionStrategy,
)

DIM = 8
BATCH = 2
VISION_LEN = 5
TEXT_LEN = 3

ALL_STRATEGIES = list(FusionStrategy.__args__)
LENGTH_SENSITIVE = [
    s for s in ALL_STRATEGIES
    if s not in MultiModalFusion.LENGTH_AGNOSTIC_STRATEGIES
]
LENGTH_AGNOSTIC = sorted(MultiModalFusion.LENGTH_AGNOSTIC_STRATEGIES)


def _layer(strategy, **kwargs):
    return MultiModalFusion(
        dim=DIM, fusion_strategy=strategy,
        attention_config={"num_heads": 2}, **kwargs
    )


def _pair(vision_len=VISION_LEN, text_len=TEXT_LEN):
    return [
        keras.random.normal((BATCH, vision_len, DIM)),
        keras.random.normal((BATCH, text_len, DIM)),
    ]


def test_every_strategy_is_classified():
    """No strategy may be silently outside both buckets."""
    assert set(LENGTH_SENSITIVE) | set(LENGTH_AGNOSTIC) == set(ALL_STRATEGIES)
    assert len(ALL_STRATEGIES) == 8


class TestUnequalLengthsAreRefusedByName:

    @pytest.mark.parametrize("strategy", LENGTH_SENSITIVE)
    def test_named_value_error(self, strategy):
        layer = _layer(strategy)
        with pytest.raises(ValueError) as excinfo:
            layer(_pair())
        message = str(excinfo.value)
        assert strategy in message, "the error must name the offending strategy"
        assert "same sequence length" in message
        assert "cross_attention" in message, "and point at a way out"

    @pytest.mark.parametrize("strategy", LENGTH_AGNOSTIC)
    def test_length_agnostic_strategies_still_accept_unequal_lengths(self, strategy):
        """Anti-vacuity control: the guard must not fire for these two."""
        layer = _layer(strategy)
        output = layer(_pair())
        assert output is not None

    @pytest.mark.parametrize("strategy", LENGTH_SENSITIVE)
    def test_equal_lengths_still_run(self, strategy):
        """Anti-vacuity control: the guard must not refuse the legal case."""
        layer = _layer(strategy)
        output = layer(_pair(vision_len=4, text_len=4))
        assert np.all(np.isfinite(np.asarray(keras.ops.convert_to_numpy(output))))

    def test_symbolic_none_length_is_not_refused(self):
        """A build with an unknown sequence axis must not be guessed at."""
        layer = _layer("concatenation")
        inputs = [
            keras.Input(shape=(None, DIM)),
            keras.Input(shape=(None, DIM)),
        ]
        assert layer(inputs) is not None


class TestComputeOutputShapeMatchesTheForwardPass:
    """`MultiModalFusion.compute_output_shape` is the single source of truth for
    the fused length — `nano_vlm` now calls through it instead of re-deriving a
    vision+text sum. That only helps if the claim is true, so it is pinned here
    against the actual forward output for every registered strategy."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_claimed_shape_equals_actual_shape(self, strategy):
        layer = _layer(strategy)
        length = 4
        inputs = _pair(vision_len=length, text_len=length)
        output = layer(inputs)
        claimed = layer.compute_output_shape([tuple(t.shape) for t in inputs])

        if isinstance(output, tuple):
            actual = [tuple(t.shape) for t in output]
            assert [tuple(c) for c in claimed] == actual
        else:
            assert tuple(claimed) == tuple(output.shape)

    def test_only_cross_attention_returns_a_per_modality_shape(self):
        """The distinction nano_vlm's own compute_output_shape branches on."""
        for strategy in ALL_STRATEGIES:
            layer = _layer(strategy)
            claimed = layer.compute_output_shape([
                (BATCH, 4, DIM), (BATCH, 4, DIM)
            ])
            is_per_modality = isinstance(claimed[0], (tuple, list))
            assert is_per_modality == (strategy == "cross_attention"), strategy
