"""
A task head's own generic attention defaults must not break a non-default type.

`BaseNLPHead._create_common_layers` injected ``dim`` and ``dropout_rate`` into
every `create_attention_layer` call, and `BaseVisionHead._create_common_layers`
injected ``dim``, for ANY `attention_type`. Since
`plan-2026-08-17T183311-79c63e38/D-011` the attention factory RAISES on a key the
target type does not declare, so those unconditional injections were a hard
construction failure for the 14 registered types that declare neither key or only
one of them. Nothing in the tree sets a non-default `attention_type` on a head,
so nothing was red -- this is the same latent shape as the `free_transformer.py`
B-4 defect that D-011 fixed.

RED, measured verbatim before the pre-filter landed (these are the exact kwarg
sets the two heads used to build):

    create_attention_layer('fnet', dim=32, dropout_rate=0.1)
    -> ValueError: create_attention_layer('fnet'): 2 unsupported parameter(s)
       ['dim', 'dropout_rate']. 'fnet' (FNetFourierTransform) accepts only
       ['epsilon', 'implementation', 'normalize_dft'].

    create_attention_layer('spatial', dim=32)
    -> ValueError: create_attention_layer('spatial'): 1 unsupported parameter(s)
       ['dim']. 'spatial' (SpatialAttention) accepts only [...].

The control arms below re-run those two raw factory calls, so a future change
that made the factory permissive again could not turn these tests green by
accident.
"""

import pytest

from dl_techniques.layers.attention.factory import (
    create_attention_layer,
    STRICT_DROPPED_KEY_MARKER,
)
from dl_techniques.layers.heads.nlp.factory import BaseNLPHead
from dl_techniques.layers.heads.nlp.task_types import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.vision.factory import BaseVisionHead


class TestNLPHeadPrefiltersItsOwnDefaults:

    def test_a_type_declaring_neither_dim_nor_dropout_rate_constructs(self) -> None:
        head = BaseNLPHead(
            task_config=NLPTaskConfig(
                name="t", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=2
            ),
            input_dim=32,
            use_task_attention=True,
            attention_type="fnet",
        )
        assert head.task_attention is not None
        assert type(head.task_attention).__name__ == "FNetFourierTransform"

    def test_control_the_raw_kwargs_still_raise(self) -> None:
        """The head is what changed, not the factory."""
        with pytest.raises(ValueError) as excinfo:
            create_attention_layer("fnet", dim=32, dropout_rate=0.1)
        assert STRICT_DROPPED_KEY_MARKER in str(excinfo.value)


class TestVisionHeadPrefiltersItsOwnDefaults:

    def test_a_type_that_does_not_declare_dim_constructs(self) -> None:
        head = BaseVisionHead(
            hidden_dim=32,
            use_attention=True,
            attention_type="spatial",
        )
        assert head.attention is not None
        assert type(head.attention).__name__ == "SpatialAttention"

    def test_control_the_raw_kwargs_still_raise(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            create_attention_layer("spatial", dim=32)
        assert STRICT_DROPPED_KEY_MARKER in str(excinfo.value)


class TestTheDefaultPathIsUnchanged:
    """The pre-filter must not eat the keys `multi_head` genuinely accepts."""

    def test_nlp_multi_head_still_receives_dim_and_num_heads(self) -> None:
        head = BaseNLPHead(
            task_config=NLPTaskConfig(
                name="t", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=2
            ),
            input_dim=32,
            use_task_attention=True,
            attention_type="multi_head",
        )
        assert head.task_attention.dim == 32
        assert head.task_attention.num_heads == 8
