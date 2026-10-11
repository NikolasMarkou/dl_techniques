"""
The classifier on top of a Mamba backbone must read the whole sequence.

`Mamba` is a strictly causal selective SSM: the hidden state at position 0 is a
function of token 0 alone. A model built with ``include_top=True`` gets its task
head through `layers/heads/nlp/factory.py`, whose `BaseNLPHead` default is
``pooling_type='cls'`` -- i.e. take position 0. Under a causal backbone that
makes the classifier a function of the FIRST TOKEN ID and nothing else, which is
prior finding C-29 and structurally identical to the qwen3 defect fixed at
`plan-2026-08-14T233721-d4f9beb2/D-029`.

MEASURED before the fix (CPU, 8-token input, `d_model=32`, 2 layers): perturbing
token 5 moved the logits by exactly ``0.000000e+00`` while perturbing token 0
moved them by ``6.204727e-02``. The failure is silent -- the model trains, the
loss falls, and accuracy plateaus at the first-token prior.

A second blocker sat in front of this one in the old ``create_mamba_with_head``
wrapper: it gated on ``hasattr(task_config, 'vocab_size')``, a field
`NLPTaskConfig` has never had, so it raised `ValueError` on every call. That
wrapper is gone (plan-2026-10-11T004932-b83935e1 step 9): the head now lives in
the model, whose vocabulary is its own ``vocab_size``, so a task config needs no
``vocabulary_size`` at all. Both `Mamba` and `Mamba2` are covered.

The anti-vacuity arm (token 0 DOES move the logits) is what makes the first
assertion evidence rather than a statement about a dead model. RED-proved
2026-10-11 by building the head with ``default_pooling="cls"``: the late-token
and mechanism arms fail for both classes, the anti-vacuity arm stays green.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.mamba import create_mamba, create_mamba2


@pytest.fixture(scope="module", params=["mamba", "mamba2"])
def tiny_classifier(request) -> keras.Model:
    """A deliberately tiny Mamba text classifier built through the real factory.

    The task config sets no ``vocabulary_size``: the vocabulary is the model's
    own ``vocab_size``.
    """
    task_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.TEXT_CLASSIFICATION,
        num_classes=3,
    )
    if request.param == "mamba":
        return create_mamba(
            "base", vocab_size=64, d_model=32, num_layers=2,
            include_top=True, task_config=task_config,
        )
    return create_mamba2(
        "base", vocab_size=64, d_model=32, num_layers=2, d_state=8, headdim=16,
        include_top=True, task_config=task_config,
    )


def _logits(model: keras.Model, ids: np.ndarray) -> np.ndarray:
    out = model({"input_ids": keras.ops.convert_to_tensor(ids)}, training=False)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return np.asarray(keras.ops.convert_to_numpy(out))


class TestCausalHeadPooling:
    """The head must not collapse a causal backbone onto its first token."""

    def test_a_late_token_moves_the_logits(self, tiny_classifier: keras.Model) -> None:
        base = np.array([[3, 9, 14, 5, 21, 7, 2, 11]], dtype="int32")
        perturbed = base.copy()
        perturbed[0, 5] = 33  # a token strictly AFTER position 0

        delta = float(
            np.max(np.abs(_logits(tiny_classifier, base) - _logits(tiny_classifier, perturbed)))
        )
        assert delta > 1e-6, (
            "perturbing token 5 moved the classifier logits by "
            f"{delta:.6e} -- the head is pooling position 0 only, so a strictly "
            "causal Mamba backbone makes this classifier a function of the first "
            "token id alone (finding C-29 / D-029)."
        )

    def test_the_first_token_also_moves_the_logits(self, tiny_classifier: keras.Model) -> None:
        """Anti-vacuity: the model is not simply inert to its input."""
        base = np.array([[3, 9, 14, 5, 21, 7, 2, 11]], dtype="int32")
        perturbed = base.copy()
        perturbed[0, 0] = 33

        delta = float(
            np.max(np.abs(_logits(tiny_classifier, base) - _logits(tiny_classifier, perturbed)))
        )
        assert delta > 1e-6, f"the model is inert to its own input: {delta:.6e}"

    def test_the_pooling_type_is_last_not_cls(self, tiny_classifier: keras.Model) -> None:
        """Pin the mechanism, not just the symptom."""
        heads = [
            layer for layer in tiny_classifier.layers
            if hasattr(layer, "pooling_type")
        ]
        assert len(heads) == 1, f"expected exactly one pooling head, found {len(heads)}"
        assert heads[0] is tiny_classifier.task_head
        assert heads[0].pooling_type == "last"
