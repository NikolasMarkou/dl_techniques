"""
Oracle adoption for ``models/language/bert`` -- the lazy-build contract, by consequence.

``lazy_build_contract_oracle`` had **no** adopter in this package before this
file (only ``test_cliffordnet``, ``test_depth_anything`` and ``test_video_jepa``
adopt it at all), and ``models/language/bert`` is one of only TWO packages in this tree
where a lazy build was ever measured to COST something: D-049 of
``plan-2026-08-19T163559-499b6f0e`` records that ``BERT``'s missing ``build()``
silently disabled weight tying, made ``CausalLanguageModel.embedding_weights``
read ``None``, reported ``use_weight_tying`` as ``False``, and made the
``.keras`` round trip RAISE. That is exactly the defect this instrument exists
to catch, and this package had no standing guard against its return.

What the oracle asks, and what it deliberately does not
------------------------------------------------------
It does NOT assert the contract ("is the model built after ``build()``?") --
~110 packages violate that harmlessly. It asserts the CONSEQUENCE: perturb every
float weight, prove the perturbation MOVED the output (without which an exact
round trip proves nothing), then save/load and require an EXACT match at
``atol=0.0``. The materialization ratio is reported as a NUMBER, not a verdict.

MEASURED, GPU 1 (RTX 4070), TF32 on by default
-----------------------------------------------

======================================  ==========  ==========
                                        encoder     with head
======================================  ==========  ==========
weights after one call                  17          25
weights after ``.build()`` alone        17          n/a [1]
``count_params()`` after ``.build()``   11,744      n/a [1]
materialization ratio                   1.0         n/a [1]
weights perturbed                       17          25
perturbation liveness                   9.181e-01   4.871e-02
weights after reload                    17          25
round-trip ``max|delta|`` at atol=0.0   0.000e+00   0.000e+00
======================================  ==========  ==========

[1] The head factory returns a Functional model over a dict of three inputs;
    ``measure_lazy_build``'s materialization arm takes a single ``input_shape``
    and there is no honest one to give. The consequence arms -- which are the
    ones that found both real defects in this tree -- run in full.

The ratio of 1.0 is the D-049 repair, still in place. Both round trips are
EXACTLY zero, so the lazy build costs nothing today; this file is what notices
if that changes.
"""

from typing import Any, Dict

import numpy as np
import keras

from dl_techniques.models.language.bert import create_bert, create_bert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing

#: Shared with ``test_precision_arm.py``'s localisation pair on purpose: the
#: same geometry means a number measured in one file is comparable in the other.
ENCODER_KWARGS: Dict[str, Any] = {
    "vocab_size": 64,
    "max_position_embeddings": 32,
    "hidden_size": 32,
    "num_layers": 1,
    "num_heads": 2,
    "intermediate_size": 64,
}

SEQ_LEN = 16

#: Measured 2026-08-24 on GPU 1; see the module docstring.
N_WEIGHTS_ENCODER = 17
N_WEIGHTS_WITH_HEAD = 25
COUNT_PARAMS_AFTER_BUILD = 11_744


def _build_encoder() -> keras.Model:
    return create_bert("tiny", **ENCODER_KWARGS)


def _ids() -> np.ndarray:
    return np.random.RandomState(0).randint(
        0, ENCODER_KWARGS["vocab_size"], (2, SEQ_LEN)
    ).astype("int32")


def _build_with_head() -> keras.Model:
    return create_bert_with_head(
        "tiny",
        NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        ),
        bert_config_overrides=dict(ENCODER_KWARGS),
    )


def _head_inputs() -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(0)
    return {
        "input_ids": rng.randint(0, ENCODER_KWARGS["vocab_size"], (2, SEQ_LEN)).astype("int32"),
        "attention_mask": np.ones((2, SEQ_LEN), dtype="int32"),
        "token_type_ids": np.zeros((2, SEQ_LEN), dtype="int32"),
    }


def test_the_bert_encoders_lazy_build_still_costs_nothing() -> None:
    """The D-049 regression pin: ``build()`` materializes 17 of 17 and reloads exact.

    MEASURED (GPU 1): 17 weights after a call, **17 after ``.build()`` alone**
    (``count_params`` 11,744, ratio 1.0); 17 perturbed; perturbation liveness
    **9.181244e-01**; 17 weights reloaded; round trip ``max|delta|`` **exactly
    0.000000e+00** at ``atol=0.0``.

    The ratio is pinned as an equality and not as ``> 0`` on purpose: D-049 was
    a PARTIAL materialization, and a ``> 0`` assertion passes against exactly
    that.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build_encoder,
        make_inputs=_ids,
        input_shape=(None, SEQ_LEN),
    )

    assert report["n_weights"] == N_WEIGHTS_ENCODER
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3, (
        f"perturbing all {report['n_perturbed']} weights moved the output by "
        f"only {report['perturb_liveness']:.6e}; the round-trip comparison "
        f"above is close to vacuous"
    )
    materialization = report["materialization"]
    assert materialization["n_weights_after_build"] == N_WEIGHTS_ENCODER, (
        f"BERT.build() now materializes "
        f"{materialization['n_weights_after_build']} of {N_WEIGHTS_ENCODER} "
        f"weights. A partial materialization is the D-049 shape: weight tying "
        f"silently off, embedding_weights None, and a round trip that RAISES."
    )
    assert materialization["count_params_after_build"] == COUNT_PARAMS_AFTER_BUILD
    assert materialization["ratio"] == 1.0


def test_the_head_factorys_model_survives_a_round_trip_exactly() -> None:
    """The same consequence arms on ``create_bert_with_head``.

    No ``input_shape``: the factory returns a Functional model over a dict of
    three inputs and the materialization arm takes one shape, so it is skipped
    rather than fed a shape that is not the model's. The arms that found both
    real lazy-build defects in this tree -- live perturbation, then an EXACT
    reload comparison -- all run.

    MEASURED (GPU 1): 25 weights, 25 perturbed, liveness **4.870642e-02**, 25
    reloaded, round trip **0.000000e+00** at ``atol=0.0``.
    """
    report = assert_lazy_build_costs_nothing(
        build=_build_with_head,
        make_inputs=_head_inputs,
    )

    assert report["n_weights"] == N_WEIGHTS_WITH_HEAD, (
        f"the head model materialized {report['n_weights']} weights, expected "
        f"{N_WEIGHTS_WITH_HEAD}"
    )
    assert report["n_weights_reloaded"] == N_WEIGHTS_WITH_HEAD
    assert report["roundtrip_max_delta"] == 0.0
    assert report["perturb_liveness"] > 1e-3
