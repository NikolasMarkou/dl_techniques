"""Shared instrument: does a model's epsilon knob reach EVERY in-block norm?

Named without a ``test_`` prefix so pytest does not collect it. Six guard modules
(``test_models/test_bert``, ``test_distilbert``, ``test_modern_bert``,
``test_qwen``, ``test_tree_transformer`` and ``test_layers/test_transformers``)
share it, because the defect it pins is one mechanism with six call sites, not six
defects: a caller that stores an epsilon knob and constructs
:class:`~dl_techniques.layers.transformers.transformer.TransformerLayer` without
``attention_norm_args`` / ``ffn_norm_args`` silently inherits
``create_normalization_layer``'s own ``epsilon=1e-6`` default at every one of the
``2 * num_layers`` in-block norms. See decisions.md D-007
(``plan-2026-08-19-a616f581``).

Two rules this module enforces on its callers, both learned the hard way:

* **Every** in-block norm is checked, never just the first one. The pre-fix state
  had the embedding/final norm correct and every block norm wrong, so a probe
  that samples one norm can read green against a fully broken model.
* An absent-difference assertion is worthless alone. Use
  :func:`assert_epsilon_tracks_the_knob`, which builds the model at TWO knob
  values and requires the observed epsilon to MOVE. Without that arm the guard
  passes just as well against a hardcoded constant that happens to match the
  default.
"""

from typing import Any, Callable, Iterable, List, Optional, Tuple

import keras

__all__ = [
    "collect_norm_epsilons",
    "assert_every_block_norm_uses",
    "assert_epsilon_tracks_the_knob",
]


def _epsilon_of(layer: keras.layers.Layer) -> Optional[float]:
    """Return a normalization layer's epsilon, or ``None`` if it has none.

    ``eps`` is checked as well as ``epsilon`` because
    ``create_normalization_layer`` ALIASES ``epsilon`` -> ``eps`` for
    ``global_response_norm``. ``dynamic_tanh`` genuinely has neither (the factory
    pops the key), which is why ``None`` is a legitimate answer rather than a
    failure.
    """
    for attribute in ("epsilon", "eps"):
        value = getattr(layer, attribute, None)
        if value is not None:
            return float(value)
    return None


def collect_norm_epsilons(
    blocks: Iterable[keras.layers.Layer],
) -> List[Tuple[str, str, float]]:
    """Walk each block's whole sub-layer tree and report every norm's epsilon.

    :param blocks: The model's transformer blocks (e.g. ``model.encoder_layers``).
    :type blocks: Iterable[keras.layers.Layer]
    :return: ``(block_name, norm_path, epsilon)`` for every normalization layer
        that exposes an epsilon, recursively.
    :rtype: List[Tuple[str, str, float]]
    """
    found: List[Tuple[str, str, float]] = []
    for block in blocks:
        for sub in block._flatten_layers(include_self=True, recursive=True):
            if "norm" not in type(sub).__name__.lower():
                continue
            epsilon = _epsilon_of(sub)
            if epsilon is not None:
                found.append((block.name, sub.name, epsilon))
    return found


def assert_every_block_norm_uses(
    blocks: Iterable[keras.layers.Layer],
    expected: float,
    expected_count: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Assert EVERY in-block norm runs at ``expected``.

    :param blocks: The model's transformer blocks.
    :type blocks: Iterable[keras.layers.Layer]
    :param expected: The model's own epsilon knob.
    :type expected: float
    :param expected_count: If given, the exact number of norms that must be
        found. Anti-vacuity: a walk that finds ZERO norms would otherwise satisfy
        the "all of them are correct" assertion trivially.
    :type expected_count: Optional[int]
    :return: The collected ``(block, norm, epsilon)`` rows, for reuse by callers.
    :rtype: List[Tuple[str, str, float]]
    :raises AssertionError: If any norm disagrees, or the count is wrong.
    """
    found = collect_norm_epsilons(blocks)
    assert found, "no in-block normalization layer found -- the walk is vacuous"
    if expected_count is not None:
        assert len(found) == expected_count, (
            f"expected {expected_count} in-block norms, found {len(found)}: "
            f"{[(b, n) for b, n, _ in found]}"
        )
    wrong = [row for row in found if row[2] != expected]
    assert not wrong, (
        f"{len(wrong)} of {len(found)} in-block norms do not use the model's own "
        f"epsilon {expected!r}: {wrong}"
    )
    return found


def assert_epsilon_tracks_the_knob(
    build: Callable[[float], Any],
    blocks_of: Callable[[Any], Iterable[keras.layers.Layer]],
    first: float,
    second: float,
    expected_count: Optional[int] = None,
) -> None:
    """Liveness arm: the observed epsilon must MOVE when the knob moves.

    Builds the model twice, at two DIFFERENT knob values, and requires every
    in-block norm to report each value in turn. Without this arm the
    correctness assertion is satisfied by an implementation that hardcodes a
    constant equal to the default under test.

    :param build: Callable taking the epsilon knob and returning a BUILT model.
    :type build: Callable[[float], Any]
    :param blocks_of: Callable returning a model's transformer blocks.
    :type blocks_of: Callable[[Any], Iterable[keras.layers.Layer]]
    :param first: First knob value. Must differ from ``second``.
    :type first: float
    :param second: Second knob value.
    :type second: float
    :param expected_count: Exact number of in-block norms expected, if known.
    :type expected_count: Optional[int]
    :raises AssertionError: If either arm disagrees with its own knob.
    """
    assert first != second, "the two knob values must differ or nothing is proven"
    for value in (first, second):
        model = build(value)
        assert_every_block_norm_uses(blocks_of(model), value, expected_count)
