"""Shared helpers, constants and validation for the vision heads.

This module holds the pure helpers that more than one head module reads:
stage-knob dropping, common-layer refusal, block-dim validation, hidden_dim
minimums, dropout refusal, the progressive-decoder constant, and the tables
of attention types that can / cannot run in a vision head.

It imports nothing from the head classes themselves, so nothing here can
participate in an import cycle.
"""

from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Flag-gated base stage knobs
# ---------------------------------------------------------------------

#: The base stage knobs each flag gates, with the value a knob takes when its
#: stage is on and the caller gave ``None``. Read by
#: :func:`~dl_techniques.layers.heads.validation.resolve_stage_knobs` (the
#: heads) and :func:`_drop_off_stage_knobs` (the presets and
#: ``build_model_vision_head``).
STAGE_KNOB_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'use_attention': {'attention_type': 'multi_head'},
    'use_ffn': {'ffn_type': 'mlp', 'ffn_expansion_factor': 4},
}


def _drop_off_stage_knobs(
        config: Dict[str, Any],
        keep: Tuple[Dict[str, Any], ...] = ()
) -> Dict[str, Any]:
    """
    Remove the knobs of every stage ``config`` turns off, unless set in ``keep``.

    Interface contract, shared by the :class:`HeadConfiguration` presets
    and :func:`build_model_vision_head`: returns a new dict; a knob whose
    gating flag is ``False`` in ``config`` (absent counts as the base
    default, on) is dropped, except when one of the ``keep`` dicts names it,
    so a value the model or the caller set explicitly still reaches the head
    and is refused there.

    :param config: A head config.
    :type config: Dict[str, Any]
    :param keep: Dicts whose knobs must not be dropped.
    :type keep: Tuple[Dict[str, Any], ...]
    :return: The config without the dead knobs.
    :rtype: Dict[str, Any]
    """
    out = dict(config)
    for flag, defaults in STAGE_KNOB_DEFAULTS.items():
        if out.get(flag, True):
            continue
        for knob in defaults:
            if not any(knob in source for source in keep):
                out.pop(knob, None)
    return out


# ---------------------------------------------------------------------
# Common-layer flag refusal
# ---------------------------------------------------------------------

def _refuse_common_layers_call_never_runs(
        kwargs: Dict[str, Any],
        head_name: str,
        flags: Tuple[str, ...]
) -> None:
    """
    Default each common-layer argument a head's ``call`` never applies to
    off (``False`` or ``0.0``), and refuse a value that would turn it on.

    Interface contract, shared by :class:`ClassificationHead` (``use_ffn``)
    and :class:`EnhancementHead` (``use_attention``, ``use_ffn``,
    ``dropout_rate``):

    * ``kwargs`` is the head's ``**kwargs`` before they reach
      :class:`BaseVisionHead`. Each name in ``flags`` that is absent is set to
      ``False`` in place (``0.0`` for ``dropout_rate``), so the base class
      builds and applies nothing for it.
    * A name passed with a truthy value (``True``, or a rate above 0) raises
      ``ValueError`` naming the head and the argument. Nothing is built then.
    * Returns ``None``.

    Call it BEFORE ``super().__init__``: the base class builds the common
    layers inside its own ``__init__``.

    :param kwargs: The head's keyword arguments, mutated in place.
    :type kwargs: Dict[str, Any]
    :param head_name: Class name, for the error message.
    :type head_name: str
    :param flags: Argument names the head's ``call`` never applies.
    :type flags: Tuple[str, ...]
    :return: None.
    :rtype: None
    :raises ValueError: If any name in ``flags`` is passed as truthy.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-020
    # Refuse, do NOT accept-and-ignore: a stored flag that builds nothing is a
    # dead knob (it silently changed no architecture for every caller that
    # passed it), and a built-but-unapplied layer is weight no gradient
    # reaches. See decisions.md D-020.
    for flag in flags:
        off = 0.0 if flag == 'dropout_rate' else False
        value = kwargs.setdefault(flag, off)
        if value:
            raise ValueError(
                f"{head_name} never applies what {flag}={value!r} asks for: "
                f"its call() runs no such layer. Pass {flag}={off!r} or leave "
                f"it unset."
            )


def _validated_block_dims(
        block_dims: Optional[Sequence[int]]
) -> Optional[Tuple[int, ...]]:
    """
    Check a head's ``block_dims`` argument and freeze it to a tuple.

    Interface contract, shared by :class:`ClassificationHead` and
    :class:`EnhancementHead`, whose ``block_dims`` means "the width of each
    hidden block, in order":

    * ``None`` is returned unchanged; each head resolves it to its own
      default stack.
    * Any other value is returned as a tuple. ``()`` is valid and means no
      hidden block.
    * An entry that is not a positive ``int`` (``bool`` included, since it is
      an ``int`` subclass) raises ``ValueError`` naming the whole argument.

    :param block_dims: The caller's ``block_dims``.
    :type block_dims: Optional[Sequence[int]]
    :return: ``None``, or the widths as a tuple.
    :rtype: Optional[Tuple[int, ...]]
    :raises ValueError: If an entry is not a positive integer.
    """
    if block_dims is None:
        return None
    block_dims = tuple(block_dims)
    bad = [d for d in block_dims
           if isinstance(d, bool) or not isinstance(d, int) or d <= 0]
    if bad:
        raise ValueError(
            f"block_dims must hold positive integers, got {block_dims}"
        )
    return block_dims


#: Registered attention types a vision head can construct but never run, and
#: why. A head calls its attention stage with one feature stream and nothing
#: else; these layers need more per call. Refused in ``BaseVisionHead`` when
#: the attention stage is on (D-089).
_UNRUNNABLE_ATTENTION_TYPES: Dict[str, str] = {
    'shared_weights_cross': (
        "SharedWeightsCrossAttention needs split_sizes on every call to split "
        "its input into modalities, and a vision head attends over one stream"
    ),
}

#: Values per box of a :class:`DetectionHead` regression branch when the
#: caller gives none: the usual corner or centre form.
_DEFAULT_BBOX_DIMS = 4

#: The registered attention types whose layer takes only a rank-4
#: ``(batch, height, width, channels)`` map. Every other type a vision head
#: can construct takes only a ``(batch, tokens, channels)`` sequence (measured
#: at both ranks over the whole registry, 2026-10-15; types the head cannot
#: construct from ``dim`` alone are not listed). ``BaseVisionHead`` reads it to
#: run its attention stage in the layout the type takes;
#: ``test_a_vision_head_runs_attention_in_its_own_layout.py`` re-measures it.
_SPATIAL_ATTENTION_TYPES: FrozenSet[str] = frozenset({
    'area', 'cbam', 'mobile_mqa', 'spatial',
    'tripse1', 'tripse2', 'tripse3', 'tripse4',
})


def _require_hidden_dim_at_least(
        hidden_dim: int,
        minimum: int,
        owner: str,
        reason: str
) -> None:
    """
    Refuse a ``hidden_dim`` too small for the widths a head derives from it.

    Interface contract, shared by the heads that halve ``hidden_dim``
    (:class:`SegmentationHead`, :class:`ClassificationHead`, and the
    progressive decoder of :class:`DepthEstimationHead`,
    :class:`SurfaceNormalsHead` and :class:`OpticalFlowHead`):

    * ``hidden_dim >= minimum`` returns ``None``.
    * Otherwise ``ValueError`` names ``owner``, ``hidden_dim``, ``minimum``
      and ``reason`` (the halving that would reach 0), instead of the
      ``filters=0``/``units=0`` error the derived layer raised.

    :param hidden_dim: The head's validated ``hidden_dim``.
    :type hidden_dim: int
    :param minimum: Smallest width every derived layer can take.
    :type minimum: int
    :param owner: Head class name, for the error message.
    :type owner: str
    :param reason: Which derived width needs the minimum.
    :type reason: str
    :return: None.
    :rtype: None
    :raises ValueError: If ``hidden_dim < minimum``.
    """
    if hidden_dim < minimum:
        raise ValueError(
            f"{owner} needs hidden_dim >= {minimum}, got {hidden_dim}: "
            f"{reason}"
        )


def _refuse_dropout_no_layer_applies(
        kwargs: Dict[str, Any],
        head_name: str
) -> None:
    """
    Default ``dropout_rate`` to 0.0 for a head whose decoder applies no
    dropout, and refuse a positive rate that no layer would read.

    Interface contract, shared by :class:`DepthEstimationHead`,
    :class:`SurfaceNormalsHead` and :class:`OpticalFlowHead`:

    * ``kwargs`` is the head's ``**kwargs`` before they reach
      :class:`BaseVisionHead`, with ``use_attention`` and ``use_ffn`` already
      defaulted. An absent ``dropout_rate`` is set to ``0.0`` in place.
    * A positive rate is accepted only with ``use_attention`` or ``use_ffn``
      on: it is then the attention/FFN dropout. Otherwise ``ValueError``
      names the head and the argument.
    * Returns ``None``.

    :param kwargs: The head's keyword arguments, mutated in place.
    :type kwargs: Dict[str, Any]
    :param head_name: Class name, for the error message.
    :type head_name: str
    :return: None.
    :rtype: None
    :raises ValueError: If a positive ``dropout_rate`` would reach no layer.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-050
    # Do NOT hand dropout_rate to the decoder ConvBlocks to make it "live":
    # no reference decoder for these tasks has a dropout (DPT/MiDaS depth,
    # Omnidata normals, RAFT FlowHead, PWC-Net predict_flow, the Mask R-CNN
    # mask branch). Do NOT go back to storing the base default 0.1 either: it
    # reached no layer with attention and the FFN off. See decisions.md D-050.
    rate = kwargs.setdefault('dropout_rate', 0.0)
    if rate and not (kwargs.get('use_attention') or kwargs.get('use_ffn')):
        raise ValueError(
            f"{head_name} applies dropout_rate={rate!r} nowhere: its decoder "
            f"takes no dropout, and it reaches only the attention or FFN "
            f"stage, both off here. Pass dropout_rate=0.0, or turn on "
            f"use_attention or use_ffn."
        )
