"""
The H-Net ``arch_layout`` mini-language: a pure-Python parser and data model.

H-Net encodes its entire architecture -- nesting depth, mixer type at every level, and
which levels carry an MLP -- in one nested list called ``arch_layout``
(``["m4", ["T22"], "m4"]`` for the 1-stage models, ``["m4", ["T1m4", ["T27"], "m4T1"],
"m4"]`` for the 2-stage ones). A 3-element list is a non-innermost stage and reads
``[encoder, main_network, decoder]``; a 1-element list is the innermost stage and holds a
single isotropic stack. The middle slot recurses. Each *layout string* is parsed with
``re.findall(r"([mMtT])(\\d+)", s)``: the letter identity selects the mixer (``m``/``M`` =
Mamba-2, ``t``/``T`` = causal Transformer) and the letter CASE selects whether a SwiGLU MLP
accompanies it (lowercase = mixer only, uppercase = mixer + MLP).

This module contains no Keras import and builds nothing. It is a parser
(:func:`parse_arch_layout`), a per-stage config indexer (:func:`get_stage_cfg`), a depth
accounting function (:func:`n_residuals`) and a cited variant table
(:data:`MODEL_VARIANTS`). The layer and model modules in this package consume it; keeping
it Keras-free is what lets the whole mini-language be tested without a backend.

Nothing here is a ``keras.Layer`` or ``keras.Model``, so nothing here carries a
serialization registration decorator -- registration exists to let Keras reconstruct an
object from a ``.keras`` archive, and these are plain dataclasses. They round-trip instead
by value: :meth:`HNetArchConfig.to_dict` emits a JSON-compatible plain ``dict`` of lists,
scalars and nested dicts suitable for embedding directly in an enclosing model's
``get_config()``, and :meth:`HNetArchConfig.from_dict` rebuilds it in ``from_config()``.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: ``hnet/models/config_hnet.py``,
      ``hnet/models/hnet.py:44-147``, ``hnet/modules/isotropic.py:60-95``,
      ``hnet/modules/utils.py:14-17``.
"""

import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

# (none -- this module is deliberately dependency-free so that the mini-language can be
#  parsed and tested without importing Keras or any backend.)

__all__ = [
    "AttnSpec",
    "BlockSpec",
    "HNetArchConfig",
    "IsotropicSpec",
    "MODEL_VARIANTS",
    "SSMSpec",
    "StageSpec",
    "get_stage_cfg",
    "get_variant_config",
    "n_residuals",
    "n_residuals_by_stage",
    "parse_arch_layout",
    "stage_count",
]

# ---------------------------------------------------------------------
# The layout mini-language
# ---------------------------------------------------------------------

#: The four legal layout letters. ``m``/``M`` select the Mamba-2 mixer, ``t``/``T`` the
#: causal-attention mixer; the uppercase form additionally carries a SwiGLU MLP.
LAYOUT_LETTERS: Tuple[str, ...] = ("m", "M", "t", "T")

#: ``hnet/modules/isotropic.py:66`` verbatim.
_LAYOUT_RE = re.compile(r"([mMtT])(\d+)")

#: Mixer family selected by the letter identity (case-insensitive).
MIXER_MAMBA2 = "mamba2"
MIXER_ATTENTION = "attention"


@dataclass(frozen=True)
class BlockSpec:
    """One ``<letter><count>`` group of a layout string.

    :param kind: One of ``"m"``, ``"M"``, ``"t"``, ``"T"`` -- case is significant.
    :type kind: str
    :param n_layer: How many identical blocks the group expands to. Always ``>= 1``.
    :type n_layer: int
    """

    kind: str
    n_layer: int

    @property
    def mixer(self) -> str:
        """:returns: :data:`MIXER_MAMBA2` for ``m``/``M``, :data:`MIXER_ATTENTION` for
            ``t``/``T``.
        :rtype: str
        """
        return MIXER_MAMBA2 if self.kind.lower() == "m" else MIXER_ATTENTION

    @property
    def has_mlp(self) -> bool:
        """:returns: ``True`` when the letter is uppercase, i.e. the block pairs its
            mixer with a SwiGLU MLP (``hnet/modules/block.py:62-71``).
        :rtype: bool
        """
        return self.kind.isupper()

    @property
    def height(self) -> int:
        """:returns: How many tensors this group writes into the residual stream:
            ``n_layer`` when lowercase, ``2 * n_layer`` when uppercase because the mixer
            and the MLP each contribute one (``hnet/modules/isotropic.py:89-92``).
        :rtype: int
        """
        return 2 * self.n_layer if self.has_mlp else self.n_layer


@dataclass(frozen=True)
class IsotropicSpec:
    """A parsed layout string -- one isotropic stack at one stage and one position.

    :param layout: The original layout string, e.g. ``"m4T1"``.
    :type layout: str
    :param blocks: The parsed groups, in order.
    :type blocks: Tuple[BlockSpec, ...]
    :param stage_idx: The stage this stack lives at; indexes ``d_model`` and every
        per-stage list of :class:`HNetArchConfig`.
    :type stage_idx: int
    :param role: ``"encoder"``, ``"main"`` or ``"decoder"``.
    :type role: str
    """

    layout: str
    blocks: Tuple[BlockSpec, ...]
    stage_idx: int
    role: str

    @property
    def n_layer(self) -> int:
        """:returns: Total number of blocks in the stack.
        :rtype: int
        """
        return sum(b.n_layer for b in self.blocks)

    @property
    def height(self) -> int:
        """:returns: ``Isotropic.height`` -- the number of residual-stream writes in this
            stack (``hnet/modules/isotropic.py:73,89-92``).
        :rtype: int
        """
        return sum(b.height for b in self.blocks)

    @property
    def arch_full(self) -> Tuple[str, ...]:
        """:returns: The per-layer letter sequence, ``"m4T1"`` expanding to
            ``("m", "m", "m", "m", "T")`` (``hnet/modules/isotropic.py:93``).
        :rtype: Tuple[str, ...]
        """
        out: List[str] = []
        for block in self.blocks:
            out.extend([block.kind] * block.n_layer)
        return tuple(out)


@dataclass(frozen=True)
class StageSpec:
    """One level of the recursive H-Net sandwich.

    A non-innermost stage carries ``encoder`` / ``main`` / ``decoder`` where ``main`` is
    the next :class:`StageSpec` inward. The innermost stage carries only ``main``, an
    :class:`IsotropicSpec`, and both ``encoder`` and ``decoder`` are ``None``.

    :param stage_idx: Depth of this stage, 0 at the outside.
    :type stage_idx: int
    :param is_innermost: Whether ``main`` is an :class:`IsotropicSpec` rather than a
        nested :class:`StageSpec`.
    :type is_innermost: bool
    :param encoder: The pre-chunking stack, or ``None`` when innermost.
    :type encoder: Optional[IsotropicSpec]
    :param main: The next stage inward, or the innermost isotropic stack.
    :type main: Union[IsotropicSpec, StageSpec]
    :param decoder: The post-dechunking stack, or ``None`` when innermost.
    :type decoder: Optional[IsotropicSpec]
    """

    stage_idx: int
    is_innermost: bool
    encoder: Optional[IsotropicSpec]
    main: Union[IsotropicSpec, "StageSpec"]
    decoder: Optional[IsotropicSpec]


def _parse_layout_string(layout: Any, stage_idx: int, role: str) -> IsotropicSpec:
    """Parse a single layout string into an :class:`IsotropicSpec`.

    ``re.findall`` on its own is silently forgiving -- ``"m4 xyz"`` and ``"q9"`` both
    return without complaint (the first drops the junk, the second returns an empty
    list, which the reference would turn into a zero-layer stack). This parser therefore
    checks that the matched groups reconstruct the input EXACTLY, so any unmatched
    character anywhere in the string is an error rather than a silent deletion.

    :param layout: The layout string.
    :type layout: Any
    :param stage_idx: Stage this stack belongs to.
    :type stage_idx: int
    :param role: ``"encoder"``, ``"main"`` or ``"decoder"``.
    :type role: str
    :returns: The parsed stack.
    :rtype: IsotropicSpec
    :raises TypeError: if ``layout`` is not a ``str``.
    :raises ValueError: if the string is empty, contains a character outside a
        ``[mMtT]<digits>`` group, or declares a group with zero layers.
    """
    if not isinstance(layout, str):
        raise TypeError(
            f"arch_layout {role} slot at stage {stage_idx} must be a string layout "
            f"such as 'm4T1', got {type(layout).__name__}: {layout!r}"
        )

    matches = _LAYOUT_RE.findall(layout)
    reconstructed = "".join(letter + digits for letter, digits in matches)
    if reconstructed != layout:
        raise ValueError(
            f"malformed arch_layout string {layout!r} at stage {stage_idx} "
            f"({role}): only groups of a letter in {LAYOUT_LETTERS} followed by a "
            f"digit count are allowed, and the whole string must be consumed "
            f"(parsed {reconstructed!r})"
        )
    if not matches:
        raise ValueError(
            f"empty arch_layout string at stage {stage_idx} ({role}): expected at "
            f"least one group such as 'm4'"
        )

    blocks: List[BlockSpec] = []
    for letter, digits in matches:
        n_layer = int(digits)
        if n_layer < 1:
            raise ValueError(
                f"arch_layout group {letter}{digits!s} at stage {stage_idx} ({role}) "
                f"declares {n_layer} layers; a group must build at least one block"
            )
        blocks.append(BlockSpec(kind=letter, n_layer=n_layer))

    return IsotropicSpec(
        layout=layout,
        blocks=tuple(blocks),
        stage_idx=stage_idx,
        role=role,
    )


def parse_arch_layout(layout: Any, stage_idx: int = 0) -> StageSpec:
    """Recursively parse an ``arch_layout`` nested list into a :class:`StageSpec` tree.

    Mirrors ``hnet/models/hnet.py:56-67``: a 3-element list is
    ``[encoder, main_network, decoder]`` and is not innermost; a 1-element list holds
    only the innermost isotropic stack. Any other length is rejected.

    :param layout: The nested list, e.g. ``["m4", ["T22"], "m4"]``.
    :type layout: Any
    :param stage_idx: Stage index this list sits at; 0 for the outermost call.
    :type stage_idx: int
    :returns: The parsed stage tree.
    :rtype: StageSpec
    :raises TypeError: if ``layout`` is not a list/tuple, or a slot holds the wrong type.
    :raises ValueError: if the list length is neither 1 nor 3, or a layout string is
        malformed.
    """
    if isinstance(layout, str) or not isinstance(layout, (list, tuple)):
        raise TypeError(
            f"arch_layout at stage {stage_idx} must be a list, got "
            f"{type(layout).__name__}: {layout!r}"
        )

    if len(layout) == 3:
        encoder = _parse_layout_string(layout[0], stage_idx, "encoder")
        decoder = _parse_layout_string(layout[2], stage_idx, "decoder")
        main = parse_arch_layout(layout[1], stage_idx + 1)
        return StageSpec(
            stage_idx=stage_idx,
            is_innermost=False,
            encoder=encoder,
            main=main,
            decoder=decoder,
        )

    if len(layout) == 1:
        main = _parse_layout_string(layout[0], stage_idx, "main")
        return StageSpec(
            stage_idx=stage_idx,
            is_innermost=True,
            encoder=None,
            main=main,
            decoder=None,
        )

    raise ValueError(
        f"arch_layout at stage {stage_idx} has length {len(layout)}; only 3 "
        f"([encoder, main, decoder]) and 1 ([innermost]) are supported. Got {layout!r}"
    )


def stage_count(spec: StageSpec) -> int:
    """Count the stages in a parsed tree, innermost included.

    ``["m4", ["T22"], "m4"]`` has 2 stages, so ``d_model`` and every per-stage list must
    carry 2 entries.

    :param spec: A parsed stage tree.
    :type spec: StageSpec
    :returns: Number of stages.
    :rtype: int
    """
    depth = 1
    node = spec
    while not node.is_innermost:
        node = node.main
        depth += 1
    return depth


def n_residuals(spec: StageSpec) -> int:
    """Total residual-stream writes across the WHOLE nested hierarchy.

    Sums ``Isotropic.height`` over every stack at every level: the encoder and decoder of
    each non-innermost stage plus the innermost main stack. Lowercase groups contribute
    ``n``, uppercase ``2n``. This is the denominator of the depth-scaled init
    ``std = 0.02 / sqrt(n_residuals)`` applied to every residual-WRITING projection
    (``out_proj`` / ``fc2``); getting it wrong mis-scales the whole model silently, which
    is why it is pinned by a hand-derived test.

    Worked example, ``hnet_1stage_L`` with ``["m4", ["T22"], "m4"]``: encoder ``m4`` is
    lowercase so ``+4``; decoder ``m4`` lowercase ``+4``; innermost main ``T22`` is
    uppercase so ``+44``. Total ``4 + 4 + 44 = 52``.

    :param spec: A parsed stage tree.
    :type spec: StageSpec
    :returns: The hierarchy-wide block-residual count.
    :rtype: int
    """
    if spec.is_innermost:
        return spec.main.height
    return spec.encoder.height + spec.decoder.height + n_residuals(spec.main)


# DECISION plan-2026-09-09T042752-6d66ac56/D-016: TWO functions, deliberately, and
# neither is redundant. Do NOT delete `n_residuals_by_stage` as a duplicate of
# `n_residuals`: the reference does NOT scale a hierarchy by one number. `hnet.py:121-147`
# threads `parent_residuals` INWARD, so stage k is scaled by the outside-in CUMULATIVE
# count -- (8, 52) for hnet_1stage_L, (8, 20, 74) for hnet_2stage_XL -- and the two agree
# ONLY at the innermost stage. Using the total everywhere scales a 2-stage model's outer
# sandwich by 74 instead of 8, an init sqrt(74/8) = 3.04x too small, with no shape symptom.
# `n_residuals` is kept because it IS the pinned total and the last element of the tuple
# provably equals it. See decisions.md D-016 (the surprise) and D-021 (the choice).
def n_residuals_by_stage(spec: StageSpec) -> Tuple[int, ...]:
    """The reference's OUTSIDE-IN CUMULATIVE residual counts, one per stage.

    ``hnet/models/hnet.py:121-147`` does not use a single hierarchy-wide number: it
    threads ``parent_residuals`` inward, so stage ``k``'s projections are scaled by the
    sum of every OUTER stage's encoder+decoder heights plus stage ``k``'s own
    contribution. For ``hnet_1stage_L`` that is ``(8, 52)``: the outer ``m4``/``m4``
    sandwich is initialized against 8, the innermost ``T22`` against 52.

    Every level contributes exactly once along the single chain, so the LAST element is
    always equal to :func:`n_residuals`. That identity is what makes the two functions
    agree on 1-stage models and diverge, per stage, on deeper ones.

    :param spec: A parsed stage tree.
    :type spec: StageSpec
    :returns: One cumulative count per stage, outermost first.
    :rtype: Tuple[int, ...]
    """
    out: List[int] = []
    running = 0
    node = spec
    while True:
        if node.is_innermost:
            running += node.main.height
            out.append(running)
            return tuple(out)
        running += node.encoder.height + node.decoder.height
        out.append(running)
        node = node.main


# ---------------------------------------------------------------------
# Per-stage config specs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SSMSpec:
    """Mamba-2 hyper-parameters shared by every stage.

    The reference's ``SSMConfig`` also declares ``chunk_size=256``. It is deliberately
    ABSENT here: this repository's ``Mamba2Layer`` implements a sequential scan and has no
    chunked path, so a ``chunk_size`` knob would be a declared field that nothing reads --
    the exact defect class ``tests/test_train/test_config_fields_are_live.py`` exists to
    reject. Do not add it back without a real consumer.

    :param d_conv: Depthwise causal convolution width.
    :type d_conv: int
    :param expand: Inner-width expansion factor.
    :type expand: int
    :param d_state: SSM state dimension.
    :type d_state: int
    """

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128


@dataclass(frozen=True)
class AttnSpec:
    """Causal-attention hyper-parameters, one entry per stage.

    :param num_heads: Attention heads at each stage.
    :type num_heads: Tuple[int, ...]
    :param rotary_emb_dim: RoPE dimensionality at each stage.
    :type rotary_emb_dim: Tuple[int, ...]
    :param window_size: Sliding-window width at each stage; ``-1`` means unlimited
        (full causal) context.
    :type window_size: Tuple[int, ...]
    """

    num_heads: Tuple[int, ...] = ()
    rotary_emb_dim: Tuple[int, ...] = ()
    window_size: Tuple[int, ...] = ()


def get_stage_cfg(cfg: Any, stage_idx: int) -> Dict[str, Any]:
    """Index every LIST field of a config dataclass by stage; pass scalars through.

    Transcribed from ``hnet/modules/utils.py:14-17``
    (``{k: v[stage_idx] if isinstance(v, list) else v ...}``), with two deliberate
    strictness additions the reference lacks: a negative ``stage_idx`` is rejected rather
    than silently wrapping around to the last stage, and an out-of-range index raises an
    ``IndexError`` that names the field and the available length.

    :param cfg: Any dataclass instance whose per-stage fields are sequences.
    :type cfg: Any
    :param stage_idx: The stage to extract.
    :type stage_idx: int
    :returns: Field name to per-stage value.
    :rtype: Dict[str, Any]
    :raises TypeError: if ``cfg`` is not a dataclass instance.
    :raises IndexError: if ``stage_idx`` is negative or beyond a sequence field's length.
    """
    try:
        cfg_fields = fields(cfg)
    except TypeError as exc:
        raise TypeError(
            f"get_stage_cfg expects a dataclass instance, got "
            f"{type(cfg).__name__}"
        ) from exc

    if stage_idx < 0:
        raise IndexError(
            f"stage_idx must be non-negative, got {stage_idx}; a negative index would "
            f"silently select a stage counted from the innermost end"
        )

    out: Dict[str, Any] = {}
    for spec in cfg_fields:
        value = getattr(cfg, spec.name)
        if isinstance(value, (list, tuple)):
            if stage_idx >= len(value):
                raise IndexError(
                    f"stage_idx {stage_idx} is out of range for field "
                    f"{spec.name!r} of {type(cfg).__name__}, which has "
                    f"{len(value)} entries"
                )
            out[spec.name] = value[stage_idx]
        else:
            out[spec.name] = value
    return out


# ---------------------------------------------------------------------
# The architecture config
# ---------------------------------------------------------------------


def _as_int_tuple(values: Any, name: str) -> Tuple[int, ...]:
    """Normalize a per-stage sequence to a tuple of ints.

    :param values: The sequence.
    :type values: Any
    :param name: Field name, used in error messages.
    :type name: str
    :returns: The normalized tuple.
    :rtype: Tuple[int, ...]
    :raises TypeError: if ``values`` is not a non-string sequence of ints.
    """
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(
            f"{name} must be a list of per-stage integers, got "
            f"{type(values).__name__}: {values!r}"
        )
    out: List[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"{name} must contain only integers, got "
                f"{type(item).__name__}: {item!r}"
            )
        out.append(int(item))
    return tuple(out)


def _freeze_layout(layout: Any) -> Any:
    """Recursively convert a nested layout list to nested tuples.

    Module-level variant rows are shared objects; storing mutable lists in them would let
    one caller's edit reach every other caller.

    :param layout: The nested list.
    :type layout: Any
    :returns: The same structure with tuples in place of lists.
    :rtype: Any
    """
    if isinstance(layout, (list, tuple)):
        return tuple(_freeze_layout(item) for item in layout)
    return layout


def _thaw_layout(layout: Any) -> Any:
    """Recursively convert a nested layout tuple back to nested lists (JSON shape).

    :param layout: The nested tuple.
    :type layout: Any
    :returns: The same structure with lists in place of tuples.
    :rtype: Any
    """
    if isinstance(layout, (list, tuple)):
        return [_thaw_layout(item) for item in layout]
    return layout


@dataclass(frozen=True)
class HNetArchConfig:
    """The full H-Net architecture description -- the port of ``HNetConfig``.

    Every per-stage field is a sequence with one entry per stage, where the stage count is
    fixed by ``arch_layout``'s nesting depth. Construction validates the layout and every
    length, so an inconsistent config fails here rather than deep inside a layer build.

    All sequence fields are stored as tuples: instances of this class are shared as
    module-level constants in :data:`MODEL_VARIANTS`, and a mutable list inside a shared
    constant is a defect waiting to happen.

    :param arch_layout: The nested layout mini-language.
    :type arch_layout: Any
    :param d_model: Hidden width at each stage.
    :type d_model: Tuple[int, ...]
    :param d_intermediate: SwiGLU intermediate width at each stage. It is CONSUMED --
        :func:`~dl_techniques.models.language.hnet.components.build_mlp` honours any
        positive value verbatim (rounded up to a multiple of 128, as upstream also
        rounds one), and ``0`` means "derive it" as ``round_up(8 * d_model / 3, 128)``.
        ``0`` is this port's spelling of the reference's ``d_intermediate=None``
        sentinel, which an int tuple cannot carry; it does NOT mean "an MLP of width
        zero". Every shipped variant pairs its ``0`` entries with an all-lowercase
        stage, which has no MLP at all, so the sentinel is never actually reached
        there -- but a caller MAY pair ``0`` with an uppercase stage and get the
        derived width, which is what this port's own test fixtures do.
    :type d_intermediate: Tuple[int, ...]
    :param vocab_size: Byte vocabulary; 256 for the raw-byte models.
    :type vocab_size: int
    :param ssm_cfg: Mamba-2 hyper-parameters, shared across stages.
    :type ssm_cfg: SSMSpec
    :param attn_cfg: Attention hyper-parameters, one entry per stage.
    :type attn_cfg: AttnSpec
    :param tie_embeddings: Whether the LM head shares the embedding matrix.
    :type tie_embeddings: bool
    :raises TypeError: if a field has the wrong type.
    :raises ValueError: if the layout is malformed or a per-stage list length disagrees
        with the layout's stage count.
    """

    arch_layout: Any
    d_model: Tuple[int, ...]
    d_intermediate: Tuple[int, ...]
    vocab_size: int = 256
    ssm_cfg: SSMSpec = SSMSpec()
    attn_cfg: AttnSpec = AttnSpec()
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arch_layout", _freeze_layout(self.arch_layout))
        object.__setattr__(self, "d_model", _as_int_tuple(self.d_model, "d_model"))
        object.__setattr__(
            self, "d_intermediate", _as_int_tuple(self.d_intermediate, "d_intermediate")
        )
        object.__setattr__(
            self,
            "attn_cfg",
            AttnSpec(
                num_heads=_as_int_tuple(self.attn_cfg.num_heads, "attn_cfg.num_heads"),
                rotary_emb_dim=_as_int_tuple(
                    self.attn_cfg.rotary_emb_dim, "attn_cfg.rotary_emb_dim"
                ),
                window_size=_as_int_tuple(
                    self.attn_cfg.window_size, "attn_cfg.window_size"
                ),
            ),
        )

        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        spec = parse_arch_layout(self.arch_layout)
        n_stages = stage_count(spec)
        per_stage = {
            "d_model": self.d_model,
            "d_intermediate": self.d_intermediate,
            "attn_cfg.num_heads": self.attn_cfg.num_heads,
            "attn_cfg.rotary_emb_dim": self.attn_cfg.rotary_emb_dim,
            "attn_cfg.window_size": self.attn_cfg.window_size,
        }
        for name, values in per_stage.items():
            if len(values) != n_stages:
                raise ValueError(
                    f"{name} has {len(values)} entries but arch_layout "
                    f"{_thaw_layout(self.arch_layout)!r} declares {n_stages} stages"
                )
        for width in self.d_model:
            if width < 1:
                raise ValueError(f"every d_model must be positive, got {self.d_model}")
        for width in self.d_intermediate:
            if width < 0:
                raise ValueError(
                    f"every d_intermediate must be non-negative, got "
                    f"{self.d_intermediate}"
                )

    @property
    def stage_spec(self) -> StageSpec:
        """:returns: The parsed :class:`StageSpec` tree for :attr:`arch_layout`.
        :rtype: StageSpec
        """
        return parse_arch_layout(self.arch_layout)

    @property
    def num_stages(self) -> int:
        """:returns: The number of stages, innermost included.
        :rtype: int
        """
        return len(self.d_model)

    @property
    def n_residuals(self) -> int:
        """:returns: :func:`n_residuals` for this config's layout.
        :rtype: int
        """
        return n_residuals(self.stage_spec)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible plain ``dict``.

        The result contains only lists, ints, bools, strings and nested dicts, so it can
        be embedded directly in an enclosing Keras model's ``get_config()`` and survive
        the JSON round-trip a ``.keras`` archive performs. Rebuild with
        :meth:`from_dict` inside ``from_config()``.

        :returns: The plain-dict form, shaped like the reference's ``configs/*.json``
            minus ``ssm_cfg.chunk_size``.
        :rtype: Dict[str, Any]
        """
        return {
            "arch_layout": _thaw_layout(self.arch_layout),
            "d_model": list(self.d_model),
            "d_intermediate": list(self.d_intermediate),
            "vocab_size": self.vocab_size,
            "ssm_cfg": {
                "d_conv": self.ssm_cfg.d_conv,
                "expand": self.ssm_cfg.expand,
                "d_state": self.ssm_cfg.d_state,
            },
            "attn_cfg": {
                "num_heads": list(self.attn_cfg.num_heads),
                "rotary_emb_dim": list(self.attn_cfg.rotary_emb_dim),
                "window_size": list(self.attn_cfg.window_size),
            },
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "HNetArchConfig":
        """Rebuild from the output of :meth:`to_dict`.

        Unknown keys are rejected rather than ignored: a silently dropped key is how a
        config field stops being honoured without anything failing.

        :param config: The plain-dict form.
        :type config: Dict[str, Any]
        :returns: The rebuilt config.
        :rtype: HNetArchConfig
        :raises KeyError: if a required key is missing or an unknown key is present.
        """
        allowed = {
            "arch_layout",
            "d_model",
            "d_intermediate",
            "vocab_size",
            "ssm_cfg",
            "attn_cfg",
            "tie_embeddings",
        }
        unknown = set(config) - allowed
        if unknown:
            raise KeyError(
                f"unknown HNetArchConfig keys {sorted(unknown)}; expected a subset of "
                f"{sorted(allowed)}"
            )
        missing = {"arch_layout", "d_model", "d_intermediate"} - set(config)
        if missing:
            raise KeyError(f"missing required HNetArchConfig keys {sorted(missing)}")

        ssm = dict(config.get("ssm_cfg", {}))
        attn = dict(config.get("attn_cfg", {}))
        return cls(
            arch_layout=config["arch_layout"],
            d_model=config["d_model"],
            d_intermediate=config["d_intermediate"],
            vocab_size=config.get("vocab_size", 256),
            ssm_cfg=SSMSpec(**ssm),
            attn_cfg=AttnSpec(**attn),
            tie_embeddings=config.get("tie_embeddings", False),
        )


# ---------------------------------------------------------------------
# The six shipped variants
# ---------------------------------------------------------------------

# Every row below is transcribed from the correspondingly named file in the reference
# repository's `configs/` directory, with `ssm_cfg.chunk_size` dropped (see SSMSpec).
# The three `ssm_cfg` values are identical (4 / 2 / 128) in all six JSONs, so each row
# relies on the SSMSpec defaults rather than restating them.
MODEL_VARIANTS: Dict[str, HNetArchConfig] = {
    # configs/hnet_1stage_L.json
    "hnet_1stage_L": HNetArchConfig(
        arch_layout=["m4", ["T22"], "m4"],
        d_model=[1024, 1536],
        d_intermediate=[0, 4096],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16],
            rotary_emb_dim=[32, 48],
            window_size=[1023, -1],
        ),
        tie_embeddings=False,
    ),
    # configs/hnet_1stage_XL.json
    "hnet_1stage_XL": HNetArchConfig(
        arch_layout=["m4", ["T24"], "m4"],
        d_model=[1024, 2048],
        d_intermediate=[0, 5504],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16],
            rotary_emb_dim=[32, 64],
            window_size=[1023, -1],
        ),
        tie_embeddings=False,
    ),
    # configs/hnet_2stage_L.json
    "hnet_2stage_L": HNetArchConfig(
        arch_layout=["m4", ["T1m4", ["T26"], "m4T1"], "m4"],
        d_model=[1024, 1024, 1536],
        d_intermediate=[0, 2816, 4096],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16, 16],
            rotary_emb_dim=[32, 32, 48],
            window_size=[1023, 1023, -1],
        ),
        tie_embeddings=False,
    ),
    # configs/hnet_2stage_XL.json
    "hnet_2stage_XL": HNetArchConfig(
        arch_layout=["m4", ["T1m4", ["T27"], "m4T1"], "m4"],
        d_model=[1024, 1536, 2048],
        d_intermediate=[0, 4096, 5504],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16, 16],
            rotary_emb_dim=[32, 48, 64],
            window_size=[1023, 1023, -1],
        ),
        tie_embeddings=False,
    ),
    # configs/hnet_2stage_XL_chinese.json
    "hnet_2stage_XL_chinese": HNetArchConfig(
        arch_layout=["m4", ["T1m4", ["T30"], "m4T1"], "m4"],
        d_model=[1024, 1536, 2048],
        d_intermediate=[0, 4096, 5504],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16, 16],
            rotary_emb_dim=[32, 48, 64],
            window_size=[1023, 1023, -1],
        ),
        tie_embeddings=False,
    ),
    # configs/hnet_2stage_XL_code.json
    "hnet_2stage_XL_code": HNetArchConfig(
        arch_layout=["m4", ["T1m4", ["T28"], "m4T1"], "m4"],
        d_model=[1024, 1536, 2048],
        d_intermediate=[0, 4096, 5504],
        vocab_size=256,
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=128),
        attn_cfg=AttnSpec(
            num_heads=[16, 16, 16],
            rotary_emb_dim=[32, 48, 64],
            window_size=[1023, 1023, -1],
        ),
        tie_embeddings=False,
    ),
}


def get_variant_config(variant: str) -> HNetArchConfig:
    """Look up a shipped variant by name.

    :param variant: One of the keys of :data:`MODEL_VARIANTS`.
    :type variant: str
    :returns: The variant's config. Instances are immutable and shared.
    :rtype: HNetArchConfig
    :raises ValueError: if the name is unknown; the message lists every available name.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown H-Net variant {variant!r}; available variants are "
            f"{sorted(MODEL_VARIANTS)}"
        )
    return MODEL_VARIANTS[variant]
