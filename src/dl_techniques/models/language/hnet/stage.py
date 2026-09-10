"""H-Net's recursive stage: the encoder / route / chunk / inner / dechunk / decoder sandwich.

This module is where H-Net's architecture actually composes. Everything built in
``config.py`` (the ``arch_layout`` mini-language), ``components.py`` (the isotropic
mixer stacks) and ``layers/dynamic_chunking/`` (routing, chunking, dechunking) meets
here, and the meeting point is *recursive*: a stage's "inner model" is either another
:class:`HNetStage` one level deeper or, at the innermost level, a plain
:class:`~dl_techniques.models.language.hnet.components.HNetIsotropic` stack.

The forward pass, transcribed from ``hnet/models/hnet.py:204-300``::

    hidden (B, L, D_in)
      |
      +-- pad_dimension            (only when this stage is WIDER than its parent)
      |
      |   [innermost]  main_network -> slice back to D_in -> return (hidden, [])
      |
      encoder  (HNetIsotropic)
      |
      +-------------------------------+
      |                               |
      routing_module            residual_proj  (Dense, zeros, fp32)
      |    -> boundary_prob (B, L, 2)      |
      |    -> boundary_mask (B, L)         |
      |    -> selected_probs (B, L, 1)     |
      |                               |
      chunk_layer -> (B, C, D), inner_mask (B, C)
      |                               |
      inner stage (HNetStage @ stage_idx+1)
      |                               |
      dechunk_layer -> (B, L, D)      |
      |                               |
      out * ste(selected_probs) + residual      <-- the confidence gate
      |
      decoder  (HNetIsotropic)
      |
      slice back to D_in
      |
      return (hidden, [this stage's routing record, *inner records])

The returned routing records propagate OUTWARD unchanged; the ratio (load-balancing)
loss in ``losses.py`` consumes them, one term per chunking level. They are the only
reason ``call`` returns a pair rather than a tensor.

Three things in this file are load-bearing and none of them is visible in a shape test
------------------------------------------------------------------------------------

1. **The straight-through gate is ONE grouped expression** --
   :func:`straight_through_ones`. Its ungrouped algebraic rewrite is not the same
   function in floating point. See the ``# DECISION`` anchor there.
2. **``residual_proj`` is exactly zero at initialisation** -- kernel AND bias -- so the
   stage starts as a pure pass-through of the dechunked inner result, and the residual
   branch has to be *learned* into existence. A zero-initialised weight looks dead to
   any init-time probe by construction, so its liveness is asserted after one real
   optimizer step, never at init.
3. **``pad_dimension``** is a learned ``(delta_d,)`` vector concatenated on entry and
   sliced back off on exit. Without the slice a stage silently returns its own width to
   a parent expecting the parent's width.

Divergences from the reference, both deliberate and both recorded in
``plans/plan-2026-09-09T042752-6d66ac56/decisions.md``: the residual projection's bias
starts at zero here where ``torch.nn.Linear`` leaves it uniformly random (D-019), and a
narrowing hierarchy (``d_model`` decreasing inward) is rejected with a ``ValueError``
rather than silently mis-shaped (the reference has no path for it either).

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.dynamic_chunking.chunk_layer import ChunkLayer
from dl_techniques.layers.dynamic_chunking.dechunk_layer import DeChunkLayer
from dl_techniques.layers.dynamic_chunking.routing_module import RoutingModule
from dl_techniques.models.language.hnet.components import HNetIsotropic
from dl_techniques.models.language.hnet.config import (
    HNetArchConfig,
    IsotropicSpec,
    StageSpec,
    get_stage_cfg,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import materialize_sublayers

# ---------------------------------------------------------------------

__all__ = [
    "HNetStage",
    "ROUTING_RECORD_KEYS",
    "straight_through_ones",
]

#: The keys of every routing record :meth:`HNetStage.call` emits. Fixed and complete:
#: ``losses.ratio_loss`` reads all four, and a record missing one is a defect that a
#: downstream ``KeyError`` would only surface at training time.
ROUTING_RECORD_KEYS: Tuple[str, ...] = (
    "boundary_prob",
    "boundary_mask",
    "selected_probs",
    "padding_mask",
)


# ---------------------------------------------------------------------
# The straight-through gate
# ---------------------------------------------------------------------


def straight_through_ones(x: keras.KerasTensor) -> keras.KerasTensor:
    """Return ``1.0`` in the forward pass and ``x``'s own gradient in the backward one.

    This is the port of the reference's ``STE`` autograd function
    (``hnet/models/hnet.py:20-31``): ``forward`` returns ``torch.ones_like(x)`` and
    ``backward`` passes the incoming gradient through untouched. Used as
    ``out * straight_through_ones(selected_probs) + residual``, it makes the confidence
    ``selected_probs`` a pure gradient conduit -- the routing module learns from the
    residual gate without the gate scaling the forward activations at all.

    :param x: Any float tensor; only its shape, dtype and gradient matter.
    :type x: keras.KerasTensor
    :returns: A tensor of the same shape and dtype, bit-exactly ``1.0`` everywhere.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-09-09T042752-6d66ac56/D-018: this is ONE grouped expression,
    # `x + stop_gradient(hard - x)`, and the grouping is load-bearing arithmetic rather
    # than style. Do NOT "simplify" it to the algebraically identical
    #     x + stop_gradient(hard) - stop_gradient(x)      (or any reassociation)
    # which is a DIFFERENT function in floating point: `fl(x + 1) - x` loses the low
    # bits of `x` in the first addition and does not return to 1.0. MEASURED here in
    # float64 over 100000 uniform [0, 1) draws: the grouped form is bit-exactly 1.0 on
    # 100000/100000, the ungrouped form on 75042/100000 -- the remaining 24958 (24.96%)
    # land exactly one ulp away (1.110223e-16 = 2^-53). The grouped form is exact for a
    # reason, not by luck: for
    # x >= 0.5, `1 - x` is exact by Sterbenz's lemma; for x < 0.5 the rounding error of
    # `1 - x` is at most half an ulp of 1.0, which the second addition rounds back to
    # 1.0. Pinned by test_stage.py::TestStraightThroughGate::
    # test_the_grouped_form_is_bit_exactly_one_in_float64 and its ungrouped twin.
    # Rationale: decisions.md D-018.
    hard = keras.ops.ones_like(x)
    return x + keras.ops.stop_gradient(hard - x)


# ---------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.hnet.stage")
class HNetStage(keras.layers.Layer):
    """One level of H-Net's hierarchy, holding the next level inside itself.

    Architecture:

    .. code-block:: text

        (B, L, D_parent)
              |
        concat learned pad_dimension (delta_d,)      [stage_idx > 0 and wider]
              |
        (B, L, D)
              |
         +----+------------------------------------------------+
         |                                                     |
        encoder HNetIsotropic                                  |   [not innermost]
         |                                                     |
         +--> residual_proj (Dense D->D, zeros, fp32) ---------+---> residual (B, L, D)
         |                                                     |
        routing_module --> boundary_prob / boundary_mask / selected_probs
         |                                                     |
        chunk_layer  ---> (B, C, D), inner_mask (B, C)         |
         |                                                     |
        inner HNetStage (stage_idx + 1)                        |
         |                                                     |
        dechunk_layer ---> (B, L, D)                           |
         |                                                     |
        out * straight_through_ones(selected_probs) + residual <+
         |
        decoder HNetIsotropic
         |
        slice [..., :D_parent]
         |
        (B, L, D_parent),  [routing record, *inner routing records]

    At the innermost stage the whole middle collapses to a single
    :class:`~dl_techniques.models.language.hnet.components.HNetIsotropic` and the routing
    record list is empty -- there is nothing below to chunk into.

    :param arch_config: The parsed architecture. Shared by reference with every nested
        stage; it is a frozen dataclass, so sharing is safe.
    :type arch_config: HNetArchConfig
    :param stage_idx: Depth of this stage, ``0`` at the outside. Indexes ``d_model`` and
        every per-stage entry of ``attn_cfg``.
    :type stage_idx: int
    :param max_chunks: One fixed chunk cap per NON-innermost stage, outermost first, so
        ``len(max_chunks) == arch_config.num_stages - 1``. Decision D-007: the inner
        width is a constructor argument and never a function of the data.
    :type max_chunks: Sequence[int]
    :param max_seq_len: Largest position the attention RoPE tables cover, at every stage.
    :type max_seq_len: int
    :param headdim: Mamba-2 SSM head width, at every stage. Not part of the reference's
        JSON configs (it takes the upstream default of 64) so it lives here.
    :type headdim: int
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :raises TypeError: if ``arch_config`` is not an :class:`HNetArchConfig`, or
        ``stage_idx`` / ``headdim`` are not ints.
    :raises ValueError: if ``stage_idx`` is out of range, if ``max_chunks`` does not
        have one entry per non-innermost stage, or if the hierarchy NARROWS inward
        (``d_model[stage_idx] < d_model[stage_idx - 1]``), which has no defined
        ``pad_dimension``.

    Example:
        >>> from dl_techniques.models.language.hnet.config import HNetArchConfig, AttnSpec
        >>> cfg = HNetArchConfig(
        ...     arch_layout=["m1", ["T1"], "m1"],
        ...     d_model=[16, 16],
        ...     d_intermediate=[0, 0],
        ...     attn_cfg=AttnSpec(num_heads=(2, 2), rotary_emb_dim=(4, 4),
        ...                       window_size=(-1, -1)),
        ... )
        >>> stage = HNetStage(cfg, stage_idx=0, max_chunks=(4,), headdim=8)
        >>> hidden, records = stage(keras.random.normal((2, 12, 16)))
        >>> hidden.shape, len(records)
        ((2, 12, 16), 1)
    """

    def __init__(
            self,
            arch_config: HNetArchConfig,
            stage_idx: int = 0,
            max_chunks: Sequence[int] = (),
            max_seq_len: int = 2048,
            headdim: int = 64,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(arch_config, HNetArchConfig):
            raise TypeError(
                f"arch_config must be an HNetArchConfig, got "
                f"{type(arch_config).__name__}"
            )
        for name, value in (("stage_idx", stage_idx), ("headdim", headdim),
                            ("max_seq_len", max_seq_len)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}: {value!r}"
                )
        if not 0 <= stage_idx < arch_config.num_stages:
            raise ValueError(
                f"stage_idx must lie in [0, {arch_config.num_stages}), got {stage_idx}"
            )

        max_chunks = tuple(int(value) for value in max_chunks)
        expected = arch_config.num_stages - 1
        if len(max_chunks) != expected:
            raise ValueError(
                f"max_chunks needs one entry per non-innermost stage: expected "
                f"{expected} for a {arch_config.num_stages}-stage layout, got "
                f"{len(max_chunks)} ({max_chunks})"
            )

        spec: Union[StageSpec, IsotropicSpec] = arch_config.stage_spec
        for _ in range(stage_idx):
            spec = spec.main
        assert isinstance(spec, StageSpec)  # stage_count() bounds the descent

        self.arch_config = arch_config
        self.stage_idx = stage_idx
        self.max_chunks = max_chunks
        self.max_seq_len = max_seq_len
        self.headdim = headdim

        self.spec = spec
        self.is_innermost = spec.is_innermost
        self.d_model = arch_config.d_model[stage_idx]

        # The width this stage is HANDED by its parent, and the width it must hand back.
        self.parent_d_model = (
            self.d_model if stage_idx == 0 else arch_config.d_model[stage_idx - 1]
        )
        if self.d_model < self.parent_d_model:
            raise ValueError(
                f"stage {stage_idx} is NARROWER than its parent "
                f"({self.d_model} < {self.parent_d_model}); H-Net's pad_dimension only "
                f"widens a sequence on the way in, so a narrowing hierarchy has no "
                f"defined behaviour. d_model={list(arch_config.d_model)}"
            )
        self.pad_width = self.d_model - self.parent_d_model

        attn = get_stage_cfg(arch_config.attn_cfg, stage_idx)
        ssm = get_stage_cfg(arch_config.ssm_cfg, stage_idx)
        self._stack_kwargs: Dict[str, Any] = {
            "d_model": self.d_model,
            # Per-stage, list-indexed exactly like `attn`/`ssm` above and exactly like
            # the reference's `config.d_intermediate[self.stage_idx]`
            # (`hnet/modules/isotropic.py:81`). See D-031 and `build_mlp`.
            "d_intermediate": arch_config.d_intermediate[stage_idx],
            "num_heads": attn["num_heads"],
            "rotary_emb_dim": attn["rotary_emb_dim"],
            "window_size": attn["window_size"],
            "max_seq_len": max_seq_len,
            "d_state": ssm["d_state"],
            "d_conv": ssm["d_conv"],
            "expand": ssm["expand"],
            "headdim": headdim,
        }

        # `pad_dimension` is created in `build`, never here: it is an `add_weight` and a
        # Layer may not own weights before it is built.
        self.pad_dimension = None

        if self.is_innermost:
            self.encoder = None
            self.decoder = None
            self.routing_module = None
            self.chunk_layer = None
            self.dechunk_layer = None
            self.residual_proj = None
            self.main_network: Any = HNetIsotropic(
                layout=spec.main.layout, name="main_network", **self._stack_kwargs
            )
            return

        self.encoder = HNetIsotropic(
            layout=spec.encoder.layout, name="encoder", **self._stack_kwargs
        )
        self.decoder = HNetIsotropic(
            layout=spec.decoder.layout, name="decoder", **self._stack_kwargs
        )
        self.routing_module = RoutingModule(d_model=self.d_model, name="routing_module")
        self.chunk_layer = ChunkLayer(
            max_chunks=max_chunks[stage_idx], name="chunk_layer"
        )
        self.dechunk_layer = DeChunkLayer(name="dechunk_layer")

        # DECISION plan-2026-09-09T042752-6d66ac56/D-019: this projection is exactly
        # ZERO at initialisation -- kernel AND bias -- and it is pinned in float32
        # independently of the surrounding dtype policy. Do NOT give it a "sensible"
        # non-zero kernel initializer and do NOT drop `dtype="float32"`:
        #   * the zero kernel is what makes a freshly built stage a pure pass-through of
        #     the dechunked inner result, so the residual branch is LEARNED rather than
        #     imposed; the reference flags this weight `_no_reinit` precisely so its
        #     depth-scaled init pass cannot overwrite it (`hnet.py:106-107`);
        #   * `dtype="float32"` is the reference's "do the residual in fp32" comment
        #     (`hnet.py:102-105`) -- under `mixed_float16` the residual accumulator is
        #     the one place where the gate's small corrections would otherwise be lost.
        # This DIVERGES from the reference in one respect, deliberately: `nn.Linear`
        # leaves its bias uniformly random in +/-1/sqrt(D), so the reference's residual
        # branch is a small random constant at step 0 rather than zero. Keras' default
        # `bias_initializer="zeros"` is kept, which is what the zero-init residual idiom
        # actually intends and what makes "starts at exactly zero" a checkable claim.
        # A zero-init weight is INDISTINGUISHABLE from a dead one at init, so liveness
        # is asserted after one real optimizer step, never at init:
        # test_stage.py::TestResidualProjection. Rationale: decisions.md D-019.
        self.residual_proj = keras.layers.Dense(
            self.d_model,
            kernel_initializer="zeros",
            dtype="float32",
            name="residual_proj",
        )

        self.main_network = HNetStage(
            arch_config=arch_config,
            stage_idx=stage_idx + 1,
            max_chunks=max_chunks,
            max_seq_len=max_seq_len,
            headdim=headdim,
            name="main_network",
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``pad_dimension`` and materialise EXACTLY the tree ``call`` runs.

        The sub-layer tree is materialised by tracing :meth:`call` on symbolic
        placeholders rather than by a hand-written chain of ``sublayer.build`` calls.
        A hand-walk is a second encoding of the forward topology and drifts from it
        silently; the failure mode is a sub-layer that is marked built, saved and
        reloaded while holding no weights, which reproduces identical shapes and
        identical parameter totals and is visible only as a value diff.

        :param input_shape: ``(batch, seq_len, d_model_of_the_parent)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: if the input is not rank 3, or its last axis is not the
            width this stage's parent hands it.
        """
        if self.built:
            return
        if len(input_shape) != 3:
            raise ValueError(
                f"HNetStage expects rank-3 (batch, seq_len, d_model) input, got shape "
                f"{input_shape}"
            )
        if input_shape[-1] is not None and int(input_shape[-1]) != self.parent_d_model:
            raise ValueError(
                f"stage {self.stage_idx} expects an input width of "
                f"{self.parent_d_model} (its parent's d_model), got {input_shape[-1]} "
                f"from shape {input_shape}"
            )

        if self.pad_width > 0:
            # `hnet.py:111-118`. Zeros, and produced by the INITIALIZER: an
            # `add_weight(...)` followed by `.assign()` inside `build()` is recorded and
            # discarded by the `StatelessScope` Keras 3 runs a parent-triggered build
            # in, leaving the weight at its initializer value in every real model while
            # every direct-`build` unit test still passes.
            self.pad_dimension = self.add_weight(
                name="pad_dimension",
                shape=(self.pad_width,),
                initializer="zeros",
                trainable=True,
            )

        materialize_sublayers(self, input_shape, batch_size=1)
        super().build(input_shape)

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

    def _widen(self, hidden: keras.KerasTensor) -> keras.KerasTensor:
        """Concatenate the learned ``pad_dimension`` vector onto the last axis.

        :param hidden: ``(B, L, D_parent)``.
        :type hidden: keras.KerasTensor
        :returns: ``(B, L, D)`` when this stage is wider than its parent, else ``hidden``
            unchanged.
        :rtype: keras.KerasTensor
        """
        if self.pad_dimension is None:
            return hidden
        # `hnet.py:227-230` expands the parameter over the leading axes. Adding it to a
        # zero column broadcasts it over a DYNAMIC batch and length without ever reading
        # `keras.ops.shape`, so the whole path traces at `(None, None, D)`.
        pad = keras.ops.zeros_like(hidden[..., :1]) + keras.ops.cast(
            self.pad_dimension, hidden.dtype
        )
        return keras.ops.concatenate([hidden, pad], axis=-1)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            padding_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, List[Dict[str, keras.KerasTensor]]]:
        """Run this stage and every stage nested inside it.

        :param hidden_states: ``(B, L, D_parent)`` hidden states at this stage's
            resolution.
        :type hidden_states: keras.KerasTensor
        :param padding_mask: Optional ``(B, L)`` validity mask, truthy for a real token.
            Padded positions are excluded from boundary selection, from chunking, from
            the dechunk partition and from attention. Spelled ``padding_mask`` and not
            ``mask`` because Keras reserves ``mask`` on ``call`` and auto-populates it
            from an upstream layer's ``_keras_mask`` -- matching
            :class:`~dl_techniques.models.language.hnet.components.HNetIsotropic`.
        :type padding_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :returns: ``(hidden (B, L, D_parent), routing_records)``. ``routing_records`` is
            one dict per chunking level from this stage inward, outermost FIRST, each
            carrying exactly :data:`ROUTING_RECORD_KEYS`. It is empty at the innermost
            stage.
        :rtype: Tuple[keras.KerasTensor, List[Dict[str, keras.KerasTensor]]]
        """
        hidden = self._widen(hidden_states)

        if self.is_innermost:
            hidden = self.main_network(
                hidden, padding_mask=padding_mask, training=training
            )
            return hidden[..., : self.parent_d_model], []

        # `hnet.py:244-251`
        hidden = self.encoder(hidden, padding_mask=padding_mask, training=training)

        # `hnet.py:253-256` -- the residual is read from the ENCODER output, before any
        # chunking, and it is accumulated in the projection's own (float32) dtype.
        # No `training=`: `keras.layers.Dense.call` has no training argument.
        residual = self.residual_proj(hidden)

        # `hnet.py:258-263`
        boundary_prob, boundary_mask, selected_probs = self.routing_module(
            hidden, mask=padding_mask, training=training
        )

        # `hnet.py:264-266`
        inner_hidden, inner_mask = self.chunk_layer(
            hidden,
            boundary_mask=boundary_mask,
            mask=padding_mask,
            training=training,
        )

        # `hnet.py:268-275` -- the recursion. The inner stage's own validity mask is the
        # chunk layer's `inner_mask`: columns past a row's real chunk count hold
        # non-boundary hidden states and must not be attended or routed on.
        inner_hidden, inner_records = self.main_network(
            inner_hidden, padding_mask=inner_mask, training=training
        )

        # `hnet.py:277-284`
        dechunked = self.dechunk_layer(
            inner_hidden,
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            mask=padding_mask,
            training=training,
        )

        # `hnet.py:286-288` -- `out * ste(p) + residual`, evaluated in the residual's
        # dtype and cast back. The gate multiplies the DECHUNKED result only; adding the
        # residual first and gating the sum is a different model (the residual would be
        # scaled by the gate's gradient path as well).
        gated = keras.ops.cast(dechunked, residual.dtype) * straight_through_ones(
            keras.ops.cast(selected_probs, residual.dtype)
        )
        hidden = keras.ops.cast(gated + residual, dechunked.dtype)

        # `hnet.py:290-297`
        hidden = self.decoder(hidden, padding_mask=padding_mask, training=training)

        # `hnet.py:299` -- undo `pad_dimension`. Without this a stage hands its parent
        # its OWN width.
        hidden = hidden[..., : self.parent_d_model]

        if padding_mask is None:
            valid = keras.ops.cast(keras.ops.ones_like(hidden[..., 0]), "bool")
        else:
            valid = keras.ops.cast(padding_mask, "bool")

        record = {
            "boundary_prob": boundary_prob,
            "boundary_mask": boundary_mask,
            "selected_probs": selected_probs,
            "padding_mask": valid,
        }
        # `hnet.py:300` -- this stage's record FIRST, then everything below it.
        return hidden, [record, *inner_records]

    # -----------------------------------------------------------------
    # shapes and serialization
    # -----------------------------------------------------------------

    # DECISION plan-2026-09-09T042752-6d66ac56/D-020: `compute_output_shape` is
    # deliberately NOT implemented. Do NOT add one "for completeness". Keras 3 branches
    # on `utils.is_default(self.compute_output_shape)` (`layers/layer.py:1059-1061`):
    # while it is absent, `compute_output_spec` falls back to tracing `call` on the meta
    # backend, which propagates the real DTYPES of this layer's mixed output structure
    # -- `boundary_mask` and `padding_mask` are BOOL. The moment a shape method exists,
    # Keras builds the output specs from shapes alone and types every one of them
    # float32, so a symbolically-built model hands `ChunkLayer`/`DeChunkLayer` a float
    # tensor where they expect a boolean one. Guarded by
    # test_stage.py::TestSymbolicBuild::test_the_routing_records_keep_their_dtypes.
    # Rationale: decisions.md D-020.

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, JSON-safe.

        The architecture is serialized through
        :meth:`~dl_techniques.models.language.hnet.config.HNetArchConfig.to_dict`, which
        emits only lists, ints, bools and nested dicts, so a nested stage survives the
        JSON round trip a ``.keras`` archive performs.

        :returns: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "arch_config": self.arch_config.to_dict(),
            "stage_idx": self.stage_idx,
            "max_chunks": list(self.max_chunks),
            "max_seq_len": self.max_seq_len,
            "headdim": self.headdim,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HNetStage":
        """Rebuild from :meth:`get_config`, reconstructing the architecture dataclass.

        :param config: The dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :returns: The rebuilt stage, nested stages included.
        :rtype: HNetStage
        """
        config = dict(config)
        arch_config = config.pop("arch_config")
        if not isinstance(arch_config, HNetArchConfig):
            arch_config = HNetArchConfig.from_dict(arch_config)
        return cls(arch_config=arch_config, **config)
