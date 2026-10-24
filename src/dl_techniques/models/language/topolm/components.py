"""The decoder block that carries TopoLM's spatial taps.

Why this block exists
---------------------
Every architectural piece a TopoLM block needs already exists in
``dl_techniques.layers.transformers`` -- and none of them can be used here.
:class:`~dl_techniques.layers.transformers.transformer.TransformerLayer` computes
the attention branch and the feed-forward branch, and then adds each straight
into the residual stream; in the pre-norm path the branch tensor is a local
variable overwritten by stochastic depth and layer scale before the residual add,
so it is gone by the time ``call`` returns. The spatial smoothness loss has to be
computed on that tensor, at exactly that point, before normalization and before
the add. There is no flag, no hook and no ``activity_regularizer`` that reaches it.

So the block is written here, and it composes the shared factories
(``create_attention_layer``, ``create_normalization_layer`` and the feed-forward
path ``TransformerLayer`` uses: ``build_transformer_ffn_config`` into
``create_ffn_from_config``) rather than reimplementing any of them. That is the
same shape ``hnet``, ``wave_field`` and ``tree_transformer`` use for their own
blocks, and the reuse order is unchanged: the factories first, a bespoke layer
last.

The feed-forward is ONE layer, ``ffn``: for the default ``ffn_type='mlp'`` it is
``Linear(H -> I)``, the activation, ``Linear(I -> H)``, exactly GPT-2's MLP. Its
type, its arguments, its bias, the normalization type of both norms and their
arguments are constructor options, serialized by :meth:`TopoLMBlock.get_config`.

.. code-block:: text

    x
    │
    ├── attention_norm ──► attention ──► attn_tap ──┐   tap: identity forward,
    │                                               │   alpha * SL when training
    └───────────────────────────────────────────────┤
                                                    ▼
                                                  x + attn
                                                    │
                            ┌───────────────────────┘
                            ▼
    x' ──► ffn_norm ──► ffn ──► mlp_tap ──┐
                                          ▼
                                    x' + mlp

The taps sit ON the branch outputs, not around them: their forward pass is the
identity, so the residual arithmetic is that of a plain GPT-2 block and any
difference in the loss curve is attributable to the added term alone. The branch
dropout (GPT-2's ``resid_pdrop``) sits between the feed-forward and its tap and is
the only dropout in the feed-forward path.

The block carries ``grid_dim`` (2, the paper's layout, or 3) and forwards it to
every tap it makes; a tap resolves its grid lazily at its own build, so a dropped
argument would leave a silently 2D tap in a 3D model.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.activation_serialization import (
    deserialize_activation,
    serialize_activation,
)
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.ffn import create_ffn_from_config
from dl_techniques.layers.ffn.factory import FFN_REGISTRY
from dl_techniques.layers.norms import (
    create_normalization_layer,
    get_normalization_info,
)
from dl_techniques.layers.transformers.transformer import (
    build_transformer_ffn_config,
)
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Which branch outputs a block can tap. The paper taps both.
TAP_SITES = ("attention", "mlp")


def accepted_ffn_types() -> List[str]:
    """The feed-forward types a residual branch can use, sorted.

    A type qualifies when the factory registry says its output width is set by
    ``output_dim``: the branch is added to the residual stream, so it has to emit
    ``hidden_size`` features, and ``hidden_size`` is handed to the factory as
    ``output_dim``. Types that size their output by ``filters``, ``features`` or
    ``units``, or keep the input width (``gated_dconv``, ``mixer``), are refused
    by name rather than built into a width mismatch.

    :return: Registry keys, e.g. ``['bilinear', 'geglu', ..., 'mlp', ..., 'swiglu']``.
    :rtype: List[str]
    """
    return sorted(
        name for name, info in FFN_REGISTRY.items()
        if info.get("output_dim_param") == "output_dim"
    )


def accepted_normalization_types() -> List[str]:
    """The normalization types of the shared factory, sorted."""
    return sorted(get_normalization_info())


def _args_dict(name: str, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Copy an optional args mapping into a dict, refusing anything else by name."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a dict of keyword arguments or None, got "
            f"{type(value).__name__}: {value!r}"
        )
    return dict(value)


@register_dl_technique("dl_techniques.models.topolm.components")
class TopoLMBlock(keras.layers.Layer):
    """A pre-norm decoder block whose two branch outputs carry spatial taps.

    Two things distinguish it from a stock block, and both are load-bearing:

    * ``attn_out`` and ``mlp_out`` are held as named locals and passed through
      :class:`~dl_techniques.layers.regularization.spatial_smoothness.SpatialSmoothness`
      before the residual add, which is where the paper computes the loss.
    * ``alpha = 0`` produces a block whose weight set, tap count and layout seeds
      are IDENTICAL to an ``alpha > 0`` block. The non-topographic control is only
      a control if everything except the loss term is held fixed, and "held fixed"
      has to mean the weights too -- otherwise the two arms differ in RNG draws as
      well as in objective and no comparison between them means anything.

    :param hidden_size: Model width, and the tapped axis' length.
    :type hidden_size: int
    :param num_heads: Attention heads. Must divide ``hidden_size``.
    :type num_heads: int
    :param intermediate_size: Feed-forward hidden width. For ``ffn_type='mlp'`` it
        is the width between the two projections (GPT-2's ``n_inner``). A type
        that sizes itself, ``swiglu`` (a 2/3 rule from ``ffn_expansion_factor``,
        rounded to ``ffn_multiple_of``), does not read it: the shared helper
        withholds it, as it does for ``TransformerLayer``; resize through
        ``ffn_args``.
    :type intermediate_size: int
    :param alpha: Weight on each tap's spatial loss. ``0`` gives the control.
    :type alpha: float
    :param radius: Neighbourhood radius for each tap.
    :type radius: int
    :param num_neighborhoods: Neighbourhoods sampled per tap per step.
    :type num_neighborhoods: int
    :param distance: Distance metric behind each tap's prior.
    :type distance: str
    :param permute: Whether each tap draws its own unit permutation. ``False`` is
        the paper's Fig. 12 ablation.
    :type permute: bool
    :param grid_shape: Explicit grid for the taps, or ``None`` to factor
        ``hidden_size``: ``(height, width)`` for ``grid_dim=2``, three axes in
        the order given (never reordered) for ``grid_dim=3``.
    :type grid_shape: Optional[Tuple[int, ...]]
    :param grid_dim: Number of grid axes of every tap, 2 (the paper's layout,
        the default) or 3. Forwarded to each
        :class:`~dl_techniques.layers.regularization.spatial_smoothness.SpatialSmoothness`
        and written by :meth:`get_config`; a config without the key is a 2D block.
    :type grid_dim: int
    :param tap_sites: Which branch outputs to tap, a subset of
        :data:`TAP_SITES`. ``()`` builds the block with no taps at all, which is
        how a plain GPT-2 block is recovered from the same code.
    :type tap_sites: Tuple[str, ...]
    :param normalization_position: ``'pre'`` (the paper's) or ``'post'``.
    :type normalization_position: str
    :param activation: Feed-forward activation, applied between the two
        projections of an MLP. Types whose activation is their identity
        (``reglu``, ``bilinear``, ``squared_relu``, ``swiglu``) do not read it.
    :type activation: Union[str, Any]
    :param dropout_rate: Dropout on the feed-forward branch output, applied once,
        after the output projection and before the tap and the residual add:
        GPT-2's ``resid_pdrop``. The feed-forward layer itself is built with no
        dropout, so the rate never stacks with a second one inside it.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout on the attention output.
    :type attention_dropout_rate: float
    :param initializer_range: Stddev for the block's ``TruncatedNormal`` inits.
    :type initializer_range: float
    :param layer_norm_eps: Normalization epsilon, passed to BOTH norms. The
        factory default is 1e-6; GPT-2 uses 1e-5, and a stack whose two norms run
        at different epsilons is a silent 1000x spread inside one forward pass.
    :type layer_norm_eps: float
    :param seed: Base seed for this block's taps.
    :type seed: Optional[int]
    :param ffn_type: The feed-forward type, a key of the shared FFN registry whose
        output width is ``output_dim`` (:func:`accepted_ffn_types`): ``'mlp'``
        (the default, GPT-2's), ``'swiglu'``, ``'geglu'`` and the rest.
    :type ffn_type: str
    :param ffn_args: Extra arguments for the feed-forward factory, merged last and
        never filtered, so a misspelled key raises. ``dropout_rate`` is refused:
        the block's own ``dropout_rate`` is the only dropout.
    :type ffn_args: Optional[Dict[str, Any]]
    :param use_bias: Whether the feed-forward projections carry a bias. The
        attention projections never do. ``swiglu`` is bias-free by its own
        default and the shared helper does not forward this flag to it
        (``ffn_args={'use_bias': True}`` does).
    :type use_bias: bool
    :param normalization_type: Type of BOTH norms of the block (and, in
        :class:`~dl_techniques.models.topolm.model.TopoLM`, of the final norm), a
        key of the shared normalization factory: ``'layer_norm'`` (default),
        ``'rms_norm'`` (no ``beta``) and the rest.
    :type normalization_type: str
    :param attention_norm_args: Extra arguments for the attention-side norm,
        e.g. ``{'epsilon': 1e-6}``, which overrides ``layer_norm_eps`` for that
        norm alone.
    :type attention_norm_args: Optional[Dict[str, Any]]
    :param ffn_norm_args: The same for the feed-forward-side norm.
    :type ffn_norm_args: Optional[Dict[str, Any]]
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :raises ValueError: If ``hidden_size`` does not divide among ``num_heads``,
        if a name in ``tap_sites`` is not in :data:`TAP_SITES`, or if
        ``normalization_position`` is not ``'pre'`` or ``'post'``, or if
        ``grid_dim`` is not 2 or 3, or if ``ffn_type`` or ``normalization_type`` is
        not accepted, an args argument is not a dict, ``ffn_args`` sets
        ``dropout_rate``, or a factory refuses its arguments -- naming the
        offending value in each case.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        alpha: float = 2.5,
        radius: int = 5,
        num_neighborhoods: int = 5,
        distance: str = "linf",
        permute: bool = True,
        grid_shape: Optional[Tuple[int, ...]] = None,
        grid_dim: int = 2,
        tap_sites: Tuple[str, ...] = TAP_SITES,
        normalization_position: str = "pre",
        activation: Union[str, Any] = "gelu",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        seed: Optional[int] = None,
        ffn_type: str = "mlp",
        ffn_args: Optional[Dict[str, Any]] = None,
        use_bias: bool = True,
        normalization_type: str = "layer_norm",
        attention_norm_args: Optional[Dict[str, Any]] = None,
        ffn_norm_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if normalization_position not in ("pre", "post"):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got "
                f"{normalization_position!r}"
            )
        # The layer's own validator, so the block, the model and the tap accept
        # the same values. Imported here like the other spatial_smoothness
        # imports in this file.
        from dl_techniques.layers.regularization.spatial_smoothness import (
            validate_grid_dim,
        )

        grid_dim = validate_grid_dim(grid_dim)
        ffn_args = _args_dict("ffn_args", ffn_args)
        attention_norm_args = _args_dict("attention_norm_args", attention_norm_args)
        ffn_norm_args = _args_dict("ffn_norm_args", ffn_norm_args)
        if ffn_type not in accepted_ffn_types():
            raise ValueError(
                f"ffn_type must be one of {accepted_ffn_types()}, got {ffn_type!r}"
            )
        if "dropout_rate" in ffn_args:
            raise ValueError(
                "ffn_args must not set dropout_rate: the block's own dropout_rate "
                "is applied once, to the branch output, and a second rate inside "
                "the feed-forward would stack with it"
            )
        if normalization_type not in accepted_normalization_types():
            raise ValueError(
                f"normalization_type must be one of "
                f"{accepted_normalization_types()}, got {normalization_type!r}"
            )
        tap_sites = tuple(tap_sites)
        unknown = sorted(set(tap_sites) - set(TAP_SITES))
        if unknown:
            raise ValueError(
                f"tap_sites contains unknown site(s) {unknown}; "
                f"accepted: {list(TAP_SITES)}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.alpha = float(alpha)
        self.radius = int(radius)
        self.num_neighborhoods = int(num_neighborhoods)
        self.distance = distance
        self.permute = bool(permute)
        self.grid_shape = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.grid_dim = grid_dim
        self.tap_sites = tap_sites
        self.normalization_position = normalization_position
        self.activation = deserialize_activation(activation)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.seed = seed
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args
        self.use_bias = bool(use_bias)
        self.normalization_type = normalization_type
        self.attention_norm_args = attention_norm_args
        self.ffn_norm_args = ffn_norm_args

        # One initializer INSTANCE per block is deliberate: Keras hands the same
        # instance to identically-shaped sublayers inside a layer, so sharing it
        # would give bit-identical draws to the two residual projections and the
        # feed-forward pair would start as a rank-deficient map.
        kernel_initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range
        )
        # The two residual-path projections are scaled by depth, matching GPT-2's
        # _init_weights, which divides by sqrt(2 * n_layer) because each block
        # performs two residual additions.
        residual_initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range / math.sqrt(2.0)
        )

        self.attention_norm = self._make_norm(
            "attention_norm", self.attention_norm_args
        )
        self.attention = create_attention_layer(
            "multi_head",
            name="attention",
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            output_kernel_initializer=residual_initializer,
            use_bias=False,
        )
        self.ffn_norm = self._make_norm("ffn_norm", self.ffn_norm_args)

        # DECISION plan-2026-10-23T190557-972a9ac5/D-030: the feed-forward is ONE
        # layer, built through the helper TransformerLayer uses. Do NOT chain two
        # create_ffn_layer("mlp") calls: each is already Linear -> activation ->
        # Linear, so two made four Dense layers and two activations, 3.12x GPT-2's
        # parameters, and the layers were named as if each were one Dense. Do NOT
        # build the layer outside build_transformer_ffn_config (its per-type
        # policy and the registry intersection are shared). Do NOT pass
        # dropout_rate into it: an MLP applies dropout after the activation, and
        # the branch dropout below is GPT-2's resid_pdrop, so both would stack.
        # Guarded by test_topolm_ffn.py::TestTheFeedForwardIsOneMLP,
        # ::TestTheForwardIsDenseActivationDense and
        # ::TestTheOraclesRejectTheOldStructure. See decisions.md.
        ffn_config = build_transformer_ffn_config(
            ffn_type=self.ffn_type,
            name="ffn",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=0.0,
            kernel_initializer=clone_initializer(kernel_initializer),
            bias_initializer="zeros",
            use_bias=self.use_bias,
            output_kernel_initializer=clone_initializer(residual_initializer),
            ffn_args=self.ffn_args,
        )
        try:
            self.ffn = create_ffn_from_config(ffn_config)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Failed to create the {self.ffn_type!r} feed-forward layer. "
                f"ffn_args: {sorted(self.ffn_args)}. Original error: {error}"
            ) from error
        self.ffn_dropout = keras.layers.Dropout(
            self.dropout_rate, name="ffn_dropout"
        )

        # Taps are created UNCONDITIONALLY and gated in `call`. Conditional
        # creation would shift every auto-generated layer name after it and make
        # the alpha=0 control differ from the alpha>0 arm by more than its loss.
        self.attention_tap = self._make_tap(
            "attention_tap", seed=self._tap_seed(0)
        )
        self.mlp_tap = self._make_tap("mlp_tap", seed=self._tap_seed(1))

        logger.info(
            f"TopoLMBlock: hidden={self.hidden_size}, heads={self.num_heads}, "
            f"ffn={self.intermediate_size}, taps={list(self.tap_sites)}, "
            f"alpha={self.alpha}, radius={self.radius}, "
            f"normalization_position={self.normalization_position!r}"
        )

    def _make_norm(self, name: str, args: Mapping[str, Any]) -> keras.layers.Layer:
        """One of the block's two norms, through the shared factory.

        ``layer_norm_eps`` is the default epsilon of both; a norm's own args may
        override it (``{'epsilon': ...}``) without a duplicate-keyword error.
        """
        try:
            return create_normalization_layer(
                self.normalization_type,
                name=name,
                **{"epsilon": self.layer_norm_eps, **args},
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Failed to create the {self.normalization_type!r} normalization "
                f"layer {name!r}. Args: {sorted(args)}. Original error: {error}"
            ) from error

    def _tap_seed(self, index: int) -> int:
        """The layout seed of one of this block's taps.

        Derived from ``(base seed, depth-independent tap index)`` through the
        layer's own :func:`~dl_techniques.layers.regularization.spatial_smoothness.permutation_seed`,
        so two taps never share a layout and the derivation is reproducible from
        configuration alone.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            permutation_seed,
        )

        return permutation_seed(self.seed, index)

    def _make_tap(self, name: str, seed: int):
        """Construct one tap layer with this block's spatial configuration."""
        from dl_techniques.layers.regularization.spatial_smoothness import (
            SpatialSmoothness,
        )

        return SpatialSmoothness(
            alpha=self.alpha,
            radius=self.radius,
            num_neighborhoods=self.num_neighborhoods,
            distance=self.distance,
            permute=self.permute,
            grid_shape=self.grid_shape,
            # Forwarded even though 2 is the default: a tap resolves its grid
            # lazily at its own build, so a dropped argument builds a silently 2D
            # layer inside a block that reports grid_dim=3.
            grid_dim=self.grid_dim,
            seed=seed,
            name=name,
        )

    @property
    def taps(self) -> Tuple[Any, ...]:
        """The tap layers, in site order; empty when ``tap_sites`` is empty."""
        selected = []
        if "attention" in self.tap_sites:
            selected.append(self.attention_tap)
        if "mlp" in self.tap_sites:
            selected.append(self.mlp_tap)
        return tuple(selected)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer that ``call`` runs, and only those.

        Created-unconditionally / built-conditionally is the asymmetry this block
        depends on: an ``alpha=0`` or ``tap_sites=()`` block keeps a stable object
        graph and stable names, while building nothing it never calls.

        :param input_shape: Block input shape ``(batch, seq_len, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis is not the configured
            ``hidden_size`` -- a cross-parameter contract ``call`` relies on, and
            one that a mismatch would otherwise turn into a broadcast error
            deep inside the attention layer.
        """
        if input_shape is not None and input_shape[-1] is not None:
            if int(input_shape[-1]) != self.hidden_size:
                raise ValueError(
                    f"TopoLMBlock was built for hidden_size={self.hidden_size} "
                    f"but received an input with last axis {int(input_shape[-1])}"
                )

        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)

        self.ffn_norm.build(input_shape)
        self.ffn.build(input_shape)
        self.ffn_dropout.build(input_shape)

        if "attention" in self.tap_sites:
            self.attention_tap.build(input_shape)
        if "mlp" in self.tap_sites:
            self.mlp_tap.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        taps: Optional[Tuple[Any, ...]] = None,
    ) -> keras.KerasTensor:
        """Run the block, tapping each branch output before its residual add.

        :param inputs: ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Rank-3 attend-semantics mask, as produced by
            :func:`~dl_techniques.utils.masking.factory.create_causal_attend_mask`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training flag, forwarded explicitly to every sub-layer.
            Keras propagates ``training`` through one mutable slot that a sibling
            overwrites, so omitting it is a latent bug even where it is currently
            a no-op.
        :type training: Optional[bool]
        :param taps: Optional pair of identity layers receiving
            ``(attn_out, mlp_out)``, for post-hoc activation extraction. This is a
            measurement hook, not a mechanism: the captures return their inputs
            unchanged, so the residual arithmetic is identical whether or not they
            are supplied. Order is the ``tap_sites`` order.
        :type taps: Optional[Tuple[Any, ...]]
        :return: ``(batch, seq_len, hidden_size)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``taps`` has one entry per tapped site but the
            wrong count -- naming what was expected.
        """
        if taps is not None:
            expected = len(self.tap_sites)
            if len(taps) != expected:
                raise ValueError(
                    f"TopoLMBlock was given {len(taps)} capture layer(s) for "
                    f"{expected} tapped site(s) {list(self.tap_sites)}"
                )

        residual = inputs

        if self.normalization_position == "pre":
            normed = self.attention_norm(inputs, training=training)
            attn_out = self.attention(
                normed, attention_mask=attention_mask, training=training
            )
            if "attention" in self.tap_sites:
                attn_out = self.attention_tap(attn_out, training=training)
                if taps is not None and "attention" in self.tap_sites:
                    position = self.tap_sites.index("attention")
                    attn_out = taps[position](attn_out, training=training)
            hidden = attn_out + residual

            normed = self.ffn_norm(hidden, training=training)
            mlp_out = self.ffn(normed, training=training)
            mlp_out = self.ffn_dropout(mlp_out, training=training)
            if "mlp" in self.tap_sites:
                mlp_out = self.mlp_tap(mlp_out, training=training)
                if taps is not None and "mlp" in self.tap_sites:
                    position = self.tap_sites.index("mlp")
                    mlp_out = taps[position](mlp_out, training=training)
            return mlp_out + hidden

        attn_out = self.attention(
            inputs, attention_mask=attention_mask, training=training
        )
        if "attention" in self.tap_sites:
            attn_out = self.attention_tap(attn_out, training=training)
            if taps is not None and "attention" in self.tap_sites:
                position = self.tap_sites.index("attention")
                attn_out = taps[position](attn_out, training=training)
        hidden = self.attention_norm(attn_out + residual, training=training)

        mlp_out = self.ffn(hidden, training=training)
        mlp_out = self.ffn_dropout(mlp_out, training=training)
        if "mlp" in self.tap_sites:
            mlp_out = self.mlp_tap(mlp_out, training=training)
            if taps is not None and "mlp" in self.tap_sites:
                position = self.tap_sites.index("mlp")
                mlp_out = taps[position](mlp_out, training=training)
        return self.ffn_norm(mlp_out + hidden, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape``; the block preserves it."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "alpha": self.alpha,
            "radius": self.radius,
            "num_neighborhoods": self.num_neighborhoods,
            "distance": self.distance,
            "permute": self.permute,
            "grid_shape": self.grid_shape,
            "grid_dim": self.grid_dim,
            "tap_sites": tuple(self.tap_sites),
            "normalization_position": self.normalization_position,
            "activation": serialize_activation(self.activation),
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "seed": self.seed,
            "ffn_type": self.ffn_type,
            "ffn_args": dict(self.ffn_args),
            "use_bias": self.use_bias,
            "normalization_type": self.normalization_type,
            "attention_norm_args": dict(self.attention_norm_args),
            "ffn_norm_args": dict(self.ffn_norm_args),
        })
        return config