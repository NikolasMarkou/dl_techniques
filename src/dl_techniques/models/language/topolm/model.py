"""TopoLM, a decoder-only language model whose units are laid out on a sheet of tissue.

The idea
--------
A GPT-2 block indexes its units by nothing but their position on the last axis of
a weight matrix, so nothing about the representation is spatial: two units that
learn to encode the same thing have no reason to sit near each other, and the
emergent clustering is a property of the initialisation rather than of anything
the objective asked for. TopoLM gives the units of every attention and
feed-forward branch a distinct cell on an ``h x w`` grid (an ``h x w x d`` box
with ``grid_dim=3``, a repo-authored extension; the paper is 2D) and adds a
penalty ``SL = 0.5 * (1 - corr(r, d))`` alongside the next-token cross-entropy,
where ``r`` is the vector of pairwise activation correlations over a sampled
square (3D: cubic) neighbourhood and ``d`` the matching inverse distances. Minimising it makes
nearby units co-activate, which is an efficient proxy for short wiring length,
and functional clusters emerge as a consequence of the objective rather than as
a post-hoc clustering of whatever the model happened to learn.

Two choices that are not the obvious ones
----------------------------------------
**Each tap permutes its units independently.** Without it the residual stream
hands every layer the same spatial pattern, the loss is satisfied by copying one
map forward rather than by each layer learning its own, and the model ends up with
one map instead of ``depth`` of them (the paper's Fig. 12). ``permute=False``
reproduces that ablation.

**The taps sit on the branch outputs, before the residual add.** The paper
computes the loss "prior to normalization and addition into the residual stream",
and :class:`~dl_techniques.models.topolm.components.TopoLMBlock` holds those
tensors as named locals for exactly that reason. A tap's forward pass is the
identity, so the residual arithmetic is unchanged and any difference in the loss
curve is attributable to the added term alone. ``tap_placement='residual'`` taps
the post-add tensor instead, which is what the authors' released training config
does (``with_resid: True``) while their evaluation scripts and the paper text tap
the branch; the default is the paper's.

Defaults that follow the released code
--------------------------------------
No feed-forward bias and no LayerNorm beta (``use_bias=False``,
``norm_center=False``; the attention never has a bias), the exact (erf) GELU
(``activation='gelu'``; ``'gelu_tanh'`` is one argument away), and the two
residual-path output projections initialised at ``initializer_range / sqrt(2 *
depth)`` (the model hands its depth to every block). The geometry of the
neighbourhood is the one place the paper text and the released code differ; see
:mod:`~dl_techniques.models.topolm.config` for both readings and which this
repository defaults to. GPT-2's feed-forward biases and LayerNorm shift are
``use_bias=True, norm_center=True`` away; GPT-2's attention projections also carry
biases, and this block's never do, so the result is not a GPT-2 block.

What is and is not here
-----------------------
The auxiliary term arrives through ``add_loss``, not a custom ``train_step``, so
stock ``fit`` still scales the loss under mixed precision and still reports the
pure task loss at validation time -- the taps add nothing when ``training`` is not
``True``. A training head therefore needs
``CausalLanguageModel(aggregate_backbone_losses=True)``, which is what folds
``backbone.losses`` into the same scalar used for both the gradient and the loss
tracker; without it the smoothness term is computed every step and discarded.

``alpha = 0`` is a first-class configuration, not an absence of one: the taps are
still created, still built, and still derive their layouts from the same seeds, so
the non-topographic control has a byte-identical weight set and differs from the
trained arm in its objective alone.

By default (``include_top=False``) the model returns ``logits`` and
``last_hidden_state``. With ``include_top=True`` and a ``task_config`` it owns ONE
task head from ``layers/heads/nlp``, named ``task_head``, in place of the LM head
(no logits are computed and no untied ``lm_head`` is created), and returns only
that head's output, as ``models/language/gpt2`` does. The head pools ``'last'`` by
default, the only position of a causal decoder that has seen the whole sequence.
The blocks, and so their taps and ``add_loss`` terms, run unchanged under a head,
and the backbone's weight paths are the same either way.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM:
      Brain-like spatio-functional organization in a topographic language model.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Lee, Margalit, Jozwik, Cohen, Kanwisher & DiCarlo, 2020. Topographic deep
      artificial neural networks. (https://arxiv.org/abs/2007.09019)
    - Margalit, Lee, Finzi, DiCarlo, Grill-Spector & Yamins, 2024. A unifying
      framework for functional organization in early and higher ventral visual
      cortex. Neuron 112(14).
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
    - Press & Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    read_checkpoint_config,
)
from dl_techniques.utils.sequence_length_guard import (
    RecordedBuildShapeMixin,
    check_sequence_length,
    refuse_static_sequence_length,
)
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.activation_serialization import (
    deserialize_activation,
    serialize_activation,
)
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    TASK_HEAD_NAME,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    task_config_to_dict,
    validate_task_head_knobs,
)
from .components import TAP_SITES, TopoLMBlock
from .config import (
    DEFAULT_ALPHA,
    DEFAULT_NUM_NEIGHBORHOODS,
    DEFAULT_RADIUS,
    MODEL_VARIANTS,
    axes_fit_problem,
    radius_advice,
    validate_all_variants,
)

# ---------------------------------------------------------------------

#: GPT-2's 50,257 base ids plus the four special ids the trainer's config assigns
#: (50257 to 50260) is 50,261 rows, padded to the next multiple of 64: 50,304, the
#: padded size the authors' released code uses too. It is the number the trainer
#: resolves for its default ``gpt2`` encoding (``train.topolm.common.resolve_vocab_size``),
#: so a model built without ``vocab_size`` and a trained one agree; a model with
#: another tokenizer passes its own. The class default used to be cl100k's 100277:
#: about 50k rows no GPT-2 id ever selects (39M parameters at width 784).
DEFAULT_VOCAB_SIZE = 50304

validate_all_variants(DEFAULT_RADIUS)


@register_dl_technique("dl_techniques.models.topolm.model")
class TopoLM(RecordedBuildShapeMixin, keras.Model):
    """A GPT-2-shaped causal language model with a topographic unit layout.

    Architecture:

        .. code-block:: text

            input_ids [B, S]
                 │
                 ▼
            ┌─────────────────────────┐
            │ token + learned position│
            └─────────────────────────┘
                 │ [B, S, D]
                 ▼
            ┌─────────────────────────┐
            │ TopoLMBlock x depth     │◄── rank-3 causal mask
            │  taps on both branches  │
            └─────────────────────────┘
                 │
                 ▼  last_hidden_state [B, S, D]
            ┌─────────────────────────┐
            │ final norm (per option) │
            └─────────────────────────┘
                 │
                 ├──► {"last_hidden_state"}
                 ▼
            tied matmul(token emb.T)  |  lm_head Dense
                 │
                 ▼  logits [B, S, vocab_size]

            include_top=True: both leaves are replaced by
            ┌─────────────────────────┐
            │ task_head (pools 'last')│──► finalized head output (only output)
            └─────────────────────────┘

    Both leaves come back together, so
    :class:`~dl_techniques.models.common.masked_language_model.CausalLanguageModel`
    and the standard CLM data wrapper work unchanged. No ``None`` is echoed into
    the output dict, so ``predict({"input_ids": ...})`` does not trip Keras'
    nested-structure check.

    Variants (:data:`~dl_techniques.models.topolm.config.MODEL_VARIANTS`):

        .. code-block:: text

            variant  embed_dim  depth  heads  max_seq_len  grid          radius  provenance
            paper    784        12     16     1024         28 x 28       5       Rathi et al. 2025
            small    512        6      8      512          16 x 32       5       repo-authored
            tiny     256        4      4      256          16 x 16       5       repo-authored
            paper3d  1000       12     8      1024         10 x 10 x 10  4       repo-authored, not in the paper
            tiny3d   343        4      7      256          7 x 7 x 7     2       repo-authored, not in the paper

    Only ``paper`` is quoted data; see
    :mod:`~dl_techniques.models.topolm.config` for why the other four are not
    pinned by the upstream-variant guard. The 3D rows carry ``grid_dim=3`` and
    their own radius (the default 5 needs every axis at least 11 across), and
    784 is not a 3D width: it factors to (7, 8, 14), whose shortest axis holds a
    radius of at most 3. Variant validation checks every axis in both ranks; this
    constructor's per-axis check is 3D-only, so a directly built 2D model with an
    explicit short-axis grid constructs as before and fails at tap build.

    :param vocab_size: Vocabulary size. Default 50304: GPT-2's 50,257 ids and the
        trainer's four special ids, padded to a multiple of 64 (the size the
        trainer resolves for its ``gpt2`` encoding and the authors' released code
        uses). Pass the tokenizer's own size for another vocabulary.
    :type vocab_size: int
    :param embed_dim: Model width, and the tapped axis' length. Must be a
        product of ``grid_dim`` factors (two or three), each at least
        ``(2 * radius + 1)`` across.
    :type embed_dim: int
    :param depth: Number of decoder blocks.
    :type depth: int
    :param num_heads: Attention heads. Must divide ``embed_dim``.
    :type num_heads: int
    :param ffn_intermediate_size: Feed-forward hidden width, or ``None`` for
        ``4 * embed_dim``.
    :type ffn_intermediate_size: Optional[int]
    :param max_seq_len: Maximum sequence length; sizes the position table.
    :type max_seq_len: int
    :param alpha: Weight on every tap's spatial loss. ``0`` gives the
        non-topographic control, with an identical weight set.
    :type alpha: float
    :param radius: Neighbourhood radius for every tap.
    :type radius: int
    :param num_neighborhoods: Neighbourhoods sampled per tap per step.
    :type num_neighborhoods: int
    :param distance: Distance metric behind each tap's prior.
    :type distance: str
    :param permute: Whether each tap draws its own unit permutation.
    :type permute: bool
    :param grid_shape: Explicit grid for every tap, or ``None`` to factor
        ``embed_dim``: ``(height, width)`` for ``grid_dim=2``, three axes for
        ``grid_dim=3`` (never reordered). Its length must equal ``grid_dim``.
    :type grid_shape: Optional[Tuple[int, ...]]
    :param grid_dim: Number of grid axes, 2 (the paper's layout, the default) or
        3, where each tapped width is laid out as a box and neighbourhoods are
        cubes. The rank is this stated value and is never inferred from
        ``len(grid_shape)``. Forwarded to every block and tap and written by
        ``get_config``; a config without the key loads as 2D. A 3D width needs
        every axis of its grid at least ``2 * radius + 1`` across, and the
        default ``radius=5`` therefore needs a box of at least 11 per axis; the
        error names the largest radius the grid can use.
    :type grid_dim: int
    :param tap_sites: Which branch outputs to tap: ``("attention", "mlp")`` for
        the paper, ``("attention",)`` for a single-tap ablation, ``()`` for none.
    :type tap_sites: Tuple[str, ...]
    :param dropout_rate: Dropout on embeddings and the feed-forward branch.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout on the attention output.
    :type attention_dropout_rate: float
    :param initializer_range: Stddev for ``TruncatedNormal`` weight init.
    :type initializer_range: float
    :param layer_norm_eps: Normalization epsilon. Default 1e-5, GPT-2's value;
        Keras' own default is 1e-3, a 1000x spread in every denominator.
    :type layer_norm_eps: float
    :param activation: Feed-forward activation of every block. Defaults to
        ``'gelu'``, the exact (erf) GELU, which is what the released TopoLM code's
        ``nn.GELU()`` computes; ``'gelu_tanh'`` (or the ``gelu_tanh`` callable) is
        GPT-2's tanh approximation. One activation for the whole feed-forward.
    :type activation: Union[str, Any]
    :param ffn_type: Feed-forward type of every block: ``'mlp'`` (default), GPT-2's
        ``Linear(embed_dim -> ffn_intermediate_size)``, activation,
        ``Linear(ffn_intermediate_size -> embed_dim)``. It is the only accepted type
        (:func:`~dl_techniques.models.topolm.components.accepted_ffn_types`): the
        residual-projection init needs the type to declare
        ``output_kernel_initializer``, and every other registry type (``'swiglu'``,
        ``'geglu'`` and the rest) is refused by name instead of silently starting
        its output projection unscaled.
    :type ffn_type: str
    :param ffn_args: Extra arguments for the feed-forward factory, e.g.
        ``{'bias_initializer': 'ones'}`` for ``mlp``. ``dropout_rate`` is refused.
    :type ffn_args: Optional[Dict[str, Any]]
    :param use_bias: Whether the feed-forward projections carry a bias. Default
        ``False``, the released code's ``bias=False``; attention projections never
        have one.
    :type use_bias: bool
    :param normalization_type: Type of every norm, both in each block and the
        final one: ``'layer_norm'`` (default), ``'rms_norm'``, or another key of
        the shared normalization factory.
    :type normalization_type: str
    :param norm_center: Whether a ``'layer_norm'`` carries a ``beta``, for every
        norm of the model (both in each block and the final one). Default
        ``False``, the released code's ``LayerNorm(bias=False)``; ``'rms_norm'``
        has no ``beta`` whatever this says.
    :type norm_center: bool
    :param tap_placement: ``'branch'`` (default, the paper text: the loss is applied
        "prior to normalization and addition into the residual stream") taps each
        branch output; ``'residual'`` taps the post-add tensor, which the released
        training yaml's ``with_resid: True`` does. The forward values are the same;
        the penalised tensor and the activations ``taps=`` captures are the tapped
        one.
    :type tap_placement: str
    :param attention_norm_args: Extra arguments for every block's attention-side
        norm; ``{'epsilon': ...}`` overrides ``layer_norm_eps`` for it.
    :type attention_norm_args: Optional[Dict[str, Any]]
    :param ffn_norm_args: The same for every block's feed-forward-side norm.
    :type ffn_norm_args: Optional[Dict[str, Any]]
    :param tie_word_embeddings: Reuse the transposed token embedding as the LM
        head. Default ``True``, which builds no ``lm_head``.
    :type tie_word_embeddings: bool
    :param seed: Base seed for the taps. Every tap's layout seed is derived from
        it, so the whole model's topography is reproducible from this one integer.
    :type seed: Optional[int]
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no ``attention_mask``. Read only with ``include_top=True``; the
        blocks never derive a mask from it. Default None, which derives no
        padding (the head sees every position): id 0 is a real token in
        the GPT-2 and ``cl100k_base`` vocabularies, so with right-padded
        batches pass the tokenizer's pad id or an ``attention_mask``. An explicit int is stored and serialized as given.
    :type pad_token_id: Optional[int]
    :param include_top: Whether the model owns a task head (and no LM head).
        Default False: no head layer and no head weight exist. With a head the
        blocks and their taps run unchanged, so ``add_loss`` still fires.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored
        as a plain dict. ``MULTIPLE_CHOICE`` is refused.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (e.g. ``{"pooling_type": "mean"}``, which replaces the
        ``'last'`` default). Only valid with ``include_top=True``. Default None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Forwarded to ``keras.Model``.
    :ivar lm_head: The untied LM projection, or None when the embeddings are tied
        or ``include_top`` is True.
    :vartype lm_head: Optional[keras.layers.Dense]
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.
    :vartype task_head: Optional[dl_techniques.layers.heads.nlp.base.BaseNLPHead]
    :raises ValueError: If the width cannot host the tapped grid, does not divide
        among the heads, or if any argument is out of range -- naming the offending
        value in each case; also if ``include_top=True`` comes without a
        ``task_config``, or a ``task_config`` or ``head_config_overrides`` comes
        with ``include_top=False``.
    """

    MODEL_VARIANTS = MODEL_VARIANTS
    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_seq_len={limit}. TopoLM's position "
        "embeddings are a learned table of {limit} rows, so there is no "
        "embedding for position {limit}. Truncate the input to {limit} "
        "tokens, or build the model with max_seq_len>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = 784,
        depth: int = 12,
        num_heads: int = 16,
        ffn_intermediate_size: Optional[int] = None,
        max_seq_len: int = 1024,
        alpha: float = DEFAULT_ALPHA,
        radius: int = DEFAULT_RADIUS,
        num_neighborhoods: int = DEFAULT_NUM_NEIGHBORHOODS,
        distance: str = "linf",
        permute: bool = True,
        grid_shape: Optional[Tuple[int, ...]] = None,
        grid_dim: int = 2,
        tap_sites: Tuple[str, ...] = TAP_SITES,
        normalization_position: str = "pre",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        activation: Union[str, Any] = "gelu",
        tie_word_embeddings: bool = True,
        seed: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        ffn_type: str = "mlp",
        ffn_args: Optional[Dict[str, Any]] = None,
        use_bias: bool = False,
        normalization_type: str = "layer_norm",
        norm_center: bool = False,
        tap_placement: str = "branch",
        attention_norm_args: Optional[Dict[str, Any]] = None,
        ffn_norm_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size, embed_dim, depth, num_heads, max_seq_len, dropout_rate,
            attention_dropout_rate, radius, alpha, num_neighborhoods, distance,
            tap_sites, grid_shape, normalization_position, grid_dim,
        )
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.ffn_intermediate_size = (
            int(ffn_intermediate_size)
            if ffn_intermediate_size is not None
            else 4 * self.embed_dim
        )
        self.max_seq_len = int(max_seq_len)
        self.alpha = float(alpha)
        self.radius = int(radius)
        self.num_neighborhoods = int(num_neighborhoods)
        self.distance = distance
        self.permute = bool(permute)
        # Stored under a private name because `grid_shape` is a PROPERTY below
        # that reports the RESOLVED grid. Assigning to `self.grid_shape` would hit
        # the property and fail with no setter, and storing the argument in a
        # public attribute named the same thing is the kind of shadowing that
        # makes a config value read back as something else.
        self._grid_shape_arg = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.grid_dim = int(grid_dim)
        self._resolved_grid_shape: Optional[Tuple[int, ...]] = None
        self.tap_sites = tuple(tap_sites)
        self.normalization_position = normalization_position
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.activation = activation
        # The block validates these (type names, dict-ness, the dropout rule) and
        # raises by name; the model stores what it was given, as copies.
        self.ffn_type = ffn_type
        self.ffn_args = dict(ffn_args) if ffn_args is not None else {}
        self.use_bias = bool(use_bias)
        self.normalization_type = normalization_type
        self.norm_center = norm_center
        self.tap_placement = tap_placement
        self.attention_norm_args = (
            dict(attention_norm_args) if attention_norm_args is not None else {}
        )
        self.ffn_norm_args = dict(ffn_norm_args) if ffn_norm_args is not None else {}
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.seed = seed
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.include_top = bool(include_top)
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        self._build_architecture()

        # Created after the backbone and under its own fixed name, so the
        # backbone's layer names, weight paths and RNG draws are the same with
        # or without a head.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-10-11T004932-b83935e1/D-012: pool 'last' by default; do NOT keep the head's
            # 'cls' default -- position 0 of a causal decoder has seen only itself. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.embed_dim,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )

        logger.info(
            f"Created TopoLM: depth={self.depth}, embed_dim={self.embed_dim}, "
            f"heads={self.num_heads}, max_seq_len={self.max_seq_len}, "
            f"alpha={self.alpha}, radius={self.radius}, "
            f"grid_dim={self.grid_dim}, "
            f"taps={len(self.blocks) * len(self.tap_sites)}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
            + (f", task head '{self.task_config['name']}'" if self.include_top else "")
        )

    @staticmethod
    def _validate_config(
        vocab_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        max_seq_len: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        radius: int,
        alpha: float,
        num_neighborhoods: int,
        distance: str,
        tap_sites: Tuple[str, ...],
        grid_shape: Optional[Tuple[int, ...]],
        normalization_position: str,
        grid_dim: int,
    ) -> None:
        """Check every argument and every cross-parameter contract at once.

        The grid floor is the one that bites: a width below ``(2r+1)^grid_dim``
        builds a model that cannot construct a single neighbourhood, and the
        failure surfaces at the tap's ``build`` -- after embeddings, attention and
        a whole stack have been created. It is a construction-time error, so it is
        raised here. For ``grid_dim=3`` the floor is followed by a per-axis check,
        because a total that clears the floor can still resolve to a grid one of
        whose axes is shorter than the patch (784 units: ``(7, 8, 14)``).
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if not 0.0 <= attention_dropout_rate <= 1.0:
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got "
                f"{attention_dropout_rate}"
            )
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if num_neighborhoods < 1:
            raise ValueError(
                f"num_neighborhoods must be >= 1, got {num_neighborhoods}"
            )
        if normalization_position not in ("pre", "post"):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got "
                f"{normalization_position!r}"
            )
        unknown_sites = sorted(set(tap_sites) - set(TAP_SITES))
        if unknown_sites:
            raise ValueError(
                f"tap_sites contains unknown site(s) {unknown_sites}; "
                f"accepted: {list(TAP_SITES)}"
            )

        # Delegated so the factoring rule lives in exactly one place, and so the
        # message names the grid as well as the width.
        from dl_techniques.layers.regularization.spatial_smoothness import (
            resolve_grid_shape,
            validate_grid_dim,
        )

        # Before the floor, which raises it to the power grid_dim.
        grid_dim = validate_grid_dim(grid_dim)

        # Floor first, THEN resolve. Resolving first reports "a radius-r
        # neighbourhood needs a (2r+1)x(2r+1) patch" for a width that was simply
        # too narrow -- true, but it names the grid rather than the argument the
        # caller has to change.
        side = 2 * radius + 1
        minimum = side ** grid_dim
        if embed_dim < minimum:
            # Only the 3D message carries the radius advice (the 2D text is
            # pinned verbatim): a cube below the default radius's floor is the
            # common 3D mistake, and this is the check that catches it first.
            hint = (
                f" {radius_advice(resolve_grid_shape(embed_dim, grid_shape, grid_dim))}"
                if grid_dim != 2
                else ""
            )
            raise ValueError(
                f"embed_dim ({embed_dim}) is below the {minimum} units a "
                f"radius-{radius} neighbourhood needs "
                f"({' x '.join([str(side)] * grid_dim)}). Lower `radius`, or "
                f"choose a wider variant.{hint}"
            )
        grid = resolve_grid_shape(embed_dim, grid_shape, grid_dim)
        # DECISION plan-2026-10-23T141429-390f3815/D-016
        # Do NOT apply the per-axis check in 2D: an explicit (8, 32) grid at the
        # default radius 5 must still construct (test_topolm.py::TestConstruction::
        # test_an_explicit_grid_is_reported_and_used), and the 3D-only branch is
        # pinned by test_topolm_3d.py::TestThreeDimensionalValidationThroughTheModel.
        # See decisions.md.
        # The per-axis check is 3D only HERE. In 2D this constructor has always
        # accepted a grid with a short axis (an explicit (8, 32) at the default
        # radius 5 constructs and reports its grid; only the tap's build refuses
        # it), and tests/test_models/test_topolm/test_topolm.py pins that, so
        # tightening it would change 2D behaviour. The variant table is stricter
        # in both ranks (config.validate_variant).
        problem = axes_fit_problem(grid, radius) if grid_dim != 2 else None
        if problem is not None:
            raise ValueError(f"embed_dim ({embed_dim}) {problem}")
        logger.debug(
            f"TopoLM unit grid resolved to {'x'.join(str(n) for n in grid)}"
        )

    def _build_architecture(self) -> None:
        """Create every sub-layer. No weights here, and no shape-dependent work."""
        initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range
        )

        self.word_embeddings = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            embeddings_initializer=initializer,
            name="word_embeddings",
        )
        self.positional_embeddings = keras.layers.Embedding(
            input_dim=self.max_seq_len,
            output_dim=self.embed_dim,
            embeddings_initializer=initializer,
            name="positional_embeddings",
        )
        self.embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="embed_dropout"
        )

        # Every sub-layer carries an explicit name, including inside the loop. The
        # auto-generated names shift with depth, and a build-parity check compares
        # two separately-constructed models by path -- so an unnamed block makes
        # that comparison report a naming failure instead of a build one.
        self.blocks = [
            TopoLMBlock(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.ffn_intermediate_size,
                alpha=self.alpha,
                radius=self.radius,
                num_neighborhoods=self.num_neighborhoods,
                distance=self.distance,
                permute=self.permute,
                grid_shape=self._grid_shape_arg,
                # Forwarded even though 2 is the default: a tap resolves its grid
                # lazily at its own build, so a dropped argument builds a silently
                # 2D layer inside a model that reports grid_dim=3.
                grid_dim=self.grid_dim,
                tap_sites=self.tap_sites,
                normalization_position=self.normalization_position,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                initializer_range=self.initializer_range,
                layer_norm_eps=self.layer_norm_eps,
                activation=self.activation,
                ffn_type=self.ffn_type,
                ffn_args=self.ffn_args,
                use_bias=self.use_bias,
                normalization_type=self.normalization_type,
                norm_center=self.norm_center,
                tap_placement=self.tap_placement,
                attention_norm_args=self.attention_norm_args,
                ffn_norm_args=self.ffn_norm_args,
                # The residual-projection init depends on the stack's depth.
                num_layers=self.depth,
                # Each tap's seed is derived from (base seed, its index in the
                # whole model's tap sequence), so no two of the 2 * depth taps
                # share a layout.
                seed=self._tap_seed(2 * index),
                name=f"block_{index}",
            )
            for index in range(self.depth)
        ]

        # The same factory, type and centre flag as the blocks' norms, so one option
        # moves all of them: a stack whose final norm kept a beta while its blocks
        # dropped theirs would be a different model from the one the option names.
        final_norm_args: Dict[str, Any] = {"epsilon": self.layer_norm_eps}
        if self.normalization_type == "layer_norm":
            final_norm_args["center"] = self.norm_center
        self.final_norm = create_normalization_layer(
            self.normalization_type,
            name="final_norm",
            **final_norm_args,
        )

        # A task head replaces the LM head: call() never runs lm_head then,
        # and build() must not create what call() does not run.
        if not self.tie_word_embeddings and not self.include_top:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=initializer,
                name="lm_head",
            )
        else:
            self.lm_head = None

    def _tap_seed(self, tap_index: int) -> int:
        """The layout seed of the ``tap_index``-th tap in the whole model."""
        from dl_techniques.layers.regularization.spatial_smoothness import (
            permutation_seed,
        )

        return permutation_seed(self.seed, tap_index)

    @property
    def tap_layers(self) -> Tuple[Any, ...]:
        """Every :class:`SpatialSmoothness` layer in the model, in stack order.

        The single place a consumer -- the loss logger, an analysis script -- asks
        "which tensors are topographic". Walking ``self.blocks`` by hand would
        have to know the tap-site ordering and would silently miss a site added to
        :data:`TAP_SITES`.
        """
        found = []
        for block in self.blocks:
            for tap in block.taps:
                found.append(tap)
        return tuple(found)

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        """The unit grid this model's width sits on, resolved from config.

        The constructor ARGUMENT is stored privately and this property reports the
        RESOLVED shape, which is the one every analysis function needs -- a caller
        who left ``grid_shape=None`` still needs to know the grid (28x28 for the
        paper's 784 units). ``grid_dim`` entries: ``(height, width)`` for 2, three
        axes for 3, in the order ``cell_to_unit.shape`` has and never reordered.
        Cached after the first read.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            resolve_grid_shape,
        )

        if self._resolved_grid_shape is None:
            self._resolved_grid_shape = resolve_grid_shape(
                self.embed_dim, self._grid_shape_arg, self.grid_dim
            )
        return self._resolved_grid_shape

    def build(self, input_shape: Any) -> None:
        """Materialize exactly the sub-layer tree that ``call`` runs.

        Each sub-layer is built by hand rather than by calling ``self``. Two
        reasons, and both were found the hard way:

        ``Model.build(shape)`` on a subclassed model only marks it built and walks
        no sub-layers, so a build that stops after ``super().build()`` restores
        into nothing when weights are loaded and raises nothing when they are not.

        Calling ``self(...)`` inside ``build`` is worse: Keras invokes ``build``
        again before ``call``, and the dummy forward re-enters it, which is a
        ``RecursionError`` at 160 frames rather than a diagnosable error.

        So the tree is built explicitly, top to bottom, and each block's own
        ``build`` does the same for its attention, norms, feed-forward and taps.
        :param input_shape: Shape of the input to ``call``, ``(batch, seq_len)``.
        :type input_shape: Any
        :raises ValueError: If the sequence axis is statically known and exceeds
            ``max_seq_len``, or the last axis is not ``embed_dim`` -- naming the
            offending value in each case. Both are contracts ``call`` relies on
            and neither can be caught by ``InputSpec``, which accepts ``None``.
        """
        if self.built:
            return

        shape = self._token_shape(input_shape)
        if len(shape) != 2:
            # Token ids, not one-hot rows: the last axis is the sequence length,
            # and asserting it against `vocab_size` would be checking the wrong
            # axis -- an embedding table of `vocab_size` rows takes ids of ANY
            # width up to the table.
            raise ValueError(
                f"TopoLM expects (batch, seq_len) token ids of rank 2, got "
                f"shape {shape}"
            )
        # First, so the cached step functions are dropped and no variable exists.
        # The shared refusal is the only one: a second raise here would refuse
        # the over-long shape a stored archive recorded, which it lets through.
        refuse_static_sequence_length(
            self, shape[1], self.max_seq_len,
            self._SEQUENCE_LENGTH_MESSAGE.format(
                seq_len=shape[1], limit=self.max_seq_len
            ),
        )

        # `token_shape` is the input's OWN rank-2 shape -- dropping its last axis
        # here would build the embeddings for `(batch,)` and hand every block a
        # `(batch, embed_dim)` tensor, which the attention layer rejects three
        # frames later with a shape error that says nothing about this line.
        token_shape = shape
        hidden_shape = (*token_shape, self.embed_dim)

        self.word_embeddings.build(token_shape)
        # The position table is indexed by a single position axis; the batch is
        # broadcast by the add.
        self.positional_embeddings.build((*shape[1:], self.embed_dim))
        self.embed_dropout.build(hidden_shape)

        for block in self.blocks:
            block.build(hidden_shape)

        self.final_norm.build(hidden_shape)
        if self.lm_head is not None:
            self.lm_head.build(hidden_shape)
        if self.task_head is not None:
            self.task_head.build(
                {"hidden_states": hidden_shape, "attention_mask": token_shape}
            )

        super().build(input_shape)

    @staticmethod
    def _token_shape(input_shape: Any) -> Tuple[Optional[int], ...]:
        """The rank-2 token-id shape, from whatever Keras handed ``build``.

        Keras passes a dict-shaped input's STRUCTURE here -- the literal key names
        -- not its shapes, so a model that accepts ``{"input_ids": ...}`` cannot
        assume a flat shape. Three forms are accepted: a real rank-2 tuple, a
        dict of shapes, and the key-name structure.
        """
        if isinstance(input_shape, dict):
            # A dict WITHOUT 'input_ids' is not a shape error here; `call` raises
            # the actionable message for it. Resolving it as a shape instead
            # would replace "must contain 'input_ids'" with a KeyError.
            if "input_ids" in input_shape:
                return tuple(input_shape["input_ids"])
            return (None, None)
        if (
            isinstance(input_shape, tuple)
            and input_shape
            and all(isinstance(entry, str) for entry in input_shape)
        ):
            # The structure arrived without shapes; the batch and sequence axes
            # are then both dynamic.
            return (None, None)
        return tuple(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        taps: Optional[Tuple[Any, ...]] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass.

        :param inputs: Token IDs ``(batch, seq_len)``, or a dict carrying
            ``'input_ids'`` and optionally ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Optional padding mask ``(batch, seq_len)``,
            ``1`` for real tokens. Overridden by a dict input that carries one.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training flag, forwarded explicitly to every sub-layer.
            The taps need it: they add their loss only when it is exactly ``True``.
        :type training: Optional[bool]
        :param taps: Optional identity layers, one per entry of
            :attr:`tap_layers`, receiving each tap's branch output in stack order.
            This is how the post-hoc analysis reads activations out of a
            SUBCLASSED model: there is no ``model.inputs`` / ``layer.output`` pair
            to slice, and a Functional sub-graph cannot be built around internal
            tensors. Each capture returns its input unchanged, so supplying them
            does not alter the residual arithmetic -- asserted in
            ``tests/test_models/test_topolm/``.
        :type taps: Optional[Tuple[Any, ...]]
        :return: With ``include_top=False``, ``{"logits": (batch, seq_len,
            vocab_size), "last_hidden_state": (batch, seq_len, embed_dim)}``.
            With ``include_top=True``, only the head's output reduced by
            ``finalize_head_outputs``: a bare tensor, or a dict of two or more
            tensors (``start_logits`` / ``end_logits`` for QA).
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :raises ValueError: If a dict input has no ``'input_ids'``, or if
            ``taps`` is given the wrong number of entries for
            :attr:`tap_layers`.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        input_ids = check_sequence_length(
            self, input_ids, self.max_seq_len, "max_seq_len",
            self._SEQUENCE_LENGTH_MESSAGE,
        )

        if taps is not None and len(taps) != len(self.tap_layers):
            raise ValueError(
                f"TopoLM was given {len(taps)} capture layer(s) for "
                f"{len(self.tap_layers)} tap(s); supply one per entry of "
                f"`model.tap_layers`, in that order"
            )

        sequence_length = keras.ops.shape(input_ids)[1]
        hidden = self.word_embeddings(input_ids)
        # `arange` with a tensor stop, broadcast over the batch. The same
        # expression `TextDecoder` uses, so the dynamic sequence axis is handled
        # the one way this repository has already measured.
        position_ids = keras.ops.arange(start=0, stop=sequence_length)
        hidden = keras.ops.add(
            hidden, self.positional_embeddings(position_ids)
        )
        hidden = self.embed_dropout(hidden, training=training)

        from dl_techniques.utils.masking import create_causal_attend_mask

        attend_mask = create_causal_attend_mask(hidden, attention_mask)
        for index, block in enumerate(self.blocks):
            block_taps = None
            if taps is not None:
                start = index * len(self.tap_sites)
                block_taps = taps[start:start + len(self.tap_sites)]
            hidden = block(
                hidden,
                attention_mask=attend_mask,
                training=training,
                taps=block_taps,
            )

        hidden = self.final_norm(hidden, training=training)

        if self.include_top:
            head_outputs = self.task_head(
                {
                    "hidden_states": hidden,
                    "attention_mask": head_attention_mask(
                        input_ids, attention_mask, self.pad_token_id,
                        hidden.dtype,
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        if self.tie_word_embeddings:
            logits = tied_embedding_logits(
                hidden, self.word_embeddings.embeddings
            )
        else:
            logits = self.lm_head(hidden)

        return {"logits": logits, "last_hidden_state": hidden}

    def compute_output_shape(
        self, input_shape: Union[Tuple[Optional[int], ...], Dict[str, Any]]
    ) -> Any:
        """Return what ``call`` returns, from STORED CONFIG, on an unbuilt model.

        :param input_shape: ``(batch, seq_len)``, or a dict holding one under
            ``'input_ids'``.
        :return: Headless, both output shapes in a dict. With a head, the head's
            output shapes after the reduction ``call`` applies: one shape, or a
            mapping of two or more.
        """
        token_shape = self._token_shape(input_shape)

        if self.include_top:
            # finalize_head_outputs only drops and unwraps dict entries, so it
            # applies to a dict of shapes exactly as to a dict of tensors.
            return finalize_head_outputs(
                self.task_head.compute_output_shape({
                    "hidden_states": (*token_shape, self.embed_dim),
                    "attention_mask": token_shape,
                })
            )

        return {
            "logits": (*token_shape, self.vocab_size),
            "last_hidden_state": (*token_shape, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "ffn_intermediate_size": self.ffn_intermediate_size,
            "max_seq_len": self.max_seq_len,
            "alpha": self.alpha,
            "radius": self.radius,
            "num_neighborhoods": self.num_neighborhoods,
            "distance": self.distance,
            "permute": self.permute,
            "grid_shape": self._grid_shape_arg,
            "grid_dim": self.grid_dim,
            "tap_sites": tuple(self.tap_sites),
            "normalization_position": self.normalization_position,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "activation": serialize_activation(self.activation),
            "ffn_type": self.ffn_type,
            "ffn_args": dict(self.ffn_args),
            "use_bias": self.use_bias,
            "normalization_type": self.normalization_type,
            "norm_center": self.norm_center,
            "tap_placement": self.tap_placement,
            "attention_norm_args": dict(self.attention_norm_args),
            "ffn_norm_args": dict(self.ffn_norm_args),
            "tie_word_embeddings": self.tie_word_embeddings,
            "seed": self.seed,
            "pad_token_id": self.pad_token_id,
            "include_top": self.include_top,
            "task_config": (
                dict(self.task_config) if self.task_config is not None else None
            ),
            "head_config_overrides": (
                dict(self.head_config_overrides)
                if self.head_config_overrides is not None else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TopoLM":
        """Rebuild from :meth:`get_config`, deserializing the activation.

        Written out rather than inherited because ``activation`` may be a callable
        (``gelu_tanh``) rather than a registered string, and the base class would
        hand the raw function back to ``__init__`` on every backend that does not
        happen to accept one.
        """
        config = dict(config)
        if "activation" in config:
            config["activation"] = deserialize_activation(config["activation"])
        return cls(**config)

    @staticmethod
    def _download_weights(variant: str, cache_dir: Optional[str] = None) -> str:
        """Always raises.

        No pretrained TopoLM weights are distributed with this library, and a
        silent random-init fallback here would make a missing download
        indistinguishable from a real load.

        :param variant: Variant name (unused).
        :type variant: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained TopoLM weights are distributed for variant "
            f"{variant!r}. Train one with `python -m train.topolm.pretrain`, "
            f"or pass pretrained=<local_path> to load a local .keras file."
        )

    @staticmethod
    def _refuse_other_grid(
        weights_path: str, recorded: Dict[str, Any], model: "TopoLM"
    ) -> None:
        """Raise ``ValueError`` when a TopoLM checkpoint sits on another grid.

        Interface contract: ``recorded`` is the ``config`` dict of an archive whose
        top-level ``class_name`` is ``TopoLM`` (the caller checks that; this
        function never guesses a rank for any other archive). A missing
        ``grid_dim`` key means 2D, the layout of every archive saved before the
        key existed. The rank is compared first, so a rank mismatch keeps its own
        message; the grid is compared second, resolved from the recorded
        ``embed_dim``, ``grid_shape`` (``None`` or a JSON list) and ``grid_dim``,
        and only when the recorded width equals the model's (a different width is
        the loader's shape mismatch, which names the real cause). Returns ``None``
        when the layouts agree.

        :param weights_path: The archive path, for the message.
        :param recorded: The archive's recorded TopoLM constructor config.
        :param model: The model being warm-started.
        :raises ValueError: On a different ``grid_dim``, or on a different
            resolved grid at the same width and rank.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            resolve_grid_shape,
        )

        rank = int(recorded.get("grid_dim", 2))
        if rank != model.grid_dim:
            raise ValueError(
                f"Checkpoint {weights_path!r} was saved with grid_dim={rank}, "
                f"but this model has grid_dim={model.grid_dim}. A tap's layout "
                f"(perm, patch_table, triu_flat) belongs to the grid it was "
                f"built on, and some widths and radii give those tables the same "
                f"shape in both ranks, so the weights cannot be moved across. "
                f"Build the model with grid_dim={rank}."
            )
        if recorded.get("embed_dim") != model.embed_dim:
            return
        explicit = recorded.get("grid_shape")
        grid = resolve_grid_shape(
            model.embed_dim,
            None if explicit is None else tuple(explicit),
            rank,
        )
        if grid != model.grid_shape:
            raise ValueError(
                f"Checkpoint {weights_path!r} was built on the unit grid "
                f"{grid}, but this model's grid is {model.grid_shape}. A tap's "
                f"layout (perm, patch_table, triu_flat) belongs to the grid it "
                f"was built on, and a different grid at the same width can give "
                f"tables of identical shape (a transposed one always does), so "
                f"the weights could load and describe the wrong layout. Build "
                f"the model with grid_shape={grid}."
            )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "TopoLM":
        """Create a TopoLM from a named variant.

        :param variant: One of :data:`MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: ``True`` raises :class:`NotImplementedError`; a string
            path loads weights from that local ``.keras`` file; ``False``
            (default) random-initializes. Headed or headless, the model receives
            the backbone (and an untied ``lm_head``) by layer name: a
            ``task_head`` in the checkpoint is never copied and the model's own
            head keeps its fresh initialization, so a checkpoint of any TopoLM,
            headless or trained under any head, warm-starts a model with any
            other head or none. Every backbone layer loads, and every
            checkpoint layer finds its place, or the call raises (another
            depth, deeper or shallower, or width is refused);
            ``word_embeddings`` and an untied ``lm_head`` stay fresh only when
            the checkpoint's own ``vocab_size`` differs, and an untied
            ``lm_head`` may otherwise be left out only when one side is
            headed. Needs
            a ``.keras`` file. To restore a whole model, head included, use
            ``keras.models.load_model``.
        :type pretrained: Union[bool, str]
        :param kwargs: Override any variant parameter, e.g. ``alpha=0.0`` for the
            non-topographic control or ``tap_sites=("attention",)``, including
            ``include_top``, ``task_config``, ``head_config_overrides`` and
            ``pad_token_id``.
        :type kwargs: Any
        :return: The configured model.
        :rtype: TopoLM
        :raises ValueError: If ``variant`` is not recognized -- the message lists
            the available names.
        :raises FileNotFoundError: If ``pretrained`` is a path that does not exist.
        :raises NotImplementedError: If ``pretrained=True``.

        Example:
            .. code-block:: python

                model = TopoLM.from_variant("paper")
                control = TopoLM.from_variant("paper", alpha=0.0)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Available: {sorted(cls.MODEL_VARIANTS)}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        # Keras derives an unset model name by splitting the class name, which
        # turns `TopoLM` into `topo_lm`. The name ends up in every weight path,
        # so it is pinned here rather than left to that rule -- a caller who wants
        # their own still overrides it.
        config.setdefault("name", "topolm")
        config.update(kwargs)

        model = cls(**config)

        if pretrained:
            weights_path = pretrained if isinstance(pretrained, str) else None
            if weights_path is None:
                cls._download_weights(variant)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights file not found: {weights_path}"
                )
            # DECISION plan-2026-10-23T141429-390f3815/D-017
            # Refuse a checkpoint of another grid BY NAME, before any copy, and
            # only when the archive IS a bare TopoLM. Do NOT default a missing
            # grid_dim to 2 for any other archive (a CausalLanguageModel around a
            # 3D TopoLM has no such key and would be reported as "saved with
            # grid_dim=2", hiding the loader's real error), and do NOT compare
            # grid_dim alone (a 3D grid (10, 5, 20) loads silently into (20, 5, 10):
            # every table has the same shape). Do NOT leave the rank to the shape
            # check in load_backbone_from_checkpoint either: perm, patch_table and
            # triu_flat have a shape fixed by the width, the radius and the grid
            # axes, and a 2D and a 3D grid can give them identical shapes (729
            # units at radius 13 in 2D and radius 4 in 3D both hold one 729-unit
            # patch). Guarded by test_topolm_3d.py::TestTheGuardReadsOnlyATopoLMConfig,
            # TestACheckpointOnAnotherGridIsRefused and
            # TestACheckpointOfTheOtherRankIsRefused. See decisions.md.
            recorded_archive = read_checkpoint_config(weights_path)
            if recorded_archive.get("class_name") == "TopoLM" and isinstance(
                recorded_archive.get("config"), dict
            ):
                cls._refuse_other_grid(weights_path, recorded_archive["config"], model)
            # Built here, so the restore has somewhere to land: a build, not a
            # dummy forward, so no RNG is drawn. Without it every local path
            # raised "model must be built" (measured).
            model.build((1, min(32, model.max_seq_len)))
            # DECISION plan-2026-10-11T004932-b83935e1/D-012: by-name and strict; do NOT send a headed
            # model through load_weights_or_raise (copies an equal-shape head) nor pass strict=False
            # (a backbone that does not fit records only shape mismatches and loads nothing). See decisions.md.
            # DECISION plan-2026-10-11T004932-b83935e1/D-007: skip the checkpoint's task_head; do NOT
            # drop TASK_HEAD_NAME from this list -- a head of another task raises on shape, one of equal shape is copied silently. See decisions.md.
            # DECISION plan-2026-10-11T004932-b83935e1/D-026: complete-or-raise headed AND headless; do NOT
            # restore a headless model by whole file with skip_mismatch=True (a deeper or wider model keeps fresh
            # layers and returns normally), and key the vocabulary layers on the CHECKPOINT's vocab_size. See decisions.md.
            load_backbone_from_checkpoint(
                model,
                weights_path,
                head_prefixes=(TASK_HEAD_NAME,),
                vocab_layers=("word_embeddings",),
                tables=("word_embeddings/embeddings", "lm_head/kernel"),
                tops=("lm_head",),
            )

        return model


# ---------------------------------------------------------------------
# Module-level Factory
# ---------------------------------------------------------------------


def create_topolm(
    variant: str = "small",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> TopoLM:
    """Build a :class:`TopoLM`, delegating to :meth:`TopoLM.from_variant`.

    No logic of its own, by ruling: defaults belong in
    :data:`MODEL_VARIANTS` and validation in the constructor, so the factory and
    the classmethod can never disagree about what a variant name means. An earlier
    revision took a ``vocab_size`` shortcut here and was the only reason this
    function appeared in the repo's non-delegating-factory census.

    :param variant: One of :data:`TopoLM.MODEL_VARIANTS`. Default ``"small"``.
    :type variant: str
    :param pretrained: ``True`` raises :class:`NotImplementedError`; a string
        loads a local ``.keras`` file; ``False`` random-initializes.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to :meth:`TopoLM.from_variant`, e.g.
        ``vocab_size=50257``, ``alpha=0.0``, ``radius=3``, ``permute=False``,
        ``tap_sites=()``, or ``include_top=True`` with a ``task_config`` for a
        model with a task head. A local ``pretrained`` path never restores a
        checkpoint's task head.
    :type kwargs: Any
    :return: A configured :class:`TopoLM`.
    :rtype: TopoLM
    :raises ValueError: If ``variant`` is unrecognized, or a forwarded argument
        is invalid (including inconsistent head knobs, see :class:`TopoLM`).
    :raises NotImplementedError: If ``pretrained=True``.

    Example:
        >>> topographic = create_topolm("paper")
        >>> control = create_topolm("paper", alpha=0.0)
        >>> no_permutation = create_topolm("paper", permute=False)
        >>>
        >>> # The same backbone with a sentence-classification head (pools 'last')
        >>> clf = create_topolm(
        ...     "tiny", include_top=True,
        ...     task_config=NLPTaskConfig(
        ...         name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION,
        ...         num_classes=2))
    """
    return TopoLM.from_variant(variant, pretrained=pretrained, **kwargs)


# ---------------------------------------------------------------------