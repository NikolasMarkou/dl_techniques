"""TopoLM, a decoder-only language model whose units are laid out on a sheet of tissue.

The idea
--------
A GPT-2 block indexes its units by nothing but their position on the last axis of
a weight matrix, so nothing about the representation is spatial: two units that
learn to encode the same thing have no reason to sit near each other, and the
emergent clustering is a property of the initialisation rather than of anything
the objective asked for. TopoLM gives the units of every attention and
feed-forward branch a distinct cell on an ``h x w`` grid and adds a penalty
``SL = 0.5 * (1 - corr(r, d))`` alongside the next-token cross-entropy, where
``r`` is the vector of pairwise activation correlations over a sampled square
neighbourhood and ``d`` the matching inverse distances. Minimising it makes
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
curve is attributable to the added term alone.

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
from dl_techniques.utils.weight_transfer import load_weights_or_raise
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.activation_serialization import (
    deserialize_activation,
    serialize_activation,
)
from dl_techniques.layers.activations import gelu_tanh
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig
from .components import TAP_SITES, TopoLMBlock
from .config import (
    DEFAULT_ALPHA,
    DEFAULT_NUM_NEIGHBORHOODS,
    DEFAULT_RADIUS,
    MODEL_VARIANTS,
    validate_all_variants,
)

# ---------------------------------------------------------------------

#: Tiktoken ``cl100k_base``, the pipeline this repository's trainers tokenize with.
DEFAULT_VOCAB_SIZE = 100277

validate_all_variants(DEFAULT_RADIUS)


@register_dl_technique("dl_techniques.models.topolm.model")
class TopoLM(keras.Model):
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
            │ final LayerNorm         │
            └─────────────────────────┘
                 │
                 ├──► {"last_hidden_state"}
                 ▼
            tied matmul(token emb.T)  |  lm_head Dense
                 │
                 ▼  logits [B, S, vocab_size]

    Both leaves come back together, so
    :class:`~dl_techniques.models.common.masked_language_model.CausalLanguageModel`
    and the standard CLM data wrapper work unchanged. No ``None`` is echoed into
    the output dict, so ``predict({"input_ids": ...})`` does not trip Keras'
    nested-structure check.

    Variants (:data:`~dl_techniques.models.topolm.config.MODEL_VARIANTS`):

        .. code-block:: text

            variant  embed_dim  depth  heads  max_seq_len  grid     provenance
            paper    784        12     16     1024         28 x 28  Rathi et al. 2025
            small    512        6      8      512          16 x 32  repo-authored
            tiny     256        4      4      256          16 x 16  repo-authored

    Only ``paper`` is quoted data; see
    :mod:`~dl_techniques.models.topolm.config` for why the other two are not
    pinned by the upstream-variant guard.

    :param vocab_size: Vocabulary size. Default 100277 (cl100k_base).
    :type vocab_size: int
    :param embed_dim: Model width, and the tapped axis' length. Must be a
        product of two factors at least ``(2 * radius + 1)`` across.
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
    :param grid_shape: Explicit ``(height, width)`` for every tap, or ``None`` to
        factor ``embed_dim``.
    :type grid_shape: Optional[Tuple[int, int]]
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
    :param activation: Feed-forward activation. Defaults to ``gelu_tanh``, GPT-2's
        tanh approximation, not the exact-erf ``'gelu'`` string.
    :type activation: Union[str, Any]
    :param tie_word_embeddings: Reuse the transposed token embedding as the LM
        head. Default ``True``, which builds no ``lm_head``.
    :type tie_word_embeddings: bool
    :param seed: Base seed for the taps. Every tap's layout seed is derived from
        it, so the whole model's topography is reproducible from this one integer.
    :type seed: Optional[int]
    :param kwargs: Forwarded to ``keras.Model``.
    :raises ValueError: If the width cannot host the tapped grid, does not divide
        among the heads, or if any argument is out of range -- naming the offending
        value in each case.
    """

    MODEL_VARIANTS = MODEL_VARIANTS

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
        grid_shape: Optional[Tuple[int, int]] = None,
        tap_sites: Tuple[str, ...] = TAP_SITES,
        normalization_position: str = "pre",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        activation: Union[str, Any] = gelu_tanh,
        tie_word_embeddings: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size, embed_dim, depth, num_heads, max_seq_len, dropout_rate,
            attention_dropout_rate, radius, alpha, num_neighborhoods, distance,
            tap_sites, grid_shape, normalization_position,
        )

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
        self._resolved_grid_shape: Optional[Tuple[int, int]] = None
        self.tap_sites = tuple(tap_sites)
        self.normalization_position = normalization_position
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.activation = activation
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.seed = seed

        self._build_architecture()

        logger.info(
            f"Created TopoLM: depth={self.depth}, embed_dim={self.embed_dim}, "
            f"heads={self.num_heads}, max_seq_len={self.max_seq_len}, "
            f"alpha={self.alpha}, radius={self.radius}, "
            f"taps={len(self.blocks) * len(self.tap_sites)}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
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
        grid_shape: Optional[Tuple[int, int]],
        normalization_position: str,
    ) -> None:
        """Check every argument and every cross-parameter contract at once.

        The grid floor is the one that bites: a width below ``(2r+1)^2`` builds a
        model that cannot construct a single neighbourhood, and the failure
        surfaces at the tap's ``build`` -- after embeddings, attention and a whole
        stack have been created. It is a construction-time error, so it is raised
        here.
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
        )

        # Floor first, THEN resolve. Resolving first reports "a radius-r
        # neighbourhood needs a (2r+1)x(2r+1) patch" for a width that was simply
        # too narrow -- true, but it names the grid rather than the argument the
        # caller has to change.
        minimum = (2 * radius + 1) ** 2
        if embed_dim < minimum:
            raise ValueError(
                f"embed_dim ({embed_dim}) is below the {minimum} units a "
                f"radius-{radius} neighbourhood needs "
                f"({2 * radius + 1} x {2 * radius + 1}). Lower `radius`, or choose "
                f"a wider variant."
            )
        grid = resolve_grid_shape(embed_dim, grid_shape)
        logger.debug(f"TopoLM unit grid resolved to {grid[0]}x{grid[1]}")

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
                tap_sites=self.tap_sites,
                normalization_position=self.normalization_position,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                initializer_range=self.initializer_range,
                layer_norm_eps=self.layer_norm_eps,
                activation=self.activation,
                # Each tap's seed is derived from (base seed, its index in the
                # whole model's tap sequence), so no two of the 2 * depth taps
                # share a layout.
                seed=self._tap_seed(2 * index),
                name=f"block_{index}",
            )
            for index in range(self.depth)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="final_norm"
        )

        if not self.tie_word_embeddings:
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
    def grid_shape(self) -> Tuple[int, int]:
        """The unit grid this model's width sits on, resolved from config.

        The constructor ARGUMENT is stored privately and this property reports the
        RESOLVED shape, which is the one every analysis function needs -- a caller
        who left ``grid_shape=None`` still needs to know the grid is 28x28.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            resolve_grid_shape,
        )

        if self._resolved_grid_shape is None:
            self._resolved_grid_shape = resolve_grid_shape(
                self.embed_dim, self._grid_shape_arg
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
        if shape[1] is not None and int(shape[1]) > self.max_seq_len:
            raise ValueError(
                f"sequence length {int(shape[1])} exceeds max_seq_len "
                f"{self.max_seq_len}; the position table has no row for it"
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
    ) -> Dict[str, keras.KerasTensor]:
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
        :return: ``{"logits": (batch, seq_len, vocab_size),
            "last_hidden_state": (batch, seq_len, embed_dim)}``.
        :rtype: Dict[str, keras.KerasTensor]
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

        if self.tie_word_embeddings:
            logits = tied_embedding_logits(
                hidden, self.word_embeddings.embeddings
            )
        else:
            logits = self.lm_head(hidden)

        return {"logits": logits, "last_hidden_state": hidden}

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return both output shapes from STORED CONFIG, on an unbuilt model."""
        return {
            "logits": (*tuple(input_shape), self.vocab_size),
            "last_hidden_state": (*tuple(input_shape), self.embed_dim),
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
            "tap_sites": tuple(self.tap_sites),
            "normalization_position": self.normalization_position,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "activation": serialize_activation(self.activation),
            "tie_word_embeddings": self.tie_word_embeddings,
            "seed": self.seed,
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
            (default) random-initializes.
        :type pretrained: Union[bool, str]
        :param kwargs: Override any variant parameter, e.g. ``alpha=0.0`` for the
            non-topographic control or ``tap_sites=("attention",)``.
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
            # Built above, so the restore has somewhere to land. By-name transfer
            # does not exist in Keras 3 and its absence used to be swallowed into
            # a warning, turning a partial restore into a silent success.
            load_weights_or_raise(model, weights_path, skip_mismatch=True)

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
        ``tap_sites=()``.
    :type kwargs: Any
    :return: A configured :class:`TopoLM`.
    :rtype: TopoLM
    :raises ValueError: If ``variant`` is unrecognized.
    :raises NotImplementedError: If ``pretrained=True``.

    Example:
        >>> topographic = create_topolm("paper")
        >>> control = create_topolm("paper", alpha=0.0)
        >>> no_permutation = create_topolm("paper", permute=False)
    """
    return TopoLM.from_variant(variant, pretrained=pretrained, **kwargs)


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_topolm_with_head(
    topolm_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    topolm_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Build an end-to-end model: a TopoLM encoder plus an NLP task head.

    Takes a variant name, instantiates the encoder, builds a head from the
    ``dl_techniques.layers.heads.nlp`` factory, and joins them into one functional
    ``keras.Model``. The head pools the last position by default (causal model),
    which ``head_config_overrides`` can change.

    .. code-block:: text

        {"input_ids": [B, L] int32}
                 │
                 ├─────────────────────┐
                 ▼                     ▼
        ┌───────────────────┐     input_ids != pad_token_id
        │ TopoLM encoder    │          │
        └───────────────────┘          │
                 │ last_hidden_state   │
                 ▼                     ▼
        ┌─────────────────────────────────┐
        │ nlp head  pooling_type 'last'   │
        └─────────────────────────────────┘
                 │
                 ▼
            task outputs

    :param topolm_variant: The TopoLM variant to use (e.g., "paper", "small").
    :type topolm_variant: str
    :param task_config: An ``NLPTaskConfig`` object defining the task, which must set
        ``vocabulary_size``.
    :type task_config: NLPTaskConfig
    :param pretrained: If a string, path to a local weights file. If True, raises
        ``NotImplementedError``. Defaults to False.
    :type pretrained: Union[bool, str]
    :param topolm_config_overrides: Optional dictionary to override default TopoLM
        configuration for the chosen variant. Defaults to None.
    :type topolm_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration, including ``pooling_type``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete ``keras.Model`` ready for the specified task.
    :rtype: keras.Model
    :raises ValueError: If ``task_config`` has no ``vocabulary_size``, or the variant
        is unknown.
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        .. code-block:: python

            from dl_techniques.layers.heads.nlp import NLPTaskType

            # Define a task for sequence classification
            seq_cls_task = NLPTaskConfig(
                name="sentiment_analysis",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=3,
                vocabulary_size=100277
            )

            # Create the full model with a TopoLM Paper encoder
            model = create_topolm_with_head(
                topolm_variant="paper",
                task_config=seq_cls_task,
                pretrained=False,
                head_config_overrides={"dropout_rate": 0.15}
            )
            model.summary()
    """
    topolm_config_overrides = topolm_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(
        f"Creating TopoLM-{topolm_variant} with a '{task_config.name}' head."
    )

    if not getattr(task_config, 'vocabulary_size', None):
        raise ValueError(
            "The `task_config` must set 'vocabulary_size' "
            "to create a TopoLM model."
        )

    topolm_encoder = TopoLM.from_variant(
        topolm_variant,
        vocab_size=task_config.vocabulary_size,
        pretrained=pretrained,
        **topolm_config_overrides,
    )

    # TopoLM is causal, so use 'last' pooling
    head_kwargs = {'pooling_type': 'last'}
    head_kwargs.update(head_config_overrides)
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=topolm_encoder.embed_dim,
        **head_kwargs,
    )

    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
    }

    # TopoLM call expects input_ids as first positional arg
    encoder_outputs = topolm_encoder(
        inputs["input_ids"],
    )

    attention_mask = keras.ops.not_equal(
        inputs["input_ids"], 0
    )

    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": attention_mask,
    }
    task_outputs = task_head(head_inputs)

    model_name = f"topolm_{topolm_variant}_with_{task_config.name}_head"
    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=model_name
    )

    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model


# ---------------------------------------------------------------------