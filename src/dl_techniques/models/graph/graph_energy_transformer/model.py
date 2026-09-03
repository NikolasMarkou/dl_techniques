"""
Graph Energy Transformer: a shared trunk, :class:`GraphEnergyTransformerBackbone`,
plus two heads, :class:`GraphAnomalyDetector` and :class:`GraphClassifier`, built on
the same recurrent energy-descent block as the image Energy Transformer models.

A message-passing GNN propagates information one hop per layer and oversmooths past
a few layers, as repeated neighbourhood averaging pulls every node toward the same
value. This model instead defines one scalar energy `E` over the node states and runs
`T` steps of gradient descent on it, `x <- x - alpha * dE/dg` with `g =
EnergyLayerNorm(x)`, reusing one block's weights at every step. The energy sums an
attention term, whose gradient mixes tokens along the graph, with a Hopfield term over
a tied memory matrix, whose gradient pulls each node toward a stored pattern on its
own. That second term is what keeps the states from collapsing to a neighbourhood mean.

The adjacency is passed as each block's rank-3 `attention_mask`, restricting dense
attention to real edges; padding nodes are excluded via a rank-2 `node_mask`. Each
block runs internally in `self.dtype_policy.variable_dtype` rather than the model's
compute dtype, because `EnergyLayerNorm`'s backward pass overflows float16 under XLA
otherwise and freezes training without an error. Both heads emit logits, with no
sigmoid or softmax applied in the graph. `GraphAnomalyDetector` and `GraphClassifier`
backbones are not weight-compatible with each other.

References:
    - Hoover et al., 2023. Energy Transformer. NeurIPS 2023 (§4-§5, graph model).
      (https://arxiv.org/abs/2302.07253)
    - Ramsauer et al., 2020. Hopfield Networks is All You Need.
      (https://arxiv.org/abs/2008.02217)
    - Dwivedi & Bresson, 2020. A Generalization of Transformer Networks to Graphs.
      (https://arxiv.org/abs/2012.09699)
    - Kreuzer et al., 2021. Rethinking Graph Transformers with Spectral Attention.
      (https://arxiv.org/abs/2106.03893)
    - Welling & Teh, 2011. Bayesian Learning via Stochastic Gradient Langevin Dynamics.
      ICML 2011.
"""

import keras
from keras import layers, ops
from keras.saving import serialize_keras_object, deserialize_keras_object
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.mask_token import MaskTokenApply

# The ET block has NO factory home — direct import is the sanctioned path (D-004 of
# plan_2026-07-13_57c9833e; and D-004 here). Do not route it through a factory: this trunk
# calls the block's duck-typed `.attention.update` / `.hopfield.update` / `.norm` surface,
# which no factory type guarantees.
from dl_techniques.layers.transformers.energy_transformer import EnergyTransformer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Stable names
# ---------------------------------------------------------------------

# A warm-start matches layers by name, so any two graph models sharing this trunk
# must name their backbone identically or the transfer moves zero layers.
GRAPH_BACKBONE_NAME = "graph_et_backbone"


@register_dl_technique("dl_techniques.models.graph_energy_transformer.model")
class GraphEnergyTransformerBackbone(keras.Model):
    """Shared graph trunk giving the ``EnergyTransformer`` block one checkpointable
    home that both graph heads compose under the same name.

    Architecture:

    .. code-block:: text

        node_features [B, N, F]   adjacency, node_mask, pe (optional)
                │
                ▼
        ┌───────────────┐
        │ node_proj      │
        │ [B, N, D]      │
        └───────┬────────┘
                ▼
        (+ pe_proj(pe))         (optional, 'use_pe')
        (+ mask_token)          (optional, node_replace_mask present)
        (prepend cls_token)     (optional, 'use_cls'; N -> N+1)
                │
                ▼
        ┌───────────────┐
        │ ET block x     │  call(): num_blocks, fed adjacency + node_mask
        │ num_blocks     │  descend_capture(): blocks[0] only, manual T steps
        └───────┬────────┘
                ▼
        tokens [B, N', D]

    Variant B (node anomaly) uses one block and reads per-step LayerNormed
    states through :meth:`descend_capture`; variant C (graph classification)
    stacks ``S`` blocks and reads the final states through the standard
    :meth:`call`. Both variants share :meth:`embed` for the token-building
    stage above the blocks.

    The backbone takes a dict of graph tensors::

        {
          "node_features": (B, N, F),      # required
          "adjacency":     (B, N, N),      # required by call()
          "node_mask":     (B, N),         # 1 = real, 0 = pad
          "pe":            (B, N, pe_dim), # only when use_pe
          "node_replace_mask": (B, N) bool # optional masked-node pretext
          "target_index":  (B,)            # variant B only; accepted, unused
        }

    ``embed`` reads ``node_features``, ``pe`` and ``node_replace_mask``;
    ``call`` additionally reads ``adjacency`` and ``node_mask``, CLS-augmenting
    both when ``use_cls``.

    ``mask_token`` is created and built on every trunk, even one that never
    masks a node, so the weight surface does not depend on whether the
    masked-node pretext is used. A ``use_pe=True`` backbone (variant C) owns a
    ``pe_proj`` Dense and a ``cls_token`` that a ``use_pe=False`` backbone
    (variant B) does not, so the two are not weight-compatible and a
    cross-variant warm-start transfers nothing.

    :param node_feature_dim: Input node-feature dimension ``F``.
    :param embed_dim: Token dimension ``D``.
    :param num_heads: Attention heads ``H``.
    :param head_dim: Per-head key/query dim ``Y`` (a FREE parameter — ET has no value matrix).
    :param hopfield_dim: Hopfield memory count ``K``.
    :param num_blocks: Number of ET blocks (``1`` for variant B, ``S=4`` for variant C).
    :param num_steps: Descent steps ``T`` per block.
    :param step_size: Descent step ``alpha``.
    :param beta: Attention inverse temperature; ``None`` -> ``1/sqrt(head_dim)`` (resolved by
        ``EnergyAttention``).
    :param hopfield_activation: ``'relu'`` (default) or ``'softmax'``.
    :param hopfield_beta: Temperature of the ``'softmax'`` Hopfield branch.
    :param noise_std: eq.-27 Langevin noise std (training only). ``0.0`` (default) keeps the
        descent guarantee; variant B must keep this ``0.0`` so the manual
        :meth:`descend_capture` loop matches the block's own noiseless descent.
    :param norm_epsilon: ``epsilon`` of each block's inner ``EnergyLayerNorm``.
    :param attn_self: ``True`` (default for the graph trunk) lets a node attend to itself, so
        the graph loaders' ``add_self_loops=True`` adjacency diagonal is honored. The image
        model's default is ``False``; on a graph that would silently mask the adjacency
        diagonal, making the added self-loops a dead no-op. Pass ``attn_self=False`` to recover
        the image default. See decisions.md D-004.
    :param use_weighted_adjacency: If ``True``, each ET block learns the paper's eq.-25 per-edge
        weighted adjacency ``Ŵ`` (a Conv2D over ``X⊗X`` gated by the binary adjacency), folded
        multiplicatively into the attention logits. The binary adjacency the projector needs is
        already the rank-3 (CLS-augmented) ``attention_mask`` each block receives in :meth:`call`,
        so no new input plumbing is required here, only this flag. ``False`` (default) is
        byte-identical to the C-lite binary-adjacency model. See ``EnergyTransformer`` D-002 of
        plan ``plan-2026-07-15T053724-78001af1``.
    :param adjacency_kernel_size: Conv2D kernel of the ``Ŵ`` projector (default ``1``). Only used
        when ``use_weighted_adjacency``.
    :param adjacency_proj_dim: Optional bottleneck ``P`` on the ``X⊗X`` pair features fed to the
        projector Conv2D; ``None`` (default) uses the full ``D²`` pairing. Set a small value
        (e.g. 8-16) to avoid the ``D²``-channel memory blow-up at ``D=128``. Only used when
        ``use_weighted_adjacency``.
    :param use_pe: If ``True``, create a ``pe_proj`` Dense and add ``pe_proj(pe)`` to the tokens
        (variant C Laplacian PE). Defaults to ``False``.
    :param pe_dim: Laplacian-PE dimension (columns of the eigenvector block). Defaults to ``15``.
    :param use_cls: If ``True``, prepend a learnable CLS token (``N -> N+1``) and CLS-augment the
        adjacency + node mask inside :meth:`call` (variant C). Defaults to ``False``.
    :param seed: Seed for the ``noise_std`` RNG.
    :param name: Sub-model name; defaults to :data:`GRAPH_BACKBONE_NAME`.

    :raises ValueError: If any positive-int dimension is non-positive, ``num_blocks < 1``,
        ``num_steps < 1`` or ``pe_dim <= 0`` while ``use_pe``.

    Input shape:
        A dict of graph tensors (see above).

    Output shape:
        ``(batch, N', embed_dim)`` with ``N' = N (+ 1 if use_cls)``.
    """

    def __init__(
            self,
            node_feature_dim: int,
            embed_dim: int,
            num_heads: int,
            head_dim: int,
            hopfield_dim: int,
            num_blocks: int = 1,
            num_steps: int = 12,
            step_size: float = 0.1,
            beta: Optional[float] = None,
            hopfield_activation: str = 'relu',
            hopfield_beta: float = 1.0,
            noise_std: float = 0.0,
            norm_epsilon: float = 1e-5,
            attn_self: bool = True,   # DECISION plan-2026-07-15T015824-3c2159eb/D-004
            use_weighted_adjacency: bool = False,
            adjacency_kernel_size: int = 1,
            adjacency_proj_dim: Optional[int] = None,
            use_pe: bool = False,
            pe_dim: int = 15,
            use_cls: bool = False,
            seed: Optional[int] = None,
            name: Optional[str] = GRAPH_BACKBONE_NAME,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # ----- validation -----
        for _n, _v in (
            ("node_feature_dim", node_feature_dim), ("embed_dim", embed_dim),
            ("num_heads", num_heads), ("head_dim", head_dim), ("hopfield_dim", hopfield_dim),
        ):
            if not isinstance(_v, int) or _v <= 0:
                raise ValueError(f"{_n} must be a positive integer, got {_v}")
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError(f"num_blocks must be an integer >= 1, got {num_blocks}")
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError(f"num_steps must be an integer >= 1, got {num_steps}")
        if not isinstance(step_size, (int, float)) or step_size <= 0:
            raise ValueError(f"step_size must be a positive number, got {step_size}")
        if not isinstance(noise_std, (int, float)) or noise_std < 0:
            raise ValueError(f"noise_std must be a non-negative number, got {noise_std}")
        if use_pe and (not isinstance(pe_dim, int) or pe_dim <= 0):
            raise ValueError(f"pe_dim must be a positive integer when use_pe, got {pe_dim}")

        self.node_feature_dim = int(node_feature_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.hopfield_dim = int(hopfield_dim)
        self.num_blocks = int(num_blocks)
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.beta = beta
        self.hopfield_activation = str(hopfield_activation)
        self.hopfield_beta = float(hopfield_beta)
        self.noise_std = float(noise_std)
        self.norm_epsilon = float(norm_epsilon)
        # DECISION plan-2026-07-15T015824-3c2159eb/D-004: the graph trunk defaults
        # attn_self=True, unlike the image default. At False the self-loops the graph
        # loaders add become a measured no-op. See decisions.md.
        self.attn_self = bool(attn_self)
        # eq.-25 weighted adjacency (Branch A). Default-off preserves the C-lite binary model
        # byte-identically; the three knobs thread straight into each ET block via `_make_block`.
        self.use_weighted_adjacency = bool(use_weighted_adjacency)
        self.adjacency_kernel_size = int(adjacency_kernel_size)
        self.adjacency_proj_dim = (
            int(adjacency_proj_dim) if adjacency_proj_dim is not None else None
        )
        self.use_pe = bool(use_pe)
        self.pe_dim = int(pe_dim)
        self.use_cls = bool(use_cls)
        self.seed = seed

        # Sub-layers are created here (unbuilt), never lazily in build()/call() -- a
        # lazily-created sub-layer is not tracked at save time and silently drops its
        # weights on a `.keras` round-trip.
        self.node_proj = layers.Dense(
            self.embed_dim, name="node_proj", dtype=self.dtype_policy
        )

        # Created and built on every trunk, even one that never masks a node, so the
        # weight surface does not depend on whether the pretext is used.
        self.mask_token = MaskTokenApply(name="node_mask_token", dtype=self.dtype_policy)

        # Only when use_pe -- B (use_pe=False) and C (use_pe=True) are therefore not
        # weight-compatible (documented in the class docstring).
        self.pe_proj = (
            layers.Dense(self.embed_dim, name="pe_proj", dtype=self.dtype_policy)
            if self.use_pe else None
        )

        # cls_token is a raw learnable weight -> created in build() (add_weight's home), only
        # when use_cls. See build().

        # Built through the overridable `_make_block` seam so the fp16/XLA guard test can
        # subclass the backbone and force the fp16-unsafe construction to prove itself.
        # Production code must not override it. See _make_block for the dtype rationale.
        self.blocks: List[EnergyTransformer] = [
            self._make_block(i) for i in range(self.num_blocks)
        ]

        # created in build() when use_cls
        self.cls_token: Optional[keras.Variable] = None

        logger.info(
            f"Created GraphEnergyTransformerBackbone: F={self.node_feature_dim}, "
            f"{self.embed_dim}d, {self.num_heads}h x {self.head_dim}, K={self.hopfield_dim}, "
            f"blocks={self.num_blocks}, T={self.num_steps}, alpha={self.step_size}, "
            f"use_pe={self.use_pe}, use_cls={self.use_cls}, noise_std={self.noise_std}"
        )

    # -----------------------------------------------------------------

    def _make_block(self, index: int) -> EnergyTransformer:
        """Construct ET block ``index``. Overridable seam for the fp16 guard test.

        # DECISION plan-2026-07-15T015824-3c2159eb/D-002: build with
        # ``dtype=self.dtype_policy.variable_dtype``, not ``self.dtype_policy`` --
        # ``EnergyLayerNorm``'s backward overflows float16 under XLA otherwise. See decisions.md.
        """
        return EnergyTransformer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            hopfield_dim=self.hopfield_dim,
            num_steps=self.num_steps,
            step_size=self.step_size,
            beta=self.beta,
            attn_self=self.attn_self,
            use_weighted_adjacency=self.use_weighted_adjacency,
            adjacency_kernel_size=self.adjacency_kernel_size,
            adjacency_proj_dim=self.adjacency_proj_dim,
            hopfield_activation=self.hopfield_activation,
            hopfield_beta=self.hopfield_beta,
            noise_std=self.noise_std,
            return_energy=False,
            norm_epsilon=self.norm_epsilon,
            seed=self.seed,
            name=f"et_block_{index}",
            dtype=self.dtype_policy.variable_dtype,
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Explicitly build EVERY sub-layer from stored config.

        Shapes come from the CONFIG, never from the input's (dynamic) node count ``N`` — the
        trunk is node-count-agnostic, so every sub-layer builds with ``N = None``. A lazily
        built sub-layer silently drops its weights on a ``.keras`` round-trip.
        """
        if self.built:
            return

        node_feat_shape = (None, None, self.node_feature_dim)
        token_shape = (None, None, self.embed_dim)
        node_mask_shape = (None, None)

        self.node_proj.build(node_feat_shape)
        # Built even for a head that never masks a node.
        self.mask_token.build([token_shape, node_mask_shape])
        if self.use_pe:
            self.pe_proj.build((None, None, self.pe_dim))
        if self.use_cls:
            # Raw learnable CLS token, prepended in `embed()`. add_weight's home is build().
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="truncated_normal",
                trainable=True,
            )
        for block in self.blocks:
            block.build(token_shape)

        super().build(input_shape)

    # -----------------------------------------------------------------

    def embed(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Node projection -> [+PE] -> [mask-token] -> [CLS prepend] -> ``(B, N', D)`` tokens.

        Graph-safe: ``keras.ops`` only, static structure. The optional pieces are TRACE-TIME
        structural choices (config flags / key presence), never Python ``if`` on tensor values.

        :param inputs: The graph dict (see the class docstring). Reads ``node_features`` and,
            when configured, ``pe`` and ``node_replace_mask``.
        :param training: Keras training flag.
        :return: ``(B, N', D)`` tokens (``N' = N + 1`` when ``use_cls``).
        """
        if not self.built:
            self.build(None)

        node_features = inputs["node_features"]
        x = self.node_proj(node_features, training=training)               # (B, N, D)

        # Laplacian PE add (variant C). CLS gets NO PE — the add happens before the prepend.
        if self.use_pe:
            x = x + self.pe_proj(inputs["pe"], training=training)          # (B, N, D)

        # Masked-node pretext: apply the (always-built) mask token where requested. Trace-time
        # structural — the key is present or it is not; the token stays built either way.
        if "node_replace_mask" in inputs and inputs["node_replace_mask"] is not None:
            x = self.mask_token([x, inputs["node_replace_mask"]])          # (B, N, D)

        # CLS token (variant C): prepend -> N -> N+1. The head reads x[:, 0] after `call()`.
        if self.use_cls:
            batch = ops.shape(x)[0]
            cls = ops.broadcast_to(
                ops.cast(self.cls_token, x.dtype), (batch, 1, self.embed_dim)
            )
            x = ops.concatenate([cls, x], axis=1)                          # (B, N+1, D)

        return x

    # -----------------------------------------------------------------

    def _augment_cls_masks(
            self,
            adjacency: keras.KerasTensor,
            node_mask: Optional[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]:
        """CLS-augment the rank-3 adjacency and the rank-2 node mask (``N -> N+1``).

        The CLS row/column are all-ones (CLS is fully connected); PAD exclusion still comes
        from the rank-2 ``node_mask``, which the block ANDs into the attention keep and into
        the Hopfield energy. The CLS slot is marked valid (``1``) in the node mask. Graph-safe.
        """
        batch = ops.shape(adjacency)[0]
        n = ops.shape(adjacency)[1]

        ones_col = ops.ones((batch, n, 1), dtype=adjacency.dtype)          # (B, N, 1)
        adj = ops.concatenate([ones_col, adjacency], axis=2)               # (B, N, N+1)
        ones_row = ops.ones((batch, 1, n + 1), dtype=adjacency.dtype)      # (B, 1, N+1)
        adj = ops.concatenate([ones_row, adj], axis=1)                     # (B, N+1, N+1)

        if node_mask is not None:
            ones_cls = ops.ones((batch, 1), dtype=node_mask.dtype)         # (B, 1)
            node_mask = ops.concatenate([ones_cls, node_mask], axis=1)     # (B, N+1)

        return adj, node_mask

    # -----------------------------------------------------------------

    def call(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Standard stacked forward (variant C): embed -> ``num_blocks`` ET blocks.

        :param inputs: The graph dict; reads ``node_features``/``pe``/``adjacency``/``node_mask``.
        :param training: Keras training flag.
        :return: ``(B, N', D)`` final token states. The head reads the CLS token ``x[:, 0]``.
        """
        x = self.embed(inputs, training=training)                          # (B, N', D)

        adjacency = inputs["adjacency"]
        node_mask = inputs.get("node_mask", None)
        if self.use_cls:
            adjacency, node_mask = self._augment_cls_masks(adjacency, node_mask)

        # D-002: run the block in its variable dtype (float32 under mixed_float16), never
        # fp16 -- EnergyLayerNorm's backward overflows fp16 under XLA. See decisions.md.
        block_dtype = self.blocks[0].compute_dtype
        x = ops.cast(x, block_dtype)
        adj = ops.cast(adjacency, block_dtype)                             # rank-3 keep mask

        # Static Python `for` over a fixed number of blocks — graph-safe.
        for i in range(self.num_blocks):
            x = self.blocks[i](x, attention_mask=adj, mask=node_mask, training=training)

        return ops.cast(x, self.compute_dtype)

    # -----------------------------------------------------------------

    def descend_capture(
            self,
            tokens: keras.KerasTensor,
            adjacency_mask: keras.KerasTensor,
            node_mask_2d: Optional[keras.KerasTensor],
            capture_steps,
            training: Optional[bool] = None,
    ) -> Dict[int, keras.KerasTensor]:
        """Variant B: run ``blocks[0]``'s descent manually and capture LayerNormed states.

        # DECISION plan-2026-07-15T015824-3c2159eb/D-002: call the block's public
        # .norm / .attention.update / .hopfield.update, never its internal descent math,
        # so a future block fix propagates for free. See decisions.md.

        This replicates ``EnergyTransformer.call``'s descent loop through the block's
        public methods only, so the block's own source stays unchanged::

            W_hat = block._weighted_adjacency(x0, adj)          # None unless flag on
            g     = block.norm(x)
            upd   = block.attention.update(g, attention_mask=adj, mask=node_mask,
                                           adjacency_weight=W_hat)
                    + block.hopfield.update(g, mask=node_mask)
            x     = x + block.step_size * upd

        After each step ``t`` in ``capture_steps`` the LayerNormed state
        ``g_t = block.norm(x_t)`` is recorded. ``noise_std`` stays 0 here (variant B is
        deterministic); enabling it would diverge from the block's own noisy ``call()``.

        Runs in the block's variable dtype (the sub-layers were built there); captured
        states are cast back to ``self.compute_dtype``.

        :param tokens: ``(B, N, D)`` embedded tokens (typically :meth:`embed`'s output).
        :param adjacency_mask: rank-3 ``(B, N, N)`` binary adjacency (the ``attention_mask``).
        :param node_mask_2d: rank-2 ``(B, N)`` per-node validity mask (or ``None``).
        :param capture_steps: iterable of 1-based step indices ``t`` to record (e.g. ``{1, T}``).
        :param training: Keras training flag (unused at ``noise_std=0`` — the descent is
            deterministic).
        :return: ``{t: (B, N, D) g_t}`` for every requested ``t`` (in the compute dtype).
        """
        block = self.blocks[0]
        capture = set(int(t) for t in capture_steps)

        x = ops.cast(tokens, block.compute_dtype)
        captured: Dict[int, keras.KerasTensor] = {}

        # DECISION plan-2026-07-15T053724-78001af1/D-003: hoist Ŵ once from the input
        # tokens, not per step from the evolving g -- a per-step recompute adds a dŴ/dg
        # term the closed form does not carry. See decisions.md.
        adjacency_weight = block._weighted_adjacency(x, adjacency_mask)

        # t in capture is a trace-time Python test over the step set, not a tensor value.
        for t in range(1, self.num_steps + 1):
            g = block.norm(x)
            update = (
                block.attention.update(
                    g, attention_mask=adjacency_mask, mask=node_mask_2d,
                    adjacency_weight=adjacency_weight,
                )
                + block.hopfield.update(g, mask=node_mask_2d)
            )
            x = x + block.step_size * update
            if t in capture:
                captured[t] = ops.cast(block.norm(x), self.compute_dtype)

        return captured

    # -----------------------------------------------------------------

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape ``(batch, N', embed_dim)`` from stored config — valid UNBUILT."""
        nf_shape: Any = None
        if isinstance(input_shape, dict):
            nf_shape = input_shape.get("node_features")
        elif (isinstance(input_shape, (list, tuple)) and len(input_shape) > 0
              and isinstance(input_shape[0], (list, tuple))):
            nf_shape = input_shape[0]
        else:
            nf_shape = input_shape

        batch: Optional[int] = None
        n: Optional[int] = None
        if isinstance(nf_shape, (list, tuple)) and len(nf_shape) >= 2:
            batch, n = nf_shape[0], nf_shape[1]
        if self.use_cls and n is not None:
            n = n + 1
        return (batch, n, self.embed_dim)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "node_feature_dim": self.node_feature_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "hopfield_dim": self.hopfield_dim,
            "num_blocks": self.num_blocks,
            "num_steps": self.num_steps,
            "step_size": self.step_size,
            "beta": self.beta,
            "hopfield_activation": self.hopfield_activation,
            "hopfield_beta": self.hopfield_beta,
            "noise_std": self.noise_std,
            "norm_epsilon": self.norm_epsilon,
            "attn_self": self.attn_self,
            "use_weighted_adjacency": self.use_weighted_adjacency,
            "adjacency_kernel_size": self.adjacency_kernel_size,
            "adjacency_proj_dim": self.adjacency_proj_dim,
            "use_pe": self.use_pe,
            "pe_dim": self.pe_dim,
            "use_cls": self.use_cls,
            "seed": self.seed,
        })
        return config


# ---------------------------------------------------------------------
# Variant B — GraphAnomalyDetector head
# ---------------------------------------------------------------------


def _coerce_graph_backbone(backbone: Any) -> GraphEnergyTransformerBackbone:
    """Accept a live backbone or its serialized config dict (the ``from_config`` path)."""
    if isinstance(backbone, GraphEnergyTransformerBackbone):
        return backbone
    if isinstance(backbone, dict):
        obj = deserialize_keras_object(backbone)
        if not isinstance(obj, GraphEnergyTransformerBackbone):
            raise TypeError(
                f"Deserialized backbone is a {type(obj).__name__}, expected "
                "GraphEnergyTransformerBackbone"
            )
        return obj
    raise TypeError(
        "backbone must be a GraphEnergyTransformerBackbone (or its serialized config dict), "
        f"got {type(backbone).__name__}"
    )


@register_dl_technique("dl_techniques.models.graph_energy_transformer.model")
class GraphAnomalyDetector(keras.Model):
    """Variant B (node anomaly): shared graph trunk, then a target-node
    ``g_1 || g_T`` readout through a 2-layer MLP.

    Architecture:

    .. code-block:: text

        graph dict (1 block, T steps)
                │
                ▼
        ┌────────────────────┐
        │ backbone.embed +   │  descend_capture() -> {1: g_1, T: g_T}
        │ descend_capture    │
        └─────────┬───────────┘
                  ▼
        target node g_1[0], g_T[0]  (static index-0 slice)
                  │
        ┌─────────┴─────────┐
        │ ln1(g_1)  lnT(g_T) │  normed separately, then concatenated
        └─────────┬─────────┘
                  ▼ [B, 2D]
        ┌────────────────────┐
        │ Dense(gelu) + drop  │
        │ Dense(1)            │
        └─────────┬───────────┘
                  ▼
             logit [B, 1]

    A single backbone block descends for ``T`` steps; the head reads the
    target node's LayerNormed state at the first step (``g_1``) and the last
    (``g_T``) and concatenates them, following the paper. ``g_1`` still
    carries the raw one-hop neighbourhood signal, and ``g_T`` the settled
    attractor, so the pair is a richer readout than either alone -- the
    reason :meth:`~GraphEnergyTransformerBackbone.descend_capture` exists.

    The fraud subgraph sampler always places the target node at index 0, so
    the readout is a static ``g[:, 0, :]`` slice rather than a runtime
    gather; a runtime-broadcast gather does not compile under
    ``jit_compile=True``. ``target_index`` stays in the input contract for
    forward compatibility but is ignored. The head emits a logit with no
    sigmoid applied; compile with ``BinaryCrossentropy(from_logits=True)``.
    The backbone has no ``return_energy`` flag -- its blocks always run with
    ``return_energy=False`` -- so there is no energy trace for this head to guard against.

    :param backbone: A :class:`GraphEnergyTransformerBackbone` (typically ``num_blocks=1``,
        ``use_cls=False``, ``use_pe=False``, ``noise_std=0.0``), or its serialized config dict.
    :type backbone: GraphEnergyTransformerBackbone
    :param mlp_hidden_dim: Hidden width of the readout MLP. Must be positive.
    :type mlp_hidden_dim: int
    :param mlp_dropout_rate: Dropout between the MLP's two Dense layers. Defaults to ``0.0``.
    :type mlp_dropout_rate: float

    :raises ValueError: If ``mlp_hidden_dim <= 0`` or ``mlp_dropout_rate`` is outside ``[0, 1]``.

    Input shape:
        The variant-B graph dict ``{"node_features": (B, N, F), "adjacency": (B, N, N),
        "node_mask": (B, N), "target_index": (B,)}``. ``target_index`` is accepted for
        forward compatibility and is ignored: the readout is a static ``g[:, 0, :]``
        slice, so the target node must be at index 0, which the fraud subgraph sampler
        guarantees. Passing any other value changes nothing. See D-003.

    Output shape:
        ``(batch, 1)`` — a single anomaly logit per target node.
    """

    def __init__(
            self,
            backbone: GraphEnergyTransformerBackbone,
            mlp_hidden_dim: int,
            mlp_dropout_rate: float = 0.0,
            name: Optional[str] = "graph_anomaly_detector",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        backbone = _coerce_graph_backbone(backbone)

        if not isinstance(mlp_hidden_dim, int) or mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be a positive integer, got {mlp_hidden_dim}")
        if not (0.0 <= mlp_dropout_rate <= 1.0):
            raise ValueError(f"mlp_dropout_rate must be in [0, 1], got {mlp_dropout_rate}")

        self.backbone = backbone
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.mlp_dropout_rate = float(mlp_dropout_rate)
        self.embed_dim = backbone.embed_dim

        # `head_` prefix so a B-pretext -> B warm-start (name-matched) moves only the
        # shared trunk and never the head. Built in build(), not lazily -- a lazily-created
        # sub-layer silently drops its weights on a `.keras` round-trip.
        # g_1 and g_T get separate norm layers (paper: "both layernormed"): the first-step
        # and converged states have different statistics.
        self.ln1 = layers.LayerNormalization(
            epsilon=1e-6, name="head_ln1", dtype=self.dtype_policy
        )
        self.lnT = layers.LayerNormalization(
            epsilon=1e-6, name="head_lnT", dtype=self.dtype_policy
        )
        self.mlp_hidden = layers.Dense(
            self.mlp_hidden_dim, activation="gelu", name="head_mlp_hidden",
            dtype=self.dtype_policy,
        )
        # Created at every dropout rate, including 0, so the layer structure does not
        # depend on the numeric value.
        self.head_dropout = layers.Dropout(
            self.mlp_dropout_rate, name="head_mlp_dropout", dtype=self.dtype_policy
        )
        self.mlp_out = layers.Dense(1, name="head_mlp_out", dtype=self.dtype_policy)

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Explicitly build the trunk and every head sub-layer from stored config."""
        if self.built:
            return
        self.backbone.build(input_shape)

        target_shape = (None, self.embed_dim)          # one target node's state (B, D)
        concat_shape = (None, 2 * self.embed_dim)       # g_1 || g_T  (B, 2D)
        hidden_shape = (None, self.mlp_hidden_dim)      # MLP hidden  (B, H)

        self.ln1.build(target_shape)
        self.lnT.build(target_shape)
        self.mlp_hidden.build(concat_shape)
        self.head_dropout.build(hidden_shape)
        self.mlp_out.build(hidden_shape)
        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Embed -> capture ``g_1``/``g_T`` -> target-node ``g_1 || g_T`` -> MLP logit.

        :param inputs: The variant-B graph dict (see the class docstring).
        :param training: Keras training flag.
        :return: ``(B, 1)`` anomaly logits (no sigmoid — ``from_logits`` loss).
        """
        x0 = self.backbone.embed(inputs, training=training)            # (B, N, D)

        t_last = self.backbone.num_steps
        caps = self.backbone.descend_capture(
            x0,
            adjacency_mask=inputs["adjacency"],
            node_mask_2d=inputs.get("node_mask", None),
            capture_steps={1, t_last},
            training=training,
        )                                                              # {1: g_1, T: g_T}

        # DECISION plan-2026-07-15T015824-3c2159eb/D-003: static g[:, 0, :] readout, not
        # a dynamic take_along_axis gather -- tf2xla rejects a runtime-broadcast index. See decisions.md.
        g1_t = caps[1][:, 0, :]                                       # (B, D)
        gT_t = caps[t_last][:, 0, :]                                  # (B, D)

        # LayerNorm each separately, then concat (paper: "both layernormed").
        g1_n = self.ln1(g1_t, training=training)                      # (B, D)
        gT_n = self.lnT(gT_t, training=training)                      # (B, D)
        h = ops.concatenate([g1_n, gT_n], axis=-1)                    # (B, 2D)

        h = self.mlp_hidden(h, training=training)                     # (B, H) gelu
        h = self.head_dropout(h, training=training)
        return self.mlp_out(h)                                        # (B, 1) logit — no sigmoid

    # -----------------------------------------------------------------

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], 1)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "mlp_dropout_rate": self.mlp_dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphAnomalyDetector":
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------
# Variant C-lite — GraphClassifier head
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.graph_energy_transformer.model")
class GraphClassifier(keras.Model):
    """Variant C-lite (graph classification): shared graph trunk, then a
    CLS-token readout mapped to class logits.

    Architecture:

    .. code-block:: text

        graph dict (S blocks, CLS + Laplacian PE + Langevin noise)
                │
                ▼
        ┌────────────────────┐
        │ backbone.call()    │  CLS prepended, adjacency/mask CLS-augmented,
        │ [B, N+1, D]         │  all inside the backbone
        └─────────┬───────────┘
                  ▼
        CLS token tokens[:, 0]  (static index-0 slice)
                  │
        ┌─────────┴─────────┐
        │ LayerNorm + Dropout │
        │ Dense(num_classes)  │
        └─────────┬───────────┘
                  ▼
             logits [B, C]

    The default is the binary-adjacency "C-lite" form; the paper's eq.-25
    learned per-edge weighted adjacency is available opt-in through
    ``use_weighted_adjacency=True``. A stack of ``S``
    :class:`GraphEnergyTransformerBackbone` blocks (``num_blocks=S=4``, Table
    9) descends the graph tokens with a prepended learnable CLS token,
    Laplacian positional encodings, and eq.-27 saddle-escape Langevin noise
    (training only). The graph-level representation is the final CLS token.

    The CLS token, its mask augmentation and the PE add all live in the
    backbone, not this head: :meth:`GraphEnergyTransformerBackbone.embed`
    prepends the CLS token after the PE add (so CLS gets no PE), and
    :meth:`~GraphEnergyTransformerBackbone.call` CLS-augments the rank-3
    adjacency and rank-2 node mask, leaving the original adjacency's
    ``[1:, 1:]`` block unchanged. This head only reads the CLS token at a
    static ``tokens[:, 0, :]`` slice, which keeps it compilable under
    ``jit_compile=True`` (a runtime-broadcast gather would not be). The head
    emits logits with no softmax applied.

    :param backbone: A :class:`GraphEnergyTransformerBackbone` (typically ``num_blocks=4``,
        ``use_cls=True``, ``use_pe=True``, ``noise_std=0.02``), or its serialized config dict.
    :type backbone: GraphEnergyTransformerBackbone
    :param num_classes: Number of graph classes ``C``. Must be ``>= 2``.
    :type num_classes: int
    :param head_dropout_rate: Dropout on the CLS representation before the classifier Dense. Defaults
        to ``0.0``.
    :type head_dropout_rate: float

    :raises ValueError: If ``num_classes < 2`` or ``head_dropout_rate`` is outside ``[0, 1]``.

    Input shape:
        The variant-C graph dict ``{"node_features": (B, N, F), "adjacency": (B, N, N),
        "pe": (B, N, pe_dim), "node_mask": (B, N)}``.

    Output shape:
        ``(batch, num_classes)`` — per-graph class logits (no softmax).
    """

    def __init__(
            self,
            backbone: GraphEnergyTransformerBackbone,
            num_classes: int,
            head_dropout_rate: float = 0.0,
            name: Optional[str] = "graph_classifier",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        backbone = _coerce_graph_backbone(backbone)

        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes must be an integer >= 2, got {num_classes}")
        if not (0.0 <= head_dropout_rate <= 1.0):
            raise ValueError(f"head_dropout_rate must be in [0, 1], got {head_dropout_rate}")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.head_dropout_rate = float(head_dropout_rate)
        self.embed_dim = backbone.embed_dim

        # `head_` prefix so a warm-start (name-matched) moves only the shared trunk and
        # never the head. Built in build(), not lazily -- a lazily-created sub-layer
        # silently drops its weights on a `.keras` round-trip.
        self.head_norm = layers.LayerNormalization(
            epsilon=1e-6, name="head_norm", dtype=self.dtype_policy
        )
        # Created at every dropout rate, including 0, so the layer structure does not
        # depend on the numeric value.
        self.head_dropout = layers.Dropout(
            self.head_dropout_rate, name="head_dropout", dtype=self.dtype_policy
        )
        self.head_dense = layers.Dense(
            self.num_classes, name="head_dense", dtype=self.dtype_policy
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Explicitly build the trunk and every head sub-layer from stored config."""
        if self.built:
            return
        self.backbone.build(input_shape)

        cls_shape = (None, self.embed_dim)              # CLS token state (B, D)
        self.head_norm.build(cls_shape)
        self.head_dropout.build(cls_shape)
        self.head_dense.build(cls_shape)
        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Stacked backbone forward -> CLS token -> LayerNorm -> Dropout -> Dense logits.

        :param inputs: The variant-C graph dict (see the class docstring).
        :param training: Keras training flag. eq.-27 saddle-escape noise inside the backbone is
            active ONLY when ``training`` is true, so two inference forwards are deterministic.
        :return: ``(B, num_classes)`` class logits (no softmax — ``from_logits`` loss).
        """
        tokens = self.backbone(inputs, training=training)             # (B, N+1, D)

        # STATIC index-0 CLS slice — the backbone prepends the CLS token first, so index 0 is
        # always the CLS token. XLA-safe (unlike variant B's dynamic take_along_axis gather).
        cls = tokens[:, 0, :]                                         # (B, D)

        h = self.head_norm(cls, training=training)                   # (B, D)
        h = self.head_dropout(h, training=training)                  # (B, D)
        return self.head_dense(h, training=training)                 # (B, C) logits — no softmax

    # -----------------------------------------------------------------

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        token_shape = self.backbone.compute_output_shape(input_shape)
        return (token_shape[0], self.num_classes)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "backbone": serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "head_dropout_rate": self.head_dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphClassifier":
        config = dict(config)
        config["backbone"] = deserialize_keras_object(config["backbone"])
        return cls(**config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_graph_energy_transformer_backbone(
        node_feature_dim: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        hopfield_dim: int,
        **overrides: Any,
) -> GraphEnergyTransformerBackbone:
    """Create a standalone :class:`GraphEnergyTransformerBackbone`.

    :param node_feature_dim: Input node-feature dim ``F``.
    :param embed_dim: Token dim ``D``.
    :param num_heads: Attention heads ``H``.
    :param head_dim: Per-head key/query dim ``Y``.
    :param hopfield_dim: Hopfield memory count ``K``.
    :param overrides: Any ctor kwarg (``num_blocks``, ``num_steps``, ``use_pe``, ``use_cls``,
        ``noise_std``, ...).
    :return: The backbone, named :data:`GRAPH_BACKBONE_NAME`.
    """
    return GraphEnergyTransformerBackbone(
        node_feature_dim=node_feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        hopfield_dim=hopfield_dim,
        name=GRAPH_BACKBONE_NAME,
        **overrides,
    )


def create_graph_anomaly_detector(
        node_feature_dim: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        hopfield_dim: int,
        mlp_hidden_dim: int,
        num_steps: int = 12,
        mlp_dropout_rate: float = 0.0,
        **overrides: Any,
) -> GraphAnomalyDetector:
    """Create a variant-B :class:`GraphAnomalyDetector` (backbone + target-node readout head).

    Builds the shared graph trunk with the variant-B topology fixed — ``num_blocks=1``,
    ``use_cls=False``, ``use_pe=False``, ``noise_std=0.0`` (so the head's manual
    :meth:`~GraphEnergyTransformerBackbone.descend_capture` matches the block's own noiseless
    descent) — and names it :data:`GRAPH_BACKBONE_NAME` so a future B-pretext -> B warm-start
    name-matches the trunk. Remaining backbone kwargs (``step_size``, ``beta``,
    ``hopfield_activation``, ``hopfield_beta``, ``norm_epsilon``, ``attn_self``, ``seed`` ...)
    pass through via ``overrides``.

    :param node_feature_dim: Input node-feature dim ``F``.
    :param embed_dim: Token dim ``D``.
    :param num_heads: Attention heads ``H``.
    :param head_dim: Per-head key/query dim ``Y``.
    :param hopfield_dim: Hopfield memory count ``K``.
    :param mlp_hidden_dim: Hidden width of the readout MLP.
    :param num_steps: Descent steps ``T`` (the head captures ``g_1`` and ``g_T``).
    :param mlp_dropout_rate: Dropout between the MLP's two Dense layers.
    :param overrides: Any other backbone ctor kwarg (``step_size``, ``beta``,
        ``hopfield_activation``, ``hopfield_beta``, ``norm_epsilon``, ``attn_self``, ``seed``).
    :return: A :class:`GraphAnomalyDetector` whose trunk is named :data:`GRAPH_BACKBONE_NAME`.

    Example:
        >>> model = create_graph_anomaly_detector(
        ...     node_feature_dim=25, embed_dim=64, num_heads=4, head_dim=16,
        ...     hopfield_dim=128, mlp_hidden_dim=64, num_steps=12,
        ... )
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.BinaryCrossentropy(from_logits=True),
        ... )
    """
    backbone = GraphEnergyTransformerBackbone(
        node_feature_dim=node_feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        hopfield_dim=hopfield_dim,
        num_blocks=1,
        num_steps=num_steps,
        use_pe=False,
        use_cls=False,
        noise_std=0.0,
        name=GRAPH_BACKBONE_NAME,
        **overrides,
    )
    return GraphAnomalyDetector(
        backbone=backbone,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout_rate=mlp_dropout_rate,
    )


def create_graph_classifier(
        node_feature_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 12,
        head_dim: int = 64,
        hopfield_dim: int = 512,
        num_blocks: int = 4,
        num_steps: int = 12,
        step_size: float = 0.01,
        beta: Optional[float] = None,
        noise_std: float = 0.02,
        pe_dim: int = 15,
        head_dropout_rate: float = 0.0,
        use_weighted_adjacency: bool = False,
        adjacency_kernel_size: int = 1,
        adjacency_proj_dim: Optional[int] = None,
        **overrides: Any,
) -> GraphClassifier:
    """Create a variant-C-lite :class:`GraphClassifier` (backbone + CLS-token readout head).

    Builds the shared graph trunk with the variant-C topology fixed — ``use_cls=True``,
    ``use_pe=True`` — and the paper's Table 9 defaults (``embed_dim=128``, ``num_heads=12``,
    ``head_dim=64``, ``hopfield_dim=512``, ``num_blocks=S=4``, ``step_size=0.01``,
    ``beta=None`` -> ``1/sqrt(head_dim)``, ``noise_std=0.02`` eq.-27 saddle-escape, ``pe_dim=15``
    Laplacian PE). The trunk is named :data:`GRAPH_BACKBONE_NAME` so a future warm-start
    name-matches it. Remaining backbone kwargs (``hopfield_activation``, ``hopfield_beta``,
    ``norm_epsilon``, ``attn_self``, ``seed`` ...) pass through via ``overrides``.

    :param node_feature_dim: Input node-feature dim ``F``.
    :param num_classes: Number of graph classes ``C`` (``>= 2``).
    :param embed_dim: Token dim ``D`` (Table 9: 128).
    :param num_heads: Attention heads ``H`` (Table 9: 12).
    :param head_dim: Per-head key/query dim ``Y`` (Table 9: 64).
    :param hopfield_dim: Hopfield memory count ``K`` (Table 9: 512).
    :param num_blocks: Number of stacked ET blocks ``S`` (Table 9: 4).
    :param num_steps: Descent steps ``T`` per block.
    :param step_size: Descent step ``alpha`` (Table 9: 0.01).
    :param beta: Attention inverse temperature; ``None`` -> ``1/sqrt(head_dim)``.
    :param noise_std: eq.-27 Langevin saddle-escape noise std (training only; Table 9: 0.02).
    :param pe_dim: Laplacian-PE width ``k`` (matches the dataset's ``k_pe``; default 15).
    :param head_dropout_rate: Dropout on the CLS representation before the classifier Dense.
    :param use_weighted_adjacency: If ``True``, each ET block learns the paper's eq.-25 per-edge
        weighted adjacency ``Ŵ`` (default ``False`` -> the C-lite binary-adjacency model,
        byte-identical to today).
    :param adjacency_kernel_size: Conv2D kernel of the ``Ŵ`` projector (default ``1``; only used
        when ``use_weighted_adjacency``).
    :param adjacency_proj_dim: Optional ``X⊗X`` bottleneck ``P`` for the projector; ``None``
        (default) uses the full ``D²`` pairing. Set small (8-16) to avoid the ``D²``-channel
        memory blow-up at ``D=128`` (only used when ``use_weighted_adjacency``).
    :param overrides: Any other backbone ctor kwarg (``hopfield_activation``, ``hopfield_beta``,
        ``norm_epsilon``, ``attn_self``, ``seed``).
    :return: A :class:`GraphClassifier` whose trunk is named :data:`GRAPH_BACKBONE_NAME`.

    Example:
        >>> model = create_graph_classifier(node_feature_dim=7, num_classes=2)
        >>> model.compile(
        ...     optimizer='adamw',
        ...     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ... )
    """
    backbone = GraphEnergyTransformerBackbone(
        node_feature_dim=node_feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        hopfield_dim=hopfield_dim,
        num_blocks=num_blocks,
        num_steps=num_steps,
        step_size=step_size,
        beta=beta,
        noise_std=noise_std,
        use_weighted_adjacency=use_weighted_adjacency,
        adjacency_kernel_size=adjacency_kernel_size,
        adjacency_proj_dim=adjacency_proj_dim,
        use_pe=True,
        pe_dim=pe_dim,
        use_cls=True,
        name=GRAPH_BACKBONE_NAME,
        **overrides,
    )
    return GraphClassifier(
        backbone=backbone,
        num_classes=num_classes,
        head_dropout_rate=head_dropout_rate,
    )
