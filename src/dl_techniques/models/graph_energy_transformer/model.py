"""
Graph Energy Transformer — shared backbone (and, in later steps, its two heads).

This module gives the ``EnergyTransformer`` block
(``layers/transformers/energy_transformer.py``, arXiv:2302.07253) a GRAPH-domain trunk,
the direct analog of ``models/energy_transformer/model.py``'s image backbone. It is the
shared trunk consumed by BOTH graph heads:

* variant **B** — ``GraphAnomalyDetector`` (node anomaly; single block, ``descend_capture``),
* variant **C-lite** — ``GraphClassifier`` (graph classification; ``S`` stacked blocks, a CLS
  token, Laplacian positional encodings and eq.-27 saddle-escape noise).

Only :class:`GraphEnergyTransformerBackbone` lives here for now; steps 5/7 add the two heads
to this same file, so the seams below are intentional.

**THE GRAPH ADJACENCY IS THE RANK-3 ``attention_mask``.** A binary ``(B, N, N)`` adjacency is
fed to each block as its rank-3 ``attention_mask`` (a KEY x QUERY keep). PAD nodes are excluded
from ``E_HN`` (and from attention) via the rank-2 ``(B, N)`` ``node_mask``. There is NO new
gradient and NO ``layers/`` change — the block already supports exactly this masking
(``EnergyTransformer.call`` masking semantics; D-001/D-002 of this plan).

**THE fp16/XLA DTYPE FIX IS REPLICATED VERBATIM FROM THE IMAGE BACKBONE (D-010/D-011).** Each
block is built with ``dtype=self.dtype_policy.variable_dtype`` (NOT ``self.dtype_policy``) and
:meth:`call` casts tokens IN to the block's variable dtype and back OUT. This is the fix for
``EnergyLayerNorm``'s fp16 backward overflow under XLA, which SILENTLY freezes training. It is
NOT fixed at the layer source — the consumer must do it, exactly as here.

References:
    - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253 (§4, graph model).
"""

import keras
from keras import layers, ops
from typing import Any, Dict, List, Optional, Tuple, Union

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

# ---------------------------------------------------------------------
# Stable names
# ---------------------------------------------------------------------

# The stable sub-model name. A warm-start matches layers BY NAME, so any two graph models that
# share this trunk MUST name their backbone identically or the transfer moves zero layers.
GRAPH_BACKBONE_NAME = "graph_et_backbone"


@keras.saving.register_keras_serializable()
class GraphEnergyTransformerBackbone(keras.Model):
    """Shared Graph Energy Transformer trunk: node-project -> [PE] -> [mask token] ->
    [CLS] -> ``num_blocks`` ET blocks.

    **Intent**: give the ``EnergyTransformer`` block a single, separately-checkpointable GRAPH
    trunk that BOTH graph heads compose under the same name. Variant B (node anomaly) runs a
    SINGLE block and reads its per-step LayerNormed states via :meth:`descend_capture`; variant
    C-lite (graph classification) runs ``S`` stacked blocks via the standard :meth:`call` and
    reads the CLS token.

    **Call signature** — the backbone consumes a DICT of graph tensors::

        {
          "node_features": (B, N, F),     # required — raw per-node features
          "adjacency":     (B, N, N),     # required by `call()` — binary graph adjacency
          "node_mask":     (B, N),        # rank-2 per-node validity (1 = real, 0 = PAD)
          "pe":            (B, N, pe_dim), # only when `use_pe` (Laplacian eigenvectors)
          "node_replace_mask": (B, N) bool # optional — masked-node pretext (mask-token apply)
          "target_index":  (B,)            # variant B only; read by the head, not the trunk
        }

    ``embed`` reads ``node_features`` / ``pe`` / ``node_replace_mask``; ``call`` additionally
    reads ``adjacency`` / ``node_mask`` (and CLS-augments both when ``use_cls``).

    **Three public methods (the seam for the two heads).**

    * :meth:`embed` -> ``(B, N', D)`` embedded tokens (node projection, optional PE add,
      optional mask-token apply, optional CLS prepend). Used by BOTH heads.
    * :meth:`call` -> ``(B, N', D)`` — the STANDARD stacked forward (variant C): embed, then
      ``num_blocks`` blocks each fed the (CLS-augmented) rank-3 adjacency + rank-2 node mask.
    * :meth:`descend_capture` -> ``dict[int, (B, N', D)]`` — variant B: run the single block's
      descent MANUALLY through its PUBLIC ``.norm`` / ``.attention.update`` / ``.hopfield.update``
      and capture the LayerNormed state ``g_t`` after selected steps. NO copy of block internals.

    **``MaskTokenApply`` is ALWAYS created and ALWAYS built** (guide §9 "ALWAYS CREATE /
    CONDITIONALLY USE"), even for a head that never masks nodes — so the trunk weight surface
    does not depend on whether a masked-node pretext is used.

    **Weight-compatibility caveat.** A ``use_pe=True`` backbone (variant C) owns a ``pe_proj``
    Dense and a ``cls_token`` that a ``use_pe=False`` backbone (variant B) does NOT. The two are
    INTENTIONALLY NOT weight-compatible — B and C are different models, and a cross-variant
    warm-start is expected to transfer nothing (different ``N`` too, via the CLS token).

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
        descent guarantee — **variant B MUST keep this ``0.0``** so the manual
        :meth:`descend_capture` loop matches the block's own noiseless descent.
    :param norm_epsilon: ``epsilon`` of each block's inner ``EnergyLayerNorm``.
    :param attn_self: ``False`` (default) is the paper's ET-Full (a token does not attend to
        itself).
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
            attn_self: bool = False,
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

        # ----- store ALL configuration (serialization contract) -----
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
        self.attn_self = bool(attn_self)
        self.use_pe = bool(use_pe)
        self.pe_dim = int(pe_dim)
        self.use_cls = bool(use_cls)
        self.seed = seed

        # ----- CREATE all sub-layers in __init__ (unbuilt), never in build()/call() -----
        # A lazily-created sub-layer is not tracked at save time and SILENTLY DROPS ITS
        # WEIGHTS on a `.keras` round-trip (MEMORY: subclassed lazy-build serialization).
        self.node_proj = layers.Dense(
            self.embed_dim, name="node_proj", dtype=self.dtype_policy
        )

        # ALWAYS CREATE / CONDITIONALLY USE (guide §9): the mask token exists on every trunk,
        # even one that never masks a node, so the weight surface is use-independent.
        self.mask_token = MaskTokenApply(name="node_mask_token", dtype=self.dtype_policy)

        # PE projection: only when use_pe. B (use_pe=False) and C (use_pe=True) are therefore
        # NOT weight-compatible by design (documented in the class docstring).
        self.pe_proj = (
            layers.Dense(self.embed_dim, name="pe_proj", dtype=self.dtype_policy)
            if self.use_pe else None
        )

        # cls_token is a raw learnable weight -> created in build() (add_weight's home), only
        # when use_cls. See build().

        # DECISION plan-2026-07-15T015824-3c2159eb/D-002
        # Each block is built with `dtype=self.dtype_policy.variable_dtype`, NOT
        # `self.dtype_policy`. Under `mixed_float16` this runs the ET block in float32 (its
        # VARIABLE dtype) rather than float16; `call()`/`descend_capture()` cast the tokens IN
        # to that dtype and back OUT. Under float32/float64 the two spellings are identical
        # (compute == variable), so nothing outside a mixed policy changes and every checkpoint
        # is untouched. This is the image backbone's fix, replicated VERBATIM (D-010/D-011 of
        # plan-2026-07-14T163315-29a4fef4): `EnergyLayerNorm`'s backward forms
        # `(var + eps)^(-3/2)`, which OVERFLOWS fp16's 65504 under XLA and SILENTLY freezes
        # training (loss finite, all weights move exactly 0.0). It is NOT fixed at the layer —
        # the consumer MUST do it here.
        # WHAT NOT TO DO: (1) do NOT "simplify" this to `dtype=self.dtype_policy`; (2) do NOT
        # drop the token casts in `call()`/`descend_capture()`; (3) do NOT "fix" it instead by
        # raising `norm_epsilon` to 1e-3 (that trains a DIFFERENT network than fp32). See
        # decisions.md D-002.
        self.blocks: List[EnergyTransformer] = [
            EnergyTransformer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                hopfield_dim=self.hopfield_dim,
                num_steps=self.num_steps,
                step_size=self.step_size,
                beta=self.beta,
                attn_self=self.attn_self,
                hopfield_activation=self.hopfield_activation,
                hopfield_beta=self.hopfield_beta,
                noise_std=self.noise_std,
                return_energy=False,
                norm_epsilon=self.norm_epsilon,
                seed=self.seed,
                name=f"et_block_{i}",
                dtype=self.dtype_policy.variable_dtype,
            )
            for i in range(self.num_blocks)
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
        # ALWAYS built — even for a head that never masks a node.
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

        # D-002: run the block in its VARIABLE dtype (float32 under mixed_float16), never fp16
        # — its EnergyLayerNorm backward overflows fp16 under XLA and silently kills training.
        # Both casts are no-ops under float32/float64. DO NOT REMOVE THEM.
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
        """Variant B: run ``blocks[0]``'s descent MANUALLY and capture LayerNormed states.

        # DECISION plan-2026-07-15T015824-3c2159eb/D-002
        This replicates ``EnergyTransformer.call``'s descent loop EXACTLY, through the block's
        PUBLIC methods only — ``.norm`` / ``.attention.update`` / ``.hopfield.update`` — so the
        block's ``layers/`` source stays frozen (0 changes) and no gradient is re-derived. The
        per-step update is byte-for-byte the block's own::

            g   = block.norm(x)
            upd = block.attention.update(g, attention_mask=adj, mask=node_mask)
                  + block.hopfield.update(g, mask=node_mask)
            x   = x + block.step_size * upd

        (The block internally derives the Hopfield's per-token keep as
        ``_hopfield_token_mask(rank3 adj, node_mask)`` == ``_token_keep(node_mask)``; passing
        ``node_mask_2d`` straight to ``hopfield.update`` is identical because ``_token_keep`` is
        an idempotent cast.) After each step ``t`` in ``capture_steps`` the LayerNormed state
        ``g_t = block.norm(x_t)`` is recorded.

        WHAT NOT TO DO: (1) do NOT copy the block's internal descent math here — call its public
        methods, so a future block fix propagates for free; (2) do NOT enable ``noise_std`` for
        variant B — this noiseless manual loop would then diverge from the block's own (noisy)
        ``call()``; keep ``noise_std=0`` (default). See decisions.md D-002.

        Runs in the block's VARIABLE dtype (the sub-layers were built there); captured states
        are cast back to ``self.compute_dtype``.

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

        # Static `range` over the fixed step count — graph-safe. `t in capture` is a
        # trace-time Python test over the captured-step SET, never a test on tensor values.
        for t in range(1, self.num_steps + 1):
            g = block.norm(x)
            update = (
                block.attention.update(g, attention_mask=adjacency_mask, mask=node_mask_2d)
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
            "use_pe": self.use_pe,
            "pe_dim": self.pe_dim,
            "use_cls": self.use_cls,
            "seed": self.seed,
        })
        return config


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
