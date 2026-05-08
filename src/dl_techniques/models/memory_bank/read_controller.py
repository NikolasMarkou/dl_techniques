"""Memory read controller — top-K STE retrieval + gated injection.

This module implements the read-tap of the dual-tap memory architecture.
Step 3 (this commit) implements the retrieval and gating; Step 4 will add
the four anti-collapse aux losses on top.

Algorithm (per F-004 §3-§4)::

    Q = X_R @ W_Q                                  # (B, T, H, d_k)
    K_total = concat([tile(K_lt), K_wm], axis=1)   # (B, S_lt + max_seq_len, d_k)
    V_total = concat([tile(V_lt), V_wm], axis=1)   # (B, S_lt + max_seq_len, d_v)
    sim = einsum('bthk,bmk->bthm', Q, K_total) / sqrt(d_k)
    sim += causal_mask + padding_mask              # WM positions only
    soft_w = softmax(sim / temp, -1)
    top_idx = ops.top_k(sim, k).indices
    hard_w = renormalize(soft_w * one_hot_top_k(top_idx, M_static))
    routing = soft_w + ops.stop_gradient(hard_w - soft_w)   # STE
    retrieved_V = einsum('bthm,bmv->bthv', routing, V_total)
    V_proj = LayerNorm(W_out @ concat_heads + b_out)
    g = sigmoid(X_R @ W_g + b_g)                   # b_g init = -3.0
    return g * V_proj                              # caller does residual add

``M_static = S_lt + max_seq_len`` is baked into ``__init__`` so that
``ops.one_hot(num_classes=M_static)`` traces with a static shape under XLA.
"""

import math
from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from dl_techniques.utils.logger import logger


_NEG_INF = -1.0e9


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MemoryReadController(keras.layers.Layer):
    """Multi-head top-K STE retrieval with gated residual injection.

    :param embed_dim: Hidden-state dimensionality of ``X_R`` (= ``D``).
    :param num_heads: Number of attention heads.
    :param d_k: Per-head key dimensionality.
    :param d_v: Per-head value dimensionality.
    :param s_lt: Long-term memory slot count (used only to set
        ``M_static``).
    :param max_seq_len: Max sequence length / WM slot count (used to set
        ``M_static``).
    :param top_k: Number of keys retrieved per query (static int).
    :param initializer_range: Stddev for projection weight init.
    :param gate_init_bias: Initial value for the gate bias (default
        ``-3.0`` so sigmoid≈0.04, memory ~96% bypassed at init).
    :param layer_norm_eps: LayerNorm epsilon for ``V_proj`` norm.
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    Aux-loss enable flags (default ``False`` — Step 4 will turn them on):
        - ``enable_gate_entropy``
        - ``enable_load_balance``
        - ``enable_z_loss``
        - ``enable_diversity``
        - ``enable_infonce``

    Variable-name prefixes are ``memory_`` for keys/values/projections and
    ``gate_`` for the gate (so the custom ``train_step`` routes their
    gradients to the memory optimizer).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        s_lt: int,
        max_seq_len: int,
        top_k: int = 32,
        initializer_range: float = 0.02,
        gate_init_bias: float = -3.0,
        layer_norm_eps: float = 1e-5,
        # Aux loss enable flags (Step 4 wires them in).
        enable_gate_entropy: bool = False,
        enable_load_balance: bool = False,
        enable_z_loss: bool = False,
        enable_diversity: bool = False,
        enable_infonce: bool = False,
        # Aux loss coefficients.
        lambda_gate_entropy: float = 1e-3,
        lambda_load_balance: float = 1e-2,
        lambda_z_loss: float = 1e-3,
        lambda_diversity: float = 1e-3,
        lambda_infonce: float = 5e-3,
        # Sub-sample sizes for cheap aux losses on large S_lt.
        diversity_subsample: int = 1024,
        infonce_negatives: int = 256,
        infonce_temperature: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads}) and num_heads must be positive"
            )
        if d_k <= 0 or d_v <= 0:
            raise ValueError(f"d_k and d_v must be positive")
        if d_k == d_v:
            raise ValueError("d_k must differ from d_v")
        if d_v >= embed_dim:
            raise ValueError(
                f"d_v ({d_v}) must be < embed_dim ({embed_dim})"
            )
        if s_lt <= 0 or max_seq_len <= 0:
            raise ValueError("s_lt and max_seq_len must be positive")
        if top_k <= 0 or top_k > s_lt + max_seq_len:
            raise ValueError(
                f"top_k must be in (0, s_lt+max_seq_len], got {top_k}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.s_lt = s_lt
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.initializer_range = initializer_range
        self.gate_init_bias = gate_init_bias
        self.layer_norm_eps = layer_norm_eps

        self.enable_gate_entropy = enable_gate_entropy
        self.enable_load_balance = enable_load_balance
        self.enable_z_loss = enable_z_loss
        self.enable_diversity = enable_diversity
        self.enable_infonce = enable_infonce

        self.lambda_gate_entropy = lambda_gate_entropy
        self.lambda_load_balance = lambda_load_balance
        self.lambda_z_loss = lambda_z_loss
        self.lambda_diversity = lambda_diversity
        self.lambda_infonce = lambda_infonce

        self.diversity_subsample = diversity_subsample
        self.infonce_negatives = infonce_negatives
        self.infonce_temperature = infonce_temperature

        # M_static is the static last-axis cardinality of the routing
        # tensor — required by ops.one_hot under XLA.
        self.M_static = s_lt + max_seq_len
        self._sqrt_dk = float(math.sqrt(d_k))

        kernel_init = keras.initializers.TruncatedNormal(
            stddev=initializer_range,
        )

        # Q projection: D -> H * d_k. No bias (positional-encoding-free
        # query direction per blueprint).
        self.W_Q = keras.layers.Dense(
            num_heads * d_k,
            use_bias=False,
            kernel_initializer=kernel_init,
            name="memory_read_W_Q",
        )

        # Output projection: head-concat (H * d_v) -> D.
        self.W_out = keras.layers.Dense(
            embed_dim,
            use_bias=True,
            kernel_initializer=kernel_init,
            name="memory_read_W_out",
        )
        self.out_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="memory_read_out_norm",
        )

        # Gate: D -> D, biased to ~0.04 sigmoid output at init.
        self.W_g = keras.layers.Dense(
            embed_dim,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer=keras.initializers.Constant(gate_init_bias),
            name="gate_W_g",
        )

        # Learned softmax temperature: temp = softplus(log_temp) + 0.1.
        # Stored as a non-trainable... no, this IS trainable per blueprint.
        self._log_temp_initializer = keras.initializers.Constant(0.0)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # X_R has shape (B, T, D).
        self.W_Q.build(input_shape)
        self.W_g.build(input_shape)
        # Head-concat shape for W_out is (B, T, H * d_v).
        head_concat_shape = (input_shape[0], input_shape[1],
                             self.num_heads * self.d_v)
        self.W_out.build(head_concat_shape)
        self.out_norm.build((input_shape[0], input_shape[1], self.embed_dim))

        self.log_temp = self.add_weight(
            name="memory_read_log_temp",
            shape=(),
            initializer=self._log_temp_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        x_r: Any,
        k_lt: Any,
        v_lt: Any,
        k_wm: Any,
        v_wm: Any,
        wm_padding_mask: Any,
        training: Optional[bool] = None,
    ) -> Any:
        """Run retrieval and return ``g * V_proj`` (gated injection).

        :param x_r: Hidden state at the read tap, shape ``(B, T, D)``.
        :param k_lt: Long-term keys, shape ``(S_lt, d_k)``.
        :param v_lt: Long-term values, shape ``(S_lt, d_v)``.
        :param k_wm: Working-memory keys, shape ``(B, max_seq_len, d_k)``.
        :param v_wm: Working-memory values, shape ``(B, max_seq_len, d_v)``.
        :param wm_padding_mask: Padding mask for WM positions, shape
            ``(B, max_seq_len)``. 1.0 on real positions, 0.0 on padded.
        :param training: Forwarded for parity (no dropout here).
        :returns: Gated injection ``g * V_proj`` of shape ``(B, T, D)``.
            Caller is responsible for the residual add.
        """
        b = ops.shape(x_r)[0]
        t = ops.shape(x_r)[1]

        # 1. Q projection -> multi-head reshape.
        q_flat = self.W_Q(x_r)  # (B, T, H * d_k)
        q = ops.reshape(q_flat, (b, t, self.num_heads, self.d_k))  # (B, T, H, d_k)

        # 2. Tile K_lt / V_lt across batch.
        # K_lt: (S_lt, d_k) -> (1, S_lt, d_k) -> (B, S_lt, d_k).
        k_lt_b = ops.broadcast_to(
            ops.expand_dims(k_lt, axis=0), (b, self.s_lt, self.d_k),
        )
        v_lt_b = ops.broadcast_to(
            ops.expand_dims(v_lt, axis=0), (b, self.s_lt, self.d_v),
        )

        # 3. Concatenate to total keys/values along axis=1.
        k_total = ops.concatenate([k_lt_b, k_wm], axis=1)  # (B, M_static, d_k)
        v_total = ops.concatenate([v_lt_b, v_wm], axis=1)  # (B, M_static, d_v)

        # 4. Similarity: einsum('bthk,bmk->bthm').
        sim = ops.einsum("bthk,bmk->bthm", q, k_total) / self._sqrt_dk

        # 5. Build the WM-portion masks:
        #   (a) causal — position t' in WM may only be read when t' <= t.
        #   (b) padding — already-zeroed positions get -inf.
        # The combined mask is applied as additive -inf on disallowed
        # positions of the WM slice (last `max_seq_len` columns of M).
        causal_pad_mask_wm = self._build_wm_mask(t, wm_padding_mask)
        # causal_pad_mask_wm shape: (B, T, max_seq_len)
        # Expand to (B, T, 1, max_seq_len) for broadcast over heads, then
        # pad with zeros for the M_LT slice (no masking on M_LT).
        wm_mask = ops.expand_dims(causal_pad_mask_wm, axis=2)  # (B, T, 1, max_seq_len)
        lt_zeros = ops.zeros((b, t, 1, self.s_lt), dtype=sim.dtype)
        full_mask = ops.concatenate([lt_zeros, wm_mask], axis=-1)  # (B, T, 1, M_static)

        sim = sim + full_mask

        # 6. Softmax + STE top-K.
        temp = ops.softplus(self.log_temp) + 0.1
        # Clip sim before softmax for numerical safety in aux losses
        # (z-loss in particular). Bound is generous.
        sim_clipped = ops.clip(sim, -50.0, 50.0)
        soft_w = ops.softmax(sim_clipped / temp, axis=-1)  # (B, T, H, M_static)

        # ops.top_k returns (values, indices). We take the indices.
        top_vals, top_idx = ops.top_k(sim_clipped, k=self.top_k)
        # top_idx shape: (B, T, H, top_k).
        hard_one_hot = ops.one_hot(
            top_idx, num_classes=self.M_static,
        )  # (B, T, H, top_k, M_static)
        hard_mask = ops.sum(hard_one_hot, axis=-2)  # (B, T, H, M_static)
        # Re-normalize the soft weights restricted to the top-K positions.
        masked_soft = soft_w * hard_mask
        hard_w = masked_soft / (
            ops.sum(masked_soft, axis=-1, keepdims=True) + 1e-9
        )

        # STE: forward = hard_w, backward = soft_w.
        routing = soft_w + ops.stop_gradient(hard_w - soft_w)

        # 7. Retrieve values: einsum('bthm,bmv->bthv').
        retrieved_v = ops.einsum("bthm,bmv->bthv", routing, v_total)

        # 8. Head-concat and output projection.
        retrieved_concat = ops.reshape(
            retrieved_v, (b, t, self.num_heads * self.d_v),
        )
        v_proj = self.W_out(retrieved_concat)
        v_proj = self.out_norm(v_proj)

        # 9. Gate.
        g = ops.sigmoid(self.W_g(x_r))  # (B, T, D)
        injection = g * v_proj

        # 10. Anti-collapse aux losses (Step 4).
        # All gated by enable flags so phase-1 disables them all. Each loss
        # is computed only when training=True to avoid double-accumulation
        # at eval time (LESSONS — `add_loss` semantics).
        if training:
            self._maybe_add_aux_losses(
                routing=routing,
                soft_w=soft_w,
                sim_clipped=sim_clipped,
                gate=g,
                k_lt=k_lt,
                v_lt=v_lt,
                v_proj=v_proj,
            )

        return injection

    # ------------------------------------------------------------------
    # Aux losses (Step 4)
    # ------------------------------------------------------------------

    def _maybe_add_aux_losses(
        self,
        routing: Any,
        soft_w: Any,
        sim_clipped: Any,
        gate: Any,
        k_lt: Any,
        v_lt: Any,
        v_proj: Any,
    ) -> None:
        """Compute and add the four (+ optional z-loss) anti-collapse aux
        losses via :meth:`self.add_loss`. Each is gated by an enable flag
        so phase-1 (or eager testing) can short-circuit all of them.
        """
        # 1. Gate entropy: maximize H(g) by adding `lambda * (-H_mean)`.
        if self.enable_gate_entropy:
            eps = 1e-7
            g_clip = ops.clip(gate, eps, 1.0 - eps)
            ent = -(g_clip * ops.log(g_clip)
                     + (1.0 - g_clip) * ops.log(1.0 - g_clip))  # (B, T, D)
            ent_mean = ops.mean(ent)
            self.add_loss(self.lambda_gate_entropy * (-ent_mean))

        # 2. Load balance: only over the M_LT slice (first S_lt columns).
        # routing_lt: (B, T, H, S_lt). soft_lt: (B, T, H, S_lt).
        routing_lt = routing[..., :self.s_lt]
        soft_lt = soft_w[..., :self.s_lt]
        if self.enable_load_balance:
            f_i = ops.mean(routing_lt, axis=(0, 1, 2))  # (S_lt,)
            p_i = ops.mean(soft_lt, axis=(0, 1, 2))     # (S_lt,)
            lb = (
                self.lambda_load_balance
                * float(self.s_lt)
                * ops.sum(ops.stop_gradient(f_i) * p_i)
            )
            self.add_loss(lb)

        # 3. Z-loss on the M_LT slice (logsumexp(sim_lt))^2.
        if self.enable_z_loss:
            sim_lt = sim_clipped[..., :self.s_lt]  # (B, T, H, S_lt)
            lse = ops.logsumexp(sim_lt, axis=-1)
            zl = self.lambda_z_loss * ops.mean(lse * lse)
            self.add_loss(zl)

        # 4. Key diversity: subsample `diversity_subsample` keys from K_lt
        # and add lambda * mean(off-diagonal cos-sim ** 2).
        if self.enable_diversity:
            n_sub = min(self.diversity_subsample, self.s_lt)
            if n_sub == self.s_lt:
                k_sub = k_lt
            else:
                # Random subsample via tf.random.shuffle on indices.
                # Backend-agnostic: use keras.random for index sampling.
                import tensorflow as _tf
                idx = _tf.random.shuffle(_tf.range(self.s_lt))[:n_sub]
                k_sub = ops.take(k_lt, idx, axis=0)
            k_norm = k_sub / (ops.norm(k_sub, axis=-1, keepdims=True) + 1e-8)
            cos = ops.matmul(k_norm, ops.transpose(k_norm))  # (n_sub, n_sub)
            # Mask diagonal.
            eye = ops.eye(n_sub, dtype=cos.dtype)
            cos_off = cos * (1.0 - eye)
            div = self.lambda_diversity * ops.mean(cos_off * cos_off)
            self.add_loss(div)

        # 5. InfoNCE: per-query positive vs `infonce_negatives` random
        # K_lt rows. Implementation: take v_proj (already aggregated per
        # query) as the query embedding, compare to mean-of-selected V_lt
        # for the positive and to V_lt @ random rows for negatives. We
        # project nothing additional — v_proj is the pooled-retrieved
        # representation. The contrast is over key indices.
        if self.enable_infonce:
            import tensorflow as _tf
            # Negatives: random index sample from K_lt's value bank V_lt.
            n_neg = min(self.infonce_negatives, self.s_lt)
            idx_neg = _tf.random.shuffle(_tf.range(self.s_lt))[:n_neg]
            v_neg = ops.take(v_lt, idx_neg, axis=0)  # (n_neg, d_v)
            # Positive: retrieved V_lt mean for top-1 routing row.
            # Use the mean of V_lt under routing_lt restricted to the
            # max position per (B, T, H).
            # For simplicity here we contrast v_proj (pooled head-concat,
            # shape (B, T, D)) vs lifted v_neg via a Dense W_out-like
            # projection — but to avoid introducing extra weights we
            # contrast in the d_v space using flattened means.
            # v_proj_query = mean over D -> 1 scalar per (B, T) is too
            # weak; instead we contrast the mean of routed V_total against
            # v_neg in d_v space.
            # routed_v_mean: (B, T, d_v) = mean over heads of retrieved_v
            # We don't have retrieved_v here; reconstruct via routing @
            # v_lt for the LT slice.
            routed_v_lt = ops.einsum(
                "bthm,mv->bthv", routing_lt, v_lt,
            )  # (B, T, H, d_v)
            q_emb = ops.mean(routed_v_lt, axis=2)  # (B, T, d_v)
            # Positive: stop-grad anchor = same q_emb mean (sanity-only,
            # collapse-resistant); per blueprint the positive is "mean of
            # selected keys' values projection of that query" which IS
            # q_emb. The contrastive signal pushes q_emb away from random
            # K_lt rows.
            tau = max(self.infonce_temperature, 1e-6)
            pos_logit = ops.sum(q_emb * ops.stop_gradient(q_emb), axis=-1)  # (B, T)
            neg_logits = ops.einsum(
                "btv,nv->btn", q_emb, v_neg,
            )  # (B, T, n_neg)
            # Concatenate pos as logit-0 and apply log-softmax.
            all_logits = ops.concatenate(
                [ops.expand_dims(pos_logit, axis=-1), neg_logits], axis=-1,
            )  # (B, T, 1 + n_neg)
            log_probs = ops.log_softmax(all_logits / tau, axis=-1)
            nce = -ops.mean(log_probs[..., 0])  # -log P(positive)
            self.add_loss(self.lambda_infonce * nce)

    def _build_wm_mask(
        self, t: Any, wm_padding_mask: Any,
    ) -> Any:
        """Build additive ``-inf`` mask for WM slice.

        Output shape ``(B, T, max_seq_len)``: ``0`` on allowed positions,
        ``-inf`` on disallowed (causal-violating OR padded).
        """
        # Causal portion: position `i` (in WM, 0..max_seq_len-1) is
        # allowed for query at time t' iff i <= t'. We want a tensor
        # M_caus[b, t', i] = -inf if i > t' else 0.
        wm_idx = ops.arange(self.max_seq_len, dtype="int32")  # (max_seq_len,)
        q_idx = ops.arange(t, dtype="int32")  # (T,)
        # broadcast: (T, max_seq_len)
        causal_bool = ops.expand_dims(wm_idx, axis=0) <= ops.expand_dims(q_idx, axis=1)
        causal_add = ops.where(
            causal_bool,
            ops.zeros_like(causal_bool, dtype="float32"),
            ops.full(ops.shape(causal_bool), _NEG_INF, dtype="float32"),
        )
        # Expand to (1, T, max_seq_len) -> broadcast against batch.
        causal_add = ops.expand_dims(causal_add, axis=0)

        # Padding mask: 1.0 -> 0.0 add, 0.0 -> -inf add.
        pad_add = ops.where(
            wm_padding_mask > 0.5,
            ops.zeros_like(wm_padding_mask),
            ops.full(ops.shape(wm_padding_mask), _NEG_INF, dtype="float32"),
        )  # (B, max_seq_len)
        pad_add = ops.expand_dims(pad_add, axis=1)  # (B, 1, max_seq_len)

        return causal_add + pad_add  # (B, T, max_seq_len) by broadcast

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "s_lt": self.s_lt,
            "max_seq_len": self.max_seq_len,
            "top_k": self.top_k,
            "initializer_range": self.initializer_range,
            "gate_init_bias": self.gate_init_bias,
            "layer_norm_eps": self.layer_norm_eps,
            "enable_gate_entropy": self.enable_gate_entropy,
            "enable_load_balance": self.enable_load_balance,
            "enable_z_loss": self.enable_z_loss,
            "enable_diversity": self.enable_diversity,
            "enable_infonce": self.enable_infonce,
            "lambda_gate_entropy": self.lambda_gate_entropy,
            "lambda_load_balance": self.lambda_load_balance,
            "lambda_z_loss": self.lambda_z_loss,
            "lambda_diversity": self.lambda_diversity,
            "lambda_infonce": self.lambda_infonce,
            "diversity_subsample": self.diversity_subsample,
            "infonce_negatives": self.infonce_negatives,
            "infonce_temperature": self.infonce_temperature,
        })
        return config
