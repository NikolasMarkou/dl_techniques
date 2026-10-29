"""
The recursive reasoning core of the Tiny Recursive Model (TRM): a
hierarchical latent state refined by repeated passes of a small
transformer stack, with the number of passes decided at run time rather
than fixed by the model's depth.

A conventional transformer adds capacity only by adding layers, so every
input pays for the worst case. TRM instead applies one small stack many
times to a persistent state: the same weights are re-entered, and a
halting head decides per sample how many times. The state has two halves
on different schedules: `z_L` is refreshed against the token embeddings
every pass and does the fast, local work; `z_H` only ever sees the
refreshed `z_L`, so it accumulates across passes instead of being
rewritten. Gradients are cut at each step boundary (`call` returns its
carry through `stop_gradient`), so training covers one step at a time
rather than the full trajectory, and the halting head reads a (halt,
continue) logit pair from position 0 of `z_H`.

`HierarchicalReasoningModule` (shared with HRM, configured here with biases,
Glorot-uniform kernels and a float FFN expansion) is the raw computation: a
stack of `TransformerLayer` instances applied to an injected state, used for
both halves. `TRMInner` orchestrates embeddings, the `z_L`-then-`z_H` update order, the
prediction head and the halting head, and owns the learnable initial
states `H_init` and `L_init`. Positional information enters only through
attention, so the attention type must carry RoPE; the default is
`'group_query'` with `num_kv_heads == num_heads`, plain multi-head
attention that also carries RoPE.

References:
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (the TRM architecture this module implements)
    - Wang et al., 2025. Hierarchical Reasoning Model. (the two-timescale
      `z_H` / `z_L` latent split and the one-step gradient approximation)
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.reasoning.hrm_reasoning_module import (
    HierarchicalReasoningModule
)
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    NormalizationType,
    NormalizationPositionType
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.trm.components")
class TRMInner(keras.layers.Layer):
    """
    One step of TRM's hierarchical reasoning, plus the ACT halting signal.

    Encapsulates a single pass of the outer Adaptive Computation Time loop: the
    low-level state ``z_L`` is refreshed against the token embeddings, the
    high-level state ``z_H`` is then refreshed against that updated ``z_L``, and
    two heads read ``z_H`` -- ``lm_head`` for prediction logits over the token
    positions, ``q_head`` for the ``(halt, continue)`` logit pair the outer loop
    uses to decide whether this sample is done.

    The order matters: ``z_H`` never sees the tokens directly, only the freshly
    updated ``z_L``, so it accumulates across steps rather than being rewritten
    by each one. The returned carry passes through ``stop_gradient``, so
    backpropagation covers one step and stops -- memory per step is independent
    of trajectory length, at the cost of a one-step-truncated gradient. The
    initial states ``H_init`` and ``L_init`` are learnable weights owned by this
    layer, so a trajectory begins where the model chose to begin it.

    Architecture:

    .. code-block:: text

        ┌──────────────┐   ┌─────────────────────┐   ┌──────────────┐
        │ carry['z_L'] │   │  data['inputs']     │   │ carry['z_H'] │
        │ [B, S+P, H]  │   │  [B, S]  token ids  │   │ [B, S+P, H]  │
        └──────┬───────┘   └──────────┬──────────┘   └──────┬───────┘
               │                      ▼                     │
               │        ┌──────────────────────────┐        │
               │        │  token_embedding         │        │
               │        └──────────┬───────────────┘        │
               │                   ▼                        │
               │        ┌──────────────────────────┐        │
               │        │  zero-pad P on the left  │        │
               │        │  → [B, S+P, H]           │        │
               │        └──────────┬───────────────┘        │
               │                   │ injection              │
               ▼                   ▼                        │
        ┌──────────────────────────────────────┐            │
        │  L_level (reasoning module)          │            │
        │  l_layers blocks                     │            │
        └──────────────────┬───────────────────┘            │
                           │ z_L (updated)                  │
                           │  ── injection ──┐              ▼
                           │                 ▼   ┌──────────────────┐
                           │      ┌──────────────────────────────────┐
                           │      │  H_level (reasoning module)      │
                           │      │  h_layers blocks                 │
                           │      └──────────────┬───────────────────┘
                           │                     │ z_H (updated)
                           │        ┌────────────┼────────────┐
                           │        ▼            ▼            ▼
                           │  ┌───────────┐ ┌─────────┐ ┌───────────┐
                           │  │ lm_head   │ │ q_head  │ │stop_grad  │
                           │  │ [:, P:]   │ │ z_H[:,0]│ │           │
                           │  └─────┬─────┘ └────┬────┘ └─────┬─────┘
                           └────────┼────────────┼────────────┤
                                    ▼            ▼            ▼
                              logits        (q_halt,      new_carry
                            [B, S, V]      q_continue)   {z_H, z_L}
                                            each [B]      (detached)

        B = batch, S = seq_len, P = puzzle_emb_len, H = hidden_size, V = vocab_size

    Data flow:

    .. code-block:: text

        1. emb   = pad_left(token_emb(inputs), P)
        2. z_L   = L_level(z_L, emb)
        3. z_H   = H_level(z_H, z_L)
        4. logits            = lm_head(z_H)[:, P:]
           q_halt, q_continue = q_head(z_H[:, 0])
        5. carry = stop_gradient(z_H), stop_gradient(z_L)

    :param vocab_size: Size of the vocabulary. Positive.
    :type vocab_size: int
    :param hidden_size: Dimensionality of all hidden states. Positive, and
        divisible by ``num_heads``.
    :type hidden_size: int
    :param num_heads: Number of attention heads in the transformer layers.
        Positive.
    :type num_heads: int
    :param expansion: FFN width multiplier for both reasoning modules.
    :type expansion: float
    :param seq_len: Length of the input token sequence.
    :type seq_len: int
    :param puzzle_emb_len: Length of the puzzle embedding prefix. The latent
        states are ``seq_len + puzzle_emb_len`` long; ``lm_head`` output is
        sliced back to the token positions. Defaults to ``16``.
    :type puzzle_emb_len: int
    :param h_layers: Number of transformer layers in the high-level module. At
        least 1. Defaults to ``2``.
    :type h_layers: int
    :param l_layers: Number of transformer layers in the low-level module. At
        least 1. Defaults to ``2``.
    :type l_layers: int
    :param rope_theta: RoPE base frequency. Defaults to ``10000.0``.
    :type rope_theta: float
    :param attention_type: Attention mechanism identifier. Defaults to
        ``'group_query'``; ``'multi_head'`` carries no RoPE. See the anchored
        decision in ``__init__``.
    :type attention_type: AttentionType
    :param ffn_type: Feed-forward network identifier. Defaults to ``'swiglu'``.
    :type ffn_type: FFNType
    :param normalization_type: Normalization layer identifier. Defaults to
        ``'rms_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'``. Defaults to
        ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param dropout_rate: General dropout rate. Defaults to ``0.0``.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-specific dropout rate. Defaults to
        ``0.0``.
    :type attention_dropout_rate: float
    :param kwargs: Additional keyword arguments for the ``keras.layers.Layer``
        base class.

    :raises ValueError: If ``vocab_size``, ``hidden_size`` or ``num_heads`` is
        non-positive, if ``hidden_size`` is not divisible by ``num_heads``, or if
        ``h_layers`` or ``l_layers`` is less than 1.

    Input shape:
        A ``(carry, data)`` pair of dicts. ``carry['z_H']`` and ``carry['z_L']``
        are ``(batch, seq_len + puzzle_emb_len, hidden_size)``;
        ``data['inputs']`` is ``(batch, seq_len)`` of token ids.

    Output shape:
        - ``new_carry``: dict of two tensors,
          ``(batch, seq_len + puzzle_emb_len, hidden_size)``, detached.
        - ``logits``: ``(batch, seq_len, vocab_size)``.
        - ``(q_halt, q_continue)``: two tensors of shape ``(batch,)``.

    Example:
        >>> inner = TRMInner(vocab_size=512, hidden_size=256, num_heads=8,
        ...                  expansion=4.0, seq_len=81)
        >>> inner.build(None)
        >>> batch = 2
        >>> z = keras.ops.repeat(inner.H_init, batch, axis=0)
        >>> carry = {"z_H": z, "z_L": keras.ops.repeat(inner.L_init, batch, axis=0)}
        >>> carry, logits, (q_halt, q_continue) = inner(carry, {"inputs": tokens})

    Note:
        This is a composite layer that also owns weights of its own. ``build``
        is responsible both for building every sub-layer and for creating
        ``H_init`` / ``L_init``, which is what keeps the layer fully
        serializable.

    Attributes:
        token_emb: Embedding layer for the input token ids.
        H_level: ``HierarchicalReasoningModule`` performing the high-level state update.
        L_level: ``HierarchicalReasoningModule`` performing the low-level state update.
        lm_head: Dense head producing prediction logits.
        q_head: Dense head producing the ACT halting logits.
        H_init: Learnable initial value of ``z_H``.
        L_init: Learnable initial value of ``z_L``.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        puzzle_emb_len: int = 16,
        h_layers: int = 2,
        l_layers: int = 2,
        rope_theta: float = 10000.0,
        # DECISION plan-2026-08-17T183311-79c63e38/D-007: 'group_query', not
        # 'multi_head': RoPE is only reachable through attention, and a stack
        # without it is exactly permutation-equivariant.
        attention_type: AttentionType = 'group_query',
        ffn_type: FFNType = 'swiglu',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the embeddings, modules and heads.

        Sub-layers are instantiated here and remain unbuilt until :meth:`build`,
        which also creates ``H_init`` and ``L_init``. See the class docstring for
        the full parameter reference.
        """
        super().__init__(**kwargs)

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError(
                f"hidden_size and num_heads must be positive, got "
                f"hidden_size={hidden_size}, num_heads={num_heads}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if h_layers < 1 or l_layers < 1:
            raise ValueError(
                f"h_layers and l_layers must be >= 1, got "
                f"h_layers={h_layers}, l_layers={l_layers}"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.puzzle_emb_len = puzzle_emb_len
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.rope_theta = rope_theta
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name="token_embedding"
        )
        # The two halves are the shared HierarchicalReasoningModule, configured
        # to TRM's math: biased projections and the TransformerLayer default
        # Glorot-uniform kernels, a float FFN expansion, attention dropout set
        # independently of dropout_rate, and RoPE tables covering the puzzle
        # prefix.
        self.H_level = self._make_level(h_layers, "H_level")
        self.L_level = self._make_level(l_layers, "L_level")
        self.lm_head = keras.layers.Dense(
            vocab_size, use_bias=False, name="lm_head"
        )
        self.q_head = keras.layers.Dense(2, name="q_head")

    def _make_level(self, num_layers: int, name: str) -> HierarchicalReasoningModule:
        """Create one reasoning half as a ``HierarchicalReasoningModule``.

        :param num_layers: Number of transformer layers in the half.
        :type num_layers: int
        :param name: Layer name, ``'H_level'`` or ``'L_level'``.
        :type name: str
        :return: The unbuilt reasoning module.
        :rtype: HierarchicalReasoningModule
        """
        return HierarchicalReasoningModule(
            num_layers=num_layers,
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            ffn_expansion_factor=self.expansion,
            attention_type=self.attention_type,
            max_seq_len=self.seq_len + self.puzzle_emb_len,
            rope_theta=self.rope_theta,
            normalization_type=self.normalization_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name=name,
        )

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build every sub-layer and create the learnable initial states.

        Two jobs, both required for serializability. Each composite sub-layer is
        built explicitly so its weights exist before restoration, and this
        layer's own ``H_init`` / ``L_init`` weights -- the starting values of the
        ``z_H`` and ``z_L`` trajectories -- are created here.

        :param input_shape: Unused; every shape is derived from the stored
            configuration, so the layer can be built with ``None``.
        :type input_shape: Optional[Any]
        """
        full_seq_len = self.seq_len + self.puzzle_emb_len
        latent_shape = (None, full_seq_len, self.hidden_size)

        if not self.token_emb.built:
            self.token_emb.build((None, self.seq_len))
        if not self.H_level.built:
            self.H_level.build([latent_shape, latent_shape])
        if not self.L_level.built:
            self.L_level.build([latent_shape, latent_shape])
        if not self.lm_head.built:
            self.lm_head.build(latent_shape)
        if not self.q_head.built:
            self.q_head.build((None, self.hidden_size))

        self.H_init = self.add_weight(
            name="H_init",
            shape=(1, full_seq_len, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.L_init = self.add_weight(
            name="L_init",
            shape=(1, full_seq_len, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(
            self,
            carry: Dict[str, keras.KerasTensor],
            data: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Perform a single inner reasoning step.

        Embeddings are left-padded by ``puzzle_emb_len`` zeros so they align with
        the latent states, ``z_L`` is updated before ``z_H``, and the carry is
        detached on the way out so the ACT loop is not unrolled by
        backpropagation.

        :param carry: Recurrent latent states from the previous step, with keys
            ``'z_H'`` and ``'z_L'``, each
            ``(batch, seq_len + puzzle_emb_len, hidden_size)``.
        :type carry: Dict[str, keras.KerasTensor]
        :param data: Input data for this step, with key ``'inputs'`` holding
            token ids of shape ``(batch, seq_len)``.
        :type data: Dict[str, keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: ``(new_carry, logits, (q_halt, q_continue))`` -- the updated
            states with gradients stopped, prediction logits of shape
            ``(batch, seq_len, vocab_size)``, and the ACT halting and
            continuation logits, each of shape ``(batch,)``.
        :rtype: Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        z_H = carry["z_H"]
        z_L = carry["z_L"]

        input_emb = self.token_emb(data["inputs"])

        batch_size = keras.ops.shape(input_emb)[0]
        puzzle_emb_padding = keras.ops.zeros(
            (batch_size, self.puzzle_emb_len, self.hidden_size),
            dtype=input_emb.dtype
        )
        input_emb_padded = keras.ops.concatenate([puzzle_emb_padding, input_emb], axis=1)

        z_L = self.L_level([z_L, input_emb_padded], training=training)
        z_H = self.H_level([z_H, z_L], training=training)

        new_carry = {
            "z_H": keras.ops.stop_gradient(z_H),
            "z_L": keras.ops.stop_gradient(z_L)
        }

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0])
        q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]

        return new_carry, logits, (q_halt, q_continue)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[
        Dict[str, Tuple[Optional[int], ...]],
        Tuple[Optional[int], ...],
        Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]],
    ]:
        """Return the output shapes of one inner reasoning step.

        Mirrors :meth:`call`'s return structure, derived purely from stored
        config, so it works before ``build``. The batch dimension is left
        dynamic.

        :param input_shape: Structure of the ``(carry, data)`` inputs. Unused:
            the output shapes are fully determined by the layer config.
        :type input_shape: Any
        :return: ``(new_carry_shape, logits_shape, (q_halt_shape,
            q_continue_shape))``, where ``new_carry_shape`` is
            ``{"z_H": (B, full_seq, hidden), "z_L": (B, full_seq, hidden)}``,
            ``logits_shape`` is ``(B, seq_len, vocab_size)``, and each q-head
            shape is ``(B,)``.
        :rtype: Tuple[Dict[str, Tuple[Optional[int], ...]], Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]
        """
        full_seq_len = self.seq_len + self.puzzle_emb_len
        latent_shape = (None, full_seq_len, self.hidden_size)
        new_carry_shape = {"z_H": latent_shape, "z_L": latent_shape}
        logits_shape = (None, self.seq_len, self.vocab_size)
        q_shape = (None,)
        return new_carry_shape, logits_shape, (q_shape, q_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Every ``__init__`` parameter is included, so both reasoning modules and
        both heads are reconstructed from config alone.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'expansion': self.expansion,
            'seq_len': self.seq_len,
            'puzzle_emb_len': self.puzzle_emb_len,
            'h_layers': self.h_layers,
            'l_layers': self.l_layers,
            'rope_theta': self.rope_theta,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
        })
        return config

# ---------------------------------------------------------------------