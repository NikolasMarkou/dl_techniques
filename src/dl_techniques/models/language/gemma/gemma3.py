"""
Gemma 3 decoder with interleaved sliding-window / global attention and sandwich
normalization, plus generation and classification task heads.

Gemma 3 targets inference memory rather than raw quality: attention costs
O(n^2) in time, and its key-value cache costs O(n * layers * kv_heads) in
memory, which is what fails to fit at long context. Grouped-query attention
shrinks the cache by storing only `num_key_value_heads` distinct KV heads and
repeating them at score time. Most layers also use a bounded attention window
(`sliding_window_size`), with the `layer_types` list controlling how often a
layer instead gets full, unbounded attention. A token's receptive field through
the windowed layers still grows with depth; the full-attention layers add an
exact single-hop route to any earlier position.

Each block uses sandwich normalization: RMSNorm before and after the sublayer,
`x = x + PostNorm(Attn(PreNorm(x)))`, so each branch's contribution is capped
before it joins the residual stream, while the residual path itself stays
unnormalized. Masking is built per block, since it depends on that block's
attention type, then inverted once to the attend-semantics the attention layer
expects. Token embeddings are scaled by `sqrt(hidden_size)` before the first
block to match the unit-scale activations the rest of the network expects.

This implementation omits QK normalization and uses one shared RoPE base for
every layer rather than a larger base in the global layers, so long-context
behavior differs from the published report; a reader should not assume
checkpoint compatibility. The LM head is an independent `Dense`, not the tied
embedding matrix. This package is text-only; the multimodal vision tower is
not included. `from_variant` takes no `pretrained` argument, and `create_gemma3`'s
`pretrained=True` raises `NotImplementedError`, so there is no path to a silently
random-initialized "pretrained" model; `pretrained=<path>` restores the backbone
of a local `.keras` checkpoint.

With `include_top=True` and a `task_config` the model owns ONE task head from
`layers/heads/nlp`, named `task_head`, in place of the LM head (no `lm_head`
is created and no vocabulary logits are computed), fed the final-normed
hidden states, as `models/language/gpt2` does. The head pools `last` by
default, the last position kept by the mask, because the causally masked
blocks make `cls` pool a position that attended only to itself.
`create_gemma3_classification` is that model behind a two-input functional
wrapper; it used to re-trace the backbone into a functional graph and pool
with a hand-rolled `SequencePooling` + `Dense`, so no classifier checkpoint
written earlier is portable.

References:
    - Gemma Team, 2025. Gemma 3 Technical Report.
      (https://arxiv.org/abs/2503.19786)
    - Gemma Team, 2024. Gemma 2: Improving Open Language Models at a Practical Size.
      (https://arxiv.org/abs/2408.00118)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need.
      (https://arxiv.org/abs/1911.02150)
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.
      (https://arxiv.org/abs/2104.09864)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
"""


import keras
from keras import initializers, layers, ops
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.heads.nlp.model_head import (
    head_output_shape,
    ids_shape_of,
    task_model_name,
)
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.sequence_length_guard import check_sequence_length
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.layers.transformers.gemma3_transformer import Gemma3TransformerBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.gemma.gemma3")
class Gemma3(keras.Model):
    """Gemma 3 language model with sandwich normalization and mixed attention.

    Token embeddings, scaled by sqrt(hidden_size), feed a stack of transformer
    blocks that alternate sliding-window and full attention per `layer_types`,
    then a final RMSNorm and a Dense projection to vocabulary logits. With
    ``include_top=True`` the projection is replaced by one NLP task head, fed
    ``{"hidden_states": <final-normed states>, "attention_mask": <mask>}``,
    where the mask is the input's ``attention_mask`` or, without one,
    ``input_ids != pad_token_id`` (all ones when it is None), cast to the hidden-state dtype; ``call``
    returns what the head returns, reduced by
    :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`.

    Architecture:

        .. code-block:: text

            input_ids [B, S]
                 │
                 ▼
            Embedding * sqrt(hidden_size)
                 │
                 ▼
            ┌────────────────────────────┐
            │ Gemma3TransformerBlock x N │  sandwich norm, per-layer
            │ (sliding_window|full)      │  attention_type from layer_types
            └────────────────────────────┘
                 │
                 ▼
            final_norm (RMSNorm)
                 │
                 ▼
            lm_head (Dense, untied)
                 │
                 ▼
            logits [B, S, vocab_size]

            include_top=True: lm_head is replaced by
            task_head (pools 'last') ──► finalized head output (only output)

    :param vocab_size: Vocabulary size. Must be positive.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Must be positive.
    :type hidden_size: int
    :param num_layers: Number of transformer blocks. Must be positive.
    :type num_layers: int
    :param num_attention_heads: Number of attention heads.
    :type num_attention_heads: int
    :param num_key_value_heads: Number of key-value heads for GQA.
    :type num_key_value_heads: int
    :param ffn_hidden_size: FFN intermediate size. Must be positive.
    :type ffn_hidden_size: int
    :param max_seq_len: Maximum sequence length. Must be positive.
    :type max_seq_len: int
    :param sliding_window_size: Window size for local attention layers.
    :type sliding_window_size: int
    :param layer_types: Attention type per layer, each ``'sliding_window'``
        or ``'full_attention'``. Length must match ``num_layers``. ``None``
        (default) yields all ``'full_attention'``.
    :type layer_types: Optional[List[str]]
    :param norm_eps: Epsilon for normalization layers.
    :type norm_eps: float
    :param dropout_rate: Dropout rate, in [0, 1].
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear layers.
    :type use_bias: bool
    :param initializer_range: Stddev for TruncatedNormal weight initialization.
    :type initializer_range: float
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no ``attention_mask``. Read only with ``include_top=True``; the
        blocks never derive a mask from it. Default: None, which derives no
        padding (the head sees every position): id 0 is a real token, so with
        right-padded batches pass the tokenizer's pad id or an ``attention_mask``.
        An explicit int is stored and serialized as given.
    :type pad_token_id: Optional[int]
    :param include_top: Whether the model owns a task head (and no LM head).
        Default: False: no head layer and no head weight exist.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored
        as a plain dict. ``MULTIPLE_CHOICE`` is refused.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (e.g. ``{"pooling_type": "mean"}``, which replaces the
        ``'last'`` default). Only valid with ``include_top=True``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for ``keras.Model``.

    :ivar embeddings: Token embedding layer.
    :ivar blocks: List of ``Gemma3TransformerBlock`` layers.
    :ivar final_norm: Final RMSNorm layer before output projection.
    :ivar lm_head: Language modeling head (Dense layer), or None when
        ``include_top`` is True.
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.

    :raises ValueError: On an invalid configuration value, if
        ``include_top=True`` comes without a ``task_config``, or if a
        ``task_config`` or ``head_config_overrides`` comes with
        ``include_top=False``.

    Input shape:
        2D tensor of shape ``(batch_size, sequence_length)`` of token IDs.

    Output shape:
        3D tensor of shape ``(batch_size, sequence_length, vocab_size)`` of
        logits; with ``include_top=True`` the finalized head output.
    """

    # Model variant configurations following Gemma 3 specifications
    MODEL_VARIANTS = {
        "270m": {
            "vocab_size": 262144,
            "hidden_size": 640,
            "num_layers": 18,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "ffn_hidden_size": 2048,
            "max_seq_len": 32768,
            "sliding_window_size": 512,
            "layer_types": [
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": (
                "Gemma 3 270M: Original model with mixed attention patterns."
            ),
        },
        "small": {
            "vocab_size": 50000,
            "hidden_size": 512,
            "num_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "ffn_hidden_size": 1536,
            "max_seq_len": 8192,
            "sliding_window_size": 256,
            "layer_types": [
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": "Gemma 3 Small: Reduced model for experimentation.",
        },
        "tiny": {
            "vocab_size": 32000,
            "hidden_size": 384,
            "num_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "ffn_hidden_size": 1024,
            "max_seq_len": 4096,
            "sliding_window_size": 128,
            "layer_types": [
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": (
                "Gemma 3 Tiny: Minimal model for mobile/edge deployment."
            ),
        },
    }

    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_seq_len={limit}. Gemma3 "
        "precomputes its RoPE tables for {limit} positions, so there is no "
        "rotation for position {limit}. Truncate the input to {limit} "
        "tokens, or build the model with max_seq_len>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 640,
        num_layers: int = 18,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 32768,
        sliding_window_size: int = 512,
        layer_types: Optional[List[str]] = None,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        initializer_range: float = 0.02,
        pad_token_id: Optional[int] = None,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if layer_types is None:
            layer_types = ["full_attention"] * num_layers
        self._validate_config(
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads,
            ffn_hidden_size,
            max_seq_len,
            sliding_window_size,
            layer_types,
            norm_eps,
            dropout_rate,
            initializer_range,
        )
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.sliding_window_size = sliding_window_size
        self.layer_types = layer_types
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.include_top = include_top
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
                input_dim=self.hidden_size,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )
        self.emb_scale = ops.sqrt(
            ops.cast(self.hidden_size, dtype=self.compute_dtype)
        )
        self._log_model_creation()

    def _validate_config(self, *args) -> None:
        """Comprehensive model configuration parameter validation."""
        (
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, ffn_hidden_size, max_seq_len,
            sliding_window_size, layer_types, norm_eps, dropout_rate,
            initializer_range,
        ) = args
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be positive, got "
                f"{num_attention_heads}"
            )
        if num_key_value_heads <= 0:
            raise ValueError(
                "num_key_value_heads must be positive, got "
                f"{num_key_value_heads}"
            )
        if ffn_hidden_size <= 0:
            raise ValueError(
                f"ffn_hidden_size must be positive, got {ffn_hidden_size}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
            )
        if sliding_window_size <= 0:
            raise ValueError(
                "sliding_window_size must be positive, got "
                f"{sliding_window_size}"
            )
        if norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}")
        if initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {initializer_range}"
            )

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({num_key_value_heads})"
            )
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match "
                f"num_layers ({num_layers})"
            )
        if any(t not in {"sliding_window", "full_attention"} for t in layer_types):
            raise ValueError(
                "Invalid layer_type found. Must be one of 'sliding_window', "
                "'full_attention'."
            )

    def _build_architecture(self) -> None:
        """Create all model components."""
        initializer = initializers.TruncatedNormal(stddev=self.initializer_range)

        self.embeddings = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=initializer,
            name="token_embedding",
        )

        self.blocks = [
            Gemma3TransformerBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                ffn_hidden_size=self.ffn_hidden_size,
                max_seq_len=self.max_seq_len,
                attention_type=self.layer_types[i],
                sliding_window_size=self.sliding_window_size,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                norm_eps=self.norm_eps,
                kernel_initializer=initializer,
                name=f"transformer_block_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.final_norm = create_normalization_layer(
            "rms_norm", epsilon=self.norm_eps, name="final_norm"
        )
        # A task head replaces the LM head: call() never runs lm_head then,
        # and build() must not create what call() does not run.
        self.lm_head = None
        if not self.include_top:
            self.lm_head = layers.Dense(
                units=self.vocab_size,
                use_bias=self.use_bias,
                kernel_initializer=initializer,
                name="lm_head",
            )

    def _log_model_creation(self) -> None:
        """Log comprehensive model creation information."""
        sliding_count = self.layer_types.count("sliding_window")
        full_count = self.num_layers - sliding_count
        logger.info(
            f"Created Gemma3 model with {self.num_layers} transformer layers:"
        )
        logger.info(
            f"  - Mixed Attention: {sliding_count} sliding window, "
            f"{full_count} full attention"
        )
        logger.info(
            f"  - Vocabulary: {self.vocab_size:,} | "
            f"Hidden Size: {self.hidden_size} | "
            f"FFN Hidden: {self.ffn_hidden_size}"
        )
        logger.info(
            f"  - Attention: {self.num_attention_heads} heads | "
            f"GQA: {self.num_key_value_heads} KV heads"
        )
        logger.info(
            f"  - Context: {self.max_seq_len:,} tokens | "
            f"Sliding Window: {self.sliding_window_size}"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method Gemma3 inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Compute the shape of what ``call`` returns.

        Keras builds a symbolic call of this model from this method, never
        from ``call``, so it must report what ``call`` returns in both modes.
        It describes the default ``return_dict=False`` form.

        :param input_shape: Shape of ``input_ids`` ``(batch, seq_len)``, or a
            dict holding it under ``'input_ids'``.
        :type input_shape: Any
        :return: The LM logits shape ``(batch, seq_len, vocab_size)``. With a
            head, the head's output shapes after the reduction ``call`` applies:
            one shape, or a mapping of two or more.
        :rtype: Any
        """
        ids_shape = ids_shape_of(input_shape)
        if self.include_top:
            return head_output_shape(
                self.task_head, ids_shape, self.hidden_size)
        return (*ids_shape, self.vocab_size)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass of the Gemma3 model.

        :return: The LM logits ``(batch, seq_len, vocab_size)``; with
            ``include_top=True``, only the head's output reduced by
            ``finalize_head_outputs``: a bare tensor, or a dict of two or more
            tensors (``start_logits`` / ``end_logits`` for QA).
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key."
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        input_ids = check_sequence_length(
            self, input_ids, self.max_seq_len, "max_seq_len",
            self._SEQUENCE_LENGTH_MESSAGE,
        )

        hidden_states = self.embeddings(input_ids) * self.emb_scale

        for block in self.blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, training=training
            )

        hidden_states = self.final_norm(hidden_states)

        if self.include_top:
            # The blocks saw only the caller's mask; the head pools under it,
            # else under input_ids != pad_token_id (all ones if None).
            return finalize_head_outputs(self.task_head(
                {
                    "hidden_states": hidden_states,
                    "attention_mask": head_attention_mask(
                        input_ids, attention_mask, self.pad_token_id,
                        hidden_states.dtype,
                    ),
                },
                training=training,
            ))
        return self.lm_head(hidden_states)

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "Gemma3":
        """Create a Gemma3 model from a predefined variant."""
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {available}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        config.update(kwargs)

        logger.info(f"Creating Gemma3 model from variant: {variant.upper()}")
        logger.info(f"Description: {description}")
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "max_seq_len": self.max_seq_len,
                "sliding_window_size": self.sliding_window_size,
                "layer_types": self.layer_types,
                "norm_eps": self.norm_eps,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "initializer_range": self.initializer_range,
                "pad_token_id": self.pad_token_id,
                "include_top": self.include_top,
                "task_config": (
                    dict(self.task_config) if self.task_config is not None
                    else None
                ),
                "head_config_overrides": (
                    dict(self.head_config_overrides)
                    if self.head_config_overrides is not None else None
                ),
            }
        )
        return config


def _two_input_model(core: Gemma3, name: str) -> keras.Model:
    """Put the ``[input_ids, attention_mask]`` functional interface around ``core``.

    Interface contract (shared by the ``create_gemma3*`` factories):

    :param core: An unbuilt ``Gemma3``, named ``gemma3_backbone`` by every
        caller.
    :param name: Name of the returned model.
    :return: A functional ``keras.Model`` with two int32 inputs whose only
        non-input layer is ``core``. ``load_weights_from_checkpoint``
        therefore compares ``core``'s layers, so a backbone moves between such
        a model and a bare ``Gemma3`` either way.
    """
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(
        shape=(None,), dtype="int32", name="attention_mask"
    )
    outputs = core(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )
    model = keras.Model(
        inputs=[input_ids, attention_mask], outputs=outputs, name=name
    )
    logger.info(f"Created {name} with {model.count_params():,} parameters.")
    return model


def create_gemma3_generation(config: Dict[str, Any]) -> keras.Model:
    """Creates a Gemma3 model for text generation tasks."""
    logger.info("Creating Gemma3 model for text generation.")
    return _two_input_model(
        Gemma3(**config, name="gemma3_backbone"), "gemma3_for_generation")


def create_gemma3_classification(
    config: Dict[str, Any],
    num_labels: int,
    pooling_strategy: str = "last",
    classifier_dropout_rate: Optional[float] = None,
) -> keras.Model:
    """Creates a Gemma3 model for sequence classification tasks.

    ``pooling_strategy`` is one of ``"last"`` (default — the last position kept
    by ``attention_mask``, the only one that has attended to the whole sequence
    under this backbone's causal mask), ``"mean"`` (mask-aware mean) or
    ``"cls"`` (first position; a function of the first token id alone here, kept
    only for bidirectional-era checkpoints). The classifier is the model's own
    task head (``Gemma3(include_top=True)`` with a ``TEXT_CLASSIFICATION``
    task); this function only puts the two-input functional interface around
    it.
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-029: default "last", not "cls" —
    # causal masking makes 'cls' pool a position that attended only to itself. See decisions.md.
    if pooling_strategy not in ["last", "cls", "mean"]:
        raise ValueError(
            f"pooling_strategy must be 'last', 'cls' or 'mean', got "
            f"'{pooling_strategy}'"
        )

    logger.info(
        f"Creating Gemma3 classification model "
        f"(Labels: {num_labels}, Pooling: {pooling_strategy})."
    )
    classifier = Gemma3(
        **config,
        include_top=True,
        task_config=NLPTaskConfig(
            name="classification",
            task_type=NLPTaskType.TEXT_CLASSIFICATION,
            num_classes=num_labels,
            dropout_rate=(
                classifier_dropout_rate if classifier_dropout_rate is not None
                else config.get("dropout_rate", 0.0)
            ),
        ),
        head_config_overrides={"pooling_type": pooling_strategy},
        name="gemma3_backbone",
    )
    return _two_input_model(
        classifier, task_model_name("gemma3", NLPTaskType.TEXT_CLASSIFICATION))


def create_gemma3(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str = "generation",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> keras.Model:
    """High-level factory to create Gemma3 models for common tasks.

    ``task_type="generation"`` (default) gives ``gemma3_for_generation`` and
    ``"classification"`` (with ``num_labels``, optionally ``pooling_strategy``
    and ``classifier_dropout_rate``) gives ``gemma3_for_classification``. Any
    other keyword overrides the config. ``include_top=True`` with a
    ``task_config`` (and optional ``head_config_overrides``) replaces the LM
    top with that task head: the result is named
    ``gemma3_for_<task_type value>`` (e.g. ``gemma3_for_token_classification``),
    except that ``TEXT_CLASSIFICATION`` gives ``gemma3_for_classification``, the
    name :func:`create_gemma3_classification` gives the same task; combining it with ``task_type="classification"`` or the classification
    settings raises ``ValueError``. Every result takes
    ``[input_ids, attention_mask]``.

    :param pretrained: ``False`` (default) or any other falsy value: random
        initialization. A ``str``: the path of a local ``.keras`` checkpoint (of the
        ``Gemma3`` itself or of a model this factory returned, headed or headless)
        whose backbone is restored into the result by layer name. A task head in
        the checkpoint is never copied, and the result's own head (if any) keeps
        its fresh initialization. The backbone loads completely or the call raises
        (another depth or width is refused); on a vocabulary change only the token
        table and the ``lm_head`` keep their initialization. ``True`` raises
        ``NotImplementedError``: no weights are distributed with this package.
    """
    weights_path = pretrained_path_or_raise(pretrained, "Gemma3")
    model = _dispatch_gemma3(config_or_variant, task_type, dict(kwargs))
    if weights_path is not None:
        # DECISION plan-2026-10-11T170958-dc4d2202/D-015: the shared load rule on the
        # wrapper's core, never `model.load_weights` (it restores nothing on a
        # name or shape mismatch and reports success); do NOT drop TASK_HEAD_NAME
        # (a head of equal shape would be copied). See decisions.md.
        load_backbone_from_checkpoint(
            model,
            weights_path,
            head_prefixes=(TASK_HEAD_NAME,),
            vocab_layers=("token_embedding",),
            tables=("token_embedding/embeddings", "lm_head/kernel"),
            tops=("lm_head",),
        )
    return model


def _dispatch_gemma3(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str,
    kwargs: Dict[str, Any],
) -> keras.Model:
    """The task dispatch of :func:`create_gemma3`, without its `pretrained` route.

    :param config_or_variant: As for :func:`create_gemma3`.
    :param task_type: As for :func:`create_gemma3`.
    :param kwargs: Overrides and task settings; may be modified.
    :return: The two-input functional model.
    """
    if isinstance(config_or_variant, str):
        if config_or_variant not in Gemma3.MODEL_VARIANTS:
            available = list(Gemma3.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{config_or_variant}'. "
                f"Available: {available}"
            )
        config = Gemma3.MODEL_VARIANTS[config_or_variant].copy()
        config.pop("description", None)
        logger.info(f"Using Gemma3 variant: {config_or_variant}")
    elif isinstance(config_or_variant, dict):
        config = config_or_variant.copy()
        logger.info("Using custom Gemma3 configuration")
    else:
        raise TypeError("config_or_variant must be a string or a dictionary.")

    task_keys = ["num_labels", "pooling_strategy", "classifier_dropout_rate"]
    task_kwargs = {k: kwargs.pop(k) for k in task_keys if k in kwargs}
    # The rest of kwargs are model overrides.
    config.update(kwargs)

    # include_top=True replaces the LM top with the task_config's head, so the
    # model is that task's, named for it, and never "generation".
    if config.get("include_top"):
        if task_type != "generation" or task_kwargs:
            raise ValueError(
                "include_top=True builds the head named by task_config; it "
                "cannot be combined with task_type='classification' or with "
                "num_labels / pooling_strategy / classifier_dropout_rate, "
                f"which build their own head. Got task_type={task_type!r} "
                f"and {sorted(task_kwargs)}."
            )
        core = Gemma3(**config, name="gemma3_backbone")
        return _two_input_model(
            core, task_model_name("gemma3", core.task_config["task_type"]))

    if task_type == "generation":
        return create_gemma3_generation(config)
    if task_type == "classification":
        if "num_labels" not in task_kwargs:
            raise ValueError(
                "num_labels must be provided for the 'classification' task"
            )
        return create_gemma3_classification(config, **task_kwargs)

    raise ValueError(
        f"Unknown task_type '{task_type}'. "
        "Supported: ['generation', 'classification']"
    )