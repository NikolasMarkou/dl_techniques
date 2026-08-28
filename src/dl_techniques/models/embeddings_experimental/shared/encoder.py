"""The encoder skeleton every arm of the embeddings study shares.

:class:`EmbeddingEncoder` is a BERT-shaped bidirectional encoder whose
sequence-mixing block is resolved through
:data:`~...blocks.BLOCK_REGISTRY` rather than hard-coded. Everything else --
the ASCII embeddings, the depth and width ladder, the pooling, the output
contract -- is identical across arms, so a difference in a reported metric is
attributable to the block.

Relationship to ``models/language/bert``
----------------------------------------
The structure is deliberately the same, and the differences are only the ones
the study needs:

- The block is a registry lookup, not a ``TransformerLayer`` literal.
- ``call()`` additionally returns ``pooled_output``, because an embedding model
  needs one vector per sequence. Upstream BERT owns no pooler, so pooling is a
  genuinely open seam; it is filled with the existing
  :class:`~dl_techniques.layers.sequence_pooling.sequence_pooling.SequencePooling`
  layer rather than a new one.
- ``pad_token_id`` is READ here, to derive an attention mask when the caller
  supplies none. In upstream BERT the same argument is stored, serialized and
  deliberately never read (its D-007 anchor says so explicitly, pinned by a
  test). That divergence is intentional and is stated here so nobody
  "restores consistency" in either direction: this family's pooling is
  mask-dependent, so a silently-absent mask would silently pool over padding.

References:
    - Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Clark et al., 2022. CANINE: Pre-training an Efficient
      Tokenization-Free Encoder for Language Representation.
      (https://arxiv.org/abs/2103.06874)
    - Xue et al., 2022. ByT5: Towards a Token-Free Future with
      Pre-trained Byte-to-Byte Models. (https://arxiv.org/abs/2105.13626)
    - Reimers and Gurevych, 2019. Sentence-BERT: Sentence Embeddings
      using Siamese BERT-Networks. (https://arxiv.org/abs/1908.10084)
"""

from typing import Any, Dict, List, Optional

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings
from dl_techniques.layers.sequence_pooling.sequence_pooling import SequencePooling
from dl_techniques.layers.tokenizers.ascii_char import (
    PAD_ID as ASCII_PAD_ID,
    VOCAB_SIZE as ASCII_VOCAB_SIZE,
)
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.logger import logger

from .blocks import create_encoder_block

# ---------------------------------------------------------------------

__all__ = ["EmbeddingEncoder"]

#: Defaults shared by every arm, so the ladder is comparable across blocks.
DEFAULT_VOCAB_SIZE: int = ASCII_VOCAB_SIZE
DEFAULT_MAX_POSITION_EMBEDDINGS: int = 512
DEFAULT_PAD_TOKEN_ID: int = ASCII_PAD_ID
DEFAULT_INITIALIZER_RANGE: float = 0.02
DEFAULT_LAYER_NORM_EPSILON: float = 1e-12

#: Pooling strategies exposed as a study axis. Any ``SequencePooling`` strategy
#: works; these are the three the study sweeps.
SUPPORTED_POOLING: tuple = ("cls", "mean", "attention", "mean_max", "max", "last")


@keras.saving.register_keras_serializable()
class EmbeddingEncoder(keras.Model):
    """
    Block-agnostic bidirectional text encoder producing one embedding per sequence.

    **Architecture:**

    .. code-block:: text

        input_ids (B, S)
              │
        BertEmbeddings          token + position (+ type), norm, dropout
              │
        N x block               resolved from BLOCK_REGISTRY by block_type
              │                 called as (x, attention_mask=, layer_idx=, training=)
              ▼
        last_hidden_state (B, S, H)
              │
        SequencePooling(mask=attention_mask)
              ▼
        pooled_output (B, P)

    :param vocab_size: Vocabulary size. Defaults to the ASCII vocabulary (101).
    :type vocab_size: int
    :param hidden_size: Model width ``H``.
    :type hidden_size: int
    :param num_layers: Number of stacked blocks.
    :type num_layers: int
    :param block_type: Key into :data:`~...blocks.BLOCK_REGISTRY`.
    :type block_type: str
    :param block_config: Keyword arguments forwarded to the block builder. A
        key the builder does not declare raises, it is never dropped.
    :type block_config: dict[str, Any] | None
    :param pooling_strategy: A :class:`SequencePooling` strategy name.
    :type pooling_strategy: str
    :param max_position_embeddings: Longest sequence the position table covers.
    :type max_position_embeddings: int
    :param type_vocab_size: Token-type vocabulary size, or ``None`` to disable
        token-type embeddings. Defaults to ``None``: this family encodes single
        segments, and an always-zero table is a dead parameter.
    :type type_vocab_size: int | None
    :param hidden_dropout_rate: Dropout on embeddings and inside blocks.
    :type hidden_dropout_rate: float
    :param stochastic_depth_rate: Maximum drop-path rate; per-block rates are
        linearly spaced from 0 to this value.
    :type stochastic_depth_rate: float
    :param initializer_range: Stddev of the truncated-normal initializer.
    :type initializer_range: float
    :param layer_norm_eps: Normalization epsilon, threaded explicitly into the
        embeddings and the blocks.
    :type layer_norm_eps: float
    :param pad_token_id: Padding id, used to derive an attention mask when the
        caller supplies none. Unlike upstream BERT, this IS read.
    :type pad_token_id: int
    :param normalization_type: Normalization registry key for the embeddings.
    :type normalization_type: str
    :param position_embedding_type: Position embedding kind.
    :type position_embedding_type: str
    :param pooling_config: Extra keyword arguments for :class:`SequencePooling`.
    :type pooling_config: dict[str, Any] | None
    :param kwargs: Additional keyword arguments for the Model base class.
    :raises ValueError: If any dimension is non-positive, a dropout rate falls
        outside ``[0, 1)``, or ``pooling_strategy`` is unsupported.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 256,
        num_layers: int = 4,
        block_type: str = "transformer",
        block_config: Optional[Dict[str, Any]] = None,
        pooling_strategy: str = "mean",
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        type_vocab_size: Optional[int] = None,
        hidden_dropout_rate: float = 0.1,
        stochastic_depth_rate: float = 0.0,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: str = "layer_norm",
        position_embedding_type: str = "learned",
        pooling_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_dropout_rate=hidden_dropout_rate,
            stochastic_depth_rate=stochastic_depth_rate,
            max_position_embeddings=max_position_embeddings,
            pooling_strategy=pooling_strategy,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_type = block_type
        self.block_config = dict(block_config or {})
        self.pooling_strategy = pooling_strategy
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.position_embedding_type = position_embedding_type
        self.pooling_config = dict(pooling_config or {})

        self._build_architecture()

        logger.info(
            f"Created EmbeddingEncoder (block={block_type}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"pooling={pooling_strategy}, vocab_size={vocab_size})"
        )

    @staticmethod
    def _validate_config(
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        hidden_dropout_rate: float,
        stochastic_depth_rate: float,
        max_position_embeddings: int,
        pooling_strategy: str,
    ) -> None:
        """Reject an unusable configuration at construction time.

        :param vocab_size: Vocabulary size.
        :type vocab_size: int
        :param hidden_size: Model width.
        :type hidden_size: int
        :param num_layers: Block count.
        :type num_layers: int
        :param hidden_dropout_rate: Hidden dropout.
        :type hidden_dropout_rate: float
        :param stochastic_depth_rate: Maximum drop-path rate.
        :type stochastic_depth_rate: float
        :param max_position_embeddings: Position table size.
        :type max_position_embeddings: int
        :param pooling_strategy: Pooling strategy name.
        :type pooling_strategy: str
        :raises ValueError: On any invalid value.
        """
        for name, value in (
            ("vocab_size", vocab_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("max_position_embeddings", max_position_embeddings),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        for name, value in (
            ("hidden_dropout_rate", hidden_dropout_rate),
            ("stochastic_depth_rate", stochastic_depth_rate),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        if pooling_strategy not in SUPPORTED_POOLING:
            raise ValueError(
                f"pooling_strategy must be one of {list(SUPPORTED_POOLING)}, "
                f"got {pooling_strategy!r}"
            )

    def _build_architecture(self) -> None:
        """Create the embeddings, the block stack and the pooler.

        Sub-layers are created here rather than in :meth:`build`, matching
        ``models/language/bert``.
        """
        self.embeddings = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.hidden_dropout_rate,
            normalization_type=self.normalization_type,
            use_token_type_embeddings=self.type_vocab_size is not None,
            position_embedding_type=self.position_embedding_type,
            name="embeddings",
        )

        drop_path_rates = linear_drop_path_rates(
            self.num_layers, self.stochastic_depth_rate
        )
        initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range
        )

        self.encoder_layers: List[keras.layers.Layer] = []
        for i in range(self.num_layers):
            block_kwargs = dict(self.block_config)
            block_kwargs.setdefault("drop_path_rate", drop_path_rates[i])
            block_kwargs.setdefault("kernel_initializer", initializer)
            self.encoder_layers.append(
                create_encoder_block(
                    self.block_type,
                    hidden_size=self.hidden_size,
                    name=f"encoder_layer_{i}",
                    **block_kwargs,
                )
            )

        self.pooler = SequencePooling(
            strategy=self.pooling_strategy,
            name="pooler",
            **self.pooling_config,
        )

    def build(self, input_shape: Any) -> None:
        """Materialise the embeddings, every block and the pooler.

        Deliberately shape-driven rather than a dummy forward pass: Keras
        rebuilds a model under a ``StatelessScope`` on load, where variables
        created by a forward pass are discarded.

        :param input_shape: Shape of ``input_ids`` -- ``(batch, seq_len)`` --
            or a mapping/sequence whose ``input_ids`` entry has that shape.
        :type input_shape: Any
        :raises ValueError: If the resolved shape is not rank 2.
        """
        if self.built:
            return

        ids_shape = input_shape
        if isinstance(ids_shape, dict):
            ids_shape = ids_shape.get("input_ids", ids_shape)
        if (
            isinstance(ids_shape, (list, tuple))
            and ids_shape
            and isinstance(ids_shape[0], (list, tuple))
        ):
            ids_shape = ids_shape[0]
        ids_shape = tuple(ids_shape)

        if len(ids_shape) != 2:
            raise ValueError(
                "EmbeddingEncoder.build expects the shape of `input_ids`, "
                f"i.e. (batch_size, seq_length); got {ids_shape}"
            )

        self.embeddings.build(ids_shape)

        hidden_shape = (ids_shape[0], ids_shape[1], self.hidden_size)
        for encoder_layer in self.encoder_layers:
            encoder_layer.build(hidden_shape)
        self.pooler.build(hidden_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        attention_mask: Optional[keras.KerasTensor] = None,
        token_type_ids: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Encode token ids into per-token states and one pooled embedding.

        :param inputs: ``input_ids`` of shape ``(batch, seq_len)``, or a
            dictionary carrying ``input_ids`` and optionally
            ``attention_mask`` / ``token_type_ids`` / ``position_ids``.
        :type inputs: Any
        :param attention_mask: ``(batch, seq_len)``, 1 for kept positions. When
            omitted it is derived from ``pad_token_id``.
        :type attention_mask: keras.KerasTensor | None
        :param token_type_ids: ``(batch, seq_len)`` segment ids.
        :type token_type_ids: keras.KerasTensor | None
        :param position_ids: ``(batch, seq_len)`` explicit positions.
        :type position_ids: keras.KerasTensor | None
        :param training: Keras training flag.
        :type training: bool | None
        :return: ``last_hidden_state`` ``(batch, seq_len, hidden_size)``,
            ``attention_mask`` ``(batch, seq_len)`` and ``pooled_output``
            ``(batch, pooled_dim)``.
        :rtype: dict[str, keras.KerasTensor]
        :raises ValueError: If a dictionary input carries no ``input_ids``.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain an 'input_ids' key"
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Unlike upstream BERT, the mask is resolved BEFORE the stack, because
        # pooling and the maskless Clifford block both consume it. A missing
        # mask would otherwise silently pool over padding.
        if attention_mask is None:
            attention_mask = keras.ops.cast(
                keras.ops.not_equal(input_ids, self.pad_token_id),
                dtype="int32",
            )

        hidden_states = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training,
        )

        for i, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,
                training=training,
            )

        pooled_output = self.pooler(
            hidden_states,
            mask=keras.ops.cast(attention_mask, "bool"),
            training=training,
        )

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": attention_mask,
            "pooled_output": pooled_output,
        }

    def compute_output_shape(self, input_shape: Any) -> Dict[str, Any]:
        """Return the shapes of the three outputs.

        :param input_shape: Shape of ``input_ids``.
        :type input_shape: Any
        :return: Shapes keyed as in :meth:`call`.
        :rtype: dict[str, Any]
        """
        ids_shape = input_shape
        if isinstance(ids_shape, dict):
            ids_shape = ids_shape.get("input_ids", ids_shape)
        ids_shape = tuple(ids_shape)
        hidden_shape = (ids_shape[0], ids_shape[1], self.hidden_size)
        return {
            "last_hidden_state": hidden_shape,
            "attention_mask": (ids_shape[0], ids_shape[1]),
            "pooled_output": self.pooler.compute_output_shape(hidden_shape),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "block_type": self.block_type,
                "block_config": keras.saving.serialize_keras_object(
                    self.block_config
                ),
                "pooling_strategy": self.pooling_strategy,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "normalization_type": self.normalization_type,
                "position_embedding_type": self.position_embedding_type,
                "pooling_config": keras.saving.serialize_keras_object(
                    self.pooling_config
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EmbeddingEncoder":
        """Rebuild from a serialized configuration.

        ``block_config`` and ``pooling_config`` are nested dictionaries that
        may carry serialized Keras objects (an initializer, for instance), so
        they are deserialized rather than passed through.

        :param config: Configuration produced by :meth:`get_config`.
        :type config: dict[str, Any]
        :return: The reconstructed model.
        :rtype: EmbeddingEncoder
        """
        config = dict(config)
        for key in ("block_config", "pooling_config"):
            if config.get(key) is not None:
                config[key] = keras.saving.deserialize_keras_object(config[key])
        return cls(**config)
