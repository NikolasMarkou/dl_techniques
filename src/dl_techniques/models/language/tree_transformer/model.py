"""
Tree Transformer: an encoder that induces a soft constituency tree and uses it to
constrain its own attention.

Defines :class:`TreeTransformer` and the factories that build it. Standard
self-attention scores every token pair on its own, with nothing preferring "the old
man" as a unit over "man who is". Tree Transformer computes, for each adjacent token
pair, the probability that a constituent boundary falls between them, then turns those
neighbour scores into the probability that each span is one constituent through a
dynamic-programming recurrence written as matrix products rather than an enumeration
over spans. That span matrix multiplies the ordinary attention weights element-wise,
so a pair the model reads as straddling a boundary is attenuated instead of
hard-masked. Each block hands its group probabilities to the next as a prior, so
boundaries persist and constituents grow with depth, and every block's break
probabilities come back stacked as the induced grammar. Blocks are Pre-LN, ordered
GroupAttention, TreeMHA, then FFN.

By default (``include_top=False``) the model also owns its masked-LM projection
``lm_head`` and returns ``last_hidden_state``, ``logits`` and ``break_probs``. With
``include_top=True`` and a ``task_config`` it owns ONE task head from
``layers/heads/nlp``, named ``task_head``, in place of ``lm_head`` (which is then not
created), and returns only that head's output, exactly as ``models/language/bert``
does. The backbone (every weight but ``lm_head`` and ``task_head``) has the same weight
paths either way, so an encoder trained under one head is reused, weights unchanged,
under another. No pretrained weights ship with this package: ``pretrained=True``
raises ``NotImplementedError``, so pass a local ``.keras`` path instead.

References:
    - Wang et al., 2019. Tree Transformer: Integrating Tree Structures into
      Self-Attention. EMNLP-IJCNLP. (https://arxiv.org/abs/1909.06639)
    - Shen et al., 2019. Ordered Neurons: Integrating Tree Structures into
      Recurrent Neural Networks. (https://arxiv.org/abs/1810.09536)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""


import numbers
import os
import keras
from typing import Optional, Union, Dict, Any, List, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import FFNType
from dl_techniques.layers.norms import (
    create_normalization_layer,
    NormalizationType,
)
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    TASK_HEAD_NAME,
    build_model_task_head,
    finalize_head_outputs,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.sequence_length_guard import (
    RecordedBuildShapeMixin,
    check_sequence_length,
)
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    load_weights_from_checkpoint,
)

from .components import (  # noqa: F401
    PositionalEncoding,
    GroupAttention,
    TreeMHA,
    TreeTransformerBlock,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.tree_transformer.model")
class TreeTransformer(RecordedBuildShapeMixin, keras.Model):
    """Encode tokens and induce break probabilities in the same forward pass.

    Headless (``include_top=False``, the default) it is an encoder with no
    task-specific layers beyond the language-modelling head. ``call`` accepts either
    a tensor of token ids or a dictionary with ``input_ids`` and an optional
    ``attention_mask``, and returns a dictionary with ``last_hidden_state``,
    ``logits`` and the per-layer ``break_probs``. With ``include_top=True`` the
    ``lm_head`` is not created; one NLP task head is fed ``{"hidden_states":
    last_hidden_state, "attention_mask": <the mask below, cast to the hidden-state
    dtype>}`` and ``call`` returns what the head returns, reduced by
    :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`: a bare tensor, or
    a dict for question answering. The knobs, their validation and the head-swap
    route are those of :class:`~dl_techniques.models.language.bert.model.BERT`.

    Architecture:

    .. code-block:: text

        {"input_ids": [B, L]}   (+ optional "attention_mask")
                 │
                 ▼
        ┌───────────────────┐
        │ embedding         │  x sqrt(hidden_size)
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ pos_encoding      │
        └───────────────────┘
                 │  [B, L, H]
                 ▼
        ┌───────────────────┐
        │ block_0           │──► break_prob
        └───────────────────┘
                 │  group_prob carried forward
                 ▼
                ...
                 │
                 ▼
        ┌───────────────────┐
        │ block_{n-1}       │──► break_prob
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ final_norm        │──► "last_hidden_state"  [B, L, H]
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ lm_head           │──► "logits"  [B, L, V]
        └───────────────────┘

        every break_prob stacked ──► "break_probs"  [B, N, L, L]

        include_top=True: lm_head is replaced by
        ┌───────────────────┐
        │ task_head         │──► finalized head output (only output)
        └───────────────────┘

    Each block takes the previous block's group_prob as its prior; the first gets 0.

    Mask:

    .. code-block:: text

        dict with "attention_mask"?
                    │
            ┌───────┴───────┐
            ▼               ▼
           yes              no
        cast to int32   input_ids != pad_token_id
            │               │
            └───────┬───────┘
                    ▼
           expand_dims axis 1 ──► [B, 1, L]

    Variants:

    .. code-block:: text

        variant  hidden  layers  heads  intermediate
        tiny        128       4      4           512
        small       256       6      4          1024
        base        512      10      8          2048
        large      1024      16     16          4096

    base follows the original paper.

    :param vocab_size: Size of the vocabulary. Defaults to 30000.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Must be divisible by
        ``num_heads``. Defaults to 512.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 10.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 2048.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param hidden_dropout_rate: Dropout for embeddings/encoder, in [0, 1]. Defaults to 0.1.
    :type hidden_dropout_rate: float
    :param attention_dropout_rate: Dropout for attention scores, in [0, 1]. Defaults to 0.1.
    :type attention_dropout_rate: float
    :param max_len: Maximum sequence length the positional encoding covers. Defaults to 256.
    :type max_len: int
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token, used to derive the mask when the
        caller passes none. Defaults to 0.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer. Defaults to "layer_norm".
    :type normalization_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param include_top: Whether the model owns a task head (and no ``lm_head``).
        Defaults to False: no head layer and no head weight exist.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored as
        a plain dict. ``MULTIPLE_CHOICE`` is refused.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head constructor
        (e.g. ``{"pooling_type": "mean"}``). Only valid with ``include_top=True``.
        Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param **kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar lm_head: The masked-LM projection, or None when ``include_top`` is True.
    :vartype lm_head: Optional[keras.layers.Dense]
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.
    :vartype task_head: Optional[dl_techniques.layers.heads.nlp.base.BaseNLPHead]

    :raises ValueError: If ``vocab_size``, ``hidden_size``, ``num_layers`` or
        ``num_heads`` is not positive, if ``hidden_size`` is not divisible by
        ``num_heads``, if either dropout rate falls outside [0, 1], if
        ``include_top=True`` comes without a ``task_config``, or if a
        ``task_config`` or ``head_config_overrides`` comes with
        ``include_top=False``.
    """

    MODEL_VARIANTS = {
        "large": {
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_size": 4096,
            "description": "TreeTransformer-Large: High capacity for large datasets",
        },
        "base": {
            "hidden_size": 512,
            "num_layers": 10,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "TreeTransformer-Base: Balanced performance, based on original paper",
        },
        "small": {
            "hidden_size": 256,
            "num_layers": 6,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "TreeTransformer-Small: Lightweight for faster training",
        },
        "tiny": {
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4,
            "intermediate_size": 512,
            "description": "TreeTransformer-Tiny: Ultra-lightweight for research",
        },
    }

    DEFAULT_VOCAB_SIZE = 30000
    DEFAULT_MAX_LEN = 256
    DEFAULT_LAYER_NORM_EPSILON = 1e-6
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0
    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_len={limit}. TreeTransformer's "
        "sinusoidal positional encoding is precomputed for {limit} positions, "
        "so there is no encoding for position {limit}. Truncate the input to "
        "{limit} tokens, or build the model with max_len>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        hidden_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        max_len: int = DEFAULT_MAX_LEN,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer.

        Arguments are documented on the class.
        """
        super().__init__(**kwargs)

        self._validate_config(
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            hidden_dropout_rate,
            attention_dropout_rate,
        )
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        self.vocab_size = int(vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.max_len = max_len
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        self._build_architecture()

        # Created after every encoder layer and under its own fixed name, so
        # the backbone's layer names, weight paths and RNG draws are the same
        # with or without a head.
        self.task_head = None
        if self.include_top:
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.hidden_size,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
            )
        logger.info(
            f"Created Tree Transformer foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
            + (f", task head '{self.task_config['name']}'" if self.include_top else "")
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dropout_rate: float,
        attention_dropout_rate: float,
    ) -> None:
        """Check the size arguments and the two dropout rates.

        :raises ValueError: If a size is not positive, ``hidden_size`` is not
            divisible by ``num_heads``, or a dropout rate is outside [0, 1].
        """
        # A bool is an int, and None / a float / a str would fail the <= with a
        # TypeError that names no argument; a NumPy integer is accepted and
        # stored as int.
        if (isinstance(vocab_size, bool)
                or not isinstance(vocab_size, numbers.Integral)
                or vocab_size <= 0):
            raise ValueError(
                f"vocab_size must be positive and an integer, got {vocab_size!r}"
            )
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= hidden_dropout_rate <= 1.0):
            raise ValueError(
                f"hidden_dropout_rate must be in [0, 1], got {hidden_dropout_rate}"
            )
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Create the embedding, positional encoding, blocks, final norm and LM head.

        The LM head is created only without a task head: ``call`` never runs it
        then, and ``build`` must not create what ``call`` does not run.
        """
        self.embedding = keras.layers.Embedding(
            self.vocab_size, self.hidden_size, name="embedding"
        )
        self.pos_encoding = PositionalEncoding(
            hidden_size=self.hidden_size,
            dropout_rate=self.hidden_dropout_rate,
            max_len=self.max_len,
            name="pos_encoding",
        )
        self.blocks: List[TreeTransformerBlock] = [
            TreeTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                hidden_dropout_rate=self.hidden_dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            epsilon=self.layer_norm_eps,
            name="final_norm",
        )
        # DECISION plan-2026-10-11T004932-b83935e1/D-011: no lm_head under a task head; do NOT build it
        # always -- call() never runs it then, and it is not backbone, so no transfer may depend on it. See decisions.md.
        self.lm_head = (
            None if self.include_top
            else keras.layers.Dense(self.vocab_size, name="lm_head_projection")
        )

    def compute_output_shape(
        self, input_shape: Any
    ) -> Any:
        """Compute the shapes of what ``call`` returns.

        Keras builds a symbolic call of this model from this method, never from
        ``call``, so it must report what ``call`` returns in both modes.

        :param input_shape: A ``(batch, seq_len)`` shape, or a dict holding one under
            ``'input_ids'``.
        :type input_shape: Any
        :return: Headless, shapes for ``last_hidden_state``, ``logits`` and
            ``break_probs``. With a head, the head's output shapes after the same
            reduction ``call`` applies: one shape, or a mapping of two or more.
        :rtype: Any
        """
        if isinstance(input_shape, dict):
            input_shape = input_shape["input_ids"]

        batch_size, seq_len = input_shape

        if self.include_top:
            # finalize_head_outputs only drops and unwraps dict entries, so it
            # applies to a dict of shapes exactly as to a dict of tensors.
            return finalize_head_outputs(
                self.task_head.compute_output_shape({
                    "hidden_states": (batch_size, seq_len, self.hidden_size),
                    "attention_mask": (batch_size, seq_len),
                })
            )

        return {
            "last_hidden_state": (batch_size, seq_len, self.hidden_size),
            "logits": (batch_size, seq_len, self.vocab_size),
            "break_probs": (batch_size, self.num_layers, seq_len, seq_len),
        }

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method TreeTransformer inherits ``Layer.build``, which marks the
        model built while its sub-layers are still unbuilt, and Keras warns about it.
        The shared helper traces ``call()`` on symbolic inputs, so what gets built
        matches what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Embed the ids, run every block, and return hidden states, logits and breaks.

        :param inputs: Token ids ``(B, L)``, or a dictionary with ``'input_ids'`` and
            optionally ``'attention_mask'``. Without a mask, one is derived from
            ``input_ids != pad_token_id``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call runs in training mode.
        :type training: Optional[bool]
        :return: With ``include_top=False``, ``last_hidden_state``
            ``(B, L, hidden_size)``, ``logits`` ``(B, L, vocab_size)`` and
            ``break_probs`` ``(B, num_layers, L, L)``. With ``include_top=True``,
            only the head's output reduced by ``finalize_head_outputs``: a bare
            tensor, or a dict of two or more tensors (``start_logits`` /
            ``end_logits`` for QA). The head's pooling sees the same resolved mask
            the blocks do.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :raises ValueError: If a dictionary input has no ``'input_ids'`` key.
        """
        explicit_attention_mask = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key"
                )
            # An explicit attention_mask wins, so a caller passing only input_ids
            # still gets the pad-derived mask.
            explicit_attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs

        input_ids = check_sequence_length(
            self, input_ids, self.max_len, "max_len",
            self._SEQUENCE_LENGTH_MESSAGE,
        )

        if explicit_attention_mask is not None:
            token_mask = keras.ops.cast(explicit_attention_mask, "int32")
        else:
            token_mask = keras.ops.cast(
                keras.ops.not_equal(input_ids, self.pad_token_id), "int32"
            )
        mask = keras.ops.expand_dims(token_mask, axis=1)

        x = self.embedding(input_ids)
        x *= keras.ops.cast(self.hidden_size, x.dtype) ** 0.5
        x = self.pos_encoding(x, training=training)

        # A tensor, not a float, so Keras can trace the first block.
        group_prob: keras.KerasTensor = keras.ops.convert_to_tensor(
            0.0, dtype=self.compute_dtype
        )
        all_break_probs = []

        for block in self.blocks:
            x, group_prob, break_prob = block(
                (x, mask, group_prob), training=training
            )
            all_break_probs.append(break_prob)

        last_hidden_state = self.final_norm(x)

        if self.include_top:
            # Cast so a float16 hidden state never meets an int mask inside
            # the head's pooling (mixed precision; QA needs a float mask).
            head_outputs = self.task_head(
                {
                    "hidden_states": last_hidden_state,
                    "attention_mask": keras.ops.cast(
                        token_mask, last_hidden_state.dtype
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        logits = self.lm_head(last_hidden_state)
        stacked_break_probs = keras.ops.stack(all_break_probs, axis=1)

        return {
            "last_hidden_state": last_hidden_state,
            "logits": logits,
            "break_probs": stacked_break_probs,
        }

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_prefixes: Sequence[str] = (),
        strict: bool = True,
    ) -> None:
        """Load weights from a ``.keras`` checkpoint by layer name.

        Uses :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        which walks layers by name and calls ``set_weights`` when shapes match. An
        unbuilt model is first called on a dummy batch, since name matching needs the
        layers to exist.

        A ``task_head`` in the checkpoint is never copied, and this model's own head
        (if any) keeps its fresh initialization. A checkpoint's ``lm_head`` has no
        target in a headed model and is left out; a headless model takes it when
        the checkpoint has one. So a checkpoint of any TreeTransformer, headless or
        trained under any head, warm-starts a model with any other head or none. To
        restore a whole model, head included, use ``keras.models.load_model``.

        On a vocabulary change (the checkpoint's own ``vocab_size`` differs from
        this model's) only the variables shaped by the vocabulary keep their
        init; with the same vocabulary every layer loads, whatever the value.

        :param weights_path: Path to a ``.keras`` checkpoint.
        :param skip_prefixes: Further layer-name prefixes to skip during transfer,
            on top of ``task_head``.
        :param strict: If True (default), the shared language load rule
            (:func:`dl_techniques.utils.weight_transfer.load_backbone_from_checkpoint`):
            every backbone layer loads and every checkpoint layer finds its
            place, or this raises; ``lm_head_projection`` may be left out only
            when one side carries a task head, or the vocabulary changed. If
            False, a layer that does not fit is recorded in the logged report
            and skipped.
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If ``strict`` and a backbone layer is left unloaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not self.built:
            seq_len = min(64, self.max_len)
            dummy_input = {
                "input_ids": keras.ops.cast(
                    keras.random.uniform(
                        (1, seq_len), 0, self.vocab_size
                    ),
                    "int32",
                )
            }
            self(dummy_input, training=False)
        logger.info(f"Loading pretrained weights from {weights_path}")
        # DECISION plan-2026-10-11T004932-b83935e1/D-007: skip the checkpoint's task_head; do NOT
        # drop TASK_HEAD_NAME from this list -- a head of another task raises on shape, one of equal shape is copied silently. See decisions.md.
        head_prefixes = (TASK_HEAD_NAME, *skip_prefixes)
        if not strict:
            load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=head_prefixes,
                strict=False,
            )
            return
        # DECISION plan-2026-10-11T004932-b83935e1/D-022: the vocabulary exception is keyed on the
        # CHECKPOINT's vocab_size (inside the shared rule); do NOT compare against DEFAULT_VOCAB_SIZE --
        # any custom vocabulary then drops a same-vocabulary embedding. See decisions.md.
        load_backbone_from_checkpoint(
            self,
            weights_path,
            head_prefixes=head_prefixes,
            vocab_layers=("embedding",),
            tables=(
                "embedding/embeddings",
                "lm_head_projection/kernel",
                "lm_head_projection/bias",
            ),
            tops=("lm_head",),
        )

    @staticmethod
    def _download_weights(
        variant: str, dataset: str = "uncased", cache_dir: Optional[str] = None
    ) -> str:
        """No public pretrained weights exist for TreeTransformer; always raises.

        Pass ``pretrained=<path/to/checkpoint.keras>`` to ``from_variant`` to
        load local weights, or omit ``pretrained`` to initialize randomly.

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No public pretrained TreeTransformer weights are distributed with "
            f"dl_techniques (requested variant '{variant}', dataset '{dataset}'). "
            f"Pass a local checkpoint instead: TreeTransformer.from_variant("
            f"'{variant}', pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "TreeTransformer":
        """Create a TreeTransformer from a predefined variant.

        A download that fails with an I/O or value error logs a warning and leaves the
        model randomly initialized; a missing local file or a failed transfer raises.

        :param variant: One of ``cls.MODEL_VARIANTS`` (e.g. ``"tiny"``, ``"small"``, ``"base"``, ``"large"``).
        :param pretrained: ``False`` (default) for random init. ``True`` to attempt downloading hosted weights — this currently raises :class:`NotImplementedError` because no public Tree Transformer weights are hosted. A string path is treated as a local ``.keras`` archive to load (the by-name backbone route reads the archive's layer structure, so a bare ``.weights.h5`` file is not accepted).
        :param weights_dataset: Dataset key for hosted weights (kept for API parity with BERT / DistilBERT — currently unused since no public weights are hosted).
        :param cache_dir: Optional cache directory for downloaded weights.
        :param **kwargs: Forwarded to ``TreeTransformer.__init__``, including
            ``include_top``, ``task_config`` and ``head_config_overrides``. A
            pretrained transfer restores every backbone layer or raises; it skips
            the embedding and LM head only when the checkpoint's ``vocab_size``
            differs from the model's, and a checkpoint's task head always.

        :return: The model, with weights loaded if a path was given.
        :rtype: TreeTransformer
        :raises ValueError: If ``variant`` is not a known name.
        :raises NotImplementedError: If ``pretrained=True``. Use ``pretrained="path/to/weights.keras"`` to load local weights.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(
            f"Creating TreeTransformer-{variant.upper()} model: {description}"
        )
        load_weights_path = None
        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
            else:
                # DECISION plan_2026-05-11_0a5779e8/D-001: catch I/O errors only; a bare
                # Exception swallows NotImplementedError into a silent random init.
                # See decisions.md.
                try:
                    load_weights_path = cls._download_weights(
                        variant, weights_dataset, cache_dir
                    )
                except (IOError, OSError, ValueError) as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {e}. "
                        "Continuing with random initialization."
                    )
        config.update(kwargs)
        model = cls(**config)
        if load_weights_path:
            try:
                model.load_pretrained_weights(load_weights_path, strict=True)
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration for serialization.

        :return: Dict holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
                "hidden_dropout_rate": self.hidden_dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "max_len": self.max_len,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "normalization_type": self.normalization_type,
                "ffn_type": self.ffn_type,
                "include_top": self.include_top,
                "task_config": (
                    dict(self.task_config) if self.task_config is not None else None
                ),
                "head_config_overrides": (
                    dict(self.head_config_overrides)
                    if self.head_config_overrides is not None else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TreeTransformer":
        """Create a model instance from its configuration.

        :param config: Dict as returned by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model.
        :rtype: TreeTransformer
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the Keras summary, then log the architecture settings.

        :param **kwargs: Forwarded to ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("Tree Transformer Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, {self.hidden_size} hidden size"
        )
        logger.info(f"  - Attention: {self.num_heads} heads")
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(f"  - Max sequence length: {self.max_len}")
        logger.info(
            f"  - Normalization: {self.normalization_type} (Pre-LN in blocks)"
        )
        logger.info(
            f"  - Feed-forward: {self.ffn_type}, {self.intermediate_size} intermediate size"
        )


def create_tree_transformer(
        variant: str = "base",
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
) -> "TreeTransformer":
    """Create a Tree Transformer encoder from a variant name.

    A thin module-level factory that delegates to
    :meth:`TreeTransformer.from_variant`, mirroring
    :func:`dl_techniques.models.vision.resnet.model.create_resnet`.

    :param variant: String, model variant ("tiny", "small", "base", "large").
    :param pretrained: Boolean or string. If ``True``, attempts to load pretrained weights for the chosen ``weights_dataset`` (currently raises :class:`NotImplementedError` — no public Tree Transformer weights are hosted). If a string, treated as a path to a local ``.keras`` archive (a bare ``.weights.h5`` file is not accepted).
    :param weights_dataset: String, dataset key for pretrained weights (kept for API parity with other foundation models).
    :param cache_dir: Optional string, directory to cache downloaded weights.
    :param **kwargs: Additional arguments forwarded to ``TreeTransformer.__init__`` through :meth:`TreeTransformer.from_variant`, laid over the variant's entry. ``vocab_size`` overrides the variant's vocabulary; with ``pretrained=<path>``, a value different from the CHECKPOINT's ``vocab_size`` skips the vocab-dependent layers (embedding, LM head) and an equal one restores them. Others: ``hidden_dropout_rate``, ``max_len``, ``pad_token_id``, or ``include_top=True`` with a ``task_config`` for a model with a task head. A local ``pretrained`` path never restores a checkpoint's task head.

    :return: TreeTransformer instance.
    :raises ValueError: If ``variant`` is unknown, or a forwarded argument is invalid (including inconsistent head knobs, see :class:`TreeTransformer`).
    :raises NotImplementedError: If ``pretrained=True``.

    Example:
        >>> # Create a Tree Transformer base encoder with random init
        >>> model = create_tree_transformer("base")
        >>>
        >>> # Smaller variant with a custom vocabulary
        >>> model = create_tree_transformer("tiny", vocab_size=8000)
        >>>
        >>> # Load from local weights file
        >>> model = create_tree_transformer("base", pretrained="path/to/weights.keras")
        >>>
        >>> # Same encoder with a NER head instead of the LM head
        >>> ner = create_tree_transformer(
        ...     "tiny", include_top=True,
        ...     task_config=NLPTaskConfig(
        ...         name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        ...         num_classes=9))
    """
    return TreeTransformer.from_variant(
        variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **kwargs,
    )
