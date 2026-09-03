"""
DistilBERT compresses a BERT encoder by knowledge distillation, in the `DistilBERT` class
and its `create_distilbert_with_head` factory. Where a small model trained from scratch on
the same corpus sees only hard targets, a distilled model fits the teacher's full soft
output distribution, which carries near-miss structure a from-scratch model would have to
rediscover from data. The compression is entirely in depth: layers are halved (6 instead of
12 for base) while hidden size, head count and FFN width stay at BERT's values, so a student
initialized from the teacher can copy every other layer verbatim. Token type embeddings and
the pooler are removed rather than shrunk, since DistilBERT trains on single-segment input
and drops the next-sentence-prediction objective the pooler served. The class is a pure
foundation model, emitting `{"last_hidden_state", "attention_mask"}` with no head attached;
`create_distilbert_with_head` wires it to a task head that returns a dict, not a single
tensor. Three preset variants: base (the paper's configuration), small and tiny. No
pretrained weights ship with this package -- `pretrained=True` raises
`NotImplementedError`. `pretrained="<path>.keras"` loads a local checkpoint into this
configuration; `keras.models.load_model(path)` is simpler when you just want a saved model
back as it was.

References:
    - Sanh et al., 2019. DistilBERT, a distilled version of BERT: smaller,
      faster, cheaper and lighter. (https://arxiv.org/abs/1910.01108)
    - Hinton et al., 2015. Distilling the Knowledge in a Neural Network.
      (https://arxiv.org/abs/1503.02531)
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""


import os
import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_or_raise
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
)
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.distilbert.model")
class DistilBERT(keras.Model):
    """DistilBERT (Distilled BERT) model.

    DistilBERT is a smaller, faster, and lighter transformer model, obtained in
    the paper through knowledge distillation during pre-training. See the module
    docstring for what the published checkpoint's reported numbers do and do not
    say about this class (it ships no weights).

    Key differences from BERT:
        - Number of layers reduced by half
        - Token type embeddings removed
        - Optional sinusoidal position embeddings
        - No pooler layer

    The model expects inputs as a dictionary containing 'input_ids', and
    optionally 'attention_mask' and 'position_ids'. It outputs a dictionary
    containing the 'last_hidden_state' and the forwarded 'attention_mask'. That
    second entry is never ``None``: an omitted mask is echoed back as all ones
    so the output structure does not depend on the input (which is what makes
    ``predict({"input_ids": ...})`` work). The encoder still sees ``None``, so
    the echo is numerically inert — it is not a mask being applied.

    Architecture:

    .. code-block:: text

        Input(input_ids, attention_mask)
               │
               ▼
        Embeddings(Word + Position) -> Norm(normalization_type) -> Dropout
               │
               ▼
        TransformerLayer₁ (Self-Attention -> FFN)
               │
               ▼
              ...
               │
               ▼
        TransformerLayerₙ (Self-Attention -> FFN)
               │
               ▼
        Output Dictionary {
            "last_hidden_state": [batch, seq_len, hidden_size],
            "attention_mask": [batch, seq_len]
        }

    :param vocab_size: Size of the vocabulary. Defaults to 30522.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Defaults to 768.
    :type hidden_size: int
    :param num_layers: Number of hidden transformer layers. Defaults to 6.
    :type num_layers: int
    :param num_heads: Number of attention heads. Defaults to 12.
    :type num_heads: int
    :param intermediate_size: Dimensionality of the FFN layer. Defaults to 3072.
    :type intermediate_size: int
    :param hidden_act: Activation function in the encoder. Defaults to "gelu".
    :type hidden_act: str
    :param dropout_rate: Dropout probability for embeddings/encoder.
        Defaults to 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout for attention probabilities.
        Defaults to 0.1.
    :type attention_dropout_rate: float
    :param max_position_embeddings: Maximum sequence length. Defaults to 512.
    :type max_position_embeddings: int
    :param sinusoidal_pos_embds: Whether to use sinusoidal position embeddings.
        Defaults to False.
    :type sinusoidal_pos_embds: bool
    :param initializer_range: Stddev for weight initialization. Defaults to 0.02.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon for normalization layers. Defaults to 1e-12.
        Applies to the embedding norm and to every one of the
        ``2 * num_layers`` in-block norms. Before 2026-08-19 the in-block
        norms ignored this value and ran at the normalization factory's
        ``1e-6`` default instead; weight shapes are unchanged, so existing
        ``.keras`` files still load, but forward values move slightly. See
        ``decisions.md`` D-007.
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token. Defaults to 0. Nothing
        in this model reads it -- it is stored and serialized for
        tokenizer/config round-tripping only. No attention mask is
        derived from it: a call without an ``attention_mask`` attends to
        padding tokens like real tokens, so supply ``attention_mask``
        explicitly. See ``decisions.md`` D-003.
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer, applied at both the
        embedding stage and the encoder blocks. Defaults to "layer_norm".
        The embedding stage is the shared ``BertEmbeddings``, which accepts
        exactly four types — ``layer_norm``, ``rms_norm``, ``band_rms``,
        ``batch_norm`` (``bert_embeddings.VALID_NORMALIZATION_TYPES``) — so this
        parameter is constrained to those four even though ``TransformerLayer``
        alone would accept more. Anything else raises ``ValueError`` at
        construction, from the embedding stage.
    :type normalization_type: str
    :param normalization_position: Position of normalization ('pre' or 'post').
        Defaults to "post".
    :type normalization_position: str
    :param attention_type: Type of attention mechanism.
        Defaults to "multi_head".
    :type attention_type: str
    :param ffn_type: Type of feed-forward network. Defaults to "mlp".
    :type ffn_type: str
    :param use_stochastic_depth: Whether to enable stochastic depth.
        Defaults to False.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Drop path rate for stochastic depth.
        Defaults to 0.1.
    :type stochastic_depth_rate: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.

    :ivar embeddings: The embedding layer instance — the shared
        ``layers/embedding/bert_embeddings.py::BertEmbeddings``, built through
        ``create_embedding_layer('bert_embeddings', ...)`` with token type
        embeddings disabled.
    :vartype embeddings: BertEmbeddings
    :ivar encoder_layers: A list of `TransformerLayer` instances.
    :vartype encoder_layers: list[TransformerLayer]

    :raises ValueError: If invalid configuration parameters are provided.

    Example:
        .. code-block:: python

            # Create a standard, randomly initialized DistilBERT model
            model = DistilBERT.from_variant("base")

            # Use the model. Supply attention_mask yourself -- see pad_token_id.
            inputs = {
                "input_ids": keras.random.randint((2, 128), 0, 30522, dtype="int32"),
                "attention_mask": keras.ops.ones((2, 128), dtype="int32")
            }
            outputs = model(inputs, training=False)
            print(outputs["last_hidden_state"].shape)
            # (2, 128, 768)

        ``pretrained="path/to/weights.keras"`` loads a local checkpoint; see
        :meth:`from_variant`. ``pretrained=True`` raises
        ``NotImplementedError`` — no checkpoint ships with this repo.

    """

    # Model variant configurations following DistilBERT specifications
    MODEL_VARIANTS = {
        "base": {
            "hidden_size": 768,
            "num_layers": 6,
            "num_heads": 12,
            "intermediate_size": 3072,
            "description": "DistilBERT-Base: the paper's configuration (parameter counts: README section 7)"
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 8,
            "intermediate_size": 2048,
            "description": "DistilBERT-Small: Lightweight variant for resource-constrained environments"
        },
        "tiny": {
            "hidden_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 1024,
            "description": "DistilBERT-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Default architecture constants
    DEFAULT_VOCAB_SIZE = 30522
    DEFAULT_MAX_POSITION_EMBEDDINGS = 512
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPSILON = 1e-12
    DEFAULT_HIDDEN_ACT = "gelu"
    DEFAULT_PAD_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = DEFAULT_HIDDEN_ACT,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS,
        sinusoidal_pos_embds: bool = False,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPSILON,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: NormalizationPositionType = "post",
        attention_type: AttentionType = "multi_head",
        ffn_type: FFNType = "mlp",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        """Initialize the DistilBERT model instance.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of hidden transformer layers.
        :type num_layers: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param intermediate_size: Dimensionality of the FFN layer.
        :type intermediate_size: int
        :param hidden_act: Activation function in the encoder.
        :type hidden_act: str
        :param dropout_rate: Dropout probability for embeddings/encoder.
        :type dropout_rate: float
        :param attention_dropout_rate: Dropout for attention scores.
        :type attention_dropout_rate: float
        :param max_position_embeddings: Maximum sequence length.
        :type max_position_embeddings: int
        :param sinusoidal_pos_embds: Whether to use sinusoidal position embeddings.
        :type sinusoidal_pos_embds: bool
        :param initializer_range: Stddev for weight initialization.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for normalization layers.
        :type layer_norm_eps: float
        :param pad_token_id: ID of the padding token. Stored and
            serialized but never read; no attention mask is derived from
            it -- without an explicit ``attention_mask`` padding is fully
            attended to. See the class docstring for the full rule.
        :type pad_token_id: int
        :param normalization_type: Type of normalization layer. Constrained to
            the four types the shared ``BertEmbeddings`` accepts; see the class
            docstring.
        :type normalization_type: str
        :param normalization_position: Position of normalization ('pre'/'post').
        :type normalization_position: str
        :param attention_type: Type of attention mechanism.
        :type attention_type: str
        :param ffn_type: Type of feed-forward network.
        :type ffn_type: str
        :param use_stochastic_depth: Whether to enable stochastic depth.
        :type use_stochastic_depth: bool
        :param stochastic_depth_rate: Drop rate for stochastic depth.
        :type stochastic_depth_rate: float
        :param kwargs: Additional keyword arguments for the `keras.Model`.
        """
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_heads,
            dropout_rate, attention_dropout_rate
        )

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        # DECISION plan-2026-08-10T183739-b007f435/D-003: pad_token_id stays unread;
        # deriving a mask from it would silently change every mask-less forward pass. See decisions.md.
        self.pad_token_id = pad_token_id
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build the model architecture
        self._build_architecture()

        logger.info(
            f"Created DistilBERT foundation model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, heads={self.num_heads}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float
    ) -> None:
        """Validate model configuration parameters.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Dimensionality of encoder layers.
        :type hidden_size: int
        :param num_layers: Number of transformer layers.
        :type num_layers: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param dropout_rate: Dropout probability for hidden layers.
        :type dropout_rate: float
        :param attention_dropout_rate: Dropout for attention scores.
        :type attention_dropout_rate: float
        :raises ValueError: If any configuration value is invalid.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, "
                f"got {dropout_rate}"
            )
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(
                f"attention_dropout_rate must be between 0 and 1, "
                f"got {attention_dropout_rate}"
            )

    def _build_architecture(self) -> None:
        """Build all model components (embeddings and encoder layers)."""
        # DECISION plan-2026-08-10T183739-b007f435/D-011: pass use_token_type_embeddings,
        # type_vocab_size, mask_zero and layer_norm_eps explicitly; each differs from BertEmbeddings' own default. See decisions.md.
        # DECISION plan-2026-08-10T183739-b007f435/D-018: mask_zero has no live masking effect
        # (BertEmbeddings never propagates a Keras mask); it only records DistilBERT's no-auto-mask intent. See decisions.md.
        self.embeddings = create_embedding_layer(
            'bert_embeddings',
            name="embeddings",
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=None,
            use_token_type_embeddings=False,
            position_embedding_type=(
                'sinusoidal' if self.sinusoidal_pos_embds else 'learned'
            ),
            mask_zero=False,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            dropout_rate=self.dropout_rate,
            normalization_type=self.normalization_type,
        )

        # DECISION plan-2026-08-19T070627-a616f581/D-007: pass layer_norm_eps into
        # attention_norm_args/ffn_norm_args — without it every block norm ran at the factory's 1e-6 default, not DistilBERT's 1e-12. See decisions.md.
        _norm_args = {'epsilon': self.layer_norm_eps}

        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                normalization_type=self.normalization_type,
                attention_norm_args=dict(_norm_args),
                ffn_norm_args=dict(_norm_args),
                normalization_position=self.normalization_position,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=self.stochastic_depth_rate,
                activation=self.hidden_act,
                use_bias=True,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                bias_initializer="zeros",
                name=f"transformer_layer_{i}"
            )
            self.encoder_layers.append(transformer_layer)

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method DistilBERT inherits ``Layer.build``, which marks the
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

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        position_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the DistilBERT foundation model.

        :param inputs: Input token IDs or a dictionary containing 'input_ids'
            and other optional tensors like 'attention_mask'.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param attention_mask: Mask to avoid attention on padding tokens, ``1``
            for tokens to keep. If omitted, NO mask is applied and padding is
            attended to like any other token — none is inferred from
            ``pad_token_id``. A ``token_type_ids`` entry in a dict input is
            accepted and silently ignored (DistilBERT has no token-type
            embeddings); passing ``token_type_ids=`` as a keyword argument
            raises ``TypeError``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param position_ids: Position IDs for positional embeddings.
        :type position_ids: Optional[keras.KerasTensor]
        :param training: Indicates if the model is in training mode.
        :type training: Optional[bool]
        :return: A dictionary with the following keys:
                 - ``last_hidden_state``: The sequence of hidden states at the
                   output of the final layer. Shape:
                   `(batch, seq_len, hidden_size)`.
                 - ``attention_mask``: The attention mask, passed through for
                   convenience in downstream models. When the caller omitted
                   it, an all-ones mask of the same shape as ``input_ids`` is
                   returned rather than ``None``, so the output structure is
                   independent of the input and ``predict()`` works on a
                   single-key input dict. The encoder itself still receives
                   ``None`` — the echo does not change any numerics.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If dictionary input does not contain 'input_ids'.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
            position_ids = inputs.get("position_ids", position_ids)
        else:
            input_ids = inputs

        # Get embeddings (no token_type_ids for DistilBERT)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            training=training
        )

        # Pass through encoder layers
        hidden_states = embedding_output
        for i, encoder_layer in enumerate(self.encoder_layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_idx=i,
                training=training
            )

        # DECISION plan-2026-08-18T140459-7991552f/D-062: resolve the echoed mask here, at the
        # return, never earlier or conditionally — a possibly-None key broke predict()'s batch concatenation. See decisions.md.
        return {
            "last_hidden_state": hidden_states,
            "attention_mask": (
                attention_mask if attention_mask is not None
                else keras.ops.ones_like(input_ids)
            ),
        }

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = False,
    ) -> None:
        """Load weights from a ``.keras`` file into this model, in place.

        Use this when you already have a `DistilBERT` instance configured the
        way you want it — typically because you are transplanting weights into
        a *differently sized* model, e.g. loading a 30522-vocab checkpoint into
        a 50000-vocab model with ``skip_mismatch=True``. If you just want the
        saved model back exactly as it was, prefer
        ``keras.models.load_model(path)``, which restores architecture and
        weights together and needs no configuration to match.

        The model is built with a dummy forward pass first if it is not built
        already, since Keras can only restore weights into materialized
        variables.

        .. note::
           ``skip_mismatch=True`` makes a *partial* load succeed silently — any
           variable whose shape does not line up is left at its initialized
           value. To keep that from being indistinguishable from a real load,
           the shared loader
           (:func:`dl_techniques.utils.weight_transfer.load_weights_or_raise`)
           counts the variables whose values actually changed and logs the
           count, and it raises if *nothing* changed.

        :param weights_path: Path to the weights file (``.keras`` format).
        :type weights_path: str
        :param skip_mismatch: Skip variables whose shape does not match instead
            of raising. Defaults to ``False`` (strict) — a mismatch is far more
            often a bug than an intent.
        :type skip_mismatch: bool
        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If the weights cannot be loaded, or if the load
            completed without changing a single variable.

        Example:
            .. code-block:: python

                model = DistilBERT.from_variant("base", vocab_size=50000)
                model.load_pretrained_weights(
                    "distilbert_base_uncased.keras",
                    skip_mismatch=True,
                )
        """
        # DECISION plan-2026-08-10T183739-b007f435/D-024: dummy input uses randint, not
        # uniform(dtype="int32") -- Keras rejects an integer dtype on uniform. See decisions.md.
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-070: restored-variable count lives in
        # load_weights_or_raise, not inline here, since gpt2/wave_field had the same unchecked load. See decisions.md.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Build first -- weights can only be restored into variables that exist.
        # The probe length is clamped to `max_position_embeddings`: a hardcoded
        # 128 overflows the position-embedding table of any model configured
        # shorter than that (`DistilBERT.from_variant("tiny",
        # max_position_embeddings=64)` died in a GatherV2 with
        # `indices[0,64] = 64 is not in [0, 64)`), so the unbuilt route was
        # unreachable for exactly the small configurations tests use.
        if not self.built:
            seq_len = min(128, self.max_position_embeddings)
            dummy_input = {
                "input_ids": keras.random.randint(
                    (1, seq_len), 0, self.vocab_size, dtype="int32"
                ),
                "attention_mask": keras.ops.ones((1, seq_len), dtype="int32")
            }
            self(dummy_input, training=False)

        load_weights_or_raise(self, weights_path, skip_mismatch=skip_mismatch)

    # No public DistilBERT weights are distributed with dl_techniques; this raises rather
    # than falling back to a random-init model. Pass a local path via `pretrained="/path/to/file.keras"`.
    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "uncased",
        cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights; always raises.

        Not implemented: no public DistilBERT weights ship with
        ``dl_techniques``. Pass a local ``.keras`` path to
        :meth:`from_variant` instead.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset/version identifier (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained DistilBERT weights are distributed with "
            f"dl_techniques (requested variant '{variant}', dataset "
            f"'{dataset}'). Pass a local checkpoint instead: "
            f"DistilBERT.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "uncased",
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ) -> "DistilBERT":
        """Create a DistilBERT model from a predefined variant.

        .. warning::
           ``pretrained=True`` raises ``NotImplementedError``. No public
           DistilBERT checkpoint ships with this repo. It used to warn and
           return a RANDOMLY INITIALIZED model, which a caller who ignores logs
           could not distinguish from a successful load.

        .. note::
           ``pretrained="<path>.keras"`` DOES work — it forwards to
           :meth:`load_pretrained_weights`, which was repaired 2026-08-11
           (D-024; it had raised on both of its paths). It loads weights into
           THIS configuration, so the architecture must match the file. If you
           just want the saved model back as it was, ``keras.models.load_model(path)``
           is simpler and needs no matching config.

        :param variant: The name of the variant, one of "base", "small", "tiny".
            Measured parameter counts live in one place only, README section 7.
        :type variant: str
        :param pretrained: If a string, a path to a local ``.keras`` weights
            file, loaded via :meth:`load_pretrained_weights` into this
            configuration. If ``True``, raises ``NotImplementedError``. If
            ``False`` (default), the model is randomly initialized.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset/version for pretrained weights.
            Options: "uncased", "cased", "multilingual".
            Only used if pretrained=True.
        :type weights_dataset: str
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param kwargs: Additional arguments to override the variant's defaults.
        :type kwargs: Any
        :return: A DistilBERT model instance configured for the specified variant.
        :rtype: DistilBERT
        :raises ValueError: If the specified variant is not recognized.
        :raises NotImplementedError: If ``pretrained`` is True.

        Example:
            .. code-block:: python

                # Randomly initialized DistilBERT-base -- the only route that
                # works with no checkpoint on disk
                model = DistilBERT.from_variant("base")

                # Custom vocabulary
                model = DistilBERT.from_variant("base", vocab_size=50000)

                # Warm-start from a local checkpoint
                model = DistilBERT.from_variant(
                    "base", pretrained="path/to/weights.keras"
                )

                # To restore a model you saved yourself, Keras is simpler --
                # it needs no matching config:
                #   model.save("distilbert.keras")
                #   model = keras.models.load_model("distilbert.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating DistilBERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )

            pretrained_vocab_size = cls.DEFAULT_VOCAB_SIZE
            custom_vocab_size = kwargs.get("vocab_size", config.get("vocab_size"))

            if custom_vocab_size and custom_vocab_size != pretrained_vocab_size:
                skip_mismatch = True
                logger.info(
                    f"vocab_size ({custom_vocab_size}) differs from pretrained "
                    f"({pretrained_vocab_size}). Will skip embedding layer weights."
                )

            pretrained_config_keys = ["hidden_size", "num_layers", "num_heads", "intermediate_size"]
            for key in pretrained_config_keys:
                if key in kwargs and kwargs[key] != config.get(key):
                    skip_mismatch = True
                    logger.info(
                        f"{key} differs from pretrained configuration. "
                        f"Will skip layers with shape mismatches."
                    )

        config.update(kwargs)
        model = cls(**config)

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch,
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration for serialization.

        :return: A dictionary containing the model's configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "max_position_embeddings": self.max_position_embeddings,
            "sinusoidal_pos_embds": self.sinusoidal_pos_embds,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DistilBERT":
        """Create a model instance from its configuration.

        :param config: A dictionary containing the model's configuration.
        :type config: Dict[str, Any]
        :return: A new DistilBERT model instance.
        :rtype: DistilBERT
        """
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional DistilBERT-specific information.

        :param kwargs: Additional arguments passed to `keras.Model.summary`.
        """
        super().summary(**kwargs)
        logger.info("DistilBERT Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, "
            f"{self.hidden_size} hidden size"
        )
        logger.info(
            f"  - Attention: {self.num_heads} heads, {self.attention_type}"
        )
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(
            f"  - Max sequence length: {self.max_position_embeddings}"
        )
        logger.info(
            f"  - Position embeddings: "
            f"{'sinusoidal' if self.sinusoidal_pos_embds else 'learned'}"
        )
        logger.info(
            f"  - Normalization: {self.normalization_type} "
            f"({self.normalization_position})"
        )
        logger.info(
            f"  - Feed-forward: {self.ffn_type}, "
            f"{self.intermediate_size} intermediate size"
        )
        if self.use_stochastic_depth:
            logger.info(
                "  - Stochastic depth enabled: "
                f"rate={self.stochastic_depth_rate}"
            )


# ---------------------------------------------------------------------
# Integration with NLP Task Heads
# ---------------------------------------------------------------------


def create_distilbert_with_head(
    distilbert_variant: str,
    task_config: NLPTaskConfig,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "uncased",
    cache_dir: Optional[str] = None,
    distilbert_config_overrides: Optional[Dict[str, Any]] = None,
    head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Factory function to create a DistilBERT model with a task-specific head.

    It instantiates a foundational `DistilBERT` model, a task-specific head
    from the `dl_techniques.layers.heads.nlp` factory, and combines them into a
    single end-to-end `keras.Model`.

    The returned model's output is whatever the head returns — for a
    classification head a DICT (e.g. ``{'logits', 'probabilities'}``), not a
    single tensor.

    :param distilbert_variant: The DistilBERT variant to use (e.g., "base", "small").
    :type distilbert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: Forwarded to :meth:`DistilBERT.from_variant`. A string
        path loads a local ``.keras`` checkpoint into this configuration;
        ``True`` raises ``NotImplementedError``; ``False`` (default) gives
        random initialization.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for pretrained weights ("uncased", "cased", etc.).
    :type weights_dataset: str
    :param cache_dir: Directory to cache downloaded weights.
    :type cache_dir: Optional[str]
    :param distilbert_config_overrides: Optional dictionary to override default
        DistilBERT configuration for the chosen variant. Defaults to None.
    :type distilbert_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model
    :raises NotImplementedError: If ``pretrained`` is True.

    Example:
        .. code-block:: python

            from dl_techniques.layers.heads.nlp import NLPTaskType

            # Define a task
            ner_task = NLPTaskConfig(
                name="ner",
                task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
                num_classes=9
            )

            # Create the full model (randomly initialized DistilBERT)
            ner_model = create_distilbert_with_head(
                distilbert_variant="base",
                task_config=ner_task,
                head_config_overrides={"use_task_attention": True}
            )
            ner_model.summary()
    """
    distilbert_config_overrides = distilbert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(
        f"Creating DistilBERT-{distilbert_variant} with a '{task_config.name}' head."
    )

    distilbert_encoder = DistilBERT.from_variant(
        distilbert_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **distilbert_config_overrides
    )

    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=distilbert_encoder.hidden_size,
        **head_config_overrides,
    )

    # DistilBERT has no token_type_ids input.
    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
        "attention_mask": keras.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        ),
    }

    # Get hidden states from the encoder
    encoder_outputs = distilbert_encoder(inputs)

    # Pass encoder outputs to the task head
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
    model_name = f"distilbert_{distilbert_variant}_with_{task_config.name}_head"
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