"""
DistilBERT Model Implementation
===============================

A complete implementation of the DistilBERT (Distilled BERT) architecture.
DistilBERT is a smaller, faster, cheaper and lighter version of BERT that,
as reported by its authors, retains 97% of BERT's language understanding
capabilities while being 40% smaller and 60% faster. Those are the *paper's*
numbers for the *published checkpoint*; this module builds the architecture and
ships no trained weights, so nothing here reproduces them (see
``DistilBERT.PRETRAINED_WEIGHTS``).

Based on: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
(Sanh et al., 2019) https://arxiv.org/abs/1910.01108

Key Architectural Differences from BERT:
    - Number of layers reduced by half (6 vs 12 for base)
    - Token type embeddings removed
    - Pooler layer removed
    - Optional sinusoidal position embeddings

The embedding stage is NOT a DistilBERT-private layer: it is the shared
``dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings``, built through
``create_embedding_layer('bert_embeddings', ...)`` with
``use_token_type_embeddings=False`` and ``mask_zero=False``. See
``DistilBERT._build_architecture``.

Usage Examples:
--------------

Every statement in the block below is executed as written (verified
2026-08-11); it needs no checkpoint and no network access.

.. code-block:: python

    import keras
    import numpy as np
    from dl_techniques.models.distilbert import DistilBERT
    from dl_techniques.layers.heads.nlp import (
        create_nlp_head,
        NLPTaskConfig,
        NLPTaskType,
    )

    # 1. Create a DistilBERT encoder from a named variant (randomly initialized)
    distilbert_encoder = DistilBERT.from_variant("base")

    # 2. Create one with configuration overrides
    distilbert_encoder = DistilBERT.from_variant("base", vocab_size=50000)

    # 3. Combine with a task-specific head
    sentiment_config = NLPTaskConfig(
        name="sentiment",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=3
    )
    sentiment_head = create_nlp_head(
        task_config=sentiment_config,
        input_dim=distilbert_encoder.hidden_size
    )

    # 4. Build the complete model. The head consumes a dict keyed
    #    'hidden_states' / 'attention_mask'; the encoder emits
    #    'last_hidden_state' / 'attention_mask', so the two are wired
    #    explicitly rather than passed straight through.
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    }
    distilbert_outputs = distilbert_encoder(inputs)
    task_outputs = sentiment_head({
        "hidden_states": distilbert_outputs["last_hidden_state"],
        "attention_mask": distilbert_outputs["attention_mask"],
    })
    sentiment_model = keras.Model(inputs, task_outputs)

    # 5. Run it. The head returns a DICT, not a single tensor.
    batch = {
        "input_ids": np.zeros((2, 16), dtype="int32"),
        "attention_mask": np.ones((2, 16), dtype="int32"),
    }
    outputs = sentiment_model.predict(batch, verbose=0)   # {'logits', 'probabilities'}

``create_distilbert_with_head`` performs steps 3-4 for you.

Weight loading. ``pretrained=<path>`` works; ``pretrained=True`` does not:

.. code-block:: python

    # WORKS (repaired 2026-08-11): loads a local .keras file into THIS
    # configuration, so the architecture must match. Raises if the file
    # restores nothing.
    distilbert_encoder = DistilBERT.from_variant("base", pretrained="path/to/weights.keras")

    # RETURNS A RANDOMLY INITIALIZED MODEL, silently apart from one logged
    # WARNING: every URL in PRETRAINED_WEIGHTS is an example.com placeholder,
    # the download fails, and the failure is caught.
    distilbert_encoder = DistilBERT.from_variant("base", pretrained=True)

    # Simplest route to restore a model you saved -- no config to match:
    distilbert_encoder.save("distilbert.keras")
    distilbert_encoder = keras.models.load_model("distilbert.keras")

"""

import os
import keras
import numpy as np
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import (
    FFNType,
    AttentionType,
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
)
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.heads.nlp import create_nlp_head, NLPTaskConfig

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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
    containing the 'last_hidden_state' and the forwarded 'attention_mask'.

    **Architecture Overview:**

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
    :type layer_norm_eps: float
    :param pad_token_id: ID of the padding token. Defaults to 0.
        **ADVISORY ONLY — nothing in this model reads it.** It is stored and
        serialized for tokenizer/config round-tripping and for nothing else. In
        particular NO attention mask is derived from it: if you call the model
        without an ``attention_mask``, padding tokens are attended to exactly
        like real tokens. Supply ``attention_mask`` yourself. (Measured: the
        same input with and without an explicit mask gives DIFFERENT outputs;
        see ``decisions.md`` D-003.)
    :type pad_token_id: int
    :param normalization_type: Type of normalization layer, applied at BOTH the
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

        For ``pretrained=True`` / ``pretrained="path/to/weights.keras"`` see
        :meth:`from_variant` — neither is runnable out of the box (no
        checkpoint ships with this repo, and every ``PRETRAINED_WEIGHTS`` URL
        is a placeholder).

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

    # PLACEHOLDER URLs. Not one of these hosts a checkpoint -- they are all
    # example.com. Any call that reaches them fails; `from_variant` catches the
    # failure, emits a single logger.warning and returns a RANDOMLY INITIALIZED
    # model (measured 2026-08-11: `load_pretrained_weights` is invoked 0 times).
    # Kept, rather than deleted, by explicit user decision; the honest wiring is
    # the documentation, not a raise.
    PRETRAINED_WEIGHTS = {
        "base": {
            "uncased": "https://example.com/distilbert_base_uncased.keras",
            "cased": "https://example.com/distilbert_base_cased.keras",
            "multilingual": "https://example.com/distilbert_base_multilingual.keras",
        },
        "small": {
            "uncased": "https://example.com/distilbert_small_uncased.keras",
        },
        "tiny": {
            "uncased": "https://example.com/distilbert_tiny_uncased.keras",
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
        :param pad_token_id: ID of the padding token. ADVISORY ONLY: it is
            stored and serialized but never read, and no attention mask is
            derived from it -- without an explicit ``attention_mask`` padding is
            fully attended to. See the class docstring for the full rule.
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
        # DECISION plan-2026-08-10T183739-b007f435/D-003
        # pad_token_id is stored and serialized but DELIBERATELY never read.
        # Do NOT "fix" this by deriving `attention_mask = input_ids !=
        # pad_token_id` in call() when no mask is supplied: that silently
        # changes the output of every mask-less forward pass that exists today,
        # and upstream HF DistilBERT does not do it either (it defaults the mask
        # to all-ones). The footgun is documented in the class docstring and in
        # README.md instead. See decisions.md D-003.
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
        # DECISION plan-2026-08-10T183739-b007f435/D-011
        # Every value below is passed EXPLICITLY. Do NOT "simplify" any of the
        # four that look droppable -- each one differs from the shared layer's
        # default, which is BERT's behaviour, not DistilBERT's:
        #   * use_token_type_embeddings=False -- BertEmbeddings defaults to True.
        #     Dropping it silently allocates a token_type_embeddings weight and
        #     adds a third term to the embedding sum.
        #   * type_vocab_size=None -- mandatory once token types are off (D-002).
        #   * mask_zero=False -- BertEmbeddings defaults to True. DistilBERT
        #     threads an EXPLICIT attention_mask into every TransformerLayer.
        #     What is MEASURED (step 10.2, at c6ab51084): BertEmbeddings NEVER
        #     propagates a Keras mask at EITHER setting -- supports_masking is
        #     False, it defines no compute_mask, and the inner Embedding's mask
        #     is dropped at the `word_embeds + position_embeds` sum; the forward
        #     output is bit-identical (max abs diff 0.0) with mask_zero True vs
        #     False, eagerly and in a functional graph. So this kwarg is INERT
        #     downstream today and its only observable effects are get_config()
        #     and embeddings.word_embeddings.mask_zero. It is passed anyway
        #     because it is the ONLY place DistilBERT's "no auto-mask" intent is
        #     recorded, and omitting it would flip the flag to BERT's True --
        #     silently correct today, silently wrong the day BertEmbeddings
        #     gains mask propagation. Do NOT restate this as a live
        #     two-masks-collide hazard: that claim was measured FALSE.
        #   * layer_norm_eps -- I-2. Omitting it does NOT inherit this model's
        #     1e-12; BertEmbeddings' own default is 1e-8.
        # create_embedding_layer SILENTLY DROPS any kwarg not registered in
        # EMBEDDING_REGISTRY['bert_embeddings'] (measured, findings/
        # step1-premise-rederivation.md (f)), so a misspelt kwarg here is a
        # silent no-op: any test covering these must assert the EFFECT, never
        # that construction succeeded.
        # DECISION plan-2026-08-10T183739-b007f435/D-018
        # See decisions.md D-011, whose stated mask_zero rationale is SUPERSEDED
        # by D-018: the "two masking mechanisms collide" hazard was re-measured
        # and does not exist. The executable pin on the corrected claim is
        # tests/test_layers/test_embedding/test_bert_embedding.py::
        # TestOptionalBranches::test_mask_zero_is_not_propagated_out_of_this_layer.
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

        self.encoder_layers: List[TransformerLayer] = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                normalization_type=self.normalization_type,
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
                 - ``attention_mask``: The original attention mask, passed
                   through for convenience in downstream models.
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

        return {
            "last_hidden_state": hidden_states,
            "attention_mask": attention_mask
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
           this method counts the variables whose values actually changed and
           logs the count, and it raises if *nothing* changed.

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
        # DECISION plan-2026-08-10T183739-b007f435/D-024
        # Two things here are deliberate and must not be "simplified":
        #   1. the dummy input uses keras.random.randint, NOT
        #      keras.random.uniform(..., dtype="int32") -- Keras rejects an
        #      integer dtype on uniform ("requires a floating point dtype"),
        #      which made this whole method raise on every unbuilt model;
        #   2. there is no `by_name` argument. Keras 3's Model.load_weights has
        #      NO by_name parameter and raises `Invalid keyword arguments:
        #      {'by_name': True}`; the old signature accepted one and forwarded
        #      it, so the built path raised too.
        # The changed-variable count below is not decoration: with
        # skip_mismatch=True a load that restores NOTHING is otherwise
        # indistinguishable from a load that restores everything.
        # See decisions.md D-012 (the defect) and D-024 (this fix).
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build the model if needed -- weights can only be restored into
            # variables that already exist.
            if not self.built:
                dummy_input = {
                    "input_ids": keras.random.randint(
                        (1, 128), 0, self.vocab_size, dtype="int32"
                    ),
                    "attention_mask": keras.ops.ones((1, 128), dtype="int32")
                }
                self(dummy_input, training=False)

            logger.info(f"Loading weights from {weights_path}")

            before = [keras.ops.convert_to_numpy(v) for v in self.weights]
            self.load_weights(weights_path, skip_mismatch=skip_mismatch)
            after = [keras.ops.convert_to_numpy(v) for v in self.weights]

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

        changed = sum(
            1 for b, a in zip(before, after)
            if b.shape != a.shape or not np.array_equal(b, a)
        )
        total = len(self.weights)

        if changed == 0:
            raise ValueError(
                f"Loading {weights_path} changed none of this model's {total} "
                "variables. Nothing was restored -- the file's variable names or "
                "shapes do not match this model. Check the architecture config, "
                "or use keras.models.load_model() to load the saved model as-is."
            )

        if changed == total:
            logger.info(f"Loaded {weights_path}: all {total} variables changed value.")
        elif skip_mismatch:
            logger.warning(
                f"Loaded {weights_path} with skip_mismatch=True: {changed} of "
                f"{total} variables changed value. The other {total - changed} were "
                "either skipped for a shape mismatch or already equal to the stored "
                "value -- this method cannot tell those two apart, so verify the "
                "load if a mismatch was not intended."
            )
        else:
            # Strict load: nothing was skipped, so the unchanged variables were
            # already equal to what the file holds (common for zero/one-init
            # variables of an untrained checkpoint).
            logger.info(
                f"Loaded {weights_path}: {changed} of {total} variables changed "
                f"value; the other {total - changed} already held the stored value."
            )

    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "uncased",
        cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from the URL registered for a variant.

        .. warning::
           Every URL in :attr:`PRETRAINED_WEIGHTS` is an ``example.com``
           placeholder, so this method CANNOT return real weights. Calling it
           directly raises (measured: ``Exception: URL fetch failure on
           https://example.com/distilbert_tiny_uncased.keras``). It is reached
           only from :meth:`from_variant`, which catches the failure, warns, and
           returns a randomly initialized model.

        :param variant: Model variant name.
        :type variant: str
        :param dataset: Dataset/version the weights were trained on.
            Options: "uncased", "cased", "multilingual".
        :type dataset: str
        :param cache_dir: Directory to cache downloaded weights.
            If None, uses default Keras cache directory.
        :type cache_dir: Optional[str]
        :return: Path to the downloaded weights file. In practice unreachable —
            the fetch always fails first, see the warning.
        :rtype: str
        :raises ValueError: If variant or dataset is not available.

        Example (RAISES as written — the URL is a placeholder):
            .. code-block:: python

                weights_path = DistilBERT._download_weights("base", "uncased")
        """
        if variant not in DistilBERT.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(DistilBERT.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in DistilBERT.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(DistilBERT.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = DistilBERT.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading DistilBERT-{variant} ({dataset}) weights...")

        # Download weights using Keras utility
        weights_path = keras.utils.get_file(
            fname=f"distilbert_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/distilbert"
        )

        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

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
           ``pretrained=True`` DOES NOT GIVE YOU TRAINED WEIGHTS. The URLs in
           :attr:`PRETRAINED_WEIGHTS` are ``example.com`` placeholders; the
           download fails, the failure is caught here, a single
           ``logger.warning`` ("Failed to download pretrained weights: ...
           Continuing with random initialization.") is emitted, and a
           RANDOMLY INITIALIZED model is returned. Nothing raises, so a caller
           who ignores logs cannot tell the difference. Measured 2026-08-11:
           ``load_pretrained_weights`` is invoked 0 times on that path and every
           non-constant weight differs from an independently built model.

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
            configuration. If ``True``, attempts the placeholder download
            described in the warning above and falls back to random
            initialization — it does NOT give you trained weights.
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

        Example:
            .. code-block:: python

                # Randomly initialized DistilBERT-base -- the only route that
                # works with no checkpoint on disk
                model = DistilBERT.from_variant("base")

                # Custom vocabulary
                model = DistilBERT.from_variant("base", vocab_size=50000)

                # To restore a model you saved yourself, do NOT use
                # pretrained= (it raises); use Keras directly:
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

        # Handle pretrained weights
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                # Load from local file path
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # Download from URL
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Determine if we need to skip mismatches
            # Check if vocab_size differs from default
            pretrained_vocab_size = cls.DEFAULT_VOCAB_SIZE
            custom_vocab_size = kwargs.get("vocab_size", config.get("vocab_size"))

            if custom_vocab_size and custom_vocab_size != pretrained_vocab_size:
                skip_mismatch = True
                logger.info(
                    f"vocab_size ({custom_vocab_size}) differs from pretrained "
                    f"({pretrained_vocab_size}). Will skip embedding layer weights."
                )

            # Check if other architectural parameters differ
            pretrained_config_keys = ["hidden_size", "num_layers", "num_heads", "intermediate_size"]
            for key in pretrained_config_keys:
                if key in kwargs and kwargs[key] != config.get(key):
                    skip_mismatch = True
                    logger.info(
                        f"{key} differs from pretrained configuration. "
                        f"Will skip layers with shape mismatches."
                    )

        # Override defaults with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load pretrained weights if available
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

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `DistilBERT` model (optionally pretrained).
    2. Instantiate a task-specific head from the
       `dl_techniques.layers.heads.nlp` factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    The returned model's output is whatever the head returns — for a
    classification head a DICT (e.g. ``{'logits', 'probabilities'}``), not a
    single tensor.

    :param distilbert_variant: The DistilBERT variant to use (e.g., "base", "small").
    :type distilbert_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task.
    :type task_config: NLPTaskConfig
    :param pretrained: Forwarded to :meth:`DistilBERT.from_variant`. Leave it
        ``False``: ``True`` attempts a placeholder download that always fails
        and falls back, with a logged warning, to random initialization, and a
        string path raises ``ValueError``. See that method's two warnings.
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

    # 1. Create the foundational DistilBERT model (with optional pretrained weights)
    distilbert_encoder = DistilBERT.from_variant(
        distilbert_variant,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        cache_dir=cache_dir,
        **distilbert_config_overrides
    )

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=distilbert_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    # Note: DistilBERT doesn't use token_type_ids
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