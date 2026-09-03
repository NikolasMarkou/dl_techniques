"""
Masked language model pre-trainer that wraps an arbitrary encoder and
corrupts its input on the fly.

Unlike autoregressive training, which conditions each token only on its
left context, MLM deletes a random subset of positions and predicts them
from the rest, so attention can run fully bidirectional. Of the selected
positions, 80% become `[MASK]`, 10% become a random vocabulary id, and 10%
are left unchanged; all three groups are scored, so the encoder cannot rely
on the `[MASK]` token to know a position is being predicted. Corruption is
redrawn every `train_step`/`test_step` call rather than baked into the
dataset, and the loss is the dense cross entropy over every position
reduced through the selection mask as `sample_weight`, so unselected
positions contribute no gradient.

The encoder is treated as a plug-in component: it must expose a
`hidden_size` attribute and return a mapping with `last_hidden_state`. The
prediction head does not tie its output projection to the encoder's input
embeddings, since the wrapper cannot assume one is exposed. `train_step`
and `test_step` are hand-written over `tf.GradientTape`, so this model is
TensorFlow-backend only. `call()` does no masking itself — it scores the
inputs as given, which is what makes `predict()` usable on a sequence the
caller already masked.

References:
    - Devlin et al., 2018. BERT: Pre-training of Deep Bidirectional Transformers for
      Language Understanding. (https://arxiv.org/abs/1810.04805)
    - Liu et al., 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach.
      (https://arxiv.org/abs/1907.11692)
    - Taylor, 1953. "Cloze Procedure": A New Tool for Measuring Readability.
      Journalism Quarterly.
    - Press and Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
"""

import keras
import tensorflow as tf
from keras import ops
from typing import Dict, Any, Optional, Union, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.masking.strategies import apply_mlm_masking
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.masked_language_model.mlm")
class MaskedLanguageModel(keras.Model):
    """A model-agnostic Masked Language Modeling (MLM) pre-trainer.

    This model wraps a given encoder (like BERT) and adds the necessary logic
    for MLM pre-training. It performs the following steps:

    1. Dynamically masks input tokens based on the BERT masking strategy.
    2. Processes the masked tokens through the encoder.
    3. Uses an MLM head to predict the original IDs of the masked tokens.
    4. Computes a sparse categorical cross-entropy loss only on the
       masked positions.

    The encoder is treated as a core component, making it easy to save and
    reuse for downstream fine-tuning after pre-training is complete.

    Masking strategy (BERT-style): 15% of tokens are selected. Of those, 80%
    become `[MASK]`, 10% become a random token, and 10% are left unchanged.
    Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are never masked.

    Architecture:

    .. code-block:: text

        Input(input_ids, attention_mask, ...) → Dynamic Masking
               ↓
        Encoder(masked_input_ids) → last_hidden_state
               ↓
        MLM Head(Dense → LayerNorm → Dense) → logits
               ↓
        Loss(only on masked positions) → Cross-Entropy

    :param encoder: An instance of a Keras model (e.g., BERT) that acts
        as the token encoder. It must accept a dictionary of inputs
        (`input_ids`, `attention_mask`, etc.) and return a dictionary
        containing the `last_hidden_state`.
    :type encoder: keras.Model
    :param vocab_size: The size of the vocabulary.
    :type vocab_size: int
    :param mask_ratio: The probability of a token being chosen for masking.
        Defaults to 0.15.
    :type mask_ratio: float
    :param mask_token_id: The vocabulary ID for the `[MASK]` token.
    :type mask_token_id: int
    :param random_token_ratio: The probability of replacing a masked token
        with a random token from the vocabulary. Defaults to 0.1.
    :type random_token_ratio: float
    :param unchanged_ratio: The probability of leaving a masked token as is.
        Defaults to 0.1.
    :type unchanged_ratio: float
    :param special_token_ids: A list of special token IDs (e.g., [CLS], [SEP])
        to exclude from masking. Defaults to None.
    :type special_token_ids: Optional[List[int]]
    :param mlm_head_activation: Activation function for the MLM head's
        intermediate layer. Defaults to "gelu".
    :type mlm_head_activation: str
    :param initializer_range: The standard deviation for weight initialization
        in the MLM head. Defaults to 0.02.
    :type initializer_range: float
    :param mlm_head_dropout_rate: Dropout rate for the MLM head. Defaults to 0.1.
    :type mlm_head_dropout_rate: float
    :param layer_norm_eps: Epsilon for LayerNormalization in MLM head.
        Defaults to 1e-12.
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for the `keras.Model`.
    """

    def __init__(
            self,
            encoder: keras.Model,
            vocab_size: int,
            mask_ratio: float = 0.15,
            mask_token_id: int = 103,  # Default for BERT's [MASK]
            random_token_ratio: float = 0.1,
            unchanged_ratio: float = 0.1,
            special_token_ids: Optional[List[int]] = None,
            mlm_head_activation: str = "gelu",
            initializer_range: float = 0.02,
            mlm_head_dropout_rate: float = 0.1,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any,
    ) -> None:
        """Initialize the MaskedLanguageModel."""
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_config(
            vocab_size, mask_ratio, mask_token_id, random_token_ratio,
            unchanged_ratio, initializer_range, mlm_head_dropout_rate
        )

        # Store all configuration parameters
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.random_token_ratio = random_token_ratio
        self.unchanged_ratio = unchanged_ratio
        self.special_token_ids = special_token_ids or []
        self.mlm_head_activation = deserialize_activation(mlm_head_activation)
        self.initializer_range = initializer_range
        self.mlm_head_dropout_rate = mlm_head_dropout_rate
        self.layer_norm_eps = layer_norm_eps

        # Validate encoder has required attributes
        if not hasattr(self.encoder, "hidden_size"):
            raise ValueError(
                "The provided encoder must have a 'hidden_size' attribute."
            )
        self.hidden_size = self.encoder.hidden_size

        # Create MLM head components
        self.mlm_dense = keras.layers.Dense(
            self.hidden_size,
            activation=self.mlm_head_activation,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="mlm_dense",
        )
        self.mlm_dropout = keras.layers.Dropout(
            rate=self.mlm_head_dropout_rate,
            name="mlm_dropout"
        )
        self.mlm_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="mlm_norm"
        )
        self.mlm_output = keras.layers.Dense(
            self.vocab_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="mlm_output",
        )

        # Add metric trackers for manual control
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        logger.info(
            f"Created MaskedLanguageModel: vocab_size={self.vocab_size}, "
            f"mask_ratio={self.mask_ratio}, hidden_size={self.hidden_size}"
        )

    @property
    def metrics(self):
        """Metrics Keras tracks: the two internal trackers PLUS any compiled ones.

        This property previously returned ONLY `[loss_tracker, acc_metric]`.
        Because `train_step`/`test_step` build their return dict by iterating
        this same property, anything passed to `compile(metrics=[...])` was
        built by Keras and then never updated and never reported — it vanished
        silently, with no error and no warning.

        The compiled metrics are appended rather than substituted, so the two
        tracker names `loss` and `accuracy` keep their existing meaning; a large
        body of tests asserts on them.
        """
        tracked = [self.loss_tracker, self.acc_metric]
        compiled = getattr(self, "_compile_metrics", None)
        if compiled is not None:
            names = {m.name for m in tracked}
            # DECISION plan-2026-08-19T163559-499b6f0e/D-131: dedup must stay
            # name-based, since train_step keys its dict by name. Warn on a
            # collision instead of silently dropping the live metric. See decisions.md.
            dropped = sorted(m.name for m in compiled.metrics if m.name in names)
            if dropped and not getattr(self, "_warned_metric_name_clash", False):
                self._warned_metric_name_clash = True
                logger.warning(
                    f"compile(metrics=...) supplied {dropped}, which collide with this "
                    f"model's own trackers {sorted(names)} and will NOT be reported. "
                    f"Rename them to see their values."
                )
            tracked += [m for m in compiled.metrics if m.name not in names]
        return tracked

    def _update_compiled_metrics(
            self,
            labels: keras.KerasTensor,
            logits: keras.KerasTensor,
            sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Feed the compiled metrics, which nothing updated before this.

        Kept separate from the two internal trackers because those are updated
        with an explicit `masked_positions` weight and must keep that exact
        semantics; this only forwards the same signals to whatever the caller
        compiled.
        """
        compiled = getattr(self, "_compile_metrics", None)
        if compiled is None:
            return
        compiled.update_state(labels, logits, sample_weight=sample_weight)

    def _validate_config(
            self,
            vocab_size: int,
            mask_ratio: float,
            mask_token_id: int,
            random_token_ratio: float,
            unchanged_ratio: float,
            initializer_range: float,
            mlm_head_dropout_rate: float,
    ) -> None:
        """Validate model configuration parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not (0.0 < mask_ratio <= 1.0):
            raise ValueError(
                f"mask_ratio must be between 0 and 1, got {mask_ratio}"
            )
        if mask_token_id < 0 or mask_token_id >= vocab_size:
            raise ValueError(
                f"mask_token_id must be in [0, {vocab_size}), got {mask_token_id}"
            )
        if not (0.0 <= random_token_ratio <= 1.0):
            raise ValueError(
                f"random_token_ratio must be between 0 and 1, "
                f"got {random_token_ratio}"
            )
        if not (0.0 <= unchanged_ratio <= 1.0):
            raise ValueError(
                f"unchanged_ratio must be between 0 and 1, "
                f"got {unchanged_ratio}"
            )
        if random_token_ratio + unchanged_ratio > 1.0:
            raise ValueError(
                f"random_token_ratio + unchanged_ratio cannot exceed 1.0, "
                f"got {random_token_ratio + unchanged_ratio}"
            )
        if initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be positive, got {initializer_range}"
            )
        if not (0.0 <= mlm_head_dropout_rate < 1.0):
            raise ValueError(
                f"mlm_head_dropout_rate must be between 0 and 1, "
                f"got {mlm_head_dropout_rate}"
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method MaskedLanguageModel inherits ``Layer.build``, which marks the
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
            inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
            training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """Forward pass for prediction."""
        encoder_outputs = self.encoder(inputs, training=training)
        sequence_output = encoder_outputs["last_hidden_state"]
        logits = self._apply_mlm_head(sequence_output, training=training)
        return logits

    def _apply_mlm_head(
            self,
            sequence_output: keras.KerasTensor,
            training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """Apply the MLM prediction head to encoder outputs."""
        hidden_states = self.mlm_dense(sequence_output)
        hidden_states = self.mlm_dropout(hidden_states, training=training)
        hidden_states = self.mlm_norm(hidden_states)
        logits = self.mlm_output(hidden_states)
        return logits

    def _mask_tokens(
            self, inputs: Dict[str, keras.KerasTensor]
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, keras.KerasTensor]:
        """Delegates dynamic token masking to the centralized masking strategy."""
        masked_input_ids, labels, mask = apply_mlm_masking(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            vocab_size=self.vocab_size,
            mask_ratio=self.mask_ratio,
            mask_token_id=self.mask_token_id,
            special_token_ids=self.special_token_ids,
            random_token_ratio=self.random_token_ratio,
            unchanged_ratio=self.unchanged_ratio,
        )

        # Create a new inputs dictionary with the corrupted input_ids
        new_inputs = inputs.copy()
        new_inputs["input_ids"] = masked_input_ids

        return new_inputs, labels, mask

    def train_step(
            self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        """Custom training step for MLM with dynamic masking."""
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        masked_inputs, labels, masked_positions = self._mask_tokens(inputs)

        with tf.GradientTape() as tape:
            encoder_outputs = self.encoder(masked_inputs, training=True)
            sequence_output = encoder_outputs["last_hidden_state"]
            logits = self._apply_mlm_head(sequence_output, training=True)
            loss = self.compute_loss(
                y=labels,
                y_pred=logits,
                sample_weight=masked_positions,
            )
            # DECISION plan-2026-08-19T163559-499b6f0e/D-036: scale_loss must run
            # inside the tape and tape.gradient must differentiate the scaled value —
            # omitting it silently divides the whole update under mixed_float16. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Manually update the state of the metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(
            y_true=labels,
            y_pred=logits,
            sample_weight=masked_positions
        )
        self._update_compiled_metrics(labels, logits, masked_positions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(
            self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        """Custom validation step for MLM with dynamic masking."""
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        # Perform dynamic masking just like in the training step
        masked_inputs, labels, masked_positions = self._mask_tokens(inputs)

        # Forward pass
        encoder_outputs = self.encoder(masked_inputs, training=False)
        sequence_output = encoder_outputs["last_hidden_state"]
        logits = self._apply_mlm_head(sequence_output, training=False)

        # Compute loss
        loss = self.compute_loss(
            y=labels,
            y_pred=logits,
            sample_weight=masked_positions,
        )

        # Manually update the state of the metrics
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(
            y_true=labels,
            y_pred=logits,
            sample_weight=masked_positions
        )
        self._update_compiled_metrics(labels, logits, masked_positions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compute_loss(
            self,
            x: Optional[keras.KerasTensor] = None,
            y: Optional[keras.KerasTensor] = None,
            y_pred: Optional[keras.KerasTensor] = None,
            sample_weight: Optional[keras.KerasTensor] = None,
            **kwargs: Any,
    ) -> keras.KerasTensor:
        """Computes MLM loss on masked positions."""
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_fn(y, y_pred)

        if sample_weight is not None:
            # DECISION plan-2026-08-19T163559-499b6f0e/D-083: `keras.ops` on the
            # traced path. Only the `GradientTape` in `train_step` stays raw --
            # it is the documented exception (module docstring line 61).
            sample_weight = ops.cast(sample_weight, dtype=loss.dtype)
            loss = loss * sample_weight
            num_masked = ops.maximum(ops.sum(sample_weight), 1.0)
            return ops.sum(loss) / num_masked
        else:
            return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the model for serialization."""
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "vocab_size": self.vocab_size,
                "mask_ratio": self.mask_ratio,
                "mask_token_id": self.mask_token_id,
                "random_token_ratio": self.random_token_ratio,
                "unchanged_ratio": self.unchanged_ratio,
                "special_token_ids": self.special_token_ids,
                "mlm_head_activation": serialize_activation(self.mlm_head_activation),
                "initializer_range": self.initializer_range,
                "mlm_head_dropout_rate": self.mlm_head_dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedLanguageModel":
        """Creates a model from its configuration."""
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)
        return cls(encoder=encoder, **config)

# ------------------------------------------------------------------------