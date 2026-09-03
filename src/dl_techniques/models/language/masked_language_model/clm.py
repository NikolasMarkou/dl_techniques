"""
Causal language model pre-trainer that wraps a decoder backbone, shifts
inputs against labels by one position, and projects hidden states to
vocabulary logits through a head tied to the backbone's embedding matrix
when one can be found.

Next-token prediction scores every position in one forward pass, unlike
masked language modelling which scores roughly 15% of them. Because a
model-agnostic wrapper cannot inject a causal mask into an arbitrary
backbone, `build()` instead runs a future-leak probe: two forward passes
differing only at one position, checking that every earlier position's
hidden state is unchanged. A bidirectional backbone fails this with a
`ValueError` instead of training silently toward a collapsed loss. Weight
tying looks for the backbone's embedding matrix through several attribute
paths in order and falls back to an untied `Dense` head, with a warning,
if none match.

The shift convention is `x = input_ids[:, :-1]`, `y = input_ids[:, 1:]`,
applied inside `train_step`/`test_step`. The attention mask is sliced twice:
the input-aligned half feeds the backbone, the label-aligned half feeds the
loss and metrics — using the same slice for both would score the first
padding id as if it were a real label. The perplexity tracker averages
`exp(batch_loss)` over batches, which by Jensen's inequality is an upper
bound on corpus perplexity; exponentiate the tracked loss instead when
comparing against a perplexity computed from an aggregated loss.

The backbone must expose a `hidden_size` attribute and return a mapping
containing `last_hidden_state`. `train_step` uses `tf.GradientTape`
directly, so this model is TensorFlow-backend only. Set
`verify_causality=False` to skip the future-leak probe.

References:
    - Bengio et al., 2003. A Neural Probabilistic Language Model. JMLR 3:1137-1155.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
    - Press and Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
    - Inan et al., 2016. Tying Word Vectors and Word Classifiers: A Loss Framework for
      Language Modeling. (https://arxiv.org/abs/1611.01462)
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.masked_language_model.clm")
class CausalLanguageModel(keras.Model):
    """A model-agnostic Causal Language Modeling (CLM) pre-trainer.

    This model wraps a given causal backbone (like GPT) and adds the necessary
    logic for autoregressive pre-training.

    Weight tying: the model attempts to tie the output projection to the
    backbone's input embeddings during `build()`. If `tie_weights=False` is
    set explicitly, a standard Dense layer is created during initialization
    instead, so serialization has a layer to restore into.

    Causality: the backbone must be causal. `build()` verifies it with a
    future-leak probe and raises `ValueError` if a past position moves when
    a future token changes. Pass `verify_causality=False` to skip the check.

    :param backbone: An instance of a Keras model that acts as the decoder.
    :param vocab_size: The size of the vocabulary.
    :param initializer_range: Standard deviation for weight initialization.
    :param tie_weights: Whether to tie the output layer weights. Defaults to True.
    :param verify_causality: Whether to probe the backbone for future leakage at
        build time. Defaults to True.
    :param causality_tolerance: Maximum tolerated absolute change at a past
        position. Defaults to 0.0 - a genuinely masked contribution is exactly
        zero, so any movement at all is leakage.
    """

    def __init__(
        self,
        backbone: keras.Model,
        vocab_size: int,
        initializer_range: float = 0.02,
        tie_weights: bool = True,
        verify_causality: bool = True,
        causality_tolerance: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._validate_config(vocab_size, initializer_range)

        self.backbone = backbone
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.tie_weights = tie_weights
        self.verify_causality = verify_causality
        self.causality_tolerance = causality_tolerance

        if not hasattr(self.backbone, "hidden_size"):
            raise ValueError("The provided backbone must have a 'hidden_size' attribute.")
        self.hidden_size = self.backbone.hidden_size

        # Components
        self.embedding_weights = None
        self.output_bias = None

        # If tie_weights is False, we create the layer immediately.
        # This guarantees it exists for load_model() to restore weights into it.
        if not self.tie_weights:
            self.use_weight_tying = False
            self.output_layer = keras.layers.Dense(
                self.vocab_size,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="clm_output",
            )
        else:
            self.use_weight_tying = True  # Attempting to tie
            self.output_layer = None

        # Trackers
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.perplexity_metric = keras.metrics.Mean(name="perplexity")

    def _validate_config(self, vocab_size: int, initializer_range: float) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric, self.perplexity_metric]

    # DECISION plan-2026-08-19T163559-499b6f0e/D-035: match by variable shape
    # first, not `layer.weight` — that assumes the PyTorch spelling and raises
    # on an unbuilt Keras layer. See decisions.md D-035 and D-049.
    def _embedding_variable_of(
            self, layer: Any
    ) -> Optional[keras.KerasTensor]:
        """Return `layer`'s ``(vocab_size, hidden_size)`` variable, or None.

        Shape matching works for a built layer of any provenance and
        degrades to None (tying disabled) instead of crashing.

        Interface contract (2 callers by design):
            :param layer: Any object that may own the token-embedding variable.
            :returns: The variable whose shape is exactly
                ``(vocab_size, hidden_size)``, else the value of a
                ``embeddings`` / ``weight`` attribute if one is readable, else
                ``None``.
            :raises: Nothing. An unreadable attribute is treated as absent.
        """
        for variable in getattr(layer, "variables", ()):  # built layers only
            if tuple(variable.shape) == (self.vocab_size, self.hidden_size):
                return variable
        for attribute in ("embeddings", "weight"):
            try:
                value = getattr(layer, attribute)
            except (AttributeError, ValueError):
                continue
            if value is not None:
                return value
        return None

    def _locate_embedding_weights(self) -> Optional[keras.KerasTensor]:
        """Attempts to find the embedding weights in the backbone."""
        # 1. Explicit Method
        if hasattr(self.backbone, "get_embedding_matrix"):
            return self.backbone.get_embedding_matrix()

        # 2. Token Embeddings Layer (KerasNLP / Custom)
        if hasattr(self.backbone, "token_embeddings"):
            located = self._embedding_variable_of(self.backbone.token_embeddings)
            if located is not None:
                return located

        # 3. HF Style
        embeddings = getattr(self.backbone, "embeddings", None)
        if embeddings is not None:
            word_embeddings = getattr(embeddings, "word_embeddings", None)
            if word_embeddings is not None:
                located = self._embedding_variable_of(word_embeddings)
                if located is not None:
                    return located
            located = self._embedding_variable_of(embeddings)
            if located is not None:
                return located

        return None

    def build(self, input_shape):
        """Builds the model and initializes the output head/weight tying."""
        # 1. Ensure backbone is built to access its variables.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-049: this determines which
        # weight-tying branch is chosen, so it must give the same answer on
        # save and on load. Do not restore a bare except pass. See decisions.md.
        if not self.backbone.built:
            try:
                self.backbone.build(input_shape)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                logger.warning(
                    "Could not build the backbone from input_shape "
                    f"{input_shape} ({type(exc).__name__}: {exc}). Weight "
                    "tying will be resolved against whatever variables the "
                    "backbone already has, which may differ between save and "
                    "load."
                )

        # 2. Attempt Weight Tying logic if requested
        if self.tie_weights:
            self.embedding_weights = self._locate_embedding_weights()

            if self.embedding_weights is not None:
                self.use_weight_tying = True
                if self.output_bias is None:
                    self.output_bias = self.add_weight(
                        name="output_bias",
                        shape=(self.vocab_size,),
                        initializer="zeros",
                        trainable=True,
                    )
                logger.info("CLM Head initialized with Weight Tying enabled.")
            else:
                # Fallback to untied if embeddings not found
                if self.built:
                    logger.warning(
                        "Weight tying requested but embedding weights could not "
                        "be located. Falling back to untied weights."
                    )
                self.use_weight_tying = False
                if self.output_layer is None:
                    self.output_layer = keras.layers.Dense(
                        self.vocab_size,
                        kernel_initializer=keras.initializers.TruncatedNormal(
                            stddev=self.initializer_range
                        ),
                        name="clm_output",
                    )

        # 3. Ensure the output layer is built if it exists
        if self.output_layer is not None and not self.output_layer.built:
             self.output_layer.build((None, self.hidden_size))

        super().build(input_shape)

        # 4. Refuse a backbone that leaks the future (see module docstring).
        if self.verify_causality:
            self._verify_backbone_causality()

    def _verify_backbone_causality(
        self, seq_len: int = 8, batch_size: int = 2
    ) -> None:
        """Probe the backbone for future leakage and raise if it leaks.

        Runs the backbone twice over identical random ids that differ only at
        position ``t = seq_len // 2`` and compares ``last_hidden_state`` at every
        position before ``t``. A causal backbone gives a bit-identical prefix; a
        bidirectional one moves it.

        :param seq_len: Probe sequence length.
        :param batch_size: Probe batch size.
        :raises ValueError: If any position before ``t`` moves by more than
            ``causality_tolerance``.
        """
        split = seq_len // 2
        try:
            ids = keras.random.randint(
                (batch_size, seq_len), minval=0, maxval=self.vocab_size, seed=0
            )
            perturbed = ops.concatenate(
                [
                    ids[:, :split],
                    (ids[:, split:split + 1] + 1) % self.vocab_size,
                    ids[:, split + 1:],
                ],
                axis=1,
            )
            mask = ops.ones((batch_size, seq_len), dtype="int32")
            base = self.backbone(
                {"input_ids": ids, "attention_mask": mask}, training=False
            )["last_hidden_state"]
            moved = self.backbone(
                {"input_ids": perturbed, "attention_mask": mask}, training=False
            )["last_hidden_state"]
        except Exception as exc:  # noqa: BLE001 - the probe is best-effort
            logger.warning(
                "Could not run the causality probe on the backbone "
                f"({type(exc).__name__}: {exc}). Causality is UNVERIFIED; a "
                "bidirectional backbone here trains on leaked targets."
            )
            return

        leak = float(
            ops.convert_to_numpy(
                ops.max(ops.abs(base[:, :split] - moved[:, :split]))
            )
        )
        if leak > self.causality_tolerance:
            raise ValueError(
                "The backbone passed to CausalLanguageModel is NOT causal: "
                f"changing the token at position {split} moved the hidden "
                f"states at positions < {split} by {leak:.6e} (tolerance "
                f"{self.causality_tolerance:.6e}). Under a next-token "
                "objective every position would train on the token it is "
                "asked to predict. Supply a causally masked backbone, or pass "
                "verify_causality=False if you have another reason to believe "
                "this is safe."
            )
        logger.info(
            f"Backbone causality verified: past-position delta {leak:.6e}."
        )

    def call(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
        training: Optional[bool] = False,
    ) -> keras.KerasTensor:
        """Forward pass for prediction/generation."""
        backbone_outputs = self.backbone(inputs, training=training)
        sequence_output = backbone_outputs["last_hidden_state"]
        logits = self._apply_output_head(sequence_output)
        return logits

    def _apply_output_head(self, hidden_states: keras.KerasTensor) -> keras.KerasTensor:
        """Projects hidden states to vocabulary logits."""
        # JIT Build: Ensure components exist if build() wasn't called explicitly
        if self.use_weight_tying and self.embedding_weights is None:
            self.build(hidden_states.shape)
        elif not self.use_weight_tying and self.output_layer is None:
            self.build(hidden_states.shape)

        # Application
        if self.use_weight_tying and self.embedding_weights is not None:
            logits = ops.matmul(
                hidden_states,
                ops.transpose(self.embedding_weights)
            )
            logits = logits + self.output_bias
        else:
            # Fallback for untied or if weight tying failed
            logits = self.output_layer(hidden_states)

        return logits

    def _prepare_inputs_and_labels(
        self, inputs: Dict[str, keras.KerasTensor]
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Optional[keras.KerasTensor]]:
        """Prepares causal inputs by shifting tokens.

        Returns the shifted inputs, the shifted labels and the label-aligned
        loss weights. The backbone gets the input-aligned mask slice
        ``attention_mask[:, :-1]``; the loss gets ``attention_mask[:, 1:]``,
        because a weight multiplies a label. Using the input-aligned slice for
        both scores the final real token against a padding label.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        x_input_ids = input_ids[:, :-1]
        y_labels = input_ids[:, 1:]

        x_attention_mask = None
        loss_weights = None
        if attention_mask is not None:
            x_attention_mask = attention_mask[:, :-1]
            loss_weights = attention_mask[:, 1:]

        x_inputs = inputs.copy()
        x_inputs["input_ids"] = x_input_ids
        if x_attention_mask is not None:
            x_inputs["attention_mask"] = x_attention_mask

        return x_inputs, y_labels, loss_weights

    def train_step(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        x_inputs, y_labels, loss_weights = self._prepare_inputs_and_labels(inputs)

        with tf.GradientTape() as tape:
            backbone_outputs = self.backbone(x_inputs, training=True)
            sequence_output = backbone_outputs["last_hidden_state"]
            logits = self._apply_output_head(sequence_output)
            loss = self.compute_loss(y=y_labels, y_pred=logits, sample_weight=loss_weights)
            # DECISION plan-2026-08-19T163559-499b6f0e/D-036: scale_loss must run
            # inside the tape and tape.gradient must differentiate the scaled value —
            # omitting it silently divides the whole update under mixed_float16. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y_true=y_labels, y_pred=logits, sample_weight=loss_weights)
        self.perplexity_metric.update_state(ops.exp(loss))

        return {m.name: m.result() for m in self.metrics}

    def test_step(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        x_inputs, y_labels, loss_weights = self._prepare_inputs_and_labels(inputs)

        backbone_outputs = self.backbone(x_inputs, training=False)
        sequence_output = backbone_outputs["last_hidden_state"]
        logits = self._apply_output_head(sequence_output)
        loss = self.compute_loss(y=y_labels, y_pred=logits, sample_weight=loss_weights)

        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y_true=y_labels, y_pred=logits, sample_weight=loss_weights)
        self.perplexity_metric.update_state(ops.exp(loss))

        return {m.name: m.result() for m in self.metrics}

    def compute_loss(
        self,
        x: Optional[keras.KerasTensor] = None,
        y: Optional[keras.KerasTensor] = None,
        y_pred: Optional[keras.KerasTensor] = None,
        sample_weight: Optional[keras.KerasTensor] = None,
        **kwargs: Any,
    ) -> keras.KerasTensor:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_fn(y, y_pred)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=loss.dtype)
            loss = loss * sample_weight
            num_valid_tokens = ops.maximum(ops.sum(sample_weight), 1.0)
            return ops.sum(loss) / num_valid_tokens
        else:
            return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "vocab_size": self.vocab_size,
                "initializer_range": self.initializer_range,
                "tie_weights": self.tie_weights,
                "verify_causality": self.verify_causality,
                "causality_tolerance": self.causality_tolerance,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CausalLanguageModel":
        backbone_config = config.pop("backbone")
        backbone = keras.saving.deserialize_keras_object(backbone_config)
        return cls(backbone=backbone, **config)