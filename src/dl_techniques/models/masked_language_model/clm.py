"""
Causal language model pre-trainer that wraps a decoder backbone, shifts inputs against
labels by one position, and projects hidden states to vocabulary logits through an
output head whose weights are tied to the backbone's embedding matrix when one can be
found.

Next-token prediction is the exact chain-rule factorization of the sequence likelihood,
`log p(x) = sum_t log p(x_t | x_<t)`, which is what makes it both a valid density model
and unusually efficient supervision: a single forward pass over a length-`T` sequence
produces `T - 1` scored predictions, against roughly `0.15 * T` for masked language
modelling. Nothing is held out of the input, so no capacity is spent reconstructing
deliberately destroyed tokens.

That efficiency rests entirely on one structural condition: the representation at
position `t` must not depend on `x_{>t}`. **This module does not enforce it.** It never
constructs a causal attention mask. The only mask it touches is the caller's
`attention_mask`, which it treats purely as a padding indicator - slicing it to the
input length and later reusing it as loss weights - and forwards to the backbone
unchanged. Causality is the backbone's responsibility. Attach a bidirectionally
attending encoder here and position `t` reads the very token it is asked to predict;
the failure is silent and looks like spectacular success, with the loss collapsing
toward zero and accuracy toward one within a handful of steps. A pre-training run that
converges implausibly fast is evidence about the mask, not about the data.

The shift convention is `x = input_ids[:, :-1]` and `y = input_ids[:, 1:]`, applied
inside `train_step`/`test_step` so that datasets stay unshifted and the training and
evaluation paths cannot drift apart. One consequence is worth stating because it is
easy to misread: the sample weights are the *input*-position slice
`attention_mask[:, :-1]`, not a label-aligned mask. At the last real token the input
mask is 1 while the label is the first padding id, so that transition is scored and
learned, and every position after it is weighted 0. This is the right behaviour when
the padding id doubles as an end-of-sequence marker and the wrong one otherwise; to
suppress it, hand in a mask already zeroed at the final real position.

The output head is `logits = h @ E^T + b` when weight tying succeeds. Tying rests on the
observation that the input and output vocabularies index the same objects, so a single
matrix can serve both directions; it removes `hidden_size * vocab_size` parameters,
which for a large vocabulary is a substantial fraction of the model, and typically
improves perplexity. Locating `E` in an arbitrary backbone is necessarily heuristic and
is attempted in order: an explicit `get_embedding_matrix()` method, a `token_embeddings`
sublayer (searched for a variable of shape `(vocab_size, hidden_size)`, then
`.embeddings`, then `.weight`), and finally a HuggingFace-shaped
`embeddings.word_embeddings.weight` or `embeddings.weight`. If none matches, the model
falls back to an untied `Dense` instead of failing; the warning only fires once the
model is already built, so a tying request that quietly did not take is visible in the
logs but does not stop the run. A zero-initialized bias is added in the tied path even
though the tied kernel carries no bias of its own, because tying fixes the projection
directions but leaves the per-token frequency prior unmodelled.

Two construction-order choices are deliberate. With `tie_weights=False` the `Dense` head
is created in `__init__` rather than in `build`, so that `load_model()` always finds an
existing layer to restore weights into; in the tied path there is no kernel to restore
and only the bias needs to exist, so it is created in `build`. And `build` wraps the
backbone's own `build` in a bare `except`, because a backbone taking a dict of inputs
will often reject a single `input_shape`; proceeding is safe since the first real call
builds it anyway, whereas raising would break tying for every dict-input backbone.
`_apply_output_head` re-enters `build` if the head components are still missing, so a
direct `call()` without an explicit `build()` works.

The perplexity tracker averages `exp(batch_loss)` over batches rather than exponentiating
the averaged loss. By Jensen's inequality the reported figure is an upper bound on the
true corpus perplexity and is not comparable with a perplexity computed from an
aggregated loss; for reporting, exponentiate the tracked loss instead.

The backbone contract mirrors the masked-language-model wrapper: it must expose a
`hidden_size` attribute (a missing one raises `ValueError`) and return a mapping
containing `last_hidden_state`. `train_step` uses `tf.GradientTape` directly, so this
model is TensorFlow-backend only.

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


@keras.saving.register_keras_serializable()
class CausalLanguageModel(keras.Model):
    """A model-agnostic Causal Language Modeling (CLM) pre-trainer.

    This model wraps a given causal backbone (like GPT) and adds the necessary
    logic for autoregressive pre-training.

    **Weight Tying:**
    This model attempts to tie the weights of the output projection layer with
    the backbone's input embeddings. This logic is executed during the `build()`
    phase. If `tie_weights=False` is set explicitly, a standard Dense layer
    is created during initialization to ensure robust serialization.

    :param backbone: An instance of a Keras model that acts as the decoder.
    :param vocab_size: The size of the vocabulary.
    :param initializer_range: Standard deviation for weight initialization.
    :param tie_weights: Whether to tie the output layer weights. Defaults to True.
    """

    def __init__(
        self,
        backbone: keras.Model,
        vocab_size: int,
        initializer_range: float = 0.02,
        tie_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._validate_config(vocab_size, initializer_range)

        self.backbone = backbone
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.tie_weights = tie_weights

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

    def _locate_embedding_weights(self) -> Optional[keras.KerasTensor]:
        """Attempts to find the embedding weights in the backbone."""
        # 1. Explicit Method
        if hasattr(self.backbone, "get_embedding_matrix"):
            return self.backbone.get_embedding_matrix()

        # 2. Token Embeddings Layer (KerasNLP / Custom)
        if hasattr(self.backbone, "token_embeddings"):
            layer = self.backbone.token_embeddings
            if hasattr(layer, "variables"):
                for v in layer.variables:
                    if v.shape == (self.vocab_size, self.hidden_size):
                        return v
            if hasattr(layer, "embeddings"):
                return layer.embeddings
            if hasattr(layer, "weight"):
                return layer.weight

        # 3. HF Style
        if hasattr(self.backbone, "embeddings"):
            if hasattr(self.backbone.embeddings, "word_embeddings"):
                return self.backbone.embeddings.word_embeddings.weight
            if hasattr(self.backbone.embeddings, "weight"):
                return self.backbone.embeddings.weight

        return None

    def build(self, input_shape):
        """Builds the model and initializes the output head/weight tying."""
        # 1. Ensure backbone is built to access its variables
        if not self.backbone.built:
            try:
                self.backbone.build(input_shape)
            except Exception:
                # Proceed even if strict build fails (common with complex inputs)
                pass

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

        self.built = True
        super().build(input_shape)

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
        """Prepares causal inputs by shifting tokens."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        x_input_ids = input_ids[:, :-1]
        y_labels = input_ids[:, 1:]

        x_attention_mask = None
        if attention_mask is not None:
            x_attention_mask = attention_mask[:, :-1]

        x_inputs = inputs.copy()
        x_inputs["input_ids"] = x_input_ids
        if x_attention_mask is not None:
            x_inputs["attention_mask"] = x_attention_mask

        return x_inputs, y_labels, x_attention_mask

    def train_step(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        x_inputs, y_labels, padding_mask = self._prepare_inputs_and_labels(inputs)

        with tf.GradientTape() as tape:
            backbone_outputs = self.backbone(x_inputs, training=True)
            sequence_output = backbone_outputs["last_hidden_state"]
            logits = self._apply_output_head(sequence_output)
            loss = self.compute_loss(y=y_labels, y_pred=logits, sample_weight=padding_mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y_true=y_labels, y_pred=logits, sample_weight=padding_mask)
        self.perplexity_metric.update_state(ops.exp(loss))

        return {m.name: m.result() for m in self.metrics}

    def test_step(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        x_inputs, y_labels, padding_mask = self._prepare_inputs_and_labels(inputs)

        backbone_outputs = self.backbone(x_inputs, training=False)
        sequence_output = backbone_outputs["last_hidden_state"]
        logits = self._apply_output_head(sequence_output)
        loss = self.compute_loss(y=y_labels, y_pred=logits, sample_weight=padding_mask)

        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y_true=y_labels, y_pred=logits, sample_weight=padding_mask)
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
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CausalLanguageModel":
        backbone_config = config.pop("backbone")
        backbone = keras.saving.deserialize_keras_object(backbone_config)
        return cls(backbone=backbone, **config)