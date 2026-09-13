"""
``CausalLanguageModel`` wraps a decoder backbone into a next-token
pre-trainer: it shifts inputs against labels by one position and projects
hidden states to vocabulary logits through a head tied to the backbone's
embedding matrix when one can be found. Next-token prediction scores every
position in one forward pass, unlike masked language modelling, which scores
a small share of them. A model-agnostic wrapper cannot inject a causal mask
into an arbitrary backbone, so ``build`` runs a future-leak probe instead:
two forward passes that differ at one position, checking that every earlier
hidden state is unchanged. A bidirectional backbone raises ``ValueError``
there rather than training toward a collapsed loss. The backbone must expose
a ``hidden_size`` attribute and return a mapping containing
``last_hidden_state``, and ``train_step`` uses ``tf.GradientTape`` directly,
so this model runs on the TensorFlow backend only. Pass ``skip_head=True``
for a backbone that already bakes its own head and returns logits directly
as a plain tensor; ``hidden_size`` is then not required and no output head
is built. Pass ``pre_shifted=True`` when the batch already comes pre-shifted
(e.g. from ``preprocess_clm_packed_dataset``, which yields
``(input_ids, labels)`` tuples with ``labels`` shifted by one position
relative to ``input_ids``): ``train_step``/``test_step`` then use the
batch's own ``(x, y)`` unchanged instead of shifting again, with
``loss_weights=None`` since the packed pipeline never emits an
``attention_mask``. Pass ``loss_fn`` (a ``keras.losses.Loss`` instance) to
fully replace the default cross-entropy computation -- e.g. a focal-loss
or label-smoothed variant a trainer's config selects. When set,
``compute_loss`` delegates entirely to ``loss_fn(y, y_pred,
sample_weight=sample_weight)`` -- a single call, since a
``keras.losses.Loss`` already implements its own reduction, so the class's
own masked-mean logic is not stacked on top of it.

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tied_embeddings import tied_embedding_logits

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.masked_language_model.clm")
class CausalLanguageModel(keras.Model):
    """Pre-train a causal backbone with next-token prediction.

    The model shifts each sequence against itself, runs the shifted input
    through the backbone, and projects the hidden states to vocabulary logits.
    The shift convention is ``x = input_ids[:, :-1]``, ``y = input_ids[:, 1:]``,
    applied inside ``train_step`` and ``test_step``; ``call`` scores the inputs
    as given, so it is usable for generation.

    Weight tying: the output projection is tied to the backbone's input
    embeddings during ``build``. With ``tie_weights=False`` a ``Dense`` layer is
    created during initialization instead, so serialization has a layer to
    restore into.

    Causality: the backbone has to be causal. ``build`` checks it with a
    future-leak probe and raises ``ValueError`` if a past position moves when a
    future token changes. Pass ``verify_causality=False`` to skip the check.
    The probe's own backbone call defaults to the ``{"input_ids": ...,
    "attention_mask": ...}`` dict shape gemma/qwen/mamba/GPT2/WaveFieldLLM
    all accept; pass ``causality_probe_plain_tensor=True`` for a backbone
    whose ``call()`` takes only a plain positional tensor (e.g. Zamba2Model,
    HNet) -- without it, the probe crashes inside its own ``try/except`` and
    silently degrades to a "could not run the causality probe" warning,
    leaving ``verify_causality=True`` looking honored while nothing was
    actually checked.

    Pre-shifted batches: with ``pre_shifted=True``, ``train_step``/``test_step``
    skip ``_prepare_inputs_and_labels`` entirely and unpack ``data`` via
    ``keras.utils.unpack_x_y_sample_weight`` instead -- the unpacked ``x`` is
    used AS the model input and the unpacked ``y`` AS the labels, with no
    further shift applied. This is the contract
    ``preprocess_clm_packed_dataset`` already produces: ``(input_ids, labels)``
    tuples where ``labels = input_ids`` shifted by one position, computed once
    upstream of this class. Applying the internal shift on top would shift
    the batch a second time, training every position against the WRONG
    target. ``loss_weights`` defaults to ``None`` under this flag, since the
    packed pipeline never emits an ``attention_mask`` (no padding by
    construction) -- there is nothing to derive a mask from.

    Injectable loss: with ``loss_fn`` set to a ``keras.losses.Loss``
    instance, ``compute_loss`` delegates to it entirely instead of the
    default hardcoded ``SparseCategoricalCrossentropy`` -- one call,
    ``loss_fn(y, y_pred, sample_weight=sample_weight)``, since the loss
    object already implements its own reduction. This exists for callers
    whose trainer configures a non-default loss family (e.g. a focal
    variant, or a non-zero ``label_smoothing``) that the default CE cannot
    reproduce. Defaults to ``None``, preserving the original hardcoded-CE
    behavior exactly.

    The perplexity tracker averages ``exp(batch_loss)`` over batches, which by
    Jensen's inequality is an upper bound on corpus perplexity. Exponentiate
    the tracked loss instead when comparing against a perplexity computed from
    an aggregated loss.

    Architecture:

    .. code-block:: text

         inputs {input_ids, attention_mask}
                          │
                          ▼
                  ┌───────────────┐
                  │   backbone    │
                  └───────────────┘
                          │ last_hidden_state [B, L, H]
                          ▼
                  ┌───────────────┐
                  │  output head  │
                  └───────────────┘
                          │
                          ▼
              logits [B, L, vocab_size]

    Training and evaluation step:

    .. code-block:: text

                                  input_ids, attention_mask
                                              │
                                              ▼
                                        shift by one
              ┌───────────┬───────────────────┤
              ▼           ▼                   │
          y_labels  loss_weights              │
              │           │                   ▼
              │           │               x_inputs
              │           │                   │
              │           │                   ▼
              │           │            ┌─────────────┐
              │           │            │  backbone   │
              │           │            └─────────────┘
              │           │                   │
              │           │                   ▼
              │           │            ┌─────────────┐
              │           │            │ output head │
              │           │            └─────────────┘
              │           │                   │ logits
              ▼           ▼                   ▼
        ┌───────────────────────────────────────────────┐
        │  loss = sum(ce * w) / sum(w)                  │
        │  w = loss_weights                             │
        └───────────────────────────────────────────────┘
                                │
                                ▼
                   loss, accuracy, perplexity

    Shift convention:

    .. code-block:: text

        position        0    1    2    3    4
        input_ids       t0   t1   t2   t3   t4
        x_inputs        t0   t1   t2   t3
        y_labels             t1   t2   t3   t4
        backbone mask   m0   m1   m2   m3
        loss_weights         m1   m2   m3   m4

    The attention mask is sliced twice, since a weight multiplies a label.

    Output head:

    .. code-block:: text

                          tied                       untied

                      hidden_states               hidden_states
                            │                           │
                            ▼                           ▼
                   matmul embeddings^T          ┌───────────────┐
                            │                   │  clm_output   │
                            ▼                   └───────────────┘
                      + output_bias                     │
                            │                           ▼
                            ▼                        logits
                         logits

    The tied branch reuses the backbone's embedding matrix and adds its own bias.

    Weight tying lookup, tried in order:

    .. code-block:: text

                  backbone
                      │
                      ▼
        ┌─────────────────────────────────┐
        │  get_embedding_matrix()         │
        │  token_embeddings               │
        │  embeddings.word_embeddings     │
        │  embeddings                     │
        └─────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          tied head           untied Dense

    No match leaves the head untied and logs a warning.

    :param backbone: An instance of a Keras model that acts as the decoder. By
        default (``skip_head=False``) it must expose ``hidden_size`` and
        return a mapping containing ``last_hidden_state``. With
        ``skip_head=True`` it may instead be a backbone that already bakes
        its own head, returning vocabulary logits directly from ``call()``
        as a plain tensor -- ``hidden_size`` is then not required.
    :param vocab_size: The size of the vocabulary.
    :param initializer_range: Standard deviation for weight initialization.
    :param tie_weights: Whether to tie the output layer weights. Defaults to
        True. Ignored when ``skip_head`` is True, since no head is built.
    :param skip_head: When True, the backbone is assumed to already produce
        vocabulary logits (a plain tensor, not a ``last_hidden_state``
        mapping) and no output head, weight tying, or ``hidden_size`` check
        is performed. Defaults to False, preserving the original
        headless-backbone contract.
    :param pre_shifted: When True, ``train_step``/``test_step`` treat ``data``
        as an already-shifted ``(x, y)`` pair (unpacked via
        ``keras.utils.unpack_x_y_sample_weight``) and use it unchanged instead
        of shifting it again via ``_prepare_inputs_and_labels``.
        ``loss_weights`` is then always ``None``. Defaults to False,
        preserving the original internal-shift contract.
    :param loss_fn: An optional ``keras.losses.Loss`` instance that fully
        replaces ``compute_loss``'s default hardcoded
        ``SparseCategoricalCrossentropy`` computation. When set,
        ``compute_loss`` returns ``loss_fn(y, y_pred,
        sample_weight=sample_weight)`` directly -- a single call, since the
        loss object already implements its own reduction; the class's own
        masked-mean logic is not additionally applied. Defaults to ``None``,
        preserving the original hardcoded-CE contract exactly. Intended for
        a trainer whose config selects a non-default loss family (e.g. a
        focal loss, or CE with ``label_smoothing`` set).
    :param verify_causality: Whether to probe the backbone for future leakage at
        build time. Defaults to True.
    :param causality_tolerance: Maximum tolerated absolute change at a past
        position. Defaults to 0.0, since a masked contribution is exactly zero
        and any movement is leakage.
    :param causality_probe_plain_tensor: When True, the causality probe
        (``_verify_backbone_causality``) calls the backbone with a plain
        ``ids`` tensor -- no ``{"input_ids": ..., "attention_mask": ...}``
        dict -- for a backbone whose ``call()`` accepts only a positional
        tensor (e.g. Zamba2Model, HNet). Defaults to ``False``, which
        preserves the original hardcoded dict-probe call exactly: a
        dict-accepting backbone (gemma/qwen/mamba/GPT2/WaveFieldLLM) is
        probed as before, and a plain-tensor-only backbone crashes inside the
        probe's own ``try/except``, which logs a "could not run the
        causality probe" warning and returns -- ``verify_causality=True``
        then looks honored but the probe never actually ran. Set this to
        ``True`` for a plain-tensor-only backbone so the probe genuinely
        executes instead of silently degrading.
    :raises ValueError: If ``vocab_size`` or ``initializer_range`` is not
        positive, if ``skip_head`` is False and the backbone has no
        ``hidden_size`` attribute, or if the causality probe finds leakage.

    :ivar backbone: The wrapped decoder, saved and reused for fine-tuning.
    :ivar loss_tracker: Tracker behind the reported ``loss`` metric.
    :ivar acc_metric: Tracker behind the reported ``accuracy`` metric, scored
        on the label-aligned mask.
    :ivar perplexity_metric: Tracker behind the reported ``perplexity`` metric.
    """

    def __init__(
        self,
        backbone: keras.Model,
        vocab_size: int,
        initializer_range: float = 0.02,
        tie_weights: bool = True,
        skip_head: bool = False,
        pre_shifted: bool = False,
        loss_fn: Optional[keras.losses.Loss] = None,
        verify_causality: bool = True,
        causality_tolerance: float = 0.0,
        causality_probe_plain_tensor: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the CausalLanguageModel."""
        super().__init__(**kwargs)
        self._validate_config(vocab_size, initializer_range)

        self.backbone = backbone
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.tie_weights = tie_weights
        self.skip_head = skip_head
        self.pre_shifted = pre_shifted
        self.loss_fn = loss_fn
        self.verify_causality = verify_causality
        self.causality_tolerance = causality_tolerance
        self.causality_probe_plain_tensor = causality_probe_plain_tensor

        # The head width follows the backbone, so the contract is checked
        # here -- but only when a head is actually built: `skip_head=True`
        # backbones already bake their own head and never need `hidden_size`.
        if not self.skip_head and not hasattr(self.backbone, "hidden_size"):
            raise ValueError("The provided backbone must have a 'hidden_size' attribute.")
        self.hidden_size = self.backbone.hidden_size if not self.skip_head else None

        # Both are resolved in `build`, once tying is settled.
        self.embedding_weights = None
        self.output_bias = None

        # `skip_head=True` builds no head-related state at all: the backbone's
        # own output IS the logits. Otherwise an untied head is created now,
        # so `load_model` has a layer to restore weights into; the tied
        # branch resolves in `build`.
        if self.skip_head:
            self.use_weight_tying = False
            self.output_layer = None
        elif not self.tie_weights:
            self.use_weight_tying = False
            self.output_layer = keras.layers.Dense(
                self.vocab_size,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="clm_output",
            )
        else:
            self.use_weight_tying = True
            self.output_layer = None

        # Updated by hand in `train_step` and `test_step`.
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.perplexity_metric = keras.metrics.Mean(name="perplexity")

    def _validate_config(self, vocab_size: int, initializer_range: float) -> None:
        """Validate the constructor arguments.

        :raises ValueError: If either argument is not positive.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

    @property
    def metrics(self):
        """Return the three trackers ``train_step`` and ``test_step`` report.

        :return: The metrics Keras should track.
        """
        return [self.loss_tracker, self.acc_metric, self.perplexity_metric]

    # DECISION plan-2026-08-19T163559-499b6f0e/D-035: match by variable shape
    # first, not `layer.weight`. See decisions.md D-035 and D-049.
    def _embedding_variable_of(
            self, layer: Any
    ) -> Optional[keras.KerasTensor]:
        """Return `layer`'s ``(vocab_size, hidden_size)`` variable, or None.

        Shape matching works for a built layer of any provenance and returns
        None, which disables tying, instead of raising.

        :param layer: Any object that may own the token-embedding variable.
        :return: The variable whose shape is exactly
            ``(vocab_size, hidden_size)``, else the value of an ``embeddings``
            or ``weight`` attribute if one is readable, else ``None``.
        """
        # `variables` is populated only once the layer is built.
        for variable in getattr(layer, "variables", ()):
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
        """Attempts to find the embedding weights in the backbone.

        :return: The first matching embedding variable, or ``None`` if no
            attribute path resolves.
        """
        if hasattr(self.backbone, "get_embedding_matrix"):
            return self.backbone.get_embedding_matrix()

        # KerasNLP and custom backbones.
        if hasattr(self.backbone, "token_embeddings"):
            located = self._embedding_variable_of(self.backbone.token_embeddings)
            if located is not None:
                return located

        # Hugging Face nests the matrix under `embeddings`.
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
        """Builds the model and initializes the output head/weight tying.

        :param input_shape: Shape of the input to ``call``.
        :raises ValueError: If the causality probe finds future leakage.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-049: the backbone must be
        # built before tying resolves, and a failure gets reported. See decisions.md.
        if not self.backbone.built:
            try:
                self.backbone.build(input_shape)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not build the backbone from input_shape "
                    f"{input_shape} ({type(exc).__name__}: {exc}). Weight "
                    "tying will be resolved against whatever variables the "
                    "backbone already has, which may differ between save and "
                    "load."
                )

        # `skip_head=True` builds no output head at all: the backbone's own
        # output IS the logits, so the tie/untie resolution below is skipped
        # entirely (no `output_bias`/`output_layer`/`embedding_weights`).
        if not self.skip_head and self.tie_weights:
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
                # The first build has nothing to warn about yet.
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

        if self.output_layer is not None and not self.output_layer.built:
             self.output_layer.build((None, self.hidden_size))

        super().build(input_shape)

        # A leaking backbone fails here rather than mid-training.
        if self.verify_causality:
            self._verify_backbone_causality()

    # DECISION plan-2026-09-12T195532-422091c3/D-005: one shared helper for
    # the skip_head dict-vs-tensor branch, not the conditional repeated at 3
    # call sites (`call`, `train_step`/`test_step`, the causality probe). See
    # decisions.md D-005.
    def _backbone_forward(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the backbone once and return its output as a plain tensor.

        With ``skip_head=False`` (the original contract) the backbone
        returns a mapping and this extracts ``last_hidden_state``. With
        ``skip_head=True`` the backbone already bakes its own head and
        returns vocabulary logits directly from ``call()``, so its return
        value is passed straight through with no dict-indexing.

        :param inputs: Backbone inputs.
        :param training: Whether to run the backbone in training mode.
        :return: The backbone's hidden states, or its logits directly when
            ``skip_head`` is True.
        """
        backbone_outputs = self.backbone(inputs, training=training)
        if self.skip_head:
            return backbone_outputs
        return backbone_outputs["last_hidden_state"]

    def _verify_backbone_causality(
        self, seq_len: int = 8, batch_size: int = 2
    ) -> None:
        """Probe the backbone for future leakage and raise if it leaks.

        Runs the backbone twice over identical random ids that differ only at
        position ``t = seq_len // 2`` and compares ``last_hidden_state`` at every
        position before ``t``. A causal backbone gives a bit-identical prefix; a
        bidirectional one moves it. A probe that cannot run at all warns and
        returns.

        With ``causality_probe_plain_tensor=False`` (the default) the probe
        calls the backbone with the ``{"input_ids": ..., "attention_mask":
        ...}`` dict shape gemma/qwen/mamba/GPT2/WaveFieldLLM all accept. With
        ``causality_probe_plain_tensor=True``, ``ids``/``perturbed`` are
        passed directly as plain tensors -- no dict, no ``attention_mask`` --
        for a backbone whose ``call()`` takes only a positional tensor (e.g.
        Zamba2Model, HNet), which would otherwise crash inside the dict
        probe and silently degrade to the "could not run" warning below.

        # DECISION plan-2026-09-13T052422-19022ba2/D-003: the dict probe call
        # is hardcoded and crashes for a plain-tensor-only backbone
        # (Zamba2Model, HNet); do NOT auto-detect via
        # inspect.signature(backbone.call) instead of this explicit flag --
        # a Union[Tensor, Dict] type hint is not reliably present on every
        # backbone, so introspection would silently reproduce the exact
        # failure this fix exists to close. See decisions.md D-003.

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
            if self.causality_probe_plain_tensor:
                base = self._backbone_forward(ids, training=False)
                moved = self._backbone_forward(perturbed, training=False)
            else:
                mask = ops.ones((batch_size, seq_len), dtype="int32")
                base = self._backbone_forward(
                    {"input_ids": ids, "attention_mask": mask}, training=False
                )
                moved = self._backbone_forward(
                    {"input_ids": perturbed, "attention_mask": mask}, training=False
                )
        except Exception as exc:  # noqa: BLE001
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
        """Score the inputs as given, with no shift applied.

        :param inputs: Backbone inputs, a mapping with ``input_ids`` and any
            other keys the backbone takes.
        :param training: Whether to run in training mode. Defaults to False.
        :return: Logits of shape (batch, seq_len, vocab_size).
        """
        sequence_output = self._backbone_forward(inputs, training=training)
        if self.skip_head:
            return sequence_output
        return self._apply_output_head(sequence_output)

    def _apply_output_head(self, hidden_states: keras.KerasTensor) -> keras.KerasTensor:
        """Projects hidden states to vocabulary logits.

        :param hidden_states: Backbone output, shape (batch, seq_len, hidden_size).
        :return: Logits of shape (batch, seq_len, vocab_size).
        """
        # `call` can run before an explicit build, so the head resolves here.
        if self.use_weight_tying and self.embedding_weights is None:
            self.build(hidden_states.shape)
        elif not self.use_weight_tying and self.output_layer is None:
            self.build(hidden_states.shape)

        if self.use_weight_tying and self.embedding_weights is not None:
            logits = tied_embedding_logits(
                hidden_states, self.embedding_weights, bias=self.output_bias
            )
        else:
            logits = self.output_layer(hidden_states)

        return logits

    def _prepare_inputs_and_labels(
        self, inputs: Dict[str, keras.KerasTensor]
    ) -> Tuple[Dict[str, keras.KerasTensor], keras.KerasTensor, Optional[keras.KerasTensor]]:
        """Prepares causal inputs by shifting tokens.

        The backbone gets the input-aligned mask slice
        ``attention_mask[:, :-1]``; the loss gets ``attention_mask[:, 1:]``,
        because a weight multiplies a label. Using the input-aligned slice for
        both scores the final real token against a padding label.

        :param inputs: Mapping with ``input_ids`` and optionally
            ``attention_mask``.
        :return: The shifted inputs, the shifted labels, and the label-aligned
            loss weights, which are ``None`` when no mask was supplied.
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

    def _unpack_batch(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Tuple[
        Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
        keras.KerasTensor,
        Optional[keras.KerasTensor],
    ]:
        """Resolve one batch into ``(x_inputs, y_labels, loss_weights)``.

        Shared by ``train_step`` and ``test_step`` so the ``pre_shifted``
        branch is written once, not twice.

        With ``pre_shifted=False`` (the original contract), ``data`` is
        unpacked for its ``inputs`` mapping only -- any ``y``/sample weight
        Keras also unpacked is ignored, since the labels come from
        ``_prepare_inputs_and_labels``'s internal shift instead.

        With ``pre_shifted=True``, ``data`` is unpacked into ``(x, y)`` via
        ``keras.utils.unpack_x_y_sample_weight`` and returned AS GIVEN, with
        no further shift: ``x`` is already the shifted model input and ``y``
        is already the shifted labels (the contract
        ``preprocess_clm_packed_dataset`` produces upstream). Applying
        ``_prepare_inputs_and_labels`` on top here would shift a second time.
        ``loss_weights`` is always ``None`` under this flag, since the packed
        pipeline never emits an ``attention_mask``.

        :param data: A batch of inputs, or a tuple Keras unpacks into inputs,
            targets and sample weights.
        :return: ``(x_inputs, y_labels, loss_weights)``, ready for the
            backbone and ``compute_loss``.
        """
        if self.pre_shifted:
            x_inputs, y_labels, _ = keras.utils.unpack_x_y_sample_weight(data)
            return x_inputs, y_labels, None

        if isinstance(data, tuple):
            inputs, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        else:
            inputs = data

        return self._prepare_inputs_and_labels(inputs)

    def train_step(
        self, data: Union[Dict[str, keras.KerasTensor], Tuple]
    ) -> Dict[str, keras.KerasTensor]:
        """Shift the batch, take one optimizer step, and report the metrics.

        :param data: A batch of inputs, or a tuple Keras unpacks into inputs,
            targets and sample weights. Targets are ignored, since the labels
            come from the shift.
        :return: Mapping from metric name to current value.
        """
        x_inputs, y_labels, loss_weights = self._unpack_batch(data)

        with tf.GradientTape() as tape:
            sequence_output = self._backbone_forward(x_inputs, training=True)
            logits = sequence_output if self.skip_head else self._apply_output_head(sequence_output)
            loss = self.compute_loss(y=y_labels, y_pred=logits, sample_weight=loss_weights)
            # DECISION plan-2026-08-19T163559-499b6f0e/D-036: scale_loss runs inside
            # the tape; skipping it shrinks every mixed_float16 update. See decisions.md.
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
        """Shift the batch, score it without a gradient, and report metrics.

        :param data: A batch of inputs, or a tuple Keras unpacks into inputs,
            targets and sample weights. Targets are ignored.
        :return: Mapping from metric name to current value.
        """
        x_inputs, y_labels, loss_weights = self._unpack_batch(data)

        sequence_output = self._backbone_forward(x_inputs, training=False)
        logits = sequence_output if self.skip_head else self._apply_output_head(sequence_output)
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
        """Compute cross entropy over all positions, reduced by the weights.

        :param x: Unused, present for the ``keras.Model`` signature.
        :param y: Shifted token ids, shape (batch, seq_len - 1).
        :param y_pred: Head logits, shape (batch, seq_len - 1, vocab_size).
        :param sample_weight: Label-aligned mask. Without it the result is the
            plain mean over every position.
        :return: Scalar loss.
        """
        # DECISION plan-2026-09-12T195532-422091c3/D-007: gemma/qwen already
        # support --loss-type focal/label-smoothing via create_clm_loss_fn;
        # the default hardcoded CE below cannot reproduce either, so do NOT
        # migrate them onto this class without this injection point -- that
        # would silently regress a working feature. See decisions.md D-007.
        if self.loss_fn is not None:
            # `keras.losses.Loss.__call__` already implements its own
            # reduction -- a single call, not stacked with the masked-mean
            # logic below, which would double-reduce.
            return self.loss_fn(y, y_pred, sample_weight=sample_weight)

        default_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = default_loss_fn(y, y_pred)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=loss.dtype)
            loss = loss * sample_weight
            num_valid_tokens = ops.maximum(ops.sum(sample_weight), 1.0)
            return ops.sum(loss) / num_valid_tokens
        else:
            return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the model for serialization.

        :return: Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "vocab_size": self.vocab_size,
                "initializer_range": self.initializer_range,
                "tie_weights": self.tie_weights,
                "skip_head": self.skip_head,
                "pre_shifted": self.pre_shifted,
                "loss_fn": (
                    keras.losses.serialize(self.loss_fn)
                    if self.loss_fn is not None
                    else None
                ),
                "verify_causality": self.verify_causality,
                "causality_tolerance": self.causality_tolerance,
                "causality_probe_plain_tensor": self.causality_probe_plain_tensor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CausalLanguageModel":
        """Creates a model from its configuration.

        :param config: Configuration produced by ``get_config``.
        :return: The deserialized model, with its backbone rebuilt first.
        """
        backbone_config = config.pop("backbone")
        backbone = keras.saving.deserialize_keras_object(backbone_config)
        loss_fn_config = config.pop("loss_fn", None)
        loss_fn = (
            keras.losses.deserialize(loss_fn_config)
            if loss_fn_config is not None
            else None
        )
        return cls(backbone=backbone, loss_fn=loss_fn, **config)

# ---------------------------------------------------------------------
