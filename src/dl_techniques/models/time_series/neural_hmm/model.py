"""
A Neural Hidden Markov Model: a K-state HMM whose initial distribution, transitions and
emissions are produced by neural layers, trained by exact maximum likelihood through the
forward algorithm. :class:`NeuralHMM` composes four parts, each its own module:

- an optional sequence encoder (:class:`~.encoders.NeuralHMMEncoder`) that turns the
  observations into a per-step context ``h``;
- a transition layer (:class:`~.transitions.StaticTransition`, or
  :class:`~.transitions.InputDependentTransition` reading ``h``) giving ``log pi`` and
  ``log A``;
- an emission layer: :class:`~.emissions.GenerativeEmission` (a normalized density
  ``log p(x_t | z_t = k)``, optionally reading ``h``) or
  :class:`~.emissions.DiscriminativeEmission` (CRF potentials read from ``h``);
- exact inference (:mod:`.inference`): the forward algorithm for the loss, Viterbi
  decoding and forward-backward posteriors.

.. code-block:: text

    {"obs", "mask"}
        │
        ├─ categorical and an encoder: Embedding(vocab_size, hidden_dim)
        ├─ generative: shift right one step (step t sees x_{<t})
        ▼
    NeuralHMMEncoder ── h (B, T, hidden_dim)       built only when something reads h
        │
        ├──► transition ──► log_pi (B, K), log_A (B, T, K, K)
        └──► emission   ──► log_emit (B, T, K)
                                │
    compute_loss: forward algorithm (generative: -log p(x); discriminative:
    log Z - score(gold path)), per valid step

``call`` returns the three log-potentials on one batched shape; ``log_A[:, t]`` is the
transition INTO step ``t`` and its ``t = 0`` slot is never read. Every output is in float32
(float64 under a float64 policy), also under ``mixed_float16``: the encoder and the heads run
in the compute dtype, and the log-space math does not.

**Modes.** ``generative`` maximizes ``log p(x)`` without labels; the encoder is causal and
reads the input shifted right by one step, so the emission of ``x_t`` is conditioned on
``x_{<t}`` only. ``discriminative`` is a linear-chain CRF trained on integer state labels,
``log p(y | x) = score(y) - log Z``; its encoder is non-causal and reads the whole sequence.

**Padding.** ``mask`` marks a contiguous valid prefix (right padding). A padded step
contributes nothing, so the loss of a padded sequence equals that of the truncated one.

**Training.** Stock ``fit()``: the loss is this class's ``compute_loss`` override, and
``compile(loss=...)`` is ignored. ``sample_weight`` weights whole sequences.

**Variants.** :attr:`NeuralHMM.MODEL_VARIANTS` holds five presets, each with every built
component live. No pretrained weights are distributed; ``pretrained=True`` raises
``NotImplementedError``.

References:
    - Tran, Bisk, Vaswani, Marcu and Knight, 2016. Unsupervised Neural Hidden Markov
      Models. Proceedings of the Workshop on Structured Prediction for NLP.
      (https://arxiv.org/abs/1609.09007)
    - Bengio and Frasconi, 1995. An Input Output HMM Architecture. Advances in Neural
      Information Processing Systems 7, pp. 427-434.
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected Applications in
      Speech Recognition. Proceedings of the IEEE 77(2), pp. 257-286.
    - Lample, Ballesteros, Subramanian, Kawakami and Dyer, 2016. Neural Architectures for
      Named Entity Recognition. NAACL-HLT. (https://arxiv.org/abs/1603.01360)
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

from . import inference
from .emissions import OBS_TYPES, DiscriminativeEmission, GenerativeEmission
from .encoders import ENCODER_TYPES, NeuralHMMEncoder
from .inference import LogSpaceOutputSpec
from .transitions import InputDependentTransition, StaticTransition

# ---------------------------------------------------------------------

#: Training regimes :class:`NeuralHMM` accepts.
MODES: Tuple[str, ...] = ("generative", "discriminative")

#: Whether the generative emission also reads the encoder context.
EMISSION_CONTEXTS: Tuple[str, ...] = ("none", "autoregressive")

#: Transition layer types.
TRANSITIONS: Tuple[str, ...] = ("static", "input_dependent")

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.neural_hmm.model")
class NeuralHMM(LogSpaceOutputSpec, keras.Model):
    """A Neural HMM over right-padded sequences, generative or discriminative.

    The encoder is built only when something reads its context: an autoregressive
    emission, input-dependent transitions, or the discriminative mode. Otherwise
    ``self.encoder`` is ``None`` and every argument read only by the encoder is unused.

    :param n_states: Number of hidden states ``K``; positive.
    :param mode: ``generative`` (unsupervised, maximizes ``log p(x)``) or
        ``discriminative`` (a linear-chain CRF trained on state labels).
    :param obs_type: ``gaussian`` (real ``(B, T, obs_dim)``), ``poisson`` (counts
        ``(B, T, obs_dim)``) or ``categorical`` (symbols ``(B, T)`` in ``[0, vocab_size)``).
    :param obs_dim: Feature count of a ``gaussian`` or ``poisson`` observation; required
        for those and must be ``None`` for ``categorical``.
    :param vocab_size: Symbol count of a ``categorical`` observation; required for it and
        must be ``None`` otherwise.
    :param emission_context: ``none``, or ``autoregressive`` (the generative emission also
        reads ``h_t``, built from ``x_{<t}``). Must be ``none`` in discriminative mode.
    :param transition: ``static``, or ``input_dependent`` (per-step transitions from ``h``,
        an input-output HMM).
    :param topology: ``ergodic``, ``left_to_right`` or ``banded``.
    :param band_width: Largest forward skip under ``banded``; non-negative.
    :param self_loop_init: Initial diagonal of the transition logits.
    :param min_scale: Floor of the Gaussian scale and the Poisson rate (generative only).
    :param hidden_dim: Width of the encoder, the observation embedding and the emission
        head; positive.
    :param state_emb_dim: Width of the learned state embeddings of the generative
        emission; positive.
    :param encoder_type: ``mlp``, ``transformer``, ``lstm``, ``gru`` or ``conv``.
    :param num_layers: Encoder blocks or stacked recurrent layers; positive.
    :param num_heads: Attention heads of the ``transformer`` encoder; positive.
    :param intermediate_size: Feed-forward width inside the ``mlp``/``transformer``
        encoder; positive.
    :param max_seq_len: Longest sequence the ``transformer`` encoder's learned position
        table covers; positive.
    :param kernel_size: Convolution width of the ``conv`` encoder; positive.
    :param ffn_type: Feed-forward type of the ``transformer`` encoder's blocks (see
        :func:`dl_techniques.layers.ffn.factory.create_ffn_layer`); read only when
        ``encoder_type='transformer'`` and an encoder is built. The generative emission
        head is always an ``mlp`` block and does not read it.
    :param normalization_type: Normalization type inside the encoder.
    :param dropout_rate: Encoder dropout in ``[0, 1)``.
    :param attention_dropout_rate: Attention-probability dropout of the ``transformer``
        encoder, in ``[0, 1)``.
    :param kwargs: Base ``keras.Model`` arguments.
    :raises ValueError: If an argument is out of range, unknown, missing for the
        ``obs_type``, or the combination is invalid.
    """

    # Each preset carries only the structural choices; the caller supplies the data
    # shape (n_states, obs_type, obs_dim or vocab_size). Every component a preset builds
    # is read by the forward pass, so every built weight trains (D-005).
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # Tran et al., 2016 (arXiv:1609.09007): emissions from learned state embeddings
        # through a feed-forward head; static transitions; no encoder.
        "neural-hmm": {
            "mode": "generative",
            "emission_context": "none",
            "transition": "static",
            "topology": "ergodic",
        },
        # Tran et al., 2016, with the emission also conditioned on the observed history
        # x_{<t} through a causal LSTM.
        "autoregressive-neural-hmm": {
            "mode": "generative",
            "emission_context": "autoregressive",
            "transition": "static",
            "topology": "ergodic",
            "encoder_type": "lstm",
        },
        # Bengio and Frasconi, 1995 (NIPS 7, pp. 427-434): an input-output HMM whose
        # transition into step t is driven by the recent inputs x_{t-k..t-1} (causal conv).
        "io-hmm": {
            "mode": "generative",
            "emission_context": "none",
            "transition": "input_dependent",
            "topology": "ergodic",
            "encoder_type": "conv",
        },
        # Rabiner, 1989, left-to-right models: a banded chain that only stays or moves one
        # state forward, with emissions conditioned on x_{<t} by a causal transformer.
        "left-to-right-segmenter": {
            "mode": "generative",
            "emission_context": "autoregressive",
            "transition": "static",
            "topology": "banded",
            "band_width": 1,
            "encoder_type": "transformer",
        },
        # Lample et al., 2016 (arXiv:1603.01360): a linear-chain CRF over neural potentials.
        # The paper's encoder is a BiLSTM; this preset uses a non-causal transformer.
        "neural-crf-tagger": {
            "mode": "discriminative",
            "emission_context": "none",
            "transition": "static",
            "topology": "ergodic",
            "encoder_type": "transformer",
        },
    }

    def __init__(
        self,
        n_states: int,
        mode: str = "generative",
        obs_type: str = "gaussian",
        obs_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        emission_context: str = "none",
        transition: str = "static",
        topology: str = "ergodic",
        band_width: int = 1,
        self_loop_init: float = 2.0,
        min_scale: float = 1e-3,
        hidden_dim: int = 64,
        state_emb_dim: int = 16,
        encoder_type: str = "lstm",
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 128,
        max_seq_len: int = 512,
        kernel_size: int = 3,
        ffn_type: str = "mlp",
        normalization_type: str = "layer_norm",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        for name, value, allowed in (
            ("mode", mode, MODES),
            ("obs_type", obs_type, OBS_TYPES),
            ("emission_context", emission_context, EMISSION_CONTEXTS),
            ("transition", transition, TRANSITIONS),
            ("encoder_type", encoder_type, ENCODER_TYPES),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown {name} '{value}'. Available: {list(allowed)}.")
        if obs_type == "categorical":
            if vocab_size is None or vocab_size <= 0 or obs_dim is not None:
                raise ValueError(
                    f"obs_type='categorical' needs a positive vocab_size and obs_dim=None, "
                    f"got vocab_size={vocab_size}, obs_dim={obs_dim}."
                )
        elif obs_dim is None or obs_dim <= 0 or vocab_size is not None:
            raise ValueError(
                f"obs_type='{obs_type}' needs a positive obs_dim and vocab_size=None, "
                f"got obs_dim={obs_dim}, vocab_size={vocab_size}."
            )
        if mode == "discriminative" and emission_context != "none":
            raise ValueError(
                f"mode='discriminative' scores states from the encoder context already; "
                f"emission_context must be 'none', got '{emission_context}'."
            )

        self.n_states = n_states
        self.mode = mode
        self.obs_type = obs_type
        self.obs_dim = obs_dim
        self.vocab_size = vocab_size
        self.emission_context = emission_context
        self.transition_type = transition
        self.topology = topology
        self.band_width = band_width
        self.self_loop_init = self_loop_init
        self.min_scale = min_scale
        self.hidden_dim = hidden_dim
        self.state_emb_dim = state_emb_dim
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        generative = mode == "generative"
        reads_context = emission_context == "autoregressive"
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-005: build the encoder ONLY when its
        # context is read. Do NOT build it unconditionally "for a uniform architecture": in
        # the default configuration nothing reads h, so every encoder weight would be saved,
        # counted and never receive a gradient (the old package's default variant did this).
        self.encoder = None
        self.obs_embedding = None
        if reads_context or transition == "input_dependent" or not generative:
            self.encoder = NeuralHMMEncoder(
                encoder_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                causal=generative,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                kernel_size=kernel_size,
                ffn_type=ffn_type,
                normalization_type=normalization_type,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                name="encoder",
            )
            if obs_type == "categorical":
                self.obs_embedding = keras.layers.Embedding(vocab_size, hidden_dim, name="obs_embedding")

        transition_class = InputDependentTransition if transition == "input_dependent" else StaticTransition
        self.transition = transition_class(
            n_states, topology=topology, band_width=band_width, self_loop_init=self_loop_init, name="transition"
        )
        if generative:
            self.emission = GenerativeEmission(
                n_states,
                obs_type,
                obs_dim=obs_dim,
                vocab_size=vocab_size,
                state_emb_dim=state_emb_dim,
                hidden_dim=hidden_dim,
                context_dim=hidden_dim if reads_context else 0,
                min_scale=min_scale,
                name="emission",
            )
        else:
            self.emission = DiscriminativeEmission(n_states, name="emission")

        logger.info(
            f"NeuralHMM: mode={mode}, n_states={n_states}, obs_type={obs_type}, "
            f"transition={transition}, topology={topology}, emission_context={emission_context}, "
            f"encoder={encoder_type if self.encoder is not None else None}"
        )

    # -----------------------------------------------------------------
    # Build and forward
    # -----------------------------------------------------------------

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """Build every sublayer in computational order.

        :param input_shape: ``{"obs": (B, T) or (B, T, obs_dim), "mask": (B, T)}``; the
            ``mask`` entry is optional.
        :raises ValueError: If ``input_shape`` is not a dict with an ``obs`` entry of the
            rank the ``obs_type`` needs.
        """
        if not isinstance(input_shape, dict) or "obs" not in input_shape:
            raise ValueError(f"NeuralHMM expects inputs {{'obs', 'mask'}}, got shape {input_shape}.")
        obs_shape = tuple(input_shape["obs"])
        expected_rank = 2 if self.obs_type == "categorical" else 3
        if len(obs_shape) != expected_rank:
            raise ValueError(
                f"obs_type='{self.obs_type}' expects observations of rank {expected_rank}, got {obs_shape}."
            )
        batch, steps = obs_shape[0], obs_shape[1]
        context_shape = (batch, steps, self.hidden_dim)
        if self.encoder is not None:
            features_shape = obs_shape
            if self.obs_embedding is not None:
                self.obs_embedding.build((batch, steps))
                features_shape = context_shape
            self.encoder.build(features_shape)
        self.transition.build(context_shape if self.transition_type == "input_dependent" else obs_shape)
        if self.mode == "generative":
            self.emission.build(obs_shape, context_shape if self.emission_context == "autoregressive" else None)
        else:
            self.emission.build(context_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Return the HMM's log-potentials for every sequence.

        :param inputs: ``{"obs", "mask"}``. ``obs`` is ``(B, T)`` integer symbols
            (``categorical``) or ``(B, T, obs_dim)`` floats; ``mask`` is ``(B, T)``,
            nonzero at valid steps, a contiguous valid prefix. Without ``mask`` every
            step is valid. Padded content may hold anything, including NaN or ``-1``.
        :param training: Forwarded to every sublayer.
        :return: ``{"log_pi": (B, K), "log_A": (B, T, K, K), "log_emit": (B, T, K)}`` in the
            accumulation dtype (float32 under float32 and mixed precision).
        """
        obs, valid = self._observations(inputs)
        context = None
        if self.encoder is not None:
            features = self._features(obs, valid)
            if self.mode == "generative":
                # Shift right by one step AFTER the embedding, so that the zero start
                # vector is not the embedding of a real symbol (token 0 is one).
                features = keras.ops.concatenate([keras.ops.zeros_like(features[:, :1]), features[:, :-1]], axis=1)
            context = self.encoder(features, mask=valid, training=training)

        transition_input = context if self.transition_type == "input_dependent" else obs
        log_pi, log_A = self.transition(transition_input, training=training)
        if self.mode == "generative":
            emission_context = context if self.emission_context == "autoregressive" else None
            log_emit = self.emission(obs, emission_context, training=training)
        else:
            log_emit = self.emission(context, training=training)
        return {"log_pi": log_pi, "log_A": log_A, "log_emit": log_emit}

    def _observations(self, inputs: Dict[str, keras.KerasTensor]) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return the observations with padded steps neutralized, and the boolean mask.

        Padded float steps become 0 and padded symbols ``-1`` (which every categorical
        emission scores 0). The inference functions ignore padded steps, but a NaN there
        would still reach the weights' gradients, since ``where`` passes ``0 * NaN``
        back through the branch it did not select.

        :param inputs: ``{"obs", "mask"}``.
        :return: ``(obs, valid)``, ``valid`` a ``(B, T)`` boolean tensor.
        """
        obs = inputs["obs"]
        mask = inputs.get("mask")
        if self.obs_type == "categorical":
            obs = keras.ops.cast(obs, "int32")
            valid = keras.ops.ones_like(obs, dtype="bool") if mask is None else keras.ops.cast(mask, "bool")
            return keras.ops.where(valid, obs, keras.ops.full_like(obs, -1)), valid
        valid = (
            keras.ops.ones_like(obs[:, :, 0], dtype="bool") if mask is None else keras.ops.cast(mask, "bool")
        )
        return keras.ops.where(valid[:, :, None], obs, keras.ops.zeros_like(obs)), valid

    def _features(self, obs: keras.KerasTensor, valid: keras.KerasTensor) -> keras.KerasTensor:
        """Return the encoder input: embedded symbols, or the observations themselves.

        :param obs: Neutralized observations from :meth:`_observations`.
        :param valid: ``(B, T)`` boolean mask.
        :return: ``(B, T, hidden_dim)`` for ``categorical``, else ``(B, T, obs_dim)``.
        """
        if self.obs_embedding is None:
            return obs
        in_range = keras.ops.logical_and(valid, keras.ops.logical_and(obs >= 0, obs < self.vocab_size))
        return self.obs_embedding(keras.ops.where(in_range, obs, keras.ops.zeros_like(obs)))

    # -----------------------------------------------------------------
    # Loss and inference helpers
    # -----------------------------------------------------------------

    def _valid_mask(self, inputs: Dict[str, Any], log_emit: keras.KerasTensor) -> keras.KerasTensor:
        """Return the ``(B, T)`` float mask of ``inputs`` (all ones when it has none).

        :param inputs: ``{"obs", "mask"}``.
        :param log_emit: ``(B, T, K)`` model output, read for its shape and dtype only.
        :return: ``(B, T)`` mask, 1.0 at valid steps.
        """
        mask = inputs.get("mask")
        if mask is None:
            return keras.ops.ones_like(log_emit[:, :, 0])
        return keras.ops.cast(keras.ops.cast(mask, "bool"), log_emit.dtype)

    def _sequence_log_likelihood(
        self,
        outputs: Dict[str, keras.KerasTensor],
        mask: keras.KerasTensor,
        states: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Return ``log p(x)`` (generative) or ``log p(states | x)`` (discriminative).

        :param outputs: The :meth:`call` dict.
        :param mask: ``(B, T)`` mask.
        :param states: ``(B, T)`` integer states; required in discriminative mode, must be
            ``None`` in generative mode.
        :return: ``(B,)`` log-likelihood per sequence.
        :raises ValueError: If ``states`` is missing or superfluous for the mode.
        """
        log_pi, log_A, log_emit = outputs["log_pi"], outputs["log_A"], outputs["log_emit"]
        log_z = inference.forward_log_likelihood(log_pi, log_A, log_emit, mask)
        if self.mode == "generative":
            if states is not None:
                raise ValueError(
                    "mode='generative' is trained without labels: pass y=None (got state labels)."
                )
            return log_z
        if states is None:
            raise ValueError("mode='discriminative' needs the state labels: pass y=states of shape (B, T).")
        return inference.path_log_score(log_pi, log_A, log_emit, states, mask) - log_z

    def compute_loss(
        self,
        x: Optional[Dict[str, Any]] = None,
        y: Optional[Any] = None,
        y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
        sample_weight: Optional[Any] = None,
        training: bool = True,
    ) -> keras.KerasTensor:
        """Return the negative log-likelihood per valid step, weighted per sequence.

        ``loss = sum_b w_b * nll_b / sum_b w_b * len_b``, where ``nll_b = -log p(x_b)``
        (generative) or ``log Z_b - score(y_b)`` (discriminative), ``len_b`` is the valid
        length and ``w = sample_weight`` (ones when absent). A batch of empty sequences
        has loss 0. Regularization losses of the sublayers are added.

        :param x: The input dict ``{"obs", "mask"}``.
        :param y: ``None`` in generative mode; ``(B, T)`` integer states in discriminative
            mode (values at padded steps are ignored).
        :param y_pred: The :meth:`call` dict; computed from ``x`` when ``None``.
        :param sample_weight: Optional ``(B,)`` per-sequence weights.
        :param training: Forwarded to :meth:`call` when ``y_pred`` is ``None``.
        :return: Scalar loss in the accumulation dtype.
        :raises ValueError: If ``y`` is missing in discriminative mode, or given in
            generative mode.
        """
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-004: the objective lives HERE, in the
        # compute_loss override that stock fit() calls. Do NOT move it into a
        # keras.losses.Loss passed to compile(): a Loss sees only (y, y_pred), and this
        # objective also needs the mask from x and, for the CRF, the gold labels. Do NOT add
        # a custom train_step either (repo rule). compile(loss=...) is therefore ignored.
        if y_pred is None:
            y_pred = self(x, training=training)
        mask = self._valid_mask(x, y_pred["log_emit"])
        nll = -self._sequence_log_likelihood(y_pred, mask, y)
        lengths = keras.ops.sum(mask, axis=1)
        weights = keras.ops.ones_like(nll) if sample_weight is None else keras.ops.reshape(
            keras.ops.cast(sample_weight, nll.dtype), (-1,)
        )
        total = keras.ops.sum(weights * nll)
        steps = keras.ops.sum(weights * lengths)
        loss = total / keras.ops.where(steps > 0, steps, keras.ops.ones_like(steps))
        for extra in self.losses:
            loss = loss + keras.ops.cast(keras.ops.sum(extra), loss.dtype)
        return loss

    def log_likelihood(self, inputs: Dict[str, Any], states: Optional[Any] = None) -> keras.KerasTensor:
        """Return the per-sequence log-likelihood the loss is built from.

        :param inputs: ``{"obs", "mask"}``.
        :param states: ``None`` (generative: ``log p(x)``) or ``(B, T)`` integer states
            (discriminative: ``log p(states | x)``).
        :return: ``(B,)`` log-likelihood, summed over valid steps.
        :raises ValueError: If ``states`` does not match the mode.
        """
        outputs = self(inputs, training=False)
        return self._sequence_log_likelihood(outputs, self._valid_mask(inputs, outputs["log_emit"]), states)

    def viterbi(self, inputs: Dict[str, Any]) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return the most likely state path of every sequence.

        :param inputs: ``{"obs", "mask"}``.
        :return: ``(path, score)``: ``path`` is ``(B, T)`` int32 with ``-1`` at padded
            steps, ``score`` the ``(B,)`` unnormalized log score of that path.
        """
        outputs = self(inputs, training=False)
        mask = self._valid_mask(inputs, outputs["log_emit"])
        return inference.viterbi_decode(outputs["log_pi"], outputs["log_A"], outputs["log_emit"], mask)

    def posteriors(self, inputs: Dict[str, Any]) -> keras.KerasTensor:
        """Return the posterior state marginals ``p(z_t = k | x)``.

        :param inputs: ``{"obs", "mask"}``.
        :return: ``(B, T, K)`` probabilities, summing to 1 at valid steps and 0 at padded ones.
        """
        outputs = self(inputs, training=False)
        mask = self._valid_mask(inputs, outputs["log_emit"])
        return inference.state_posteriors(outputs["log_pi"], outputs["log_A"], outputs["log_emit"], mask)

    # -----------------------------------------------------------------
    # Shapes, config and factories
    # -----------------------------------------------------------------

    def compute_output_shape(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> Dict[str, Tuple]:
        """Return the output shapes from the stored config; valid before ``build``.

        :param input_shape: ``{"obs": (B, T, ...), "mask": (B, T)}``.
        :return: ``{"log_pi": (B, K), "log_A": (B, T, K, K), "log_emit": (B, T, K)}``.
        """
        obs_shape = tuple(input_shape["obs"])
        batch, steps, k = obs_shape[0], obs_shape[1], self.n_states
        return {"log_pi": (batch, k), "log_A": (batch, steps, k, k), "log_emit": (batch, steps, k)}

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: The model config.
        """
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "mode": self.mode,
                "obs_type": self.obs_type,
                "obs_dim": self.obs_dim,
                "vocab_size": self.vocab_size,
                "emission_context": self.emission_context,
                "transition": self.transition_type,
                "topology": self.topology,
                "band_width": self.band_width,
                "self_loop_init": self.self_loop_init,
                "min_scale": self.min_scale,
                "hidden_dim": self.hidden_dim,
                "state_emb_dim": self.state_emb_dim,
                "encoder_type": self.encoder_type,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "max_seq_len": self.max_seq_len,
                "kernel_size": self.kernel_size,
                "ffn_type": self.ffn_type,
                "normalization_type": self.normalization_type,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
            }
        )
        return config

    @classmethod
    def from_variant(cls, variant: str, pretrained: bool = False, **kwargs: Any) -> "NeuralHMM":
        """Build a preset from :attr:`MODEL_VARIANTS`, with any argument overridden.

        :param variant: A key of :attr:`MODEL_VARIANTS`.
        :param pretrained: Must be ``False``: no weights are distributed.
        :param kwargs: Constructor arguments; they override the preset. ``n_states`` and the
            observation shape (``obs_type`` with ``obs_dim`` or ``vocab_size``) are needed.
        :return: An unbuilt model.
        :raises ValueError: If ``variant`` is unknown.
        :raises NotImplementedError: If ``pretrained`` is set.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {sorted(cls.MODEL_VARIANTS)}.")
        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are distributed for NeuralHMM variant '{variant}'. Train one, "
                f"then restore it with keras.models.load_model(path) or model.load_weights(path)."
            )
        config = dict(cls.MODEL_VARIANTS[variant])
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_neural_hmm(variant: str = "neural-hmm", pretrained: bool = False, **kwargs: Any) -> NeuralHMM:
    """Build a :class:`NeuralHMM` preset; delegates to :meth:`NeuralHMM.from_variant`.

    :param variant: A key of :attr:`NeuralHMM.MODEL_VARIANTS`.
    :param pretrained: Must be ``False``: no weights are distributed.
    :param kwargs: Constructor arguments, overriding the preset.
    :return: An unbuilt model.
    :raises ValueError: If ``variant`` is unknown.
    :raises NotImplementedError: If ``pretrained`` is set.
    """
    return NeuralHMM.from_variant(variant, pretrained=pretrained, **kwargs)
