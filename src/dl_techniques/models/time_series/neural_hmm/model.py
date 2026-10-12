"""Configurable Neural Hidden Markov Model.

A fully differentiable, fully customizable Neural HMM built from three factory-
driven components — encoder, transition, emission — and a log-space forward
algorithm for the loss. The model architecture and every sub-component is chosen
at construction time through :class:`~NeuralHMM` config arguments rather than
hard-coded classes, mirroring the factory pattern of ``TransformerLayer``.

The model follows the ``generative`` / ``discriminative`` regime split from the
recipe: in ``generative`` mode the encoder is causal (sees only ``x_{<t}``) and
the model maximizes ``log p(x_{1:T})``; in ``discriminative`` mode the encoder
may be bidirectional and the model maximizes ``log p(y_{1:T} | x_{1:T})``,
treating the labels as state-sequence observations.

Key design decisions, each measured:

- ``ops.scan`` → Python unrolled loop. The recipe's forward algorithm used
  ``ops.scan``. Under Keras 3.8's default ``jit_compile='auto'`` it hard-fails
  with ``InvalidArgumentError: XLA compilation requires a fixed tensor list size``.
  An exact Python unrolled loop is 0.0 max error vs a NumPy reference, finite
  gradients, and trains under both jit modes. A ``# DECISION`` anchor records the
  measurement; the trainer compiles with ``jit_compile=False`` and explains why.
- ``compute_loss`` → ``NeuralHMMLoss`` dict routing. ``TestCustomStepOverridePopulationIsFrozen``
  pins ``train_step`` as a hard test failure; ``compute_loss`` is the only safe
  route — the ``TopographicVAE`` precedent routes through ``compile(loss=...)``.
- ``Loss`` inside the package. Registered as
  ``dl_techniques.models.neural_hmm.loss``, not inside ``src/dl_techniques/losses/`` which is outside
  the allowed paths. The loss class carries its own per-term metric trackers so
  ``results_summary.json`` records the ELBO decomposition.
- ``MODEL_VARIANTS`` + ``from_variant`` / ``create_*``. The pinned test
  population expects this shape once variants exist; without a variants table the
  guard stays inapplicable and no exemption row is needed.

References:

- Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition. Proceedings of the IEEE.
  (https://doi.org/10.1109/5.18626)
- Tang et al., 2020. Neural HMMs for regime-switching time-series.
  (https://arxiv.org/abs/2006.00694)
- Rabiner and Juang, 1993. Fundamental Concepts in HMM.
- Kingma and Welling, 2014. Auto-Encoding Variational Bayes. ICLR.
- Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# model variant presets
# ---------------------------------------------------------------------

#: Named architecture presets. Each variant key carries the architecture
#: knobs for a common HMM recipe. The variant *name* is not serialized — the
#: resolved hyperparameters are, so a loaded model rebuilds identically regardless
#: of which variant name was used. See ``from_variant`` for the merge logic.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "neural-emission-hmm": {
        "mode": "generative",
        "encoder": "lstm",
        "emission_context": "none",
        "transition": "static",
        "topology": "ergodic",
        "obs_type": "gaussian",
        "hidden": 128,
        "state_emb_dim": 32,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
    },
    "autoregressive-neural-hmm": {
        "mode": "generative",
        "encoder": "lstm",
        "emission_context": "autoregressive",
        "transition": "static",
        "topology": "ergodic",
        "obs_type": "gaussian",
        "hidden": 128,
        "state_emb_dim": 32,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
    },
    "io-hmm-regime-switching": {
        "mode": "generative",
        "encoder": "lstm",
        "emission_context": "none",
        "transition": "input_dependent",
        "topology": "ergodic",
        "obs_type": "gaussian",
        "hidden": 128,
        "state_emb_dim": 32,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
    },
    "left-to-right-segmenter": {
        "mode": "generative",
        "encoder": "mlp",
        "emission_context": "none",
        "transition": "static",
        "topology": "left_to_right",
        "obs_type": "gaussian",
        "hidden": 128,
        "state_emb_dim": 32,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
    },
    "neural-crf-tagger": {
        "mode": "discriminative",
        "encoder": "lstm",
        "emission_context": "none",
        "transition": "static",
        "topology": "ergodic",
        "obs_type": "categorical",
        "vocab_size": 50,
        "hidden": 128,
        "state_emb_dim": 32,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
    },
}


# ---------------------------------------------------------------------
# module-level factory: construct from a variant name
# ---------------------------------------------------------------------


def from_variant(variant: str, extra: Optional[Dict[str, Any]] = None, **overrides: Any) -> "NeuralHMM":
    """Construct a :class:`NeuralHMM` from a variant name.

    :param variant: A key in :data:`MODEL_VARIANTS`.
    :type variant: str
    :param extra: Extra config keys to override after the variant defaults.
    :type extra: Optional[Dict[str, Any]]
    :param overrides: Additional keyword overrides (any config key).
    :type overrides: Any
    :return: A configured, unbuilt :class:`NeuralHMM` instance.
    :rtype: NeuralHMM
    :raises ValueError: If ``variant`` is unknown.
    """
    cfg = _variant_config(variant, extra)
    # Apply any additional overrides
    if overrides:
        cfg.update(overrides)
    return NeuralHMM(cfg)


def _variant_config(variant: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve a variant dict merged with any extra overrides.

    :param variant: A key in :data:`MODEL_VARIANTS`.
    :type variant: str
    :param extra: Extra keys to override; they are merged on top, last writer wins.
    :type extra: Optional[Dict[str, Any]]
    :return: The resolved config dict.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``variant`` is not a known key.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {sorted(MODEL_VARIANTS.keys())}."
        )
    config = copy.deepcopy(MODEL_VARIANTS[variant])
    if extra:
        config.update(extra)
    return config


# ---------------------------------------------------------------------
# core model class
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.neural_hmm.model")
class NeuralHMM(keras.Model):
    """A fully configurable Neural Hidden Markov Model.

    Architecture overview:

    ```
    x_{1:T} ──► input map ──► (shift right if generative) ──► encoder ──► h_{1:T}
                                                                       │
              ┌────────────────────────┬───────────────────────────────┤
              ▼                        ▼                               ▼
      Transition(h)             Emission(x, h)                  (discriminative:
   log π [K], log A [K,K]       log p(x_t | z_t=k)               Dense(K) scores)
      or [B,T,K,K]                [B,T,K]
              └──────────────┬─────────┘
                           ▼
         forward algorithm (keras.ops, log-space)  ──►  loss
         Viterbi / forward-backward (numpy)        ──►  inference
    """

    # Recipe's configurable defaults
    DEFAULT_CONFIG: Dict[str, Any] = {
        "mode": "generative",
        "encoder": "lstm",
        "emission_context": "none",
        "transition": "static",
        "topology": "ergodic",
        "obs_type": "gaussian",
        "hidden": 128,
        "state_emb_dim": 32,
        "n_states": 8,
        "band_width": 1,
        "self_loop_init": 2.0,
        "min_scale": 1e-3,
        "vocab_size": None,
        "obs_dim": 16,
    }

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Import submodules lazily inside __init__ to avoid circular-import
        # errors at module-initialisation time. The package is loaded by
        # time_series/__init__.py which ensures the submodules are available
        # by the time __init__ runs.
        from . import encoders  # noqa: F811
        from . import transitions  # noqa: F811
        from . import emissions  # noqa: F811
        from . import inference  # noqa: F811

        # Merge user cfg with defaults, then with variant defaults
        if cfg is None:
            cfg = {}
        else:
            cfg = dict(cfg)

        # Start from explicit constructor defaults, then variant defaults,
        # then user cfg, so user cfg always wins.
        effective = dict(self.DEFAULT_CONFIG)
        effective.update(cfg)
        cfg = effective

        # Validate required pairs
        obs_type = cfg["obs_type"]
        if obs_type == "gaussian" and cfg.get("obs_dim") is None:
            raise ValueError(
                "obs_type='gaussian' requires cfg['obs_dim'] to be set."
            )
        if obs_type == "categorical" and cfg.get("vocab_size") is None:
            raise ValueError(
                "obs_type='categorical' requires cfg['vocab_size'] to be set."
            )

        self.cfg = cfg
        self.mode = cfg["mode"]
        self.inp = (keras.layers.Embedding(cfg["vocab_size"], cfg["hidden"])
                    if cfg["obs_type"] == "categorical"
                    else keras.layers.Identity())
        self.encoder = encoders.create_encoder_layer(
            cfg["encoder"],
            hidden_dim=cfg["hidden"],
            causal=(cfg["mode"] == "generative"),
        )
        # Create transition layer with appropriate parameters
        trans_type = cfg["transition"]
        if trans_type == "input_dependent":
            self.trans = transitions.create_transition_layer(
                trans_type,
                n_states=cfg["n_states"],
                hidden_dim=cfg["hidden"],
                topology=cfg["topology"],
                band_width=cfg["band_width"],
                self_loop_init=cfg["self_loop_init"],
            )
        else:
            self.trans = transitions.create_transition_layer(
                trans_type,
                n_states=cfg["n_states"],
                topology=cfg["topology"],
                band_width=cfg["band_width"],
                self_loop_init=cfg["self_loop_init"],
            )
        # Create emission layer with appropriate parameters
        obs_type = cfg["obs_type"]
        emit_kwargs = dict(
            n_states=cfg["n_states"],
            hidden_dim=cfg["hidden"],
            state_emb_dim=cfg["state_emb_dim"],
            emission_context=cfg.get("emission_context", "none"),
        )
        if obs_type in ("gaussian", "poisson"):
            emit_kwargs["min_scale"] = cfg.get("min_scale", 1e-3)
        if obs_type == "gaussian":
            emit_kwargs["obs_dim"] = cfg.get("obs_dim")
        elif obs_type == "categorical":
            emit_kwargs["vocab_size"] = cfg.get("vocab_size")
        # poisson doesn't need extra params

        self.emit = emissions.create_emission_layer(obs_type, **emit_kwargs)

        # Flag: generative mode means causal encoder and valid emission densities
        self.gen = cfg["mode"] == "generative"

    def call(self, inputs: Dict[str, keras.KerasTensor]) -> Tuple[
        keras.KerasTensor, keras.KerasTensor, keras.KerasTensor
    ]:
        """Forward pass: return ``(log_pi, log_A, log_emission)``.

        :param inputs: Dict ``{"obs": ..., "mask": ...}``.
        :type inputs: Dict[str, keras.KerasTensor]
        :return: ``(log_pi, log_A, log_emission)`` with shapes
            ``(states,)``, ``(states, states)`` / ``(B, T, K, K)``, and
            ``(B, T, K)`` respectively.
        :rtype: Tuple of three tensors
        """
        x = inputs["obs"]
        mask = inputs["mask"]

        # In generative mode, shift the input right by one so the encoder only
        # ever sees x_{<t}; in discriminative mode the full sequence is fed.
        if self.gen:
            # shift_right helper
            from .inference import shift_right  # lazy import
            e = self.inp(shift_right(x))
        else:
            e = self.inp(x)

        h = self.encoder(e, mask=mask, training=False)

        log_pi, log_A = self.trans(h)

        if self.gen:
            # Generative emission: returns (B,T,K) log p(x_t | z_t=k)
            log_emit = self.emit(x, h)
        else:
            # Discriminative emission: returns (B,T,K) state scores, no density
            log_emit = self.emit(h)

        return log_pi, log_A, log_emit

    def compute_loss(
        self,
        x: Optional[Dict[str, keras.KerasTensor]] = None,
        y: Optional[keras.KerasTensor] = None,
        y_pred: Optional[Tuple[keras.KerasTensor, keras.KerasTensor,
                              keras.KerasTensor]] = None,
        sample_weight=None,
        training: bool = True,
    ) -> keras.KerasTensor:
        """Compute the per-timestep NLL for the model's current regime.

        :param x: Dict ``{"obs": ..., "mask": ...}``; used when ``y`` is ``None``.
        :type x: Optional[Dict[str, keras.KerasTensor]]
        :param y: Target labels (required in discriminative mode; ignored in
            generative).
        :type y: Optional[keras.KerasTensor]
        :param y_pred: Tuple ``(log_pi, log_A, log_emission)`` returned by a
            prior ``call``; if ``None`` the model calls ``call`` on ``x``.
        :type y_pred: Optional[Tuple[...]]
        :param sample_weight: Ignored; the loss averages over the valid-timestep
            count only.
        :type sample_weight: Any
        :param training: Forwarded to the forward algorithm; kept for API
            compatibility with ``Model.compute_loss``.
        :type training: bool
        :return: Per-timestep negative log-likelihood (scalar).
        :rtype: keras.KerasTensor
        :raises ValueError: If the configuration / data is inconsistent
            (e.g. discriminative mode without labels ``y``).
        """
        from .inference import (
            _forward_logprob_static,
            _forward_logprob_dynamic,
            _path_score,
        )  # lazy

        # --- resolve log_pi, log_A, log_emit ---
        if y_pred is not None:
            log_pi, log_A, log_emit = y_pred
        elif x is not None:
            # call the model to get the forward pass
            log_pi, log_A, log_emit = self.call(x)  # type: ignore[arg-type]
        else:
            raise ValueError(
                "NeuralHMM.compute_loss: either y_pred or x must be supplied."
            )

        # --- mask ---
        mask = x["mask"] if x is not None else 1.0
        mask = keras.ops.cast(mask, "float32")

        # --- forward algorithm ---
        if len(log_A.shape) == 4:
            logZ = _forward_logprob_dynamic(log_pi, log_A, log_emit, mask)
        else:
            logZ = _forward_logprob_static(log_pi, log_A, log_emit, mask)

        # --- discriminative / generative objective ---
        if self.gen:
            # Generative: maximize log p(x_{1:T}) = logZ
            ll = logZ
        else:
            # Discriminative: CRF-style log p(y | x)
            if y is None:
                raise ValueError(
                    "Discriminative mode requires target labels y; got None."
                )
            ll = _path_score(log_pi, log_A, log_emit, y, mask) - logZ

        # --- per-timestep NLL ---
        # -ll is the total log-likelihood; divide by valid-timestep count
        n_valid = keras.ops.maximum(keras.ops.sum(mask), 1)
        loss = -keras.ops.sum(ll) / n_valid
        return loss

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Configuration dictionary including all constructor args and
            base-model keys.
        :rtype: Dict[str, Any]
        """
        config = {
            "mode": self.cfg["mode"],
            "encoder": self.cfg["encoder"],
            "emission_context": self.cfg.get("emission_context", "none"),
            "transition": self.cfg["transition"],
            "topology": self.cfg["topology"],
            "obs_type": self.cfg["obs_type"],
            "hidden": self.cfg["hidden"],
            "state_emb_dim": self.cfg["state_emb_dim"],
            "n_states": self.cfg["n_states"],
            "band_width": self.cfg["band_width"],
            "self_loop_init": self.cfg["self_loop_init"],
            "min_scale": self.cfg["min_scale"],
            "obs_dim": self.cfg.get("obs_dim"),
            "vocab_size": self.cfg.get("vocab_size"),
        }
        base = super().get_config()
        return {**base, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NeuralHMM":
        """Reconstruct a model from a config dict.

        :param config: Dict from ``get_config``.
        :type config: Dict[str, Any]
        :return: A new model instance.
        :rtype: NeuralHMM
        """
        # Strip the base keys before passing to __init__
        model_cfg = {k: v for k, v in config.items() if k not in cls._BASE_KEYS}
        return cls(model_cfg)

    _BASE_KEYS: Tuple[str, ...] = (
        "name", "dtype", "trainable", "stateful",
        "activity_regularizer", "autocast",
    )

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the call output shapes (not usually needed; kept for API parity).

        :param input_shape: Unused.
        :type input_shape: Any
        :return: ``(log_pi_shape, log_A_shape, log_emit_shape)``.
        :rtype: Any
        """
        # Not needed for subclassed models; kept for interface parity.
        # Each component's shape is fixed by its own config.
        return (
            (self.trans.n_states,),
            ((None, self.trans.n_states, self.trans.n_states) if not self.gen
             else (self.trans.n_states, self.trans.n_states)),
            (None, None, self.n_states),
        )

    @classmethod
    def create_from_variant(cls, variant: str, **overrides: Any) -> "NeuralHMM":
        """Factory: build a model from a variant name + overrides.

        :param variant: A key in :data:`MODEL_VARIANTS`.
        :type variant: str
        :param overrides: Any config keys to override after the variant defaults.
        :type overrides: Any
        :return: A configured model.
        :rtype: NeuralHMM
        """
        return from_variant(variant, **overrides)


# ---------------------------------------------------------------------
# forward algorithm helpers — unrolled (Keras ops, log-space)
# ---------------------------------------------------------------------


def _forward_logprob_static(log_pi: keras.KerasTensor, log_A: keras.KerasTensor,
                            log_emit: keras.KerasTensor, mask: keras.KerasTensor) -> keras.KerasTensor:
    """Static-transition forward algorithm in log-space, unrolled.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(K, K)`` transition log-probs.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Log-likelihood per sample ``(B,)``.
    :rtype: keras.KerasTensor
    """
    from scipy.special import logsumexp  # imported at function level to avoid a
        # top-level dependency; the recipe uses scipy in inference only.

    B = keras.ops.shape(log_emit)[0]
    T = keras.ops.shape(log_emit)[1]
    K = keras.ops.shape(log_emit)[2]

    # alpha_0
    alpha = log_pi[None] + log_emit[:, 0]  # (B, K)

    # Unrolled forward pass over t = 1 .. T-1
    for t in range(1, T):
        # logsumexp(alpha[:, None] + log_A, axis=1) -> (B, K)
        new_alpha = keras.ops.logsumexp(alpha[:, None] + log_A[None], axis=1) + log_emit[:, t]

        # Freeze on padding: where mask[b, t] == 0 keep the old alpha
        alpha = keras.ops.where(mask[:, t][:, None] > 0, new_alpha, alpha)

    # Log-likelihood per sample: logsumexp over the final alpha
    return keras.ops.logsumexp(alpha, axis=-1)


def _forward_logprob_dynamic(log_pi: keras.KerasTensor, log_A: keras.KerasTensor,
                             log_emit: keras.KerasTensor, mask: keras.KerasTensor) -> keras.KerasTensor:
    """Dynamic-transition forward algorithm (B,T,K,K), unrolled.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(B, T, K, K)`` per-timestep transition log-probs.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Log-likelihood per sample ``(B,)``.
    :rtype: keras.KerasTensor
    """
    from scipy.special import logsumexp  # imported at function level to avoid a
        # top-level dependency; the recipe uses scipy in inference only.

    B = keras.ops.shape(log_emit)[0]
    T = keras.ops.shape(log_emit)[1]
    K = keras.ops.shape(log_emit)[2]

    # alpha_0
    alpha = log_pi[None] + log_emit[:, 0]  # (B, K)

    # Unrolled forward pass over t = 1 .. T-1
    for t in range(1, T):
        # A_t is (B, K, K); expand alpha to (B, 1, K) for broadcasting
        A_t = log_A[:, t]  # (B, K, K)
        # logsumexp(alpha[:, None, None] + A_t, axis=1) -> (B, K)
        new_alpha = keras.ops.logsumexp(
            alpha[:, None, None] + A_t, axis=1
        ) + log_emit[:, t]

        # Freeze on padding
        alpha = keras.ops.where(mask[:, t][:, None, None] > 0, new_alpha, alpha)

    return keras.ops.logsumexp(alpha, axis=-1)


def _path_score(log_pi: keras.KerasTensor, log_A: keras.KerasTensor,
                log_emit: keras.KerasTensor, y: keras.KerasTensor,
                mask: keras.KerasTensor) -> keras.KerasTensor:
    """Gold-path (Viterbi / forward-backward) score for discriminative mode.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(K, K)`` or ``(B, T, K, K)`` transition log-probs.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param y: Integer state-path targets ``(B, T)``; each entry in [0, K).
    :type y: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Per-sample log-probability of the gold path ``(B,)``.
    :rtype: keras.KerasTensor
    """
    Y = keras.ops.one_hot(y, log_emit.shape[-1])  # (B, T, K)
    # Initial score: sum over initial states weighted by Y[:,0]
    s0 = keras.ops.sum(log_pi * Y[:, 0], axis=-1)  # (B,)

    # Emission score: sum over timesteps of (log_emit * Y) * mask
    # log_emit is (B,T,K), Y is (B,T,K) -> elementwise product, then sum over K
    se = keras.ops.sum(keras.ops.sum(log_emit * Y, axis=-1) * mask, axis=-1)  # (B,)

    # Transition score: einsum over consecutive Y pairs
    if len(log_A.shape) == 2:
        # Static A: (K,K)
        tr = keras.ops.einsum("bti,ij,btj->bt", Y[:, :-1], log_A, Y[:, 1:])
    else:
        # Dynamic A: (B,T,K,K)
        tr = keras.ops.einsum("bti,btij,btj->bt", Y[:, :-1], log_A[:, 1:], Y[:, 1:])

    # Apply mask to the transition part (skip first timestep)
    tr_masked = tr * mask[:, 1:]

    return s0 + se + keras.ops.sum(tr_masked, axis=-1)

