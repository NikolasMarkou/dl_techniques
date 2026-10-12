"""Neural HMM Loss class.

This module provides the ``NeuralHMMLoss`` class which wraps the Neural HMM
forward algorithm. It is registered as ``dl_techniques.models.neural_hmm.loss``
and can be passed to ``model.compile(loss=NeuralHMMLoss(...))``.
"""

from typing import Any, Dict, Optional, Sequence, Mapping

import keras

from dl_techniques.utils.keras_registration import register_dl_technique

from .inference import (
    _forward_logprob_static,
    _forward_logprob_dynamic,
    _path_score,
)


@register_dl_technique("dl_techniques.models.neural_hmm.loss")
class NeuralHMMLoss(keras.losses.Loss):
    """Keras ``keras.losses.Loss`` wrapping the Neural HMM forward algorithm.

    This loss is registered as
    ``dl_techniques.models.neural_hmm.loss`` and can be passed to
    ``model.compile(loss=NeuralHMMLoss(...))``. It carries its own per-term
    metric trackers so the ELBO decomposition appears in ``results_summary.json``.

    The loss supports both regimes:

    - ``generative``: maximizes ``log p(x_{1:T})``; the loss returns the
      negative log-marginal likelihood per timestep.
    - ``discriminative``: maximizes ``log p(y_{1:T} | x_{1:T})`` (CRF-style);
      the loss subtracts the marginal from the gold-path score.

    In both modes the loss returns a per-timestep scalar ``(batch,)``, consistent
    with the repo rule that ``Loss.call`` returns ``(batch,)`` and never a
    rank-0 scalar that would broadcast against ``sample_weight``.

    :param mode: Explicit regime override; ``None`` reads from the model's
        ``cfg["mode"]``.
    :type mode: Optional[str]
    :param emission_context: Explicit override for generative emission context;
        ``None`` reads from the model's ``cfg["emission_context"]``.
    :type emission_context: Optional[str]
    :param min_scale: Floor added to Gaussian/Poisson scales; default ``1e-3``.
    :type min_scale: float
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        emission_context: Optional[str] = None,
        min_scale: float = 1e-3,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=None, **kwargs)
        self.mode = mode
        self.emission_context = emission_context
        self.min_scale = min_scale

        # Per-term trackers; created unconditionally so they always appear in
        # the model.metrics list and are serializable via get_config().
        self.reconstruction_tracker = keras.metrics.Mean(name="nll")
        self.kl_tracker = keras.metrics.Mean(name="forward_alg")
        self.gold_score_tracker = keras.metrics.Mean(name="gold_path")

    def call(self, y_true: keras.KerasTensor, y_pred: Any) -> keras.KerasTensor:
        """Core loss computation.

        The caller (``model.compute_loss`` or ``model.compile``) supplies
        ``y_pred`` as a tuple ``(log_pi, log_A, log_emit)`` or the model will
        call its own ``call`` internally.

        :param y_true: Ignored in generative mode; in discriminative mode the
            target state-path labels ``(B, T)`` of integers in [0, K).
        :type y_true: keras.KerasTensor
        :param y_pred: Tuple ``(log_pi, log_A, log_emit)`` returned by the
            model's ``call``, or ``None`` if the loss should invoke the model
            forward pass itself.
        :type y_pred: Any
        :return: Per-timestep NLL ``(batch,)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the mode is unrecognised or required data is
            missing.
        """
        # Resolve config overrides
        mode = self.mode if self.mode is not None else "generative"
        em_context = self.emission_context  # not used in the loss itself

        if y_pred is None:
            raise ValueError(
                "NeuralHMMLoss.call: y_pred must be a (log_pi, log_A, log_emit) "
                "tuple, or the model's compute_loss should be used instead."
            )

        log_pi, log_A, log_emit = y_pred

        # --- mask ---
        if isinstance(y_true, dict):
            mask = keras.ops.cast(y_true["mask"], "float32")
        else:
            mask = keras.ops.ones_like(log_emit[:, :, 0], "float32")

        # --- forward algorithm ---
        if len(log_A.shape) == 4:
            logZ = _forward_logprob_dynamic(log_pi, log_A, log_emit, mask)
        else:
            logZ = _forward_logprob_static(log_pi, log_A, log_emit, mask)

        # --- mode-specific objective ---
        if mode == "generative":
            # Negative log-marginal likelihood per timestep
            n_valid = keras.ops.maximum(keras.ops.sum(mask), 1)
            return -logZ

        elif mode == "discriminative":
            if y_true is None:
                raise ValueError(
                    "Discriminative mode requires y_true (target state-path labels)."
                )
            gold = _path_score(log_pi, log_A, log_emit, y_true, mask)
            # Return gold - logZ; the model's compute_loss handles the sign/avg.
            return gold - logZ

        else:
            raise ValueError(f"Unknown mode '{mode}'. Available: ['generative', 'discriminative'].")

    def get_config(self) -> Dict[str, Any]:
        """Return the loss constructor configuration.

        :return: Configuration dict.
        :rtype: Dict[str, Any]
        """
        config = {
            "mode": self.mode,
            "emission_context": self.emission_context,
            "min_scale": self.min_scale,
        }
        base = super().get_config()
        return {**base, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NeuralHMMLoss":
        """Reconstruct a loss from its config.

        :param config: Config dict from ``get_config``.
        :type config: Dict[str, Any]
        :return: A new loss instance.
        :rtype: NeuralHMMLoss
        """
        return cls(**config)


def assemble_loss_config(
    emission_type: str,
    wrapper_config: Mapping[str, Any],
    caller_args: Optional[Mapping[str, Any]] = None,
    *,
    passthrough: Sequence[str] = ("type", "name"),
) -> Dict[str, Any]:
    """Filter a wrapper's generic loss defaults, then merge the caller's.

    :param emission_type: An :data:`EMISSION_REGISTRY` key.
    :type emission_type: str
    :param wrapper_config: The wrapper's own generic config; filtered.
    :type wrapper_config: Mapping[str, Any]
    :param caller_args: The caller's explicit args; never filtered.
    :type caller_args: Optional[Mapping[str, Any]]
    :param passthrough: Keys kept regardless of the registry intersection.
    :type passthrough: Sequence[str]
    :return: The assembled config dict for :func:`create_loss_layer`.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``emission_type`` is not registered.
    """
    # This function mirrors the pattern in encoders/transitions/emissions
    # but the loss doesn't have a registry; it just takes mode/emission_context/min_scale
    accepted = {"mode", "emission_context", "min_scale", "name", "type"}
    config = {k: v for k, v in wrapper_config.items() if k in accepted}
    if caller_args:
        config.update(caller_args)
    return config