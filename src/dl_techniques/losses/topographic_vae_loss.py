"""The evidence lower bound for a Topographic VAE, as a stock ``keras.losses.Loss``.

Shape contract
--------------
``call()`` returns **one value per sample**, shape ``(batch,)`` — never a scalar.
Keras' ``reduce_weighted_values`` multiplies ``call()``'s output by
``sample_weight`` and only THEN reduces, so a scalar return does not *ignore*
``sample_weight``: it BROADCASTS against it, charging every row the batch
aggregate and discarding which rows were weighted. It would also make
``reduction=`` a dead knob. The batch mean belongs to the training loop.

Why a custom ``Loss`` and not a ``train_step``
---------------------------------------------
The Topographic VAE's objective (Eq. 12 of Keller & Welling, 2022) is

    sum_l [ E_q[log p(x_l | g_theta(t_l))] - KL(q_phi(z_l|x_l) || p(z))
                             - KL(q_gamma(u_l|x_l) || p(u)) ]

which is a standard ELBO with **two** KL terms, because the topographic prior
adds a second Gaussian ``u`` alongside the usual latent ``z``, and each is a
separate reparameterization. Keras' stock ``compile``/``fit`` machinery will not
compose two KL terms against one reconstruction out of the box, but it *will*
accept any ``keras.losses.Loss`` that takes the model's dict output and the data
as its two arguments. That is the route taken here rather than overriding
``train_step`` -- see ``research/2026_keras_custom_models_instructions_v2.md``
§11.1, which prefers stock ``fit()`` precisely because a custom ``train_step``
opts out of mixed-precision loss scaling.

So ``TopographicVAELoss`` receives ``y_pred`` = the model's output dict and
``y_true`` = the input sequence, and returns a scalar.

Reduction conventions -- stated because they change the number
---------------------------------------------------------------
Two conventions are inherited from the sibling ``dl_techniques.models.vae`` and
are **not** the textbook ones, so a reader comparing against the paper's
``log p(x)`` column needs both of them:

1. The reconstruction term is a **mean over pixels** (binary cross-entropy,
   summed over the sequence), not a sum. Combined with a KL that *is* summed
   over latent dimensions, this makes ``kl_loss_weight`` differ from the
   literature's ``beta`` by the pixel count -- the model exposes
   ``effective_kl_beta`` for exactly this reason, mirroring
   ``vae.VAE.effective_kl_beta``.
2. Both KL terms are summed over the sequence and then averaged over the batch,
   matching Eq. 12's outer ``sum_l``.

``reconstruction_sum_reduction`` and ``kl_loss_weight`` are constructor arguments
so a caller who wants the textbook sum-over-pixels form can ask for it
explicitly rather than getting whichever one a default happened to pick.

Numerics
--------
The whole objective runs in **float32** regardless of ``compute_dtype``. Two
independent reasons, both measured in the sibling VAE: ``exp`` of an unclipped
log-variance overflows float16 (max 65504), and the BCE argument clip at 1e-7 is
below float16's smallest normal, so ``log(0)`` would reach ``-inf`` without the
cast. This is not defensive decoration; each of those is a silent all-NaN loss.

References:
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
    - Kingma & Welling, 2014. Auto-Encoding Variational Bayes. ICLR.
      (https://arxiv.org/abs/1312.6114)
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: The output keys ``TopographicVAELoss`` ALWAYS reads from ``y_pred``. A missing
#: key is a construction-time contract violation, not a silently-skipped term,
#: because a dropped KL shows up only as a worse model.
REQUIRED_PREDICTION_KEYS = (
    "reconstruction",
    "z_mean",
    "z_log_var",
)

#: The output keys for the topographic prior's second Gaussian ``u``, which a
#: model built with ``use_variance_variables=False`` does not emit at all.
#:
#: They are CONDITIONAL, and the pair is conditional TOGETHER: a model that has
#: the variance variables contributes ``KL(q_gamma(u|x) || p(u))`` and a model
#: without them contributes zero for that term. Making them unconditional would
#: make the paper's own plain-VAE baseline untrainable with this loss -- the
#: baseline is the comparison the topographic result is read against, so an
#: objective it cannot be trained with is a defect in the objective, not in the
#: baseline.
#:
#: The pair cannot be halved: ``u_mean`` without ``u_log_var`` is not a
#: distribution, and silently treating its missing scale as unit variance would
#: train against a prior the model does not have.
VARIANCE_PREDICTION_KEYS = ("u_mean", "u_log_var")

#: Log-variance clamp applied before ``exp``. ``[-20, 20]`` spans standard
#: deviations of roughly [2e-5, 4.8e4], well past anything a trained VAE
#: reaches, while keeping ``exp`` inside float32's range under mixed precision.
LOG_VAR_CLIP = 20.0

#: Probability clamp for the Bernoulli likelihood. Below float16's smallest
#: normal (6.1e-5), which is why the objective is computed in float32.
PROBABILITY_CLIP = 1e-7


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.topographic_vae_loss")
class TopographicVAELoss(keras.losses.Loss):
    """Negative evidence lower bound for a Topographic VAE over an input sequence.

    Combines a Bernoulli reconstruction term with the two Gaussian KL terms of
    Eq. 12. ``y_pred`` is the model's output dict; ``y_true`` is the input
    sequence itself.

    The ``u`` pair (:data:`VARIANCE_PREDICTION_KEYS`) is **conditional**: a model
    built with ``use_variance_variables=False`` emits neither key, and its
    ``KL_u`` term is then absent from the objective rather than zero. That is the
    paper's plain-VAE baseline, which has to be trainable with this loss or the
    comparison the topographic result is read against cannot be run. Half a pair
    raises.

    The loss is **not** dependent on the order of the sequence axis: temporal
    coherence lives entirely in how ``t`` is constructed inside the model, not
    in the objective. The sum over the sequence axis is what Eq. 12 prescribes
    and is applied here.

    Architecture:

    .. code-block:: text

        y_pred = {reconstruction, z_mean, z_log_var[, u_mean, u_log_var]}
        y_true = x                              (B, S, H, W, C)

              ┌────────────────────────────────────────┐
              │ reconstruction term                    │
              │   BCE(x, reconstruction)               │
              │   mean over pixels, sum over sequence  │
              └──────────────────┬─────────────────────┘
                                 ▼
              ┌────────────────────────────────────────┐
              │  + kl_loss_weight * ( KL_z + KL_u )    │
              │  KL_q || N(0, I), summed over latent   │
              │  dims and sequence, mean over batch    │
              └──────────────────┬─────────────────────┘
                                 ▼
                          scalar loss

    :param kl_loss_weight: Weight on the summed KL terms. NOT the literature's
        ``beta`` unless ``reconstruction_sum_reduction=True``; see
        :attr:`effective_beta` and the module docstring. Must be non-negative.
    :type kl_loss_weight: float
    :param reconstruction_sum_reduction: When ``True`` the reconstruction is
        summed over pixels rather than averaged, which makes ``kl_loss_weight``
        exactly the paper's ``beta``. Defaults to ``False``, matching the
        sibling ``vae.VAE``.
    :type reconstruction_sum_reduction: bool
    :param clip_log_var: Symmetric clamp applied to both log-variance tensors
        before ``exp``. Defaults to :data:`LOG_VAR_CLIP`.
    :type clip_log_var: float
    :param name: Optional loss name.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for ``keras.losses.Loss``.

    :raises ValueError: On a negative ``kl_loss_weight`` or a non-positive
        ``clip_log_var``.

    :Example:

    >>> import numpy as np
    >>> from keras import ops
    >>> loss = TopographicVAELoss(kl_loss_weight=0.01)
    >>> y_true = ops.convert_to_tensor(np.zeros((2, 3, 4, 4, 1), "float32"))
    >>> y_pred = {
    ...     "reconstruction": ops.convert_to_tensor(np.full((2, 3, 4, 4, 1), 0.5, "float32")),
    ...     "z_mean": ops.convert_to_tensor(np.zeros((2, 3, 8), "float32")),
    ...     "z_log_var": ops.convert_to_tensor(np.zeros((2, 3, 8), "float32")),
    ...     "u_mean": ops.convert_to_tensor(np.zeros((2, 3, 8), "float32")),
    ...     "u_log_var": ops.convert_to_tensor(np.zeros((2, 3, 8), "float32")),
    ... }
    >>> loss(y_true, y_pred).shape
    (2,)
    """

    def __init__(
        self,
        kl_loss_weight: float = 1.0,
        reconstruction_sum_reduction: bool = False,
        clip_log_var: float = LOG_VAR_CLIP,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if kl_loss_weight < 0.0:
            raise ValueError(
                f"kl_loss_weight must be non-negative, got {kl_loss_weight}"
            )
        if clip_log_var <= 0.0:
            raise ValueError(f"clip_log_var must be positive, got {clip_log_var}")

        self.kl_loss_weight = kl_loss_weight
        self.reconstruction_sum_reduction = reconstruction_sum_reduction
        self.clip_log_var = clip_log_var

        # Observed from the batch shape by `call`, then read by `effective_beta`.
        # An INSTANCE attribute, deliberately: as a class attribute it leaks
        # across losses, so a fresh loss on a different frame size would inherit
        # the first one's pixel count until it saw a batch. A plain Python int is
        # right here -- it only ever builds a diagnostic number outside the tape,
        # and putting it in the graph would add a traced dependency for nothing.
        self._pixels_per_frame = 0

        # Per-term trackers, so a training run reports the decomposition rather
        # than one opaque number. Created unconditionally: they hold no weights
        # and their absence would make a caller reach for model.losses instead.
        self.reconstruction_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.z_kl_tracker = keras.metrics.Mean(name="z_kl_loss")
        self.u_kl_tracker = keras.metrics.Mean(name="u_kl_loss")

        logger.info(
            f"TopographicVAELoss: kl_loss_weight={kl_loss_weight}, "
            f"reconstruction_sum_reduction={reconstruction_sum_reduction}"
        )

    @property
    def effective_beta(self) -> float:
        """The ``beta`` this loss actually optimizes, given its reductions.

        With ``reconstruction_sum_reduction=False`` (the default) the
        reconstruction is divided by the pixel count while the KL is not, so the
        optimized ``beta`` is ``kl_loss_weight * pixels_per_frame`` -- the same
        correction the sibling ``vae.VAE.effective_kl_beta`` makes, and with the
        same caveat that the SAME nominal ``kl_loss_weight`` is a different
        regularization strength at a different input resolution.

        With ``reconstruction_sum_reduction=True`` the two terms are on the same
        scale and ``effective_beta`` is ``kl_loss_weight`` itself.

        ``pixels_per_frame`` is observed from the batch shape by :meth:`call`,
        so before the first batch this returns ``kl_loss_weight`` -- the
        sum-reduction answer, which is the conservative one when the true scale
        factor is unknown.

        :return: The effective ``beta``.
        :rtype: float
        """
        if self.reconstruction_sum_reduction or not self._pixels_per_frame:
            return float(self.kl_loss_weight)
        return float(self.kl_loss_weight) * float(self._pixels_per_frame)

    # Observed from the batch shape by `call`, then read by `effective_beta`. A
    # plain Python attribute rather than a Variable: it only ever builds a
    # diagnostic number outside the tape, and putting it in the graph would add
    # a traced dependency for no training purpose.
    _pixels_per_frame = 0

    def has_variance_variables(self, y_pred: Any) -> bool:
        """Whether ``y_pred`` carries the topographic prior's ``u`` posterior.

        :param y_pred: The model's output mapping.
        :type y_pred: Any
        :return: ``True`` when both ``u_mean`` and ``u_log_var`` are present.
        :rtype: bool
        :raises ValueError: If ``y_pred`` is not a mapping, or carries exactly one
            of the two.
        """
        if not isinstance(y_pred, dict):
            raise ValueError(
                f"TopographicVAELoss expects a dict y_pred with keys "
                f"{list(REQUIRED_PREDICTION_KEYS)}, got "
                f"{type(y_pred).__name__}"
            )
        present = [key for key in VARIANCE_PREDICTION_KEYS if key in y_pred]
        if len(present) == 1:
            missing = [k for k in VARIANCE_PREDICTION_KEYS if k not in y_pred]
            raise ValueError(
                f"TopographicVAELoss y_pred carries {present} but not "
                f"{missing}; the u posterior needs both its mean and its "
                f"log-variance, and defaulting the scale to unit variance "
                f"would train against a prior the model does not have"
            )
        return len(present) == len(VARIANCE_PREDICTION_KEYS)

    def _validate_prediction(self, y_pred: Any) -> Dict[str, Any]:
        """Check that ``y_pred`` carries every term the objective needs.

        :param y_pred: The model's output.
        :type y_pred: Any
        :return: The same mapping.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``y_pred`` is not a mapping, a always-required key
            is absent, or the ``u`` pair is half present. A silently-dropped KL
            term is a defect that trains to a worse model without any other
            symptom.
        """
        if not isinstance(y_pred, dict):
            raise ValueError(
                f"TopographicVAELoss expects a dict y_pred with keys "
                f"{list(REQUIRED_PREDICTION_KEYS)}, got "
                f"{type(y_pred).__name__}"
            )
        missing = [
            key for key in REQUIRED_PREDICTION_KEYS if key not in y_pred
        ]
        if missing:
            raise ValueError(
                f"TopographicVAELoss y_pred is missing {len(missing)} required "
                f"key(s) {missing}; expected {list(REQUIRED_PREDICTION_KEYS)}"
            )
        # Raises on the half-present pair, and is the single place that decides
        # whether the u KL term is present at all.
        self.has_variance_variables(y_pred)
        return y_pred

    def reconstruction_loss(self, y_true, y_pred_reconstruction) -> Any:
        """Bernoulli negative log-likelihood of the sequence.

        :param y_true: Target sequence on ``[0, 1]``.
        :type y_true: Any
        :param y_pred_reconstruction: Decoded means on ``[0, 1]``.
        :type y_pred_reconstruction: Any
        :return: Scalar. Summed over the sequence axis, averaged over pixels
            unless ``reconstruction_sum_reduction`` is set.
        :rtype: Any
        """
        y_true = ops.cast(y_true, "float32")
        y_pred_reconstruction = ops.cast(y_pred_reconstruction, "float32")

        # The clip is what keeps log(0) out of the objective. It must precede
        # the BCE, not follow it as a repair.
        clipped = ops.clip(y_pred_reconstruction, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
        per_element = keras.losses.binary_crossentropy(y_true, clipped)

        # binary_crossentropy has already reduced away the channel axis, so
        # per_element is (batch, sequence, height, width) and the image axes to
        # reduce are 2..rank-1 of ITS OWN rank. Reading the rank off y_true
        # instead would be off by one and would silently reduce the wrong axis
        # -- a mean over width instead of over height/width, which changes the
        # number by a factor of the image size and nothing raises.
        image_axes = list(range(2, len(per_element.shape)))
        if self.reconstruction_sum_reduction:
            if not image_axes:
                return per_element
            return ops.sum(per_element, axis=image_axes)

        # Mean over the image axes only, keeping (batch, sequence) so the
        # sequence sum in `call` can be applied on axis -1.
        if image_axes:
            return ops.mean(per_element, axis=image_axes)
        return per_element

    def gaussian_kl_loss(self, mean, log_var) -> Any:
        """``KL(N(mean, exp(log_var)) || N(0, I))``, summed over latent dims.

        :param mean: Posterior mean.
        :type mean: Any
        :param log_var: Posterior log-variance.
        :type log_var: Any
        :return: ``(batch, sequence)`` tensor of per-step KL values.
        :rtype: Any
        """
        mean = ops.cast(mean, "float32")
        log_var = ops.cast(log_var, "float32")
        clipped = ops.clip(log_var, -self.clip_log_var, self.clip_log_var)
        return -0.5 * ops.sum(
            1.0 + clipped - ops.square(mean) - ops.exp(clipped), axis=-1
        )

    def call(self, y_true, y_pred) -> Any:
        """Compute the negative ELBO for one batch.

        :param y_true: Input sequence ``(batch, sequence, height, width,
            channels)``, or a mapping whose ``"images"`` entry is that sequence.
        :type y_true: Any
        :param y_pred: The model's output dict; see
            :data:`REQUIRED_PREDICTION_KEYS` and
            :data:`VARIANCE_PREDICTION_KEYS`.
        :type y_pred: Any
        :return: The negative ELBO, **one value per sample**, shape ``(batch,)``.
            Not a scalar: see the note in :meth:`call`.
        :rtype: Any
        :raises ValueError: If ``y_pred`` is missing a required key, or carries half
            of the ``u`` pair.
        """
        y_pred = self._validate_prediction(y_pred)

        if isinstance(y_true, dict):
            y_true = y_true["images"]

        reconstruction = self.reconstruction_loss(
            y_true, y_pred["reconstruction"]
        )
        # PER SAMPLE, shape (batch,). Two reasons, and the second is the one that
        # bites: Keras multiplies by `sample_weight` and only then reduces, so a
        # scalar return BROADCASTS and charges every row the batch aggregate,
        # discarding which rows were weighted. The batch mean is the training
        # loop's job, not this loss's.
        #
        # The sequence axis is summed (Eq. 12's outer sum) and the batch axis is
        # kept.
        reconstruction = ops.sum(reconstruction, axis=-1)

        z_kl = ops.sum(
            self.gaussian_kl_loss(y_pred["z_mean"], y_pred["z_log_var"]), axis=-1
        )
        if self.has_variance_variables(y_pred):
            u_kl = ops.sum(
                self.gaussian_kl_loss(
                    y_pred["u_mean"], y_pred["u_log_var"]
                ),
                axis=-1,
            )
        else:
            # The plain-VAE baseline (`use_variance_variables=False`) has no u
            # posterior, so Eq. 12's second KL term does not exist for it rather
            # than being zero-valued by accident. Adding an explicit zero keeps
            # the returned shape `(batch,)` -- a bare `0.0` here would broadcast
            # against the other terms and be fine only by luck of their shapes.
            u_kl = ops.zeros_like(z_kl)

        # The pixel count is a SHAPE fact, read off the static shape; it never
        # depends on a tensor value, so this is a Python int by the time the
        # property reads it.
        if y_true.shape is not None and len(y_true.shape) >= 2:
            frame_elements = 1
            for dim in y_true.shape[2:]:
                if dim is not None:
                    frame_elements *= int(dim)
            if frame_elements:
                self._pixels_per_frame = frame_elements

        self.reconstruction_tracker.update_state(reconstruction)
        self.z_kl_tracker.update_state(z_kl)
        self.u_kl_tracker.update_state(u_kl)

        return reconstruction + self.kl_loss_weight * (z_kl + u_kl)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "kl_loss_weight": self.kl_loss_weight,
            "reconstruction_sum_reduction": self.reconstruction_sum_reduction,
            "clip_log_var": self.clip_log_var,
        })
        return config

# ---------------------------------------------------------------------