"""
Diffusion Schedulers for Score-Based nanoVLM

Implements various noise scheduling strategies for the forward and reverse
diffusion processes, following DDPM, DDIM, and continuous-time SDE formulations.

This is the foundation for score-based generative modeling in the VLM context.
"""

import keras
import numpy as np
from keras import ops
from typing import Tuple, Optional, Literal, Any


from dl_techniques.utils.logger import logger


# DECISION plan_2026-06-13_ae9ee2cd/D-004: DiffusionScheduler is a PLAIN Python class,
# NOT a keras.layers.Layer (cf. sd3_mmdit/scheduler.py FlowMatchEulerScheduler). It holds
# only NumPy schedule arrays and exposes pure tensor-math methods (.add_noise/.step/etc.);
# it is never invoked as scheduler(x). Do NOT re-add the Layer base or
# @register_keras_serializable: as a Layer it polluted model.layers and the .keras graph
# with a non-trainable utility. The parent ScoreBasedNanoVLM re-creates it in __init__ from
# its own inlined diffusion_config, so model round-trip does not depend on Layer serialization.
class DiffusionScheduler:
    """
    Base diffusion scheduler for score-based models.

    Implements the noise schedule for forward diffusion (data → noise) and
    provides utilities for reverse diffusion (noise → data). Following the
    framework from Ho et al. (2020) and Song et al. (2021).

    The forward process is: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
    where ᾱ_t = ∏_{s=1}^t (1 - β_s)

    Args:
        num_timesteps: Number of diffusion timesteps T. Defaults to 1000.
        beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic').
            Defaults to 'linear'.
        beta_start: Starting value of β at t=0. Defaults to 0.0001.
        beta_end: Ending value of β at t=T. Defaults to 0.02.
        clip_sample: Whether to clip predicted x_0 to [-1, 1]. Defaults to **False**,
            which is a deliberate departure from the DDPM convention this class
            otherwise follows. `[-1, 1]` is DDPM's *pixel*-space range, and nothing in
            this package diffuses pixels: `ScoreBasedNanoVLM.call` noises the output of
            `VisionEncoder`/`TextEncoder`, both of which end in a LayerNormalization,
            so a unit-variance coordinate falls outside `[-1, 1]` roughly a third of
            the time. Clipping also applied only on the reverse path — `add_noise`
            never clipped — so the train and inference domains disagreed. Pass True
            explicitly when the samples really are pixels in `[-1, 1]`.
        prediction_type: What the model predicts ('epsilon', 'sample', 'v_prediction').
            Defaults to 'sample', which is a deliberate departure from the DDPM
            convention this class otherwise follows. Every denoiser in this package is
            an x_0 predictor — ``ScoreBasedNanoVLM.call`` supervises them against the
            *clean* vision/text features — and none of the three shipped
            ``create_score_based_nanovlm`` presets overrides this argument, so an
            'epsilon' default would make ``step`` read an x_0-shaped output as noise
            and return silently wrong samples rather than raising. Pass 'epsilon'
            explicitly when the denoiser really predicts noise. Note that the value is
            not validated here: an unknown one raises only when ``step`` runs.

    References:
        - Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
        - Song et al. "Score-Based Generative Modeling through SDEs" (2021)
        - Song et al. "Denoising Diffusion Implicit Models" (2021)
    """

    # DECISION plan-2026-08-14T183218-f4c612aa/D-006: prediction_type defaults to
    # 'sample', NOT the DDPM-conventional 'epsilon'. Do NOT "restore" 'epsilon' here on
    # the grounds that it matches DDPM/diffusers: the denoisers in this package are
    # supervised against clean features by ScoreBasedNanoVLM.call, so they emit x_0, and
    # the three shipped create_score_based_nanovlm presets pass no prediction_type at
    # all — this default is their only route. With 'epsilon', step() feeds x_0 into
    # predict_start_from_noise and returns wrong samples with no error. The alternative
    # fix — rewriting call() to supervise against noise — was rejected: it would change
    # what every existing model in this package learns.
    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_schedule: Literal['linear', 'cosine', 'quadratic'] = 'linear',
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            # DECISION plan-2026-08-17T183311-79c63e38/D-029: clip_sample defaults
            # False, NOT the DDPM-conventional True. Do NOT "restore" True on the
            # grounds that it matches DDPM/diffusers: this scheduler only ever runs
            # over LayerNorm'd ENCODER FEATURES, not pixels, so [-1, 1] projects away
            # ~a third of every coordinate axis while add_noise (training) clips
            # nothing. The knob is deliberately KEPT rather than deleted — a
            # pixel-space caller may legitimately want it, and the three shipped
            # create_score_based_nanovlm presets pass no clip_sample at all, so this
            # default is their only route. See decisions.md D-029.
            clip_sample: bool = False,
            prediction_type: Literal['epsilon', 'sample', 'v_prediction'] = 'sample',
    ) -> None:
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(f"Must have 0 < beta_start < beta_end < 1, got {beta_start}, {beta_end}")

        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        # Compute noise schedule
        self.betas = self._compute_beta_schedule()

        # Precompute useful quantities (following DDPM notation)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.concatenate(
            [np.array([1.0], dtype=np.float32), self.alphas_cumprod[:-1]]
        )

        # Quantities for forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Quantities for reverse process p(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
                self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        logger.info(f"Initialized {beta_schedule} diffusion schedule with {num_timesteps} steps")

    def _compute_beta_schedule(self) -> np.ndarray:
        """Compute the beta schedule according to the specified type."""
        timesteps = np.arange(self.num_timesteps, dtype=np.float32)

        if self.beta_schedule == 'linear':
            # Linear schedule: β_t = β_start + (β_end - β_start) * t/T
            betas = np.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, dtype=np.float32
            )

        elif self.beta_schedule == 'quadratic':
            # Quadratic schedule
            betas = np.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps, dtype=np.float32
            ) ** 2

        elif self.beta_schedule == 'cosine':
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008  # offset
            steps = self.num_timesteps + 1
            x = np.linspace(0, self.num_timesteps, steps, dtype=np.float32)
            alphas_cumprod = np.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0.0001, 0.9999)

        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        return betas

    def add_noise(
            self,
            original_samples: keras.KerasTensor,
            noise: keras.KerasTensor,
            timesteps: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Forward diffusion: Add noise to samples according to q(x_t | x_0).

        Implements: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            original_samples: Clean samples x_0 of shape [batch, ...]
            noise: Gaussian noise ε of same shape as original_samples
            timesteps: Timestep indices of shape [batch] or scalar

        Returns:
            Noisy samples x_t
        """
        # Get coefficients for this timestep
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, original_samples)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples
        )

        # Apply noise: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def get_velocity(
            self,
            sample: keras.KerasTensor,
            noise: keras.KerasTensor,
            timesteps: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute velocity prediction v_t for the v-prediction objective.

        v_t = √ᾱ_t * ε - √(1-ᾱ_t) * x_0

        Args:
            sample: Clean samples x_0
            noise: Noise ε
            timesteps: Timestep indices

        Returns:
            Velocity v_t
        """
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, sample)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, sample
        )

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def predict_start_from_noise(
            self,
            x_t: keras.KerasTensor,
            t: keras.KerasTensor,
            noise: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Predict x_0 from x_t and predicted noise ε using Miyasawa's theorem.

        This is the key formula: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t

        Args:
            x_t: Noisy samples at timestep t
            t: Timestep indices
            noise: Predicted noise ε

        Returns:
            Predicted clean samples x_0
        """
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t
        )

        # x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        pred_original = (x_t - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod

        if self.clip_sample:
            pred_original = ops.clip(pred_original, -1.0, 1.0)

        return pred_original

    def predict_noise_from_start(
            self,
            x_t: keras.KerasTensor,
            t: keras.KerasTensor,
            x_0: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Predict the noise ε from x_t and a predicted x_0 — the exact inverse of
        :meth:`predict_start_from_noise`.

        Rearranging that method's `x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t` for ε gives

        `ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)`

        This is the conversion callers need when they hold an x_0 estimate and
        need the ε consistent with it — e.g. converting an x_0-predicting denoiser
        into an ε-predicting one, or recovering the noise that `add_noise` used.
        `ScoreBasedNanoVLM.compute_score_field` is its in-repo caller: it composes
        this with :meth:`get_score_from_noise` to turn a denoiser's x_0 output into
        a score, which is the only correct route because `get_score_from_noise`
        consumes ε and nothing else.

        It is deliberately not used by `ScoreBasedNanoVLM.generate_from_text`: an
        earlier version of that method converted x_0 to ε here before calling
        :meth:`step`, on the false premise (previously recorded in this docstring)
        that `step`'s `prediction_type='sample'` branch consumes ε. It does not —
        that branch consumes x_0 directly, so the conversion made the reverse
        process read noise as the clean sample. Do not reintroduce it.

        Note the inversion is exact only when `clip_sample` is False. With
        clipping enabled `predict_start_from_noise` is a projection, not a
        bijection: any x_0 driven outside [-1, 1] is clamped, and the ε that
        produced it cannot be recovered from the clamped value.

        Args:
            x_t: Noisy samples at timestep t
            t: Timestep indices
            x_0: Predicted clean samples

        Returns:
            The noise ε consistent with (x_t, x_0) at timestep t
        """
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t
        )

        # ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)
        return (x_t - sqrt_alpha_prod * x_0) / sqrt_one_minus_alpha_prod

    def get_score_from_noise(
            self,
            noise_pred: keras.KerasTensor,
            timesteps: keras.KerasTensor,
            x_t: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Convert predicted noise to score function ∇_x log p(x_t).

        Following Miyasawa/Tweedie: ∇_x log p(x_t) = -ε / √(1-ᾱ_t)

        Args:
            noise_pred: Predicted noise ε
            timesteps: Timestep indices
            x_t: Noisy samples (for shape)

        Returns:
            Score function ∇_x log p(x_t)
        """
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t
        )

        # Score: ∇ log p(x_t) = -ε / √(1-ᾱ_t)
        score = -noise_pred / sqrt_one_minus_alpha_prod

        return score

    def step(
            self,
            model_output: keras.KerasTensor,
            timestep: int,
            sample: keras.KerasTensor,
            generator: Optional[Any] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Reverse diffusion step: predict x_{t-1} from x_t and model output.

        Implements the reverse process p(x_{t-1} | x_t) using the denoiser.

        Args:
            model_output: Output from denoiser (ε, x_0, or v depending on prediction_type)
            timestep: Current timestep t
            sample: Current sample x_t
            generator: Random number generator (for stochastic sampling)

        Returns:
            Tuple of (previous sample x_{t-1}, predicted original x_0)
        """
        t = timestep

        # 1. Predict x_0 from model output
        if self.prediction_type == 'epsilon':
            # Model predicts noise ε
            pred_original_sample = self.predict_start_from_noise(sample, t, model_output)

        elif self.prediction_type == 'sample':
            # Model directly predicts x_0
            pred_original_sample = model_output

        elif self.prediction_type == 'v_prediction':
            # Model predicts velocity v
            sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, sample)
            sqrt_one_minus_alpha_prod = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, sample
            )
            pred_original_sample = (
                    sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * model_output
            )

        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            pred_original_sample = ops.clip(pred_original_sample, -1.0, 1.0)

        # 2. Compute previous sample mean using posterior q(x_{t-1} | x_t, x_0)
        pred_prev_sample = self._get_posterior_mean(sample, pred_original_sample, t)

        # 3. Add noise for stochastic sampling (not needed at t=0)
        if t > 0:
            variance = self._get_variance(t)
            noise = keras.random.normal(ops.shape(sample), dtype=sample.dtype)
            pred_prev_sample = pred_prev_sample + ops.sqrt(variance) * noise

        return pred_prev_sample, pred_original_sample

    def _get_posterior_mean(
            self,
            x_t: keras.KerasTensor,
            x_0: keras.KerasTensor,
            t: int
    ) -> keras.KerasTensor:
        """Compute posterior mean μ(x_t, x_0) for reverse process."""
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t)

        posterior_mean = coef1 * x_0 + coef2 * x_t
        return posterior_mean

    def _get_variance(self, t: int) -> float:
        """Get variance for timestep t."""
        if t == 0:
            return 0.0
        return self.posterior_variance[t]

    def _extract(
            self,
            arr: np.ndarray,
            timesteps: keras.KerasTensor,
            broadcast_shape: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Extract values from array at timestep indices and broadcast to shape.

        Args:
            arr: Array to extract from
            timesteps: Timestep indices [batch] or scalar
            broadcast_shape: Tensor with target shape [batch, ...]

        Returns:
            Extracted values broadcast to broadcast_shape
        """
        # Convert to tensor
        arr_tensor = ops.convert_to_tensor(arr, dtype='float32')

        # Handle scalar timestep
        if len(ops.shape(timesteps)) == 0:
            timesteps = ops.expand_dims(timesteps, 0)

        # Gather values
        res = ops.take(arr_tensor, timesteps, axis=0)

        # Reshape for broadcasting: [batch, 1, 1, ...]
        while len(ops.shape(res)) < len(ops.shape(broadcast_shape)):
            res = ops.expand_dims(res, -1)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-047
        # The gather runs in float32 — the schedule tables are float32 constants
        # and their tail values are small enough that float16 rounding matters —
        # but the coefficient is RETURNED in the dtype of the tensor it is about
        # to be multiplied with. Every caller writes `coef * sample`, so a
        # hard-coded float32 return raised under `mixed_float16` at the caller's
        # multiply, not here. Do NOT instead cast `arr` down before the gather.
        # See decisions.md D-047.
        target_dtype = getattr(broadcast_shape, "dtype", None)
        if target_dtype is not None:
            res = ops.cast(res, getattr(target_dtype, "name", target_dtype))

        return res