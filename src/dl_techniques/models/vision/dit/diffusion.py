"""The reverse process: sampling a latent out of a trained DiT.

:class:`DiT` is only half a generative model. It answers a single question --
"given a noised latent ``x_t`` and the step index ``t``, what noise was added?"
-- and everything that turns a thousand such answers into an image lives here:
the forward corruption ``q(x_t | x_0)`` the model was trained against, the
reverse-posterior algebra, the ancestral (DDPM) sampler, the deterministic-capable
(DDIM) sampler, and the timestep respacing that lets a model trained on 1000
steps be sampled in 50.

This module owns no tables. Every constant it reads comes from a
:class:`~dl_techniques.utils.ddpm_schedule.DDPMSchedule`, which is also what the
training loss reads, so the sampler and the objective provably agree on the same
numbers rather than agreeing by hand.

The reverse chain, and where each piece sits:

.. code-block:: text

                     x_T ~ N(0, I)        [B, H, W, C]
                          │
                          ▼
        ┌─────────────────────────────────────────────────────────┐
        │  for t = T-1, T-2, ... 1, 0   (a plain Python loop)      │
        │                                                         │
        │   ┌─ respacing ─────────────────────────────────────┐   │
        │   │ t_model = timestep_map[t]   ◄── ORIGINAL index  │   │
        │   └────────────────────┬────────────────────────────┘   │
        │                        ▼                                │
        │   ┌─ the model callable (CFG branch lives HERE) ────┐   │
        │   │ model_fn = DiT.call            (unguided)       │   │
        │   │          | DiT.forward_with_cfg (guided; the    │   │
        │   │            batch is duplicated by the CALLER,   │   │
        │   │            second half labelled num_classes)    │   │
        │   │ out = model_fn(x, t_model, **model_kwargs)      │   │
        │   │ out: [B, H, W, 2C]  when learn_sigma            │   │
        │   └────────────────────┬────────────────────────────┘   │
        │                        ▼                                │
        │   ┌─ p_mean_variance ───────────────────────────────┐   │
        │   │ eps_hat, v = split(out, 2, axis=-1)   ◄── LAST  │   │
        │   │                                                 │   │
        │   │ frac    = (v + 1) / 2                           │   │
        │   │ log_var = frac * log(beta_t)                    │   │
        │   │           ⊕ (1 - frac) * post_log_var_clip[t]   │   │
        │   │                                                 │   │
        │   │ x_0_hat = sqrt_recip[t]  * x_t                  │   │
        │   │           ⊖ sqrt_recipm1[t] * eps_hat           │   │
        │   │ (clip to [-1, 1] ONLY if clip_denoised)         │   │
        │   │                                                 │   │
        │   │ mean = coef1[t] * x_0_hat ⊕ coef2[t] * x_t      │   │
        │   └────────────────────┬────────────────────────────┘   │
        │                        ▼                                │
        │   ┌─ p_sample ──────────┐   ┌─ ddim_sample ──────────┐  │
        │   │ x = mean            │   │ sigma = eta * ...      │  │
        │   │   ⊕ [t != 0]        │   │ x = sqrt(a_prev)*x_0hat│  │
        │   │     * exp(.5*logvar)│   │   ⊕ sqrt(1-a_prev-s^2) │  │
        │   │     * noise         │   │     * eps              │  │
        │   │                     │   │   ⊕ [t != 0]*sigma*n   │  │
        │   └─────────────────────┘   └────────────────────────┘  │
        └─────────────────────────┬───────────────────────────────┘
                                  ▼
                             x_0  [B, H, W, C]

**Three things here are invisible to every shape, dtype and finiteness check.**

1. ``[t != 0]`` -- the ``nonzero_mask``. Dropping it adds one extra noise draw
   at the very last step. The output stays finite, keeps its shape, and is
   simply wrong.
2. ``sigma`` at ``eta = 0``. If the DDIM ``sigma`` is not exactly zero there,
   DDIM stops being deterministic while still producing plausible samples.
3. The respacing remap. Passing the respaced index ``t`` straight to a model
   trained on the original chain feeds it a ``t`` off by a factor of ``T / N``.
   Nothing raises; the samples are just noise-shaped garbage.

Each is pinned by a named test in
``tests/test_models/test_dit/test_dit_diffusion.py``.

**Latents are not images: ``clip_denoised`` defaults to ``False`` here.**
Upstream's ``gaussian_diffusion.py`` defaults it to ``True`` because the ADM/IDDPM
codebase it came from diffuses pixels rescaled to ``[-1, 1]``. DiT diffuses VAE
latents, whose values routinely leave that range, and upstream's own ``sample.py``
therefore passes ``clip_denoised=False`` explicitly for every DiT sample. A
sampler that silently clipped would destroy the latent while every test that only
checks shapes and finiteness stayed green, so this port makes the correct value
the default rather than a flag the caller must remember.

**The loops are eager Python loops.** ``p_sample_loop`` and ``ddim_sample_loop``
run ``num_timesteps`` model calls in a Python ``for``. Under
``tf.function``/``jit_compile`` that unrolls into ``num_timesteps`` copies of the
model graph, which is not traceable at any realistic ``T``; use the loops eagerly
(the regime upstream also uses) and respace to a small step count for speed. The
per-step methods -- :meth:`GaussianDiffusion.p_sample`,
:meth:`GaussianDiffusion.ddim_sample` -- are ordinary tensor code and are
traceable. The unrolling is documented by a test that counts model calls.

**Randomness is explicit.** Every sampling entry point takes ``seed``. Passing an
``int`` builds a fresh :class:`keras.random.SeedGenerator` for that call, which
is the only way to get a reproducible sample here:
``keras.utils.set_random_seed`` does NOT re-seed an already-created global
``SeedGenerator``, so a "reproducibility" check written against it would be
measuring nothing.

**Not registered for serialization, deliberately.** :class:`GaussianDiffusion` is
not a ``Layer`` or a ``Model``; it holds a handful of NumPy tables and no
weights, exactly like :class:`~dl_techniques.utils.ddpm_schedule.DDPMSchedule`,
which is likewise unregistered. It is rebuilt from its parameters
(:meth:`GaussianDiffusion.from_name` / :meth:`GaussianDiffusion.from_config`),
never deserialized.

References:
    - Ho, J., Jain, A. and Abbeel, P. "Denoising Diffusion Probabilistic
      Models." arXiv:2006.11239, 2020. https://arxiv.org/abs/2006.11239
      (``q_sample``, the reverse posterior, and the ancestral sampler).
    - Song, J., Meng, C. and Ermon, S. "Denoising Diffusion Implicit Models."
      arXiv:2010.02502, 2020. https://arxiv.org/abs/2010.02502 (``ddim_sample``
      and its Equation 12, the ``eta`` interpolation between DDIM and DDPM).
    - Nichol, A. and Dhariwal, P. "Improved Denoising Diffusion Probabilistic
      Models." arXiv:2102.09672, 2021. https://arxiv.org/abs/2102.09672 (the
      ``LEARNED_RANGE`` variance interpolation and timestep respacing).
    - Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers."
      arXiv:2212.09748, 2022. https://arxiv.org/abs/2212.09748 (the model this
      sampler drives, and the classifier-free-guidance sampling recipe).
    - Upstream ``fast-DiT`` reference copy staged under the plan's
      ``reference/diffusion/gaussian_diffusion.py`` and
      ``reference/diffusion/respace.py``, the arbiter for every formula above.
"""

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import keras
import numpy as np

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.vision_language.dit.config import DiffusionConfig
from dl_techniques.utils.ddpm_schedule import DDPMSchedule, space_timesteps
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------

__all__ = [
    "DEFAULT_CLIP_DENOISED",
    "GaussianDiffusion",
    "MODEL_MEAN_TYPES",
    "MODEL_VAR_TYPES",
]

#: What the model's first output half predicts. ``"epsilon"`` is DiT's setting.
MODEL_MEAN_TYPES: Tuple[str, ...] = ("epsilon", "start_x")

#: How the reverse-step variance is obtained. ``"learned_range"`` is DiT's
#: setting (``learn_sigma=True``); the two ``"fixed_*"`` names are the
#: ``learn_sigma=False`` branches and read no model output at all.
MODEL_VAR_TYPES: Tuple[str, ...] = (
    "learned_range",
    "learned",
    "fixed_small",
    "fixed_large",
)

# DECISION plan-2026-09-02T170923-1285ed83/D-017: default is False, diverging from
# upstream's `clip_denoised=True` — DiT diffuses VAE latents, which are not in
# [-1, 1], and upstream's own sample.py already passes False. See decisions.md.
#: Whether ``x_0_hat`` is clamped to ``[-1, 1]`` before the posterior mean.
#:
#: ``False``. See the module docstring: this sampler diffuses latents.
DEFAULT_CLIP_DENOISED: bool = False


# ---------------------------------------------------------------------


class GaussianDiffusion:
    """The forward and reverse Gaussian diffusion processes over a fixed schedule.

    A port of upstream's ``GaussianDiffusion`` and ``SpacedDiffusion`` fused into
    one class: upstream splits them because ``SpacedDiffusion`` subclasses and
    overrides four methods, but the whole difference is one index remap applied
    to ``t`` before the model sees it, which is a two-line private method here.

    Construct with :meth:`from_name` or :meth:`from_config` rather than the
    initializer; both handle respacing, which is what makes the ``timestep_map``
    and the shortened tables mutually consistent.

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │ GaussianDiffusion                                            │
        │                                                              │
        │  schedule : DDPMSchedule    ← every constant table, float64  │
        │  timestep_map : [N] or None ← respaced idx ⟶ ORIGINAL idx    │
        │                                                              │
        │  forward   q(x_t | x_0)   : q_sample                         │
        │  posterior q(x_{t-1}|x_t,x_0) : q_posterior_mean_variance    │
        │  reverse   p(x_{t-1} | x_t)   : p_mean_variance ⟶ p_sample   │
        │  reverse (DDIM)               : p_mean_variance ⟶ ddim_sample│
        │                                                              │
        │  x [B, H, W, C]  ⊕  model out [B, H, W, 2C]                  │
        │             │                                                │
        │             ▼                                                │
        │        x_{t-1} [B, H, W, C]                                  │
        └──────────────────────────────────────────────────────────────┘

    :param schedule: The (possibly already respaced) constant tables.
    :type schedule: DDPMSchedule
    :param model_mean_type: One of :data:`MODEL_MEAN_TYPES`.
    :type model_mean_type: str
    :param model_var_type: One of :data:`MODEL_VAR_TYPES`.
    :type model_var_type: str
    :param timestep_map: Original timestep indices retained by respacing, in
        increasing order, length ``schedule.num_timesteps``. ``None`` means no
        respacing and the identity map.
    :type timestep_map: Optional[np.ndarray]
    :param original_num_steps: Length of the chain the model was trained on.
        Informational; defaults to ``schedule.num_timesteps``.
    :type original_num_steps: Optional[int]
    :raises ValueError: If a type name is unknown, or if ``timestep_map`` does
        not match the schedule length.
    """

    def __init__(
        self,
        schedule: DDPMSchedule,
        model_mean_type: str = "epsilon",
        model_var_type: str = "learned_range",
        timestep_map: Optional[np.ndarray] = None,
        original_num_steps: Optional[int] = None,
    ) -> None:
        if model_mean_type not in MODEL_MEAN_TYPES:
            raise ValueError(
                f"model_mean_type must be one of {MODEL_MEAN_TYPES}, "
                f"got {model_mean_type!r}"
            )
        if model_var_type not in MODEL_VAR_TYPES:
            raise ValueError(
                f"model_var_type must be one of {MODEL_VAR_TYPES}, "
                f"got {model_var_type!r}"
            )

        self.schedule = schedule
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        if timestep_map is None:
            self.timestep_map: Optional[np.ndarray] = None
        else:
            mapped = np.asarray(timestep_map, dtype=np.int64)
            if mapped.ndim != 1:
                raise ValueError(
                    f"timestep_map must be 1-D, got shape {mapped.shape}"
                )
            if mapped.shape[0] != schedule.num_timesteps:
                raise ValueError(
                    f"timestep_map has {mapped.shape[0]} entries but the "
                    f"schedule covers {schedule.num_timesteps} timesteps"
                )
            self.timestep_map = mapped

        self.original_num_steps = (
            int(original_num_steps)
            if original_num_steps is not None
            else schedule.num_timesteps
        )

        # log(beta_t) is the LEARNED_RANGE upper bound. Precomputed once; it is
        # the only table this class derives rather than reads.
        self._log_betas: np.ndarray = np.log(schedule.betas)

        # The FIXED_LARGE variance is NOT posterior_variance: upstream replaces
        # every entry but the first with beta_t "to get a better decoder log
        # likelihood" (gaussian_diffusion.py:217-220).
        if schedule.num_timesteps > 1:
            self._fixed_large_variance: np.ndarray = np.append(
                schedule.posterior_variance[1], schedule.betas[1:]
            )
        else:
            self._fixed_large_variance = np.array(
                schedule.betas, dtype=np.float64
            )
        self._fixed_large_log_variance: np.ndarray = np.log(
            self._fixed_large_variance
        )

        logger.info(
            f"GaussianDiffusion: {schedule.num_timesteps} steps "
            f"(of {self.original_num_steps} original), "
            f"mean={model_mean_type}, var={model_var_type}, "
            f"respaced={self.timestep_map is not None}"
        )

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_name(
        cls,
        schedule_name: str = "linear",
        num_timesteps: int = 1000,
        timestep_respacing: Optional[Union[int, str, Sequence[int]]] = None,
        model_mean_type: str = "epsilon",
        model_var_type: str = "learned_range",
    ) -> "GaussianDiffusion":
        """Build a process from a named beta schedule, optionally respaced.

        This is upstream's ``create_diffusion`` (``reference/diffusion/__init__.py``)
        with the enum plumbing replaced by strings.

        :param schedule_name: One of
            :data:`~dl_techniques.utils.ddpm_schedule.VALID_BETA_SCHEDULES`.
        :type schedule_name: str
        :param num_timesteps: Length of the ORIGINAL chain the model was trained
            on.
        :type num_timesteps: int
        :param timestep_respacing: Section specification accepted by
            :func:`~dl_techniques.utils.ddpm_schedule.space_timesteps`, e.g.
            ``50``, ``"250"``, ``"ddim100"``. ``None`` or ``""`` keeps the full
            chain and installs no map.
        :type timestep_respacing: Optional[Union[int, str, Sequence[int]]]
        :param model_mean_type: One of :data:`MODEL_MEAN_TYPES`.
        :type model_mean_type: str
        :param model_var_type: One of :data:`MODEL_VAR_TYPES`.
        :type model_var_type: str
        :return: The configured process.
        :rtype: GaussianDiffusion
        """
        base = DDPMSchedule.from_name(schedule_name, num_timesteps)
        return cls._respace(
            base,
            timestep_respacing,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
        )

    @classmethod
    def from_config(
        cls,
        config: DiffusionConfig,
        timestep_respacing: Optional[Union[int, str, Sequence[int]]] = None,
    ) -> "GaussianDiffusion":
        """Build the process a given :class:`DiffusionConfig` describes.

        ``learn_sigma`` selects the variance type: ``True`` means the model emits
        ``2 * in_channels`` channels whose second half is a variance
        interpolation logit (``"learned_range"``); ``False`` means the variance
        is a fixed table and the model output is read whole
        (``"fixed_small"``, upstream's ``sigma_small=True`` branch -- the
        posterior variance, which is the right fixed choice for latents because
        ``"fixed_large"`` is tuned for a pixel decoder likelihood).

        :param config: The model-side diffusion configuration.
        :type config: DiffusionConfig
        :param timestep_respacing: As in :meth:`from_name`.
        :type timestep_respacing: Optional[Union[int, str, Sequence[int]]]
        :return: The configured process.
        :rtype: GaussianDiffusion
        """
        base = config.build_schedule()
        return cls._respace(
            base,
            timestep_respacing,
            model_mean_type="epsilon",
            model_var_type=(
                "learned_range" if config.learn_sigma else "fixed_small"
            ),
        )

    @classmethod
    def _respace(
        cls,
        base: DDPMSchedule,
        timestep_respacing: Optional[Union[int, str, Sequence[int]]],
        model_mean_type: str,
        model_var_type: str,
    ) -> "GaussianDiffusion":
        """Apply optional respacing and construct the instance.

        :param base: The full-length schedule.
        :type base: DDPMSchedule
        :param timestep_respacing: As in :meth:`from_name`.
        :type timestep_respacing: Optional[Union[int, str, Sequence[int]]]
        :param model_mean_type: One of :data:`MODEL_MEAN_TYPES`.
        :type model_mean_type: str
        :param model_var_type: One of :data:`MODEL_VAR_TYPES`.
        :type model_var_type: str
        :return: The configured process.
        :rtype: GaussianDiffusion
        """
        if timestep_respacing is None or timestep_respacing == "":
            return cls(
                base,
                model_mean_type=model_mean_type,
                model_var_type=model_var_type,
            )

        retained = space_timesteps(base.num_timesteps, timestep_respacing)
        spaced, timestep_map = base.respaced(retained)
        return cls(
            spaced,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            timestep_map=timestep_map,
            original_num_steps=base.num_timesteps,
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        """Number of steps THIS process runs (post-respacing).

        :return: ``schedule.num_timesteps``.
        :rtype: int
        """
        return self.schedule.num_timesteps

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _float_dtype(x: Any) -> str:
        """Floating dtype the process runs in, taken from the state tensor.

        Half precision is promoted to ``float32``: the reverse step
        exponentiates a log-variance and divides by ``1 - alphas_cumprod``,
        which underflows in ``float16`` with no finiteness symptom until it is
        far too late.

        :param x: The state tensor.
        :type x: Any
        :return: ``'float32'`` or ``'float64'``.
        :rtype: str
        """
        # DECISION plan-2026-09-02T170923-1285ed83/D-018: uses `getattr(dtype, "name",
        # None) or str(dtype)`, not `keras.backend.standardize_dtype` — a repo-wide
        # guard forbids any `keras.backend.*` call across `src/`. See decisions.md.
        raw = x.dtype
        name = getattr(raw, "name", None) or str(raw)
        return "float64" if name == "float64" else "float32"

    @staticmethod
    def _gather(
        table: np.ndarray,
        t: Any,
        n_broadcast_axes: int,
        dtype: str,
    ) -> Any:
        """Gather one table entry per sample, shaped to broadcast over the state.

        Upstream's ``_extract_into_tensor``
        (``reference/diffusion/gaussian_diffusion.py:545-550``), minus the
        explicit ``+ zeros(broadcast_shape)``: broadcasting is implicit here, and
        materializing a full-size zero tensor per lookup is pure cost.

        :param table: 1-D ``float64`` NumPy table of length ``T``.
        :type table: np.ndarray
        :param t: Integer timestep tensor of shape ``[B]``.
        :type t: Any
        :param n_broadcast_axes: Number of trailing singleton axes to append.
        :type n_broadcast_axes: int
        :param dtype: Floating dtype the result is produced in.
        :type dtype: str
        :return: Tensor of shape ``[B] + [1] * n_broadcast_axes``.
        :rtype: Any
        """
        values = keras.ops.take(
            keras.ops.cast(keras.ops.convert_to_tensor(table), dtype),
            t,
            axis=0,
        )
        return keras.ops.reshape(values, (-1,) + (1,) * n_broadcast_axes)

    def _map_t(self, t: Any) -> Any:
        """Translate a respaced timestep index into the ORIGINAL index.

        Upstream's ``_WrappedModel.__call__``
        (``reference/diffusion/respace.py:92-95``). The model was trained on the
        full chain, so it must always be handed an original index; the schedule
        tables, in contrast, are indexed by the respaced position. Getting these
        two the wrong way round raises nothing and produces noise.

        :param t: Respaced timestep tensor of shape ``[B]``.
        :type t: Any
        :return: Original-index timestep tensor of shape ``[B]``, or ``t``
            unchanged when this process is not respaced.
        :rtype: Any
        """
        if self.timestep_map is None:
            return t
        return keras.ops.take(
            keras.ops.convert_to_tensor(self.timestep_map), t, axis=0
        )

    @staticmethod
    def _seed_generator(
        seed: Optional[Union[int, keras.random.SeedGenerator]],
    ) -> Optional[keras.random.SeedGenerator]:
        """Normalize a seed argument into a generator (or ``None``).

        An ``int`` builds a FRESH generator, which is the only reproducible
        option: ``keras.utils.set_random_seed`` does not re-seed a
        ``SeedGenerator`` that already exists, so seeding globally and relying on
        the default generator reproduces nothing.

        :param seed: ``None`` (use the global generator), an ``int``, or an
            existing generator to draw from.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: The generator to pass to ``keras.random.normal``.
        :rtype: Optional[keras.random.SeedGenerator]
        """
        if seed is None or isinstance(seed, keras.random.SeedGenerator):
            return seed
        return keras.random.SeedGenerator(seed=int(seed))

    @staticmethod
    def _draw_noise(
        shape: Tuple[int, ...],
        dtype: str,
        noise: Optional[Any],
        seed: Optional[keras.random.SeedGenerator],
    ) -> Any:
        """Return the caller's noise, or draw a fresh standard-normal tensor.

        :param shape: Shape to draw.
        :type shape: Tuple[int, ...]
        :param dtype: Floating dtype of the result.
        :type dtype: str
        :param noise: Caller-supplied noise, or ``None`` to draw.
        :type noise: Optional[Any]
        :param seed: Generator to draw from, or ``None`` for the global one.
        :type seed: Optional[keras.random.SeedGenerator]
        :return: Tensor of the requested shape and dtype.
        :rtype: Any
        """
        if noise is not None:
            return keras.ops.cast(keras.ops.convert_to_tensor(noise), dtype)
        # keras.random.normal has no float64 kernel on every backend; draw in
        # float32 and widen, which is exact (float32 is a subset of float64).
        drawn = keras.random.normal(shape, dtype="float32", seed=seed)
        return keras.ops.cast(drawn, dtype)

    def _nonzero_mask(self, t: Any, n_broadcast_axes: int, dtype: str) -> Any:
        """``1.0`` where ``t != 0`` and ``0.0`` where ``t == 0``, broadcastable.

        Upstream's ``nonzero_mask``
        (``reference/diffusion/gaussian_diffusion.py:289-291,360-362``). At
        ``t == 0`` the reverse step lands on ``x_0`` itself and no noise may be
        added. Dropping this mask changes no shape, no dtype and nothing about
        finiteness -- it just corrupts the final sample.

        :param t: Timestep tensor of shape ``[B]``.
        :type t: Any
        :param n_broadcast_axes: Number of trailing singleton axes.
        :type n_broadcast_axes: int
        :param dtype: Floating dtype of the mask.
        :type dtype: str
        :return: Tensor of shape ``[B] + [1] * n_broadcast_axes``.
        :rtype: Any
        """
        mask = keras.ops.cast(
            keras.ops.not_equal(t, 0), dtype
        )
        return keras.ops.reshape(mask, (-1,) + (1,) * n_broadcast_axes)

    @staticmethod
    def _timestep_tensor(index: int, batch: int) -> Any:
        """A constant ``[B]`` timestep tensor for one loop iteration.

        :param index: The (respaced) timestep index.
        :type index: int
        :param batch: Batch size.
        :type batch: int
        :return: ``int32`` tensor of shape ``[batch]``.
        :rtype: Any
        """
        return keras.ops.full((batch,), index, dtype="int32")

    # -----------------------------------------------------------------
    # The forward process
    # -----------------------------------------------------------------

    def q_sample(
        self,
        x_start: Any,
        t: Any,
        noise: Optional[Any] = None,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Any:
        """Sample ``x_t ~ q(x_t | x_0)`` in one shot.

        ``reference/diffusion/gaussian_diffusion.py:159-167``.

        :param x_start: Clean data ``[B, ...]``.
        :type x_start: Any
        :param t: Timestep tensor ``[B]``.
        :type t: Any
        :param noise: Standard-normal noise of ``x_start``'s shape, or ``None``
            to draw.
        :type noise: Optional[Any]
        :param seed: Seed or generator used when ``noise`` is ``None``.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: The noised sample, same shape as ``x_start``.
        :rtype: Any
        """
        dtype = self._float_dtype(x_start)
        x_start = keras.ops.cast(x_start, dtype)
        n_bcast = len(keras.ops.shape(x_start)) - 1
        noise = self._draw_noise(
            tuple(keras.ops.shape(x_start)),
            dtype,
            noise,
            self._seed_generator(seed),
        )
        sched = self.schedule
        return (
            self._gather(sched.sqrt_alphas_cumprod, t, n_bcast, dtype) * x_start
            + self._gather(
                sched.sqrt_one_minus_alphas_cumprod, t, n_bcast, dtype
            )
            * noise
        )

    def q_posterior_mean_variance(
        self,
        x_start: Any,
        x_t: Any,
        t: Any,
    ) -> Tuple[Any, Any, Any]:
        """Mean and variance of the true reverse posterior ``q(x_{t-1}|x_t,x_0)``.

        ``reference/diffusion/gaussian_diffusion.py:169-186``.

        :param x_start: Clean (or predicted-clean) data ``[B, ...]``.
        :type x_start: Any
        :param x_t: Noised data at step ``t``, same shape.
        :type x_t: Any
        :param t: Timestep tensor ``[B]``.
        :type t: Any
        :return: ``(mean, variance, log_variance_clipped)``, each broadcast to
            ``x_t``'s shape by the arithmetic that produces them.
        :rtype: Tuple[Any, Any, Any]
        """
        dtype = self._float_dtype(x_t)
        x_start = keras.ops.cast(x_start, dtype)
        x_t = keras.ops.cast(x_t, dtype)
        n_bcast = len(keras.ops.shape(x_t)) - 1
        sched = self.schedule

        mean = (
            self._gather(sched.posterior_mean_coef1, t, n_bcast, dtype) * x_start
            + self._gather(sched.posterior_mean_coef2, t, n_bcast, dtype) * x_t
        )
        variance = self._gather(
            sched.posterior_variance, t, n_bcast, dtype
        ) + keras.ops.zeros_like(x_t)
        log_variance = self._gather(
            sched.posterior_log_variance_clipped, t, n_bcast, dtype
        ) + keras.ops.zeros_like(x_t)
        return mean, variance, log_variance

    # -----------------------------------------------------------------
    # The reverse process
    # -----------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t: Any, t: Any, eps: Any) -> Any:
        """Recover ``x_0`` from ``(x_t, eps)``.

        ``reference/diffusion/gaussian_diffusion.py:253-258``.

        :param x_t: Noised data ``[B, ...]``.
        :type x_t: Any
        :param t: Timestep tensor ``[B]``.
        :type t: Any
        :param eps: Predicted noise, same shape as ``x_t``.
        :type eps: Any
        :return: Predicted ``x_0``.
        :rtype: Any
        """
        dtype = self._float_dtype(x_t)
        n_bcast = len(keras.ops.shape(x_t)) - 1
        sched = self.schedule
        return (
            self._gather(sched.sqrt_recip_alphas_cumprod, t, n_bcast, dtype) * x_t
            - self._gather(sched.sqrt_recipm1_alphas_cumprod, t, n_bcast, dtype)
            * eps
        )

    def _predict_eps_from_xstart(
        self, x_t: Any, t: Any, pred_xstart: Any
    ) -> Any:
        """Recover ``eps`` from ``(x_t, x_0)``.

        ``reference/diffusion/gaussian_diffusion.py:260-263``. DDIM re-derives
        the noise from the (possibly clipped) ``x_0`` prediction rather than
        reusing the raw model output, so a clip actually reaches the update.

        :param x_t: Noised data ``[B, ...]``.
        :type x_t: Any
        :param t: Timestep tensor ``[B]``.
        :type t: Any
        :param pred_xstart: Predicted ``x_0``, same shape as ``x_t``.
        :type pred_xstart: Any
        :return: The implied noise.
        :rtype: Any
        """
        dtype = self._float_dtype(x_t)
        n_bcast = len(keras.ops.shape(x_t)) - 1
        sched = self.schedule
        return (
            self._gather(sched.sqrt_recip_alphas_cumprod, t, n_bcast, dtype) * x_t
            - pred_xstart
        ) / self._gather(sched.sqrt_recipm1_alphas_cumprod, t, n_bcast, dtype)

    def p_mean_variance(
        self,
        model_fn: Any,
        x: Any,
        t: Any,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the model and turn its output into ``p(x_{t-1} | x_t)``.

        ``reference/diffusion/gaussian_diffusion.py:188-251``, with one layout
        change: upstream splits the model output on ``dim=1`` because it is
        NCHW; this port is channels-LAST and splits on ``axis=-1``.

        The ``t`` handed to ``model_fn`` is remapped through
        :attr:`timestep_map`; the ``t`` used for every table lookup is not.

        .. code-block:: text

            model out [B, H, W, 2C]
                 │
                 ├─ eps_hat [B, H, W, C]  ──┐
                 └─ v       [B, H, W, C]    │
                        │                   │
                        ▼                   ▼
             frac = (v + 1) / 2      x_0_hat = sqrt_recip[t] * x_t
             log_var =                        ⊖ sqrt_recipm1[t] * eps_hat
               frac * log(beta[t])                     │
               ⊕ (1-frac) * post_log_var_clip[t]       ▼
                                          mean = coef1[t]*x_0_hat ⊕ coef2[t]*x_t

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``. Pass
            ``DiT.forward_with_cfg`` here for classifier-free guidance, exactly
            as upstream's ``sample.py`` does.
        :type model_fn: Any
        :param x: Current state ``[B, H, W, C]``.
        :type x: Any
        :param t: Respaced timestep tensor ``[B]``.
        :type t: Any
        :param clip_denoised: Clamp ``x_0_hat`` to ``[-1, 1]``. Defaults to
            :data:`DEFAULT_CLIP_DENOISED` (``False``) because DiT diffuses
            latents, which are not in that range.
        :type clip_denoised: bool
        :param denoised_fn: Optional map applied to ``x_0_hat`` BEFORE the clip.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: Extra keyword arguments forwarded to ``model_fn``
            (for DiT: ``y``, and ``cfg_scale`` when guided).
        :type model_kwargs: Optional[Dict[str, Any]]
        :return: ``{'mean', 'variance', 'log_variance', 'pred_xstart'}``, every
            value shaped like ``x``.
        :rtype: Dict[str, Any]
        :raises ValueError: If a learned-variance model output does not carry
            ``2 * C`` channels.
        """
        model_kwargs = {} if model_kwargs is None else model_kwargs

        dtype = self._float_dtype(x)
        x = keras.ops.cast(x, dtype)
        n_bcast = len(keras.ops.shape(x)) - 1
        channels = int(keras.ops.shape(x)[-1])
        sched = self.schedule

        model_output = model_fn(x, self._map_t(t), **model_kwargs)
        model_output = keras.ops.cast(
            keras.ops.convert_to_tensor(model_output), dtype
        )

        if self.model_var_type in ("learned", "learned_range"):
            out_channels = int(keras.ops.shape(model_output)[-1])
            if out_channels != 2 * channels:
                raise ValueError(
                    f"model_var_type={self.model_var_type!r} needs a model "
                    f"output with 2 * {channels} = {2 * channels} channels, "
                    f"got {out_channels}. A model built with learn_sigma=False "
                    f"must be sampled with a 'fixed_*' model_var_type."
                )
            model_output, model_var_values = keras.ops.split(
                model_output, 2, axis=-1
            )

            if self.model_var_type == "learned":
                model_log_variance = model_var_values
                model_variance = keras.ops.exp(model_log_variance)
            else:
                # DECISION plan-2026-09-02T170923-1285ed83/D-016: this is a second
                # copy of the learned-range interpolation in `losses/ddpm_hybrid_loss.py`
                # — do not import one from the other, that creates a models/losses cycle. See decisions.md.
                min_log = self._gather(
                    sched.posterior_log_variance_clipped, t, n_bcast, dtype
                )
                max_log = self._gather(self._log_betas, t, n_bcast, dtype)
                # model_var_values is in [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1.0) / 2.0
                model_log_variance = frac * max_log + (1.0 - frac) * min_log
                model_variance = keras.ops.exp(model_log_variance)
        else:
            if self.model_var_type == "fixed_large":
                variance_table = self._fixed_large_variance
                log_variance_table = self._fixed_large_log_variance
            else:
                variance_table = sched.posterior_variance
                log_variance_table = sched.posterior_log_variance_clipped
            model_variance = self._gather(
                variance_table, t, n_bcast, dtype
            ) + keras.ops.zeros_like(x)
            model_log_variance = self._gather(
                log_variance_table, t, n_bcast, dtype
            ) + keras.ops.zeros_like(x)

        def process_xstart(value: Any) -> Any:
            if denoised_fn is not None:
                value = denoised_fn(value)
            if clip_denoised:
                return keras.ops.clip(value, -1.0, 1.0)
            return value

        if self.model_mean_type == "start_x":
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model_fn: Any,
        x: Any,
        t: Any,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Any] = None,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Dict[str, Any]:
        """Take one ancestral (DDPM) reverse step.

        ``reference/diffusion/gaussian_diffusion.py:283-295``.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param x: Current state ``[B, H, W, C]``.
        :type x: Any
        :param t: Respaced timestep tensor ``[B]``.
        :type t: Any
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param noise: Fixed noise draw, or ``None`` to draw one.
        :type noise: Optional[Any]
        :param seed: Seed or generator used when ``noise`` is ``None``.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: ``{'sample', 'pred_xstart'}``.
        :rtype: Dict[str, Any]
        """
        out = self.p_mean_variance(
            model_fn,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        dtype = self._float_dtype(x)
        n_bcast = len(keras.ops.shape(x)) - 1
        noise = self._draw_noise(
            tuple(keras.ops.shape(x)), dtype, noise, self._seed_generator(seed)
        )
        nonzero_mask = self._nonzero_mask(t, n_bcast, dtype)
        sample = (
            out["mean"]
            + nonzero_mask
            * keras.ops.exp(0.5 * out["log_variance"])
            * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample(
        self,
        model_fn: Any,
        x: Any,
        t: Any,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
        noise: Optional[Any] = None,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Dict[str, Any]:
        """Take one DDIM reverse step (Song et al., Equation 12).

        ``reference/diffusion/gaussian_diffusion.py:334-364``.

        At ``eta = 0`` the ``sigma`` factor is exactly zero, the noise term
        vanishes and the step is DETERMINISTIC -- that is the whole point of
        DDIM, and it is invisible to any check that only looks at shape or
        finiteness. At ``eta = 1`` the step recovers the ancestral variance.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param x: Current state ``[B, H, W, C]``.
        :type x: Any
        :param t: Respaced timestep tensor ``[B]``.
        :type t: Any
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param eta: Stochasticity, ``0.0`` (deterministic) to ``1.0``.
        :type eta: float
        :param noise: Fixed noise draw, or ``None`` to draw one.
        :type noise: Optional[Any]
        :param seed: Seed or generator used when ``noise`` is ``None``.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: ``{'sample', 'pred_xstart'}``.
        :rtype: Dict[str, Any]
        """
        out = self.p_mean_variance(
            model_fn,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        dtype = self._float_dtype(x)
        x = keras.ops.cast(x, dtype)
        n_bcast = len(keras.ops.shape(x)) - 1
        sched = self.schedule

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = self._gather(sched.alphas_cumprod, t, n_bcast, dtype)
        alpha_bar_prev = self._gather(
            sched.alphas_cumprod_prev, t, n_bcast, dtype
        )
        sigma = (
            eta
            * keras.ops.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * keras.ops.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = self._draw_noise(
            tuple(keras.ops.shape(x)), dtype, noise, self._seed_generator(seed)
        )
        mean_pred = out["pred_xstart"] * keras.ops.sqrt(alpha_bar_prev) + (
            keras.ops.sqrt(1.0 - alpha_bar_prev - keras.ops.square(sigma)) * eps
        )
        nonzero_mask = self._nonzero_mask(t, n_bcast, dtype)
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # -----------------------------------------------------------------
    # The loops
    # -----------------------------------------------------------------

    def _initial_state(
        self,
        shape: Optional[Sequence[int]],
        noise: Optional[Any],
        seed: Optional[keras.random.SeedGenerator],
        dtype: str,
    ) -> Any:
        """Resolve ``x_T``: the caller's tensor, or a fresh standard-normal draw.

        :param shape: Shape to draw when ``noise`` is ``None``.
        :type shape: Optional[Sequence[int]]
        :param noise: Explicit starting state.
        :type noise: Optional[Any]
        :param seed: Generator to draw from.
        :type seed: Optional[keras.random.SeedGenerator]
        :param dtype: Floating dtype of the state.
        :type dtype: str
        :return: The starting state.
        :rtype: Any
        :raises ValueError: If both ``shape`` and ``noise`` are ``None``.
        """
        if noise is not None:
            return keras.ops.cast(keras.ops.convert_to_tensor(noise), dtype)
        if shape is None:
            raise ValueError(
                "one of `shape` or `noise` must be given so the loop knows "
                "what x_T looks like"
            )
        return self._draw_noise(tuple(int(s) for s in shape), dtype, None, seed)

    def p_sample_loop_progressive(
        self,
        model_fn: Any,
        shape: Optional[Sequence[int]] = None,
        noise: Optional[Any] = None,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield the ancestral chain one reverse step at a time.

        ``reference/diffusion/gaussian_diffusion.py:308-332``. This is an eager
        Python generator; see the module docstring on tracing.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param shape: Shape of ``x_T``, required unless ``noise`` is given.
        :type shape: Optional[Sequence[int]]
        :param noise: Explicit ``x_T``.
        :type noise: Optional[Any]
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param seed: Seed or generator for every draw in the loop, including
            ``x_T`` when ``noise`` is ``None``. An ``int`` builds ONE generator
            reused by the whole loop, so the whole chain is reproducible.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: Iterator of ``{'sample', 'pred_xstart'}`` dicts, latest first
            reverse step to last.
        :rtype: Iterator[Dict[str, Any]]
        """
        generator = self._seed_generator(seed)
        dtype = (
            self._float_dtype(noise)
            if noise is not None
            else keras.config.floatx()
        )
        img = self._initial_state(shape, noise, generator, dtype)
        batch = int(keras.ops.shape(img)[0])

        for index in reversed(range(self.num_timesteps)):
            t = self._timestep_tensor(index, batch)
            out = self.p_sample(
                model_fn,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                seed=generator,
            )
            yield out
            img = out["sample"]

    def p_sample_loop(
        self,
        model_fn: Any,
        shape: Optional[Sequence[int]] = None,
        noise: Optional[Any] = None,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Any:
        """Run the whole ancestral chain and return ``x_0``.

        ``reference/diffusion/gaussian_diffusion.py:297-306``.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param shape: Shape of ``x_T``, required unless ``noise`` is given.
        :type shape: Optional[Sequence[int]]
        :param noise: Explicit ``x_T``.
        :type noise: Optional[Any]
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param seed: See :meth:`p_sample_loop_progressive`.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: The final sample, shaped like ``x_T``.
        :rtype: Any
        """
        final: Optional[Dict[str, Any]] = None
        for out in self.p_sample_loop_progressive(
            model_fn,
            shape=shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            seed=seed,
        ):
            final = out
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model_fn: Any,
        shape: Optional[Sequence[int]] = None,
        noise: Optional[Any] = None,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield the DDIM chain one reverse step at a time.

        ``reference/diffusion/gaussian_diffusion.py:395-419``.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param shape: Shape of ``x_T``, required unless ``noise`` is given.
        :type shape: Optional[Sequence[int]]
        :param noise: Explicit ``x_T``.
        :type noise: Optional[Any]
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param eta: Stochasticity; ``0.0`` makes the whole chain deterministic
            given ``x_T``.
        :type eta: float
        :param seed: See :meth:`p_sample_loop_progressive`.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: Iterator of ``{'sample', 'pred_xstart'}`` dicts.
        :rtype: Iterator[Dict[str, Any]]
        """
        generator = self._seed_generator(seed)
        dtype = (
            self._float_dtype(noise)
            if noise is not None
            else keras.config.floatx()
        )
        img = self._initial_state(shape, noise, generator, dtype)
        batch = int(keras.ops.shape(img)[0])

        for index in reversed(range(self.num_timesteps)):
            t = self._timestep_tensor(index, batch)
            out = self.ddim_sample(
                model_fn,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                seed=generator,
            )
            yield out
            img = out["sample"]

    def ddim_sample_loop(
        self,
        model_fn: Any,
        shape: Optional[Sequence[int]] = None,
        noise: Optional[Any] = None,
        clip_denoised: bool = DEFAULT_CLIP_DENOISED,
        denoised_fn: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
        seed: Optional[Union[int, keras.random.SeedGenerator]] = None,
    ) -> Any:
        """Run the whole DDIM chain and return ``x_0``.

        ``reference/diffusion/gaussian_diffusion.py:384-393``.

        :param model_fn: Callable ``(x, t, **model_kwargs) -> tensor``.
        :type model_fn: Any
        :param shape: Shape of ``x_T``, required unless ``noise`` is given.
        :type shape: Optional[Sequence[int]]
        :param noise: Explicit ``x_T``.
        :type noise: Optional[Any]
        :param clip_denoised: See :meth:`p_mean_variance`.
        :type clip_denoised: bool
        :param denoised_fn: See :meth:`p_mean_variance`.
        :type denoised_fn: Optional[Any]
        :param model_kwargs: See :meth:`p_mean_variance`.
        :type model_kwargs: Optional[Dict[str, Any]]
        :param eta: Stochasticity, ``0.0`` (deterministic) to ``1.0``.
        :type eta: float
        :param seed: See :meth:`p_sample_loop_progressive`.
        :type seed: Optional[Union[int, keras.random.SeedGenerator]]
        :return: The final sample, shaped like ``x_T``.
        :rtype: Any
        """
        final: Optional[Dict[str, Any]] = None
        for out in self.ddim_sample_loop_progressive(
            model_fn,
            shape=shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            eta=eta,
            seed=seed,
        ):
            final = out
        return final["sample"]


# ---------------------------------------------------------------------
