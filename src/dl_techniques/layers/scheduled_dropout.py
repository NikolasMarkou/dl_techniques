"""
Dropout with a drop probability that varies over the course of training.

This layer embodies the principle of curriculum regularization, a design
paradigm that treats the strength of a regularizer as a quantity to be annealed
rather than a constant to be tuned. The core idea is that the optimal amount of
noise is not the same at every point in training: heavy dropout early prevents
premature co-adaptation while the network is still exploring, whereas the same
amount late in training merely obstructs the fine-grained convergence that
determines final accuracy. Decaying the rate over training resolves the tension
between those two regimes instead of settling on a single compromise value.

The mechanism reuses existing infrastructure rather than introducing new decay
math. The drop probability at training step `t` is produced by evaluating a
`keras.optimizers.schedules.LearningRateSchedule` at `t` and clipping the
result:

`rate(t) = clip(schedule(t), 0.0, 1.0 - 1e-6)`

Every curve Keras ships (cosine, exponential, polynomial, piecewise) therefore
becomes available verbatim as a dropout schedule. The `LearningRateSchedule`
name is a historical misnomer in this context: the interface is a generic
`step -> scalar` mapping with nothing learning-rate specific about it. A plain
float is also accepted, in which case the layer is a drop-in replacement for
`keras.layers.Dropout`. The upper clip bound exists because a rate of exactly
1.0 would divide by zero in the inverted-dropout rescale; it bounds the rate,
not the activation magnitude, and is not a safety net against a badly
parameterized schedule.

Architecturally, the significant decision is that the step index is
layer-internal rather than supplied by the trainer. A non-trainable scalar
`int64` weight is created in `build()` and incremented once per training-mode
forward pass, giving curriculum behaviour with no callback, no custom training
loop, and no coupling to the optimizer. Because the counter is a real weight it
is captured in `.keras` checkpoints, so a resumed run continues its decay from
where it stopped instead of silently restarting it. During inference the layer
is a pure identity: no mask, no rescale, no RNG draw, and no counter increment.

Two consequences of that design deserve attention. First, the counter tracks
*this instance's* training-mode calls, not global optimizer steps, so an
instance invoked multiple times per step or shared across several call sites
advances faster than the step count. This matters most for the MC-dropout idiom
`model(x, training=True)`, where repeated sampling both advances the decay and
draws each sample at a different rate, making the samples non-i.i.d.; pinning
and restoring the counter around the sampling loop avoids both problems.
Second, the counter round-trips exactly through serialization but the RNG stream
*position* does not, since `Layer.weights` excludes random-seed state by design.
A reloaded model resumes its decay at the correct step yet draws a fresh mask
sequence, matching stock `keras.layers.Dropout` behaviour.

The training horizon is owned entirely by the schedule's own `decay_steps`; the
layer never learns a total step count independently, which avoids a second
source of truth for something the schedule already specifies.

References:
    - Srivastava et al., 2014. Dropout: A Simple Way to Prevent Neural Networks
      from Overfitting. JMLR 15(56).
    - Morerio et al., 2017. Curriculum Dropout.
      (https://arxiv.org/abs/1703.06229)
    - Loshchilov and Hutter, 2017. SGDR: Stochastic Gradient Descent with Warm
      Restarts. (https://arxiv.org/abs/1608.03983)

"""


import keras
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ScheduledDropout(keras.layers.Layer):
    """Dropout whose rate is driven by a schedule over this layer's own steps.

    During training the layer draws a Bernoulli mask with probability
    ``clip(rate(t), 0.0, 1.0 - 1e-6)`` and rescales the survivors by
    ``1 / (1 - rate)``, exactly like `keras.layers.Dropout`. During inference it
    is a pure identity: no mask, no rescale, no RNG draw, no counter increment.

    Args:
        rate: Either a `keras.optimizers.schedules.LearningRateSchedule`
            instance -- evaluated at the step counter to give the drop
            probability -- or a plain float in ``[0, 1)`` meaning a constant
            rate. Anything else raises `TypeError`.
        noise_shape: Optional 1D shape for the binary mask, broadcast against
            the input. Identical semantics to `keras.layers.Dropout`.
        seed: Optional integer seed backing a `keras.random.SeedGenerator`.
        **kwargs: Additional arguments forwarded to `keras.layers.Layer`.

    Attributes:
        step_counter: Non-trainable scalar ``int64`` weight created in
            ``build()``, counting this instance's training-mode forward passes.
        seed_generator: `keras.random.SeedGenerator` created in ``build()``.

    Raises:
        TypeError: If ``rate`` is neither a schedule instance nor a number.
        ValueError: If ``rate`` is a number outside ``[0, 1)``.

    Sharp edges (each one measured, not assumed):
        1. The counter counts **this layer instance's own training-mode forward
           passes**, not global optimizer steps. An instance called twice per
           training step advances by 2 per step; one shared across N sites
           advances by N. Inference via `predict`, `evaluate` or an explicit
           ``training=False`` never increments it -- but ``model(x,
           training=True)``, the MC-dropout idiom this repo uses for
           uncertainty estimation, **does**. Ten MC samples were measured to
           move the counter ``8 -> 18`` and to be drawn at ten *different*,
           drifting rates (``[0.3, 0.275, ..., 0.075]``), so they are not
           i.i.d.; they also fast-forward the decay, corrupting a later resume.
           Workaround -- pin the counter for the whole loop, then restore it::

               saved = int(layer.step_counter)
               samples = []
               for _ in range(n_samples):
                   layer.step_counter.assign(saved)  # same rate every sample
                   samples.append(model(x, training=True))
               layer.step_counter.assign(saved)      # training state intact
        2. The training horizon lives in the schedule's own ``decay_steps`` and
           nowhere else -- there is no progress-fraction API and the layer never
           learns ``total_steps`` independently. The caller passes
           ``epochs * steps_per_epoch`` to the schedule (see the example).
        3. The counter survives `.keras` save/load exactly, but the RNG stream
           *position* does not: a reloaded model resumes its decay at the right
           step yet draws a fresh mask sequence from the seed's start. This is
           stock Keras 3 behaviour, not a defect here -- `keras.layers.Dropout`
           is identical, because ``Layer.weights`` excludes random-seed state by
           design. Two freshly built layers with the same ``seed`` are
           reproducible against each other; a reloaded layer is not
           bit-reproducible against an uninterrupted run.
        4. Multi-replica counting under TensorFlow distribution strategies is
           UNVERIFIED and untested. Single-process training is the only
           supported configuration.
        5. A plain-float ``rate`` still creates and still increments the
           counter: state layout and forward path are identical for both rate
           kinds, only the value differs.
        6. The clamp bounds the **rate** to ``[0, 1 - 1e-6]``; it does NOT bound
           the activation magnitude and it is not a safety net. At the ceiling
           the few survivors are rescaled by ``1 / (1 - rate)`` -- measured at
           986895x on an all-ones input, a ~10^6 spike that is finite and
           NaN-free but will still wreck a loss. A schedule whose range escapes
           ``[0, 1)`` (say ``initial_learning_rate=1.5`` -- exactly the mistake
           the "LearningRateSchedule" name invites) is therefore accepted
           **silently** at construction. Check your schedule's endpoints
           yourself.

           On why nothing is checked at construction, precisely:

           - A **full-range** check is genuinely impossible here. It would have
             to evaluate the schedule over a step domain, and the layer does
             not own one -- the horizon lives in the schedule's own
             ``decay_steps`` (sharp edge 2). Any horizon the layer assumed
             would be either wrong or a second source of truth for something
             ``decay_steps`` already owns.
           - A **step-0** sanity check is a different matter and needs no
             horizon at all. Evaluating ``schedule(0)`` once would catch both
             footguns named above -- measured, ``PolynomialDecay(
             initial_learning_rate=1.5, ...)`` returns 1.5 and
             ``ExponentialDecay(initial_learning_rate=1.0, ...)`` returns 1.0,
             both out of range -- with no false positive on the obvious
             candidate, since a warmup ``CosineDecay`` correctly returns 0.0.
             So the omission is a **choice**, not an impossibility.
           - That check is deliberately **out of scope for this iteration**.
             Whether it should warn, raise, or stay silent, and for which
             schedule types, is an API decision about the layer's construction
             contract; it deserves its own plan rather than being folded into a
             fix pass. Until then, the clip at call time is the only defence.

    Example:
        Cosine-decayed dropout across a whole run, the horizon being the
        schedule's ``decay_steps``::

            import keras
            from dl_techniques.layers.scheduled_dropout import ScheduledDropout

            epochs, steps_per_epoch = 50, 200
            schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.3,             # drop rate at step 0
                decay_steps=epochs * steps_per_epoch,  # the only horizon
                alpha=0.1,                             # floors at 0.03
            )
            model = keras.Sequential([
                keras.layers.Dense(64, activation="relu"),
                ScheduledDropout(schedule, seed=42),
                keras.layers.Dense(10),
            ])

        Constant rate, i.e. a drop-in for `keras.layers.Dropout`::

            layer = ScheduledDropout(0.25)
            y = layer(keras.ops.ones((2, 4)), training=True)
            float(layer.current_rate())   # -> 0.25, forever
            int(layer.step_counter)       # -> 1
    """

    def __init__(
        self,
        rate: Union[float, keras.optimizers.schedules.LearningRateSchedule],
        noise_shape: Optional[Sequence[Optional[int]]] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(rate, keras.optimizers.schedules.LearningRateSchedule):
            self.rate = rate
        elif isinstance(rate, (int, float)) and not isinstance(rate, bool):
            if not 0.0 <= float(rate) < 1.0:
                raise ValueError(
                    f"Invalid value received for argument `rate`. Expected a "
                    f"float value in [0, 1). Received: rate={rate}"
                )
            self.rate = float(rate)
        else:
            raise TypeError(
                f"`rate` must be either a float in [0, 1) or a "
                f"`keras.optimizers.schedules.LearningRateSchedule` instance. "
                f"Received: rate={rate!r} of type {type(rate).__name__}"
            )

        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

        # All state is created in build(), never here.
        self.step_counter = None
        self.seed_generator = None

        logger.info(
            f"Created ScheduledDropout layer '{self.name}' with "
            f"rate={self.rate}, noise_shape={self.noise_shape}, "
            f"seed={self.seed}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the step counter and the seed generator.

        Args:
            input_shape: Input shape (unused -- no shape-dependent state).
        """
        self.step_counter = self.add_weight(
            name="step_counter",
            shape=(),
            dtype="int64",
            initializer="zeros",
            trainable=False,
        )

        # DECISION plan-2026-07-22T184143-94cb10cb/D-011: the SeedGenerator is
        # built here, NOT in __init__ as stock Dropout does. Do not "fix" this to
        # chase RNG-state persistence -- a 3-arm control proved __init__ placement
        # is behaviourally identical (Keras excludes seed state from
        # Layer.weights by design) and it violates the repo's no-state-in-init rule.
        self.seed_generator = keras.random.SeedGenerator(self.seed)

        super().build(input_shape)

    def _rate_at(self, step: Union[int, keras.KerasTensor]) -> keras.KerasTensor:
        """Evaluate the drop probability at ``step``, clipped into range.

        The single clamp site in this class. A rate of exactly 1.0 would divide
        by zero in the backend, hence the ``1 - 1e-6`` upper bound.

        Args:
            step: Step index; an ``int64`` tensor or Variable needs no cast.

        Returns:
            Scalar float32 tensor in ``[0.0, 1.0 - 1e-6]``.
        """
        if isinstance(self.rate, keras.optimizers.schedules.LearningRateSchedule):
            raw_rate = self.rate(step)
        else:
            raw_rate = keras.ops.convert_to_tensor(self.rate, dtype="float32")

        return keras.ops.clip(raw_rate, 0.0, 1.0 - 1e-6)

    def current_rate(self) -> keras.KerasTensor:
        """Return the clipped rate at the current counter, without advancing it.

        Read-only handle for tests and training logs. After N training-mode
        calls the counter holds N, so this returns ``schedule(N)``: the rate the
        **next** training call will use, not the one the previous call used.

        Returns:
            Scalar float32 tensor in ``[0.0, 1.0 - 1e-6]``.

        Raises:
            ValueError: If the layer has not been built yet.
        """
        if not self.built:
            raise ValueError(
                f"ScheduledDropout layer '{self.name}' must be built before "
                f"`current_rate()` can be called."
            )
        return self._rate_at(self.step_counter)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = False
    ) -> keras.KerasTensor:
        """Apply scheduled dropout.

        Args:
            inputs: Input tensor of any shape.
            training: Training mode flag. Inference is a pure identity and does
                not advance the counter.

        Returns:
            Tensor with the same shape and dtype as ``inputs``.
        """
        if not training:
            return inputs

        # DECISION plan-2026-07-22T184143-94cb10cb/D-010: read the counter, THEN
        # increment. Do not reorder -- the first training call must see
        # schedule(0), and swapping these lines shifts every rate by one step.
        step = self.step_counter
        rate = self._rate_at(step)
        self.step_counter.assign_add(1)

        # DECISION plan-2026-07-22T184143-94cb10cb/D-006: no `rate == 0`
        # short-circuit. It cannot be a Python `if` on a scheduled (tensor) rate,
        # and rate 0 is already a bit-exact identity through the backend.
        # DECISION plan-2026-07-22T184143-94cb10cb/D-004: the cast is required,
        # not defensive -- without it a mixed_float16 forward pass raises
        # ValueError: "x.dtype" must be compatible with "rate.dtype".
        return keras.random.dropout(
            inputs,
            keras.ops.cast(rate, inputs.dtype),
            noise_shape=self.noise_shape,
            seed=self.seed_generator,
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which always equals ``input_shape``.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            The same shape, as a tuple.
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable configuration of the layer.

        Returns:
            Config dict. A schedule ``rate`` is nested via
            `keras.optimizers.schedules.serialize`; a float is stored verbatim.
        """
        config = super().get_config()
        if isinstance(self.rate, keras.optimizers.schedules.LearningRateSchedule):
            rate_config = keras.optimizers.schedules.serialize(self.rate)
        else:
            rate_config = self.rate
        config.update({
            "rate": rate_config,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ScheduledDropout":
        """Rebuild a layer from its configuration.

        Args:
            config: Dict from `get_config`. A dict-valued ``rate`` is a
                serialized schedule and is deserialized here.

        Returns:
            A new, unbuilt `ScheduledDropout` instance.
        """
        config = dict(config)
        if isinstance(config.get("rate"), dict):
            config["rate"] = keras.optimizers.schedules.deserialize(config["rate"])
        return cls(**config)

# ---------------------------------------------------------------------
