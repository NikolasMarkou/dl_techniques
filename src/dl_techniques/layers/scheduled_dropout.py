"""ScheduledDropout, a dropout layer whose rate follows a schedule over training.

The drop probability at training step ``t`` is ``clip(schedule(t), 0.0,
1.0 - 1e-6)``, where ``schedule`` is any
``keras.optimizers.schedules.LearningRateSchedule`` (cosine, exponential,
polynomial, piecewise all work unchanged) or a plain float for a constant
rate. The step index lives inside the layer: a non-trainable ``int64`` weight
counts this instance's training-mode forward passes, so the schedule needs no
callback, no custom training loop, and no coupling to the optimizer, and its
progress survives a `.keras` save/load. Inference is a pure identity: no mask,
no rescale, no RNG draw, no counter increment.

The counter counts this instance's own calls, not a global step. Calling the
layer more than once per optimizer step, or sharing one instance across
several call sites, advances the schedule faster than the training loop does.

References:
    - Srivastava et al., 2014. Dropout: A Simple Way to Prevent Neural
      Networks from Overfitting. (JMLR 15(56))
    - Morerio et al., 2017. Curriculum Dropout.
      (https://arxiv.org/abs/1703.06229)
    - Loshchilov and Hutter, 2017. SGDR: Stochastic Gradient Descent with
      Warm Restarts. (https://arxiv.org/abs/1608.03983)
"""

import keras
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.scheduled_dropout")
class ScheduledDropout(keras.layers.Layer):
    """Dropout whose rate is driven by a schedule over this layer's own steps.

    During training the layer draws a Bernoulli mask with probability
    ``clip(rate(t), 0.0, 1.0 - 1e-6)`` and rescales the survivors by
    ``1 / (1 - rate)``, exactly like :class:`keras.layers.Dropout`. During
    inference it is a pure identity: no mask, no rescale, no RNG draw, no
    counter increment.

    :param rate: Either a :class:`keras.optimizers.schedules.LearningRateSchedule`
        instance, evaluated at the step counter to give the drop probability,
        or a plain float in ``[0, 1)`` for a constant rate.
    :type rate: Union[float, keras.optimizers.schedules.LearningRateSchedule]
    :param noise_shape: Optional 1D shape for the binary mask, broadcast
        against the input. Same semantics as :class:`keras.layers.Dropout`.
    :type noise_shape: Optional[Sequence[Optional[int]]]
    :param seed: Optional integer seed backing a `keras.random.SeedGenerator`.
    :type seed: Optional[int]
    :param kwargs: Additional arguments forwarded to `keras.layers.Layer`.
    :raises TypeError: If ``rate`` is neither a schedule instance nor a number.
    :raises ValueError: If ``rate`` is a number outside ``[0, 1)``.

    :ivar step_counter: Non-trainable scalar ``int64`` weight created in
        ``build()``, counting this instance's training-mode forward passes.
    :vartype step_counter: keras.Variable
    :ivar seed_generator: Seed generator created in ``build()``.
    :vartype seed_generator: keras.random.SeedGenerator

    Note:
        The counter counts this instance's own training-mode forward passes,
        not a global optimizer step. Calling the layer twice per training
        step advances it by 2; sharing one instance across N call sites
        advances it by N. `predict`/`evaluate`/``training=False`` never
        increment it, but ``model(x, training=True)`` -- the MC-dropout idiom
        this repo uses for uncertainty estimation -- does. Ten MC samples
        moved the counter from 8 to 18 and were drawn at ten different,
        drifting rates (``[0.3, 0.275, ..., 0.075]``), so they are not i.i.d.
        and they fast-forward the decay. Pin and restore the counter around
        a sampling loop::

            saved = int(layer.step_counter)
            samples = []
            for _ in range(n_samples):
                layer.step_counter.assign(saved)
                samples.append(model(x, training=True))
            layer.step_counter.assign(saved)

        The training horizon lives entirely in the schedule's own
        ``decay_steps``; the layer never learns a total step count on its
        own, so pass ``epochs * steps_per_epoch`` to the schedule.

        The counter survives `.keras` save/load exactly, but the RNG stream
        position does not: a reloaded model resumes its decay at the right
        step but draws a fresh mask sequence from the seed's start. This
        matches stock `keras.layers.Dropout`, since `Layer.weights` excludes
        random-seed state.

        Multi-replica counting under TensorFlow distribution strategies is
        not verified; single-process training is the only tested
        configuration.

        A plain-float ``rate`` still creates and increments the counter --
        state layout and forward path are identical for both rate kinds.

        The clip bounds the rate to ``[0, 1 - 1e-6]``, not the activation
        magnitude. At the ceiling the few survivors are rescaled by
        ``1 / (1 - rate)`` -- measured at 986895x on an all-ones input, a
        finite, NaN-free spike that will still wreck a loss. A schedule whose
        range escapes ``[0, 1)`` (for example ``initial_learning_rate=1.5``)
        is accepted at construction without a check: a full-range check would
        need a step horizon the layer does not own (`decay_steps` lives in
        the schedule), and a step-0 check, while possible (measured:
        `PolynomialDecay(initial_learning_rate=1.5, ...)` returns 1.5,
        `ExponentialDecay(initial_learning_rate=1.0, ...)` returns 1.0, both
        out of range, with no false positive on a warmup `CosineDecay`, which
        correctly returns 0.0), is left to a future construction-contract
        change rather than folded in here. The clip at call time is the only
        defense until then.

    Example:
        Cosine-decayed dropout across a whole run, the horizon being the
        schedule's ``decay_steps``::

            import keras
            from dl_techniques.layers.scheduled_dropout import ScheduledDropout

            epochs, steps_per_epoch = 50, 200
            schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.3,
                decay_steps=epochs * steps_per_epoch,
                alpha=0.1,
            )
            model = keras.Sequential([
                keras.layers.Dense(64, activation="relu"),
                ScheduledDropout(schedule, seed=42),
                keras.layers.Dense(10),
            ])

        Constant rate, a drop-in for `keras.layers.Dropout`::

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

        :param input_shape: Input shape (unused -- no shape-dependent state).
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.step_counter = self.add_weight(
            name="step_counter",
            shape=(),
            dtype="int64",
            initializer="zeros",
            trainable=False,
        )

        # DECISION plan-2026-07-22T184143-94cb10cb/D-011: build the SeedGenerator
        # here, not in __init__ -- a 3-arm control showed __init__ placement is behaviourally identical but violates no-state-in-init. See decisions.md.
        self.seed_generator = keras.random.SeedGenerator(self.seed)

        super().build(input_shape)

    def _rate_at(self, step: Union[int, keras.KerasTensor]) -> keras.KerasTensor:
        """Evaluate the drop probability at ``step``, clipped into range.

        The single clamp site in this class. A rate of exactly 1.0 would divide
        by zero in the backend, hence the ``1 - 1e-6`` upper bound.

        :param step: Step index; an ``int64`` tensor or Variable needs no cast.
        :type step: Union[int, keras.KerasTensor]
        :return: Scalar float32 tensor in ``[0.0, 1.0 - 1e-6]``.
        :rtype: keras.KerasTensor
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
        next training call will use, not the one the previous call used.

        :return: Scalar float32 tensor in ``[0.0, 1.0 - 1e-6]``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the layer has not been built yet.
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

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag. Inference is a pure identity and
            does not advance the counter.
        :type training: Optional[bool]
        :return: Tensor with the same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        if not training:
            return inputs

        # DECISION plan-2026-07-22T184143-94cb10cb/D-010: read the counter before
        # incrementing -- swapping the order shifts every rate by one step. See decisions.md.
        step = self.step_counter
        rate = self._rate_at(step)
        self.step_counter.assign_add(1)

        # DECISION plan-2026-07-22T184143-94cb10cb/D-006: no rate==0 short-circuit
        # -- rate is a tensor, and 0 is already a bit-exact identity through the backend. See decisions.md.
        # DECISION plan-2026-07-22T184143-94cb10cb/D-004: the cast is required --
        # without it a mixed_float16 forward pass raises a dtype ValueError. See decisions.md.
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape, as a tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the serializable configuration of the layer.

        :return: Config dict. A schedule ``rate`` is nested via
            `keras.optimizers.schedules.serialize`; a float is stored verbatim.
        :rtype: Dict[str, Any]
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

        :param config: Dict from `get_config`. A dict-valued ``rate`` is a
            serialized schedule and is deserialized here.
        :type config: Dict[str, Any]
        :return: A new, unbuilt `ScheduledDropout` instance.
        :rtype: ScheduledDropout
        """
        config = dict(config)
        if isinstance(config.get("rate"), dict):
            config["rate"] = keras.optimizers.schedules.deserialize(config["rate"])
        return cls(**config)
