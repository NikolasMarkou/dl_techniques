"""``ComplexLayer`` -- the one piece of ``layers/complex/`` that is shared.

Exactly one thing in this package is common to more than one layer: the complex
weight draw, magnitude from a Rayleigh distribution and phase uniform over
``[-pi, pi]``. Only the two weight-owning layers (``ComplexConv2D``,
``ComplexDense``) need it, so it lives on a base they inherit rather than in a
free helper module; the four weightless layers of the package extend
``keras.layers.Layer`` directly and never import this file.

The serialization contract rides along with it. ``get_config`` here emits
``kernel_regularizer``/``kernel_initializer`` through the Keras ``serialize``
helpers and ``from_config`` turns them back into objects, so neither subclass
repeats that pairing.

Two constructor parameters, ``epsilon`` and ``kernel_initializer``, are accepted
and serialized but read by no computation. Both are pinned deliberately rather
than overlooked -- see the DECISION anchors at their assignment sites, and
``decisions.md`` D-053 of ``plan-2026-08-19T163559-499b6f0e`` and D-002 of
``plan-2026-09-08T070501-528ded1a``.

References:
    - Trabelsi et al., 2018. Deep Complex Networks.
    - Arjovsky et al., 2016. Unitary Evolution Recurrent Neural Networks.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.random import rayleigh
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.base")
class ComplexLayer(keras.layers.Layer):
    """Base class for complex-valued layers.

    Handles complex weight initialization: magnitude from a Rayleigh
    distribution, phase from a uniform distribution over ``[-pi, pi]``.
    Complex numbers are represented as ``z = x + iy`` throughout, computed
    with split real/imaginary arithmetic.

    :param epsilon: Accepted and serialized for config compatibility but read
        by no computation in this module. Measured on CoShNet: values from
        ``1e-30`` to ``1e+3`` move the output by exactly 0. Kept because
        existing ``.keras`` files pass it through ``from_config``; removing it
        would raise a ``TypeError`` loading those checkpoints. Defaults to
        ``1e-7``.
    :param kernel_regularizer: Regularizer applied to both real and imaginary
        parts of complex weights. Defaults to ``None``.
    :param kernel_initializer: Accepted and serialized for config compatibility
        but read by no computation in this module. Measured: a spy
        ``Initializer`` records 0 ``__call__`` invocations during ``build()``
        for both ``ComplexDense`` and ``ComplexConv2D``, and the kernel is not
        the spy's value -- ``_init_complex_weights`` draws its own Rayleigh
        magnitude and uniform phase. It is pinned to preserve behaviour, not
        because a wire-up is impossible: real and imaginary parts drawn from two
        DISTINCT-SEED ``Initializer`` instances do give a genuine complex weight,
        but at a different scale from the shipped draw, so honouring this
        parameter would move every CoShNet kernel's scale. At CoShNet's
        ``(5, 5, 3, 20)`` kernel the shipped scheme's analytic ``mean|z|`` is
        ``0.181850`` (12 seeds: mean ``0.181919``, std ``0.001900``, spread
        ``+/-0.005``) against ``HeNormal``'s ``0.204665``. (One instance called
        twice does NOT -- a keras 3.8 ``Initializer`` is deterministic per shape,
        so that yields ``imag == real``; see the DECISION anchor in ``__init__``.)
        Kept because existing ``.keras`` files pass it through ``from_config``;
        removing it would raise a ``TypeError`` loading those checkpoints.
        Defaults to ``GlorotUniform``.
    """

    def __init__(
        self,
        epsilon: float = 1e-7,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_initializer: Optional[keras.initializers.Initializer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-053: epsilon is inert -- no
        # division in this module reads it. Removing it breaks from_config on every saved .keras checkpoint. See decisions.md.
        self.epsilon = epsilon
        self.kernel_regularizer = kernel_regularizer
        # DECISION plan-2026-09-08T070501-528ded1a/D-002: kernel_initializer is inert -- a spy
        # Initializer records 0 __call__ invocations during build() on both weight-owning subclasses.
        # The pin is BEHAVIOUR PRESERVATION, not impossibility. A working wire-up DOES exist: draw the
        # real and imaginary parts from TWO DISTINCT-SEED Initializer instances (or from an explicitly
        # complex initializer). It is refused because it moves every CoShNet kernel's scale and drops
        # Trabelsi's scheme -- per-component std at CoShNet's (5, 5, 3, 20) kernel, analytic and
        # seed-free: shipped sqrt(2/(75+20)) = 0.145095, mean|z| = sigma*sqrt(pi/2) = 0.181850, against
        # HeNormal's sqrt(2/75) = 0.163299 (mean|z| 0.204665; MEASURED 0.204822 at seeds 1 and 2) and
        # GlorotUniform's 0.058977, 2.46x low because keras reads this shape as fan_in=75/fan_out=500.
        # Quote the ANALYTIC mean|z|, never one draw's: 12 seeds through this method give mean 0.181919,
        # std 0.001900, min 0.179526, max 0.184856, and an independent 5-seed run reached 0.1780/0.1863,
        # so read +/-0.005 as sampling spread. HeNormal's gap is ~4x that, which is why the ruling holds.
        # TRAP, and the reason this comment has been rewritten twice: do NOT probe a wire-up with ONE
        # instance called twice. A keras 3.8 Initializer is DETERMINISTIC PER SHAPE -- `i(s)` is
        # bit-identical to `i(s)`, seeded or not -- so imag == real, the "complex" weight is a real
        # kernel times (1+i) with exactly 2 distinct phases, and every mean|z| so measured is merely
        # sqrt(2)*mean|r| (Constant(0.1) reading exactly 0.141421 = 0.1*sqrt(2) is the tell). Pinned by
        # test_one_initializer_instance_replays_its_draw_at_a_fixed_shape. Do NOT wire this parameter
        # up under its current name; propose it as a new opt-in mode. See decisions.md D-002.
        self.kernel_initializer = kernel_initializer or keras.initializers.GlorotUniform()

    def _init_complex_weights(
        self,
        shape: Tuple[int, ...],
        dtype: tf.DType = tf.complex64
    ) -> tf.Tensor:
        """
        Initialize complex weights using Rayleigh distribution with proper scaling.

        This method creates complex-valued weights by sampling magnitudes from a
        Rayleigh distribution and phases from a uniform distribution, then combining
        them into complex numbers. The scaling follows Xavier/Glorot initialization
        principles adapted for the complex domain.

        :param shape: Shape of the weight tensor.
        :type shape: Tuple[int, ...]
        :param dtype: TensorFlow dtype for the complex weights.
        :type dtype: tf.DType
        :return: Complex-valued weight tensor.
        :rtype: tf.Tensor
        """
        fan_in = int(np.prod(shape[:-1]))
        fan_out = int(shape[-1])
        sigma = keras.ops.sqrt(2.0 / (fan_in + fan_out))

        magnitude = rayleigh(shape, sigma, dtype=tf.float32)
        phase = keras.random.uniform(shape, -np.pi, np.pi, dtype=tf.float32)

        weights = tf.complex(
            magnitude * keras.ops.cos(phase),
            magnitude * keras.ops.sin(phase)
        )

        return tf.cast(weights, dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ComplexLayer":
        """Rebuild a layer from a config produced by :meth:`get_config`.

        ``get_config`` writes ``kernel_regularizer`` and ``kernel_initializer``
        through the Keras ``serialize`` helpers, so both arrive here as plain
        dicts and must be turned back into objects; without this, the reloaded
        layer keeps the raw dict as its attribute and every later use of it
        fails far from the load site. ``kernel_initializer`` is inert in the
        forward path (see D-002 above) but is deserialized all the same,
        because surviving this round trip is precisely why the parameter was
        kept rather than deleted.

        Only those two keys are touched. Nothing is popped -- the base keys
        (``name``, ``trainable``, ``dtype``) are passed straight through to the
        constructor, and a copy is taken so the caller's dict is not consumed.

        :param config: Configuration dictionary, as returned by ``get_config``.
        :type config: Dict[str, Any]
        :return: A new layer instance built from ``config``.
        :rtype: ComplexLayer
        """
        config = dict(config)

        # The `isinstance(..., dict)` guards follow the v2 guide's own 6.1
        # template. MEASURED at keras 3.8: both `deserialize` helpers are
        # idempotent on a live object and pass `None` through, so removing the
        # guards changes nothing observable today -- they are kept for the
        # guide's shape and against a future non-idempotent helper, not as a
        # live defense.
        if isinstance(config.get('kernel_regularizer'), dict):
            config['kernel_regularizer'] = keras.regularizers.deserialize(
                config['kernel_regularizer']
            )
        if isinstance(config.get('kernel_initializer'), dict):
            config['kernel_initializer'] = keras.initializers.deserialize(
                config['kernel_initializer']
            )

        return cls(**config)

