"""FreMLP's backward pass is finite when a frequency bin is exactly zero.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-097). Measured at 561e02165:
``train_darkir.py`` on its synthetic low-light CIFAR-10 data (32x32, batch 64,
the trainer's AdamW schedule) logged a NaN training loss from step ~80 of the
first epoch and finished with ``loss = nan`` in both epochs of
``training_log.csv``. Reproduced outside the trainer with the Charbonnier term
alone (step 39), with XLA off (step 14), and with the original grouped
``Conv2D`` branch (step 118): the loss terms and D-096 are not the cause.

The cause is FreMLP's magnitude, ``sqrt(re**2 + im**2)``: at a bin whose power
is exactly zero the derivative of ``sqrt`` is infinite and the chain rule
multiplies it by ``2 * re = 0``, giving NaN. A spatially constant channel has an
exactly zero spectrum everywhere but DC, so any such channel (an all-zero input,
or a feature map a gate has flattened) turns the input gradient, and through it
every weight upstream, into NaN. The forward pass was always finite (the phasor
divide is guarded), which is why no forward test saw it.

A model-level probe (an all-zero image through a small DarkIR) passed before the
fix as well, so it is not kept as a guard; the trainer run in the step record
is the model-level evidence. The forward semantics (a magnitude-only edit that
keeps the phase) stay pinned by ``test_dilation_and_fremlp.py``.
"""

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vision.image_restoration.darkir.model import FreMLP


def _constant_channel_input() -> tf.Tensor:
    x = np.random.default_rng(0).random((2, 8, 8, 4)).astype("float32")
    x[..., 1] = 0.3  # spatially constant: every non-DC bin is exactly 0
    return tf.constant(x)


def test_the_input_gradient_is_finite_with_a_constant_channel():
    keras.utils.set_random_seed(0)
    layer = FreMLP(4)
    x = _constant_channel_input()
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(layer(x) ** 2)
    grads = tape.gradient(loss, [x] + layer.trainable_weights)
    for grad in grads:
        assert np.isfinite(np.asarray(grad)).all()
    # Liveness: the constant channel still reaches the loss.
    assert np.abs(np.asarray(grads[0])[..., 1]).max() > 0.0
