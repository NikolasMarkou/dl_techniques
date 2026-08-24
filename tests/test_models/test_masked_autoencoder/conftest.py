"""The one encoder builder this package's suites share.

Every ``MaskedAutoencoder`` test needs the same thing: the smallest possible
4D-output encoder whose total stride matches the decoder's ``2 ** len(decoder_dims)``
upsampling factor, because the constructor's scale contract rejects any other
pairing (see ``test_scale_contract.py``). Four modules had grown four private
``_tiny_encoder`` / ``_tiny_conv_encoder`` copies of that shape, differing only in
channel widths and whether the convolutions carried an activation — so a change to
the scale contract had four places to reach and no single definition to correct.

``tiny_encoder`` is that definition. It is a *factory*, not a fixture value: the
callers need different widths (one of them perturbs every weight and wants a
non-uniform tensor set) and different image sizes, and a fixture returning one
built model would force them all onto one shape.

Interface contract
------------------
``tiny_encoder(image_size=32, channels=3, filters=(16, 16, 16, 16), activation=None,
name=None) -> keras.Model``

* ``filters`` — one stride-2 ``Conv2D`` per entry, so ``len(filters)`` IS the
  downsampling exponent and must equal ``len(decoder_dims)`` (default 4, i.e. 16x).
* ``activation`` — passed straight to each ``Conv2D``; ``None`` means linear.
* Returns an unbuilt-weights-but-shape-known functional ``keras.Model`` mapping
  ``(None, image_size, image_size, channels)`` to
  ``(None, image_size >> len(filters), image_size >> len(filters), filters[-1])``.
* Raises nothing of its own; an ``image_size`` not divisible by ``2 ** len(filters)``
  produces a smaller-than-expected output rather than an error, which the scale
  contract then rejects downstream.
"""

from typing import Optional, Sequence

import keras
import pytest


def tiny_encoder(
    image_size: int = 32,
    channels: int = 3,
    filters: Sequence[int] = (16, 16, 16, 16),
    activation: Optional[str] = None,
    name: Optional[str] = None,
) -> keras.Model:
    """Build the smallest encoder satisfying MAE's scale contract. See module docstring."""
    inp = keras.Input(shape=(image_size, image_size, channels))
    x = inp
    for width in filters:
        x = keras.layers.Conv2D(
            width, 3, strides=2, padding="same", activation=activation
        )(x)
    return keras.Model(inp, x, name=name)


@pytest.fixture(scope="session")
def tiny_encoder_factory():
    """The factory above, as a fixture, for tests that prefer injection."""
    return tiny_encoder
