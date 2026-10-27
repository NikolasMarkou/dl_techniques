"""The two convs of a ``ConvStem`` draw their kernels independently.

A seedless initializer INSTANCE replays one sample at every weight it is handed
to, and a shorter draw comes out as the longer draw's prefix
(``initializers/clone.py``). A two-conv stem must therefore not hand ONE
instance to both kernels, or the second kernel is the first kernel's draw.
``stem.py`` carries the anchor that records the rule (search it for "replays
one sample"); this file is its guard, with a planted positive control so the
detector is seen to fire.
"""

import keras
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem

from ...test_models.initializer_replay_oracle import replayed_pairs
from .stem_instruments import B

_INITIALIZERS = {
    "string": lambda: "glorot_uniform",
    "instance": lambda: keras.initializers.GlorotUniform(),
    "he_normal": lambda: keras.initializers.HeNormal(),
}


class TestTheTwoConvsDrawIndependently:
    """The oracle compares the leading values of every pair of kernels, so a
    replay across DIFFERENT shapes is caught too: ``replay_r`` is 0.9999
    against a null standard deviation of ``1 / sqrt(255) = 0.063`` at 256
    values (``initializer_replay_oracle``).
    """

    @pytest.mark.parametrize("seed", (1, 2, 3))
    @pytest.mark.parametrize("which", sorted(_INITIALIZERS))
    def test_conv1_and_conv2_kernels_do_not_replay_one_draw(self, which, seed):
        keras.utils.set_random_seed(seed)
        layer = ConvStem(filters=6, num_convs=2,
                         kernel_initializer=_INITIALIZERS[which]())
        layer.build((B, 8, 8, 4))
        replayed, compared = replayed_pairs(layer)
        # Anti-vacuity: the two kernels (216 and 324 values) were compared.
        assert compared == 1, compared
        assert replayed == [], replayed

    def test_the_detector_sees_a_planted_shared_instance(self):
        """Positive control: ONE seedless instance at the stem's two shapes is caught."""
        keras.utils.set_random_seed(1)
        shared = keras.initializers.get("glorot_uniform")
        inputs = keras.Input((8, 8, 4))
        first = keras.layers.Conv2D(6, 3, kernel_initializer=shared, name="site_1")
        second = keras.layers.Conv2D(6, 3, kernel_initializer=shared, name="site_2")
        model = keras.Model(inputs, second(first(inputs)))
        replayed, compared = replayed_pairs(model)
        assert compared == 1
        assert len(replayed) == 1, replayed
