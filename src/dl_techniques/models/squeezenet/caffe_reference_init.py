"""Kernel initializers transcribed from the official SqueezeNet Caffe prototxts.

Both SqueezeNet families in this package are ports of Caffe definitions, and a
Caffe layer's ``weight_filler`` is part of the published recipe, not a framework
default the porter gets to inherit. Omitting it hands Keras' own
``glorot_uniform``, which is a DIFFERENT distribution -- measured on the v1.0
stem kernel ``(7, 7, 3, 96)``, ``fan_in = 147``: the reference draws uniformly on
``+-0.1429`` and ``glorot_uniform`` on ``+-0.0352``, a 4x too-narrow stem.

Fetched 2026-08-23 from the official DeepScale/`forresti` repository. Both
prototxts have the SAME shape: **25** convolutions carry
``weight_filler { type: "xavier" }`` (``conv1`` and all 24 fire-module squeeze /
expand convolutions) and exactly **one** carries
``weight_filler { type: "gaussian" mean: 0.0 std: 0.01 }`` -- ``conv10``, the
classification head.

    https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/train_val.prototxt
    https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt

CAFFE'S "xavier" IS NOT KERAS' "glorot_uniform". This is the trap in the
translation and the reason this module exists rather than a bare string at each
site. Caffe's ``XavierFiller`` normalizes by ``fan_in`` **by default**
(``FillerParameter.variance_norm`` defaults to ``FAN_IN``; neither prototxt
overrides it), giving ``U(-sqrt(3/fan_in), +sqrt(3/fan_in))``:

    https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp  (XavierFiller)
    https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto  (variance_norm)

Keras' ``glorot_uniform`` normalizes by ``(fan_in + fan_out) / 2``. The Keras
initializer that reproduces Caffe's default xavier is ``lecun_uniform``
(``VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')``), whose
limit is ``sqrt(3 * scale / fan_in)``. Verified by sampling, 2026-08-23:

    kernel shape        fan_in   sqrt(3/fan_in)   max|lecun_uniform|   max|glorot_uniform|
    (7, 7, 3, 96)          147         0.142857             0.142845              0.035166
    (1, 1, 96, 16)          96         0.176777             0.176532              0.231060
    (3, 3, 16, 64)         144         0.144338             0.144326              0.091283

Note the second row: ``glorot_uniform`` is not uniformly narrower, it is simply a
different distribution. "It is roughly the same" is not a defence.

Interface contract
------------------
``CAFFE_XAVIER_INITIALIZER`` : ``str``
    The Keras identifier reproducing Caffe's default (FAN_IN) ``xavier`` filler.
    Use for the stem convolution and every fire-module convolution. A string, so
    it round-trips through ``keras.initializers.serialize`` unchanged and every
    layer resolves its own fresh instance -- there is no shared-instance replay
    hazard of the kind D-072 measured in ``clip/clifford_clip.py``.

``CAFFE_HEAD_INITIALIZER`` : ``Dict[str, Any]``
    The ``conv10`` filler, ``N(0, 0.01)``, as a serialized initializer CONFIG
    rather than an ``Initializer`` instance. Deliberate on two counts: Keras
    resolves a config to a FRESH instance at every consumer, so two models built
    in one process cannot share one seedless object and replay its draw (the
    hazard D-072 measured in ``clip/clifford_clip.py``); and plain data survives
    being read off the class, reassigned and read back, which a bare function
    stored as a class attribute does not (it silently becomes a bound method and
    receives ``self``).

``CAFFE_HEAD_STDDEV`` : ``float``
    The published ``std``, exposed so a guard can assert the shipped value
    against the prototxt without reaching into an initializer's config.

These are TRAINING-ONLY: an initializer supplies a fresh model's starting weights
and is never consulted again. Loading a checkpoint (``.keras`` or
``load_weights``) overwrites every kernel it names, so no saved artifact changes
value because of this module.
"""

from typing import Any, Dict

import keras

# Caffe `weight_filler { type: "xavier" }` with the default variance_norm=FAN_IN.
# NOT `glorot_uniform` -- see the module docstring's measured table before
# "simplifying" this to the Keras default.
CAFFE_XAVIER_INITIALIZER: str = "lecun_uniform"

# Caffe `weight_filler { type: "gaussian" mean: 0.0 std: 0.01 }` on `conv10`.
CAFFE_HEAD_MEAN: float = 0.0
CAFFE_HEAD_STDDEV: float = 0.01


CAFFE_HEAD_INITIALIZER: Dict[str, Any] = keras.initializers.serialize(
    keras.initializers.RandomNormal(mean=CAFFE_HEAD_MEAN, stddev=CAFFE_HEAD_STDDEV)
)
