"""
R-057: six ``get_config`` implementations that dropped ``super().get_config()``.

Subject set (plan ``plan-2026-08-19T163559-499b6f0e``, step 18.1, rule R-057):
``coshnet``, ``fractalnet``, ``relgt``, ``vae``, ``swin_transformer``,
``fastvlm``. Each returned a bare literal dict, so every key the Keras base
class contributes -- ``name``, ``trainable``, ``dtype`` -- was silently dropped
and replaced by its DEFAULT on reload.

Why this is a defect and not bookkeeping
----------------------------------------
Two of the six had the consequence measured before the fix:

* ``coshnet``   -- ``trainable`` ``False -> True`` across a round trip;
* ``fractalnet`` -- ``trainable`` ``False -> True`` AND
  ``len(trainable_weights)`` **0 -> 18**. A caller who froze a backbone, saved
  it, and reloaded it got an UNFROZEN model back, with no warning of any kind
  and no assertion in the suite that could see it.

``coshnet`` carried a compounding defect: ``model.py`` hard-coded
``name="coshnet"`` in its ``super().__init__`` call, so ``CoShNet(name="x")``
raised ``TypeError: got multiple values for keyword argument 'name'`` -- and
once ``get_config`` started reporting ``name``, the round trip would have
raised it too. See decisions.md D-066.

What is asserted, and why it is not vacuous
-------------------------------------------
The property under test is ``config`` COMPLETENESS, so the test asserts on the
config dict AND on a reconstructed model. The ``trainable`` arm is the one that
carries the measured consequence, and it is exercised with the NON-default
value ``False`` -- a test that only ever used the default ``True`` would pass
against the very defect it names, since the default is what the broken code
restored.
"""

import keras
import pytest


def _coshnet():
    from dl_techniques.models.coshnet.model import CoShNet
    return CoShNet(num_classes=4, input_shape=(32, 32, 3),
                   conv_filters=[4, 8], dense_units=[8, 4])


def _fractalnet():
    from dl_techniques.models.fractalnet.model import FractalNet
    return FractalNet(num_classes=4, depths=[1], filters=[8], strides=[1],
                      input_shape=(32, 32, 3))


def _relgt():
    from dl_techniques.models.relgt.model import RELGT
    return RELGT(output_dim=2, embedding_dim=16, num_heads=2,
                 num_global_centroids=4, ffn_dim=32,
                 num_transformer_blocks=1, dropout_rate=0.0)


def _vae():
    from dl_techniques.models.vae.model import VAE
    return VAE(latent_dim=4, input_shape=(32, 32, 1), depths=1,
               steps_per_depth=1, filters=[8])


def _swin():
    from dl_techniques.models.swin_transformer.model import SwinTransformer
    # `NUM_STAGES` is 4 and validated, so `depths` / `num_heads` must be 4-long.
    return SwinTransformer(num_classes=4, input_shape=(32, 32, 3),
                           embed_dim=16, depths=[1, 1, 1, 1],
                           num_heads=[2, 2, 2, 2], window_size=2,
                           patch_size=4)


def _fastvlm():
    from dl_techniques.models.fastvlm.model import FastVLM
    return FastVLM(num_classes=4)


#: ``name`` is asserted for every subject; ``trainable`` only where the class
#: exposes it as a constructor keyword (all six do, via ``**kwargs``).
BUILDERS = {
    "coshnet": _coshnet,
    "fractalnet": _fractalnet,
    "relgt": _relgt,
    "vae": _vae,
    "swin_transformer": _swin,
    "fastvlm": _fastvlm,
}

BASE_KEYS = ("name", "trainable")


@pytest.mark.parametrize("package", sorted(BUILDERS))
def test_get_config_reports_the_base_class_keys(package):
    """The RED half: before the fix, none of these keys was present at all."""
    keras.utils.set_random_seed(0)
    model = BUILDERS[package]()
    config = model.get_config()
    missing = [k for k in BASE_KEYS if k not in config]
    assert not missing, (
        f"{package}.get_config() still omits {missing}; it is not calling "
        "super().get_config()"
    )


@pytest.mark.parametrize("package", sorted(BUILDERS))
def test_a_frozen_model_is_still_frozen_after_a_config_round_trip(package):
    """The consequence arm, at the NON-default value.

    ``fractalnet`` measured ``len(trainable_weights)`` 0 -> 18 here before the
    fix -- a frozen backbone silently unfreezing. The weight-count assertion is
    made in addition to the flag, because a class could honour ``trainable``
    in ``__init__`` and still hand its sublayers back unfrozen.
    """
    keras.utils.set_random_seed(0)
    model = BUILDERS[package]()
    model.trainable = False
    frozen_count = len(model.trainable_weights)
    assert frozen_count == 0, (
        f"{package}: setting .trainable = False left {frozen_count} trainable "
        "weights, so this arm cannot observe an unfreeze")

    config = model.get_config()
    assert config["trainable"] is False, (
        f"{package}.get_config() reports trainable={config['trainable']!r} for "
        "a frozen model")

    keras.utils.set_random_seed(0)
    restored = type(model).from_config(config)
    assert restored.trainable is False, (
        f"{package}: trainable flipped False -> {restored.trainable!r} across "
        "the round trip")
    assert len(restored.trainable_weights) == 0, (
        f"{package}: a frozen model came back with "
        f"{len(restored.trainable_weights)} trainable weights")


@pytest.mark.parametrize("package", sorted(BUILDERS))
def test_a_custom_name_survives_a_config_round_trip(package):
    """``coshnet``'s hard-coded ``name="coshnet"`` made this RAISE (D-066)."""
    keras.utils.set_random_seed(0)
    model = BUILDERS[package]()
    config = dict(model.get_config())
    config["name"] = f"custom_{package}_name"
    keras.utils.set_random_seed(0)
    restored = type(model).from_config(config)
    assert restored.name == f"custom_{package}_name", (
        f"{package}: name came back as {restored.name!r}")
