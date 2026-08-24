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


# ---------------------------------------------------------------------------
# Step 19.1 (D-082): the last five of the eleven genuine omissions
# ---------------------------------------------------------------------------
#
# Four `dino` classes and `KAN`. `KAN` already reported `name` by hand and was
# missing only `trainable` -- which is the key with the measured consequence.


def _dino_v1():
    from dl_techniques.models.dino.dino_v1 import DINOv1
    return DINOv1(embed_dim=16, depth=1, num_heads=2, patch_size=14,
                  image_size=28, num_classes=4)


def _dino_v2_backbone():
    from dl_techniques.models.dino.dino_v2 import DINOv2VisionTransformer
    return DINOv2VisionTransformer(image_size=28, patch_size=14, embed_dim=16,
                                   depth=1, num_heads=2, num_register_tokens=2)


def _dino_v2():
    from dl_techniques.models.dino.dino_v2 import DINOv2
    return DINOv2(image_size=28, patch_size=14, num_classes=4,
                  embed_dim=16, depth=1, num_heads=2)


def _dino_v3():
    from dl_techniques.models.dino.dino_v3 import DINOv3
    return DINOv3(image_size=28, patch_size=14, num_classes=4, embed_dim=16,
                  depth=1, num_heads=2)


def _kan():
    from dl_techniques.models.kan.model import KAN
    return KAN(layer_configs=[{"features": 8, "grid_size": 5,
                               "activation": "swish"},
                              {"features": 4, "grid_size": 4,
                               "activation": "linear"}],
               input_features=6)


#: ``name`` is asserted for every subject; ``trainable`` only where the class
#: exposes it as a constructor keyword (all six do, via ``**kwargs``).
BUILDERS = {
    "coshnet": _coshnet,
    "fractalnet": _fractalnet,
    "relgt": _relgt,
    "vae": _vae,
    "swin_transformer": _swin,
    "fastvlm": _fastvlm,
    "dino_v1": _dino_v1,
    "dino_v2": _dino_v2,
    "dino_v2_backbone": _dino_v2_backbone,
    "dino_v3": _dino_v3,
    "kan": _kan,
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


# ---------------------------------------------------------------------------
# The dtype arm, and the half of it that is REFUTED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", sorted(BUILDERS))
def test_the_dtype_policy_key_is_reported(package):
    """``dtype`` is the third base key, and it needs a different assertion.

    MEASURED and recorded in decisions.md: under the DEFAULT policy the
    consequence claimed for ``dtype`` is REFUTED -- a model reconstructed from a
    config with no ``dtype`` key still reports ``float32`` for ``model.dtype``,
    so a ``.dtype`` assertion passes against the defect and proves nothing. The
    observable that does move is the POLICY, so this arm asserts
    ``dtype_policy.name`` and the presence of the key, never ``.dtype``.
    """
    keras.utils.set_random_seed(0)
    model = BUILDERS[package]()
    config = model.get_config()
    assert "dtype" in config, (
        f"{package}.get_config() omits the dtype key")

    keras.utils.set_random_seed(0)
    restored = type(model).from_config(dict(config))
    assert restored.dtype_policy.name == model.dtype_policy.name, (
        f"{package}: dtype policy {model.dtype_policy.name!r} -> "
        f"{restored.dtype_policy.name!r} across the round trip")


#: The ONE subject of the eleven whose policy really is restored from the
#: ``dtype`` key. See the test below: this is a per-class property, not a Keras
#: guarantee, and asserting it for the other ten would convict them of a
#: behaviour they never had.
DTYPE_KEY_RESTORES_THE_POLICY = ("relgt",)


@pytest.mark.parametrize("package", sorted(BUILDERS))
def test_whether_the_dtype_key_restores_the_policy_is_a_per_class_property(package):
    """CLOSED-as-refuted, with the measurement, for ten of the eleven.

    The obvious assertion -- "a config carrying ``dtype='mixed_float16'``
    restores a mixed_float16 model" -- was written first and MEASURED FALSE for
    10 of these 11 classes. The reason is structural: all but ``relgt`` build
    their sub-layers (or their whole functional graph) BEFORE the ``dtype``
    kwarg ever reaches ``keras.Model.__init__``, so the layers took the GLOBAL
    policy at construction time and the key changes nothing.

    That is why the audit's ``dtype`` half of R-057 is not a defect these five
    ``get_config`` fixes could have repaired, and why the arm above asserts
    ``dtype_policy.name`` rather than ``model.dtype`` -- ``.dtype`` reports
    ``float32`` for a correct AND for a broken model under the default policy,
    so a ``.dtype`` assertion passes against the defect.
    """
    keras.utils.set_random_seed(0)
    model = BUILDERS[package]()
    config = dict(model.get_config())
    config["dtype"] = "mixed_float16"

    keras.utils.set_random_seed(0)
    restored = type(model).from_config(dict(config))
    restores = restored.dtype_policy.name == "mixed_float16"
    assert restores == (package in DTYPE_KEY_RESTORES_THE_POLICY), (
        f"{package}: the dtype key "
        f"{'now restores' if restores else 'does not restore'} the policy, "
        f"against the measured {package in DTYPE_KEY_RESTORES_THE_POLICY}")
