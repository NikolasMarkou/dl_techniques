"""
R-082: normalization-epsilon PROVENANCE for the ten charged packages.

Keras' ``BatchNormalization`` / ``LayerNormalization`` default is ``1e-3``;
most transformer-family and vision references use ``1e-5`` or ``1e-6``. That is
a 100-1000x spread in a denominator, with no symptom -- no raise, no NaN, no
shape change -- so nothing in a normal test suite can see it. R-082 charges the
PROVENANCE: a package must be able to say why its epsilon is what it is.

The ruling, per package
-----------------------
Ten packages were charged. **None of their VALUES is changed here**, and that
is a decision, not an omission -- see decisions.md D-067. Changing an epsilon
is a behaviour change that no test in this repository can adjudicate, and for
the packages that wrap pretrained weights it is provably WRONG:

===================== ==================================== =================
package               MEASURED census                      ruling
===================== ==================================== =================
``yolo12``            134 sites, all ``1e-03``             CITED (Ultralytics
                                                           port; the epsilon
                                                           agrees with Keras'
                                                           default by
                                                           coincidence)
``mobile_clip`` (v1)  34x ``1e-03`` + 9x ``1e-05``         STATED, NOT
                                                           CHANGED -- the
                                                           ``1e-03`` sites are
                                                           the substitute
                                                           ``keras.applications``
                                                           MobileNetV2's own
                                                           BatchNorms, whose
                                                           checkpoints were
                                                           TRAINED at ``1e-3``
``mobilenet`` (v2)    51x ``1e-06`` + 2x ``1e-03``         STATED -- the split
``vit_hmlp``          25x ``1e-06`` + 3x ``1e-03``         is factory-vs-bare
``fastvlm``           34x ``1e-03`` + 12x ``1e-06``        construction, not a
                                                           per-site choice
``qwen``              9x ``1e-05`` + 8x ``1e-06``          STATED
``accunet``           119 sites, all ``1e-03``             STATED (port;
                                                           reference is
                                                           ``1e-5``, no
                                                           checkpoint exists
                                                           to adjudicate)
``masked_autoencoder`` 4 sites, all ``1e-03``              STATED
``cbam``              2 sites, all ``1e-03``               STATED
``time_series`` (MDN) 4 sites, all ``1e-03``               STATED
===================== ==================================== =================

Three of the audit's own counts were WRONG and are corrected by the
measurement below: ``vit_hmlp`` has **3** ``1e-03`` sites, not 2; ``fastvlm``
has 46 norms rather than 16; ``accunet`` has 119 at this configuration rather
than 103. Site counts are a function of the CONFIGURATION, which is why every
census here names the exact constructor call that produced it.

What this file is for
---------------------
It is a PIN, not a verdict. Once a census is written down, a later edit that
silently moves an epsilon -- in either direction -- fails here and has to be
argued. Its liveness arm proves the census can actually observe a moved
epsilon, so a green result is not the instrument failing to look.
"""

import collections

import keras
import pytest

from ..norm_epsilon_oracle import _epsilon_of


def _census(model) -> collections.Counter:
    """``{"1e-03": n, ...}`` over every normalization layer, at 1 significant digit."""
    counts = collections.Counter()
    for layer in model._flatten_layers(include_self=True):
        eps = _epsilon_of(layer)
        if eps is not None:
            counts[f"{float(eps):.0e}"] += 1
    return counts


def _yolo12():
    from dl_techniques.models.yolo12 import create_yolov12_multitask
    return create_yolov12_multitask(num_detection_classes=4,
                                    tasks=["detection"],
                                    input_shape=(64, 64, 3), scale="n")


def _mobile_clip():
    from dl_techniques.models.mobile_clip import create_mobile_clip_model
    return create_mobile_clip_model("s0")


def _mobilenet():
    from dl_techniques.models.mobilenet import create_mobilenetv2
    return create_mobilenetv2(variant="small", num_classes=4,
                              input_shape=(32, 32, 3))


def _vit_hmlp():
    from dl_techniques.models.vit_hmlp import create_vit_hmlp
    return create_vit_hmlp(input_shape=(32, 32, 3), num_classes=4,
                           scale="tiny", patch_size=16)


def _fastvlm():
    from dl_techniques.models.fastvlm.model import FastVLM
    return FastVLM(num_classes=4)


def _qwen():
    from dl_techniques.models.qwen import Qwen3Next
    return Qwen3Next(vocab_size=64, hidden_size=32, num_layers=1,
                     num_attention_heads=2, num_key_value_heads=1,
                     max_seq_len=32, num_experts=2, num_experts_per_tok=1,
                     moe_intermediate_size=32)


def _accunet():
    from dl_techniques.models.accunet import create_acc_unet
    return create_acc_unet(input_channels=3, num_classes=1, base_filters=8,
                           input_shape=(32, 32))


def _masked_autoencoder():
    from dl_techniques.models.masked_autoencoder import create_mae_model
    inp = keras.Input((32, 32, 3))
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same")(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same")(x)
    encoder = keras.Model(inp, x, name="tiny_encoder")
    return create_mae_model(encoder=encoder, patch_size=16,
                            input_shape=(32, 32, 3), decoder_dims=[16, 16])


def _cbam():
    from dl_techniques.models.cbam import create_cbam_net
    return create_cbam_net("tiny", num_classes=4, input_shape=(32, 32, 3))


def _mdn():
    from dl_techniques.models.time_series import create_mdn_model
    return create_mdn_model(hidden_layers=[8], output_dimension=1,
                            num_mixtures=2, input_dimension=4,
                            use_batch_norm=True)


#: package -> (builder, MEASURED census). Every entry was produced by running
#: the builder beside it; none is copied from the audit, and three of the
#: audit's numbers were wrong (see the module docstring).
CENSUS = {
    "yolo12": (_yolo12, {"1e-03": 134}),
    "mobile_clip": (_mobile_clip, {"1e-03": 34, "1e-05": 9}),
    "mobilenet": (_mobilenet, {"1e-06": 51, "1e-03": 2}),
    "vit_hmlp": (_vit_hmlp, {"1e-06": 25, "1e-03": 3}),
    "fastvlm": (_fastvlm, {"1e-03": 34, "1e-06": 12}),
    "qwen": (_qwen, {"1e-05": 9, "1e-06": 8}),
    "accunet": (_accunet, {"1e-03": 119}),
    "masked_autoencoder": (_masked_autoencoder, {"1e-03": 4}),
    "cbam": (_cbam, {"1e-03": 2}),
    "time_series": (_mdn, {"1e-03": 4}),
}


@pytest.mark.parametrize("package", sorted(CENSUS))
def test_the_norm_epsilon_census_is_what_was_ruled_on(package):
    """The pin. A silently moved epsilon fails here, in either direction."""
    build, expected = CENSUS[package]
    keras.utils.set_random_seed(0)
    measured = dict(_census(build()))
    assert measured == expected, (
        f"{package}: norm-epsilon census moved from {expected} to {measured}. "
        "This is not a test to update -- state the provenance of the new "
        "value in decisions.md (R-082, D-067) first."
    )


def test_the_census_can_actually_observe_a_moved_epsilon():
    """Liveness. Without this arm a census of ``{}`` would pass every case.

    ``Qwen3Next`` is used because it is the ONE charged package with a real
    ``norm_eps`` knob; the others hard-code their values, which is what R-082
    is about. If ``norm_eps`` ever stops reaching the norms, this fails.
    """
    from dl_techniques.models.qwen import Qwen3Next

    def at(eps):
        keras.utils.set_random_seed(0)
        return dict(_census(Qwen3Next(
            vocab_size=64, hidden_size=32, num_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_seq_len=32,
            num_experts=2, num_experts_per_tok=1, moe_intermediate_size=32,
            norm_eps=eps)))

    low, high = at(1e-6), at(1e-2)
    assert low != high, (
        f"the census reported {low} at norm_eps=1e-6 and the same at 1e-2 -- "
        "it cannot see an epsilon change, so every assertion above is vacuous"
    )
    assert "1e-02" in high, high


def test_the_two_epsilon_split_packages_really_carry_two_values():
    """The 'two epsilons three orders apart in one model' shape, asserted.

    ``mobilenet``, ``vit_hmlp``, ``fastvlm`` and ``mobile_clip`` each split
    along factory-vs-bare construction. A future refactor that unified them
    would be a real change and must not pass silently as "still green".
    """
    for package in ("mobilenet", "vit_hmlp", "fastvlm", "mobile_clip"):
        _build, expected = CENSUS[package]
        assert len(expected) == 2, (
            f"{package} is recorded as a two-epsilon package but its census "
            f"has {len(expected)} distinct values: {expected}")
        values = sorted(float(k) for k in expected)
        assert values[-1] / values[0] >= 100.0, (
            f"{package}: the two epsilons are {values}, less than the 100x "
            "spread this row was charged for")
