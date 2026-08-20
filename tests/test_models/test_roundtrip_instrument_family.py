"""
The three named round-trip instruments, for EVERY ``models/`` package.

Rules R-063 (round trip on VALUES with ``rtol=0``), R-072 (build parity by
relative weight path + a no-sub-layer-config sibling), R-073 (weight-VALUE
comparison at ``atol=0.0`` before the loaded model's first call) and R-135 (all
three present, per package) were charged against essentially the whole of
``models/`` -- ~290 cells, the single largest block the audit left open, and
R-072 was measured absent in **73 of 73** packages.

They are one family and they are instrumented once. The assertions live in
``roundtrip_instrument_oracle.py``; the per-package build/input pairs live in
``precision_arm_subjects.py`` beside the precision, XLA and float64 arms; this
file is the parameterization that joins them plus the tables of what was
MEASURED per package.

What the family found, rather than merely instrumented
------------------------------------------------------
* **The substance of R-063/R-073 passes everywhere.** 72 subjects, weight-value
  delta across a save/load cycle **exactly ``0.000000e+00``** in every one. The
  charge was that the comparison was never MADE, and that was correct.
* **``depth_anything`` has 46 of 114 weights on a colliding relative path** --
  its frozen teacher is a ``clone_model`` of the student and ``clone_model``
  bakes the STUDENT's name into the clone. Pinned below rather than repaired:
  the name is fixed at clone time, and every alternative (rebuilding the clone
  from a renamed config) changes how the teacher is constructed.
* **17 packages built sub-layers with no explicit ``name=``**, so two instances
  of the same builder disagreed by weight path and R-072 could not be asserted
  at all. Five were repaired at three shared sites (D-081); twelve are pinned
  by name below.
* Two readings were the INSTRUMENT and are recorded in the oracle: a
  ``split("/", 1)[-1]`` "relative" path that pairs nothing after a reload, and
  a ``yolo12`` weight snapshot taken before a ``training=True`` forward, which
  read a 3.56e+00 delta against a true 0.0.
"""

import os

import pytest

from .precision_arm_subjects import (
    NO_HEAD_SIBLINGS,
    ROUNDTRIP_NA,
    roundtrip_charged,
    roundtrip_subject,
)
from .roundtrip_instrument_oracle import (
    assert_build_parity,
    assert_disabled_component_has_no_weights,
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_build_parity,
    measure_roundtrip,
)


#: Packages whose weight PATHS drift across a save/load cycle, and by how many
#: weights. Auto-generated layer names re-number in the loading process, so the
#: donor and the reloaded model agree on count, shapes and order but not on
#: names. Pinned EXACTLY: a package that grows new drift fires, and one that
#: repairs its drift fires too.
RELOAD_PATH_DRIFT = {
    "SAM": 82,
    # A NESTED ``keras.Sequential`` / functional sub-model is a second, distinct
    # mechanism: the donor's variables carry the sub-model's name
    # (``tiny_encoder/enc_conv_1/kernel``) and the RELOADED model's do not
    # (``enc_conv_1/kernel``), even though every layer here has an explicit
    # ``name=``. That is Keras' deserialization, not a naming defect, and it is
    # why these three do not shrink to zero when their layers are named.
    "detr": 4,
    "vq_vae": 8,
    "vq_vae_rotation": 8,
    "darkir": 24,
    "fractalnet": 42,
    "latent_gmm_registration": 20,
    "mamba": 9,
    "nam": 16,
    "nano_vlm": 16,
    "nano_vlm_world_model": 204,
    "qwen": 32,
    "tabm": 5,
    "tree_transformer": 14,
}

#: Packages that build a sub-layer with NO explicit ``name=``, and the layer
#: stems that therefore re-number per process. Every drifting path must be
#: covered by one of these stems, and every stem must still match a drifting
#: path -- a repair must DELETE its entry in the same commit.
AUTONAME_STEMS = {
    "SAM": ("image_encoder_vi_t", "mask_decoder", "prompt_encoder",
            "two_way_transformer"),
    "darkir": ("layer_normalization",),
    # `fractal_stage`'s conv blocks embed the auto-name counter in their OWN
    # name (`conv_block_conv` vs `conv_block_3_conv`), so the drifting segment
    # is not a plain `<stem>_<n>` -- the stem below matches the common prefix.
    "fractalnet": ("conv_block",),
    "latent_gmm_registration": ("dense",),
    "mamba": ("mamba_layer",),
    "masked_language_model": ("bert",),
    "nam": ("group_attention", "layer_normalization", "tree_mha"),
    "nano_vlm": ("vision_encoder",),
    "nano_vlm_world_model": ("dense", "vision_encoder"),
    "qwen": ("linear_gating", "swi_gluffn"),
    "tabm": ("n_linear", "scale_ensemble", "tab_m_backbone"),
    "tree_transformer": ("group_attention", "layer_normalization",
                         "tree_mha"),
}

#: Packages where two weights share a relative path, with the MEASURED count.
PATH_COLLISIONS = {
    # `DepthAnything` keeps a frozen teacher built with
    # `keras.models.clone_model(self.encoder)`. `clone_model` reproduces the
    # source's NAME, so student and teacher are both `encoder_vit_s` and 46 of
    # the 114 weights cannot be addressed by path. The `.keras` archive is
    # positional and round-trips at 0.0 either way; what is lost is
    # `load_weights(by_name)` and any by-name checkpoint.
    "depth_anything": 46,
}

#: Packages whose forward is NOT deterministic, with the bound the output arm
#: is held to. ``"calibrate"`` means the bound is the model's OWN call-to-call
#: spread (asserted non-zero, so it cannot be switched on for a deterministic
#: model), times ``CALIBRATION_FACTOR``.
STOCHASTIC_OUTPUT = {
    # Samples z at inference: the SAME model called twice differs by more than
    # any edit does (measured 7.45e-01 self against a 1.15e+00 round trip).
    "vae": "calibrate",
    "sd3_mmdit": "calibrate",
    "relgt": "calibrate",
    "nano_vlm_world_model": "calibrate",
    # Not stochastic in the same sense: its own two calls agree to 9.31e-10, a
    # float32 reduction-order difference, so a fixed small bound is honest and
    # `calibrate` would be flaky at exactly 0.0.
    "vq_vae": 1e-8,
}

#: The measured round-trip/self-spread ratios are 1.54 (``vae``), 1.21
#: (``nano_vlm_world_model``), 0.88 (``sd3_mmdit``) and 0.65 (``relgt``); 3.0 is
#: ~2x the worst. The WEIGHT arm stays at ``atol=0.0`` for all four, so the
#: strict guard against a lost weight is not the one being relaxed here.
CALIBRATION_FACTOR = 3.0

#: Packages whose ``.build(shape)`` half of the parity arm is skipped, and why.
EXPLICIT_BUILD_SKIP = {
    "SAM": ("its inputs are a dict of ragged shapes; a single input_shape "
            "cannot be derived from them"),
}

#: Marker substrings for the R-072(b) sibling arm -- the head weights that must
#: be ABSENT when ``include_top=False``. Each is asserted to match at least one
#: weight in the enabled instance, so a typo cannot make the arm vacuous.
NO_HEAD_MARKERS = {
    "cbam": "classifier/",
    "convnext": "classifier/",
    "coshnet": "classifier/",
    "dino": "classifier/",
    "fastvlm": "classification_head/",
    "fractalnet": "classifier/",
    "mobilenet": "classifier/",
    "resnet": "classifier/",
    "squeezenet": "conv10/",
    "swin_transformer": "classifier/",
    "vit": "head/",
    "vit_hmlp": "head/",
    "vit_siglip": "head/",
}


_ROUNDTRIP_CACHE = {}


def _report(name):
    """One :func:`measure_roundtrip` per package, shared by the R-063 and R-073
    arms. They are two sides of ONE save/load cycle; running it twice doubles a
    72-package family's cost and lets the halves disagree about what they saw."""
    if name not in _ROUNDTRIP_CACHE:
        build, make_inputs, kwargs = roundtrip_subject(name)
        _ROUNDTRIP_CACHE[name] = measure_roundtrip(build, make_inputs, **kwargs)
    return _ROUNDTRIP_CACHE[name]


def _input_shape(make_inputs):
    """A ``build()`` shape derived from the subject's own inputs."""
    import numpy as np

    def one(value):
        return (None,) + np.asarray(value).shape[1:]

    sample = make_inputs()
    if isinstance(sample, dict):
        return {key: one(value) for key, value in sample.items()}
    if isinstance(sample, (list, tuple)):
        return [one(value) for value in sample]
    return one(sample)


# ===========================================================================
# Coverage -- R-135
# ===========================================================================

def test_every_model_package_has_a_roundtrip_subject():
    """R-135 is a COVERAGE row: all three instruments, per package.

    The subject registry is asserted set-equal to the packages on disk, so a
    new ``models/`` package joins the family the day it lands and an existing
    one cannot leave it by having its ``_sub`` line deleted.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "..",
                        "src", "dl_techniques", "models")
    on_disk = {entry for entry in os.listdir(root)
               if os.path.isfile(os.path.join(root, entry, "__init__.py"))}
    covered = set(roundtrip_charged()) | set(ROUNDTRIP_NA)
    assert sorted(on_disk - covered) == [], (
        f"models/ packages with no round-trip subject and no stated reason: "
        f"{sorted(on_disk - covered)}")
    assert sorted(covered - on_disk) == [], (
        f"subjects naming a package that does not exist: "
        f"{sorted(covered - on_disk)}")
    assert len(on_disk) == 73, f"expected 73 packages, found {len(on_disk)}"


def test_every_waiver_table_names_only_real_subjects():
    """No table may carry an entry for a package that left the family."""
    known = set(roundtrip_charged())
    for label, table in (("RELOAD_PATH_DRIFT", RELOAD_PATH_DRIFT),
                         ("AUTONAME_STEMS", AUTONAME_STEMS),
                         ("PATH_COLLISIONS", PATH_COLLISIONS),
                         ("STOCHASTIC_OUTPUT", STOCHASTIC_OUTPUT),
                         ("EXPLICIT_BUILD_SKIP", EXPLICIT_BUILD_SKIP),
                         ("NO_HEAD_MARKERS", NO_HEAD_MARKERS)):
        unknown = sorted(set(table) - known)
        assert not unknown, f"{label} names unknown subjects: {unknown}"


def test_the_no_head_sibling_set_is_the_include_top_set():
    """R-072(b)'s subject set is DERIVED, not chosen.

    Every model class that accepts ``include_top`` must have a sibling builder,
    and every sibling builder must have a marker. A package that gains the knob
    therefore joins the arm without anyone remembering to add it.
    """
    import inspect

    import keras

    with_knob = set()
    for name in roundtrip_charged():
        build, _make_inputs, _kwargs = roundtrip_subject(name)
        keras.utils.set_random_seed(0)
        model = build()
        if "include_top" in inspect.signature(type(model).__init__).parameters:
            with_knob.add(name)

    assert with_knob == set(NO_HEAD_SIBLINGS), (
        f"include_top classes without a sibling: "
        f"{sorted(with_knob - set(NO_HEAD_SIBLINGS))}; siblings for classes "
        f"without the knob: {sorted(set(NO_HEAD_SIBLINGS) - with_knob)}")
    assert set(NO_HEAD_MARKERS) == set(NO_HEAD_SIBLINGS), (
        "every sibling needs a marker")


# ===========================================================================
# R-063 -- the round trip compares output VALUES, rtol=0
# ===========================================================================

@pytest.mark.parametrize("name", roundtrip_charged())
def test_the_roundtrip_compares_output_values(name):
    report = _report(name)
    bound = STOCHASTIC_OUTPUT.get(name)
    if bound == "calibrate":
        assert_roundtrip_output_values(
            report,
            atol=CALIBRATION_FACTOR * report["self_max_delta"],
            calibrate=True)
    elif bound is not None:
        assert_roundtrip_output_values(report, atol=bound)
    else:
        # The default, and the measured state of 67 of the 72 packages: the
        # forward is deterministic and the round trip is EXACT.
        assert report["self_max_delta"] == 0.0, (
            f"{name} is not deterministic (self spread "
            f"{report['self_max_delta']:.6e}) but is not declared in "
            "STOCHASTIC_OUTPUT")
        assert_roundtrip_output_values(report, atol=0.0)


# ===========================================================================
# R-073 -- weight VALUES at atol=0.0, before the loaded model's first call
# ===========================================================================

@pytest.mark.parametrize("name", roundtrip_charged())
def test_the_weights_are_restored_before_the_first_call(name):
    report = _report(name)
    assert_weights_restored_before_first_call(
        report,
        atol=0.0,
        allow_positional_match=name in RELOAD_PATH_DRIFT,
        expect_path_collisions=PATH_COLLISIONS.get(name, 0),
    )
    expected_drift = RELOAD_PATH_DRIFT.get(name, 0)
    assert report["n_path_mismatch"] == expected_drift, (
        f"{name} weight-path drift across the reload is "
        f"{report['n_path_mismatch']}, pinned at {expected_drift}")


# ===========================================================================
# R-072(a) -- build parity by relative weight path
# ===========================================================================

@pytest.mark.parametrize("name", roundtrip_charged())
def test_the_builder_has_build_parity(name):
    build, make_inputs, kwargs = roundtrip_subject(name)
    input_shape = (None if name in EXPLICIT_BUILD_SKIP
                   else _input_shape(make_inputs))
    report = measure_build_parity(build, make_inputs,
                                  input_shape=input_shape, **kwargs)
    assert_build_parity(report,
                        autoname_stems=AUTONAME_STEMS.get(name, ()),
                        expect_path_collisions=PATH_COLLISIONS.get(name, 0))


# ===========================================================================
# R-072(b) -- the no-sub-layer-config sibling
# ===========================================================================

@pytest.mark.parametrize("name", sorted(NO_HEAD_SIBLINGS))
def test_a_disabled_head_builds_no_head_weights(name):
    build, make_inputs, kwargs = roundtrip_subject(name)
    assert_disabled_component_has_no_weights(
        build, NO_HEAD_SIBLINGS[name], make_inputs,
        marker=NO_HEAD_MARKERS[name], **kwargs)
