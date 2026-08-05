"""
Tests for ``models/sam3/sam3_image.py`` -- the top-level ``Sam3Image`` assembly.

What this file guards, and why each guard is shaped the way it is:

- **The fusion**, against a float64 NumPy oracle probed at three points, one of
  which is chosen so a plain multiply-in-logit-space candidate differs
  MEASURABLY. At saturated presence the correct expression collapses to the
  identity, which is a coincidence family the probe set must avoid.
- **The eps ceiling.** ``inverse_sigmoid``'s ``eps=1e-3`` bounds its output to
  +-6.9078, so the reference's +-10 clamp is a provable no-op. Both facts are
  asserted, so a future edit to either literal fires.
- **Serialization by VALUE, never by count.** A nested sub-layer store restores
  freshly initialized kernels while the weight count, every weight path and the
  parameter total match (measured in this package on ``necks.py``). Only an
  output comparison sees it.
- **The parameter audit**, per component, EXACT, with a companion test that
  EXECUTES the non-vacuity claim rather than asserting it.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.sam3 import Sam3Image
from dl_techniques.models.sam3.sam3_image import COMPONENT_KEYS

# ---------------------------------------------------------------------
# measured constants
# ---------------------------------------------------------------------

#: `Sam3TransformerDecoder._inverse_sigmoid`'s eps, and the OUTPUT bound it
#: implies: `log(1/eps - 1)` is not the bound -- `log(1 / eps)` is, because both
#: the numerator and the denominator are clamped independently.
INVERSE_SIGMOID_EPS = 1e-3
INVERSE_SIGMOID_CEILING = float(np.log(1.0 / INVERSE_SIGMOID_EPS))

#: GPU float32 under TF32 moves an identical float64 comparison by three orders
#: relative to CPU (measured twice in this package: ~1,000x at step 5 and
#: ~1,775x at step 8). The tolerance is set from the SLOWER regime and the
#: wrong-candidate margins are pinned separately so it cannot swallow a defect.
ORACLE_TOL = 5e-3


# ---------------------------------------------------------------------
# float64 oracles -- written from the reference expression, never from the
# implementation
# ---------------------------------------------------------------------


def np_inverse_sigmoid(x, eps=INVERSE_SIGMOID_EPS):
    """The reference's numerically guarded logit, in float64."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.log(np.maximum(x, eps) / np.maximum(1.0 - x, eps))


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def np_fusion_correct(class_logit, presence_logit, clamp=10.0,
                      eps=INVERSE_SIGMOID_EPS):
    """The CORRECT candidate: multiply probabilities, then re-logit."""
    product = np_sigmoid(class_logit) * np_sigmoid(presence_logit)
    return np.clip(np_inverse_sigmoid(product, eps), -clamp, clamp)


def np_fusion_logit_multiply(class_logit, presence_logit, clamp=10.0):
    """The WRONG candidate M9.1: multiply the logits directly."""
    return np.clip(np.asarray(class_logit, dtype=np.float64)
                   * np.asarray(presence_logit, dtype=np.float64),
                   -clamp, clamp)


# ---------------------------------------------------------------------
# closed-form parameter counts -- derived from STRUCTURE, then validated
# against real instantiations at two scales
# ---------------------------------------------------------------------


def _dense(fan_in, units, bias=True):
    return fan_in * units + (units if bias else 0)


def _norm(width):
    return 2 * width


def backbone_params(table):
    """Closed-form parameter count of ``Sam3ViTDetBackbone``."""
    dim, patch = table["embed_dim"], table["patch_size"]
    hidden = int(dim * table["mlp_ratio"])
    pretrain_grid = table["pretrain_img_size"] // patch
    total = patch * patch * 3 * dim                       # patch-embed conv
    total += (pretrain_grid ** 2 + 1) * dim               # abs pos + cls row
    total += _norm(dim)                                   # ln_pre
    block = (_norm(dim) + _dense(dim, 3 * dim) + _dense(dim, dim)
             + _norm(dim) + _dense(dim, hidden) + _dense(hidden, dim))
    return total + table["depth"] * block


def neck_params(table):
    """Closed-form parameter count of ``Sam3DualViTDetNeck``."""
    dim, model = table["embed_dim"], table["d_model"]
    total = 0
    for scale in table["scale_factors"]:
        width = dim
        if scale == 4.0:
            total += _dense(4 * dim, dim // 2) + _dense(4 * (dim // 2),
                                                        dim // 4)
            width = dim // 4
        elif scale == 2.0:
            total += _dense(4 * dim, dim // 2)
            width = dim // 2
        total += _dense(width, model) + _dense(9 * model, model)
    return total * (2 if table["add_sam2_neck"] else 1)


def text_encoder_params(table):
    """Closed-form parameter count of ``Sam3TextEncoder``."""
    width, model = table["text_width"], table["d_model"]
    hidden = int(width * 4.0)
    total = table["vocab_size"] * width + table["context_length"] * width
    total += _norm(width)                                  # embed norm
    block = (_norm(width) + _dense(width, 3 * width) + _dense(width, width)
             + _norm(width) + _dense(width, hidden) + _dense(hidden, width))
    total += table["text_depth"] * block
    total += _norm(width)                                  # ln_final
    return total + _dense(width, model)                    # resizer


def transformer_params(table):
    """Closed-form parameter count of ``Sam3TransformerDecoder``."""
    dim, heads = table["d_model"], table["decoder_heads"]
    total = table["num_queries"] * dim + table["num_queries"] * 4 + dim
    layer = 4 * _dense(dim, dim) + _norm(dim)                     # self attn
    layer += (_dense(dim, dim) + _dense(dim, 2 * dim) + _dense(dim, dim)
              + _norm(dim))                                       # text cross
    layer += 4 * _dense(dim, dim) + _norm(dim)                    # image cross
    layer += _dense(dim, table["dim_feedforward"]) + _dense(
        table["dim_feedforward"], dim) + _norm(dim)               # ffn
    total += table["decoder_layers"] * layer
    total += _norm(dim)                                           # final norm
    total += 2 * _dense(dim, dim) + _dense(dim, 4)                # bbox_embed
    total += _dense(2 * dim, dim) + _dense(dim, dim)              # ref points
    total += 2 * (_dense(2, dim) + _dense(dim, heads))            # boxRPB x, y
    total += _norm(dim) + 2 * _dense(dim, dim) + _dense(dim, 1)   # presence
    return total


def scoring_params(table):
    """Closed-form parameter count of ``Sam3DotProductScoring``."""
    dim, hidden = table["d_model"], table["prompt_mlp_hidden_dim"]
    return (_dense(dim, hidden) + _dense(hidden, dim) + _norm(dim)
            + 2 * _dense(dim, table["d_proj"]))


def segmentation_params(table, stages):
    """Closed-form parameter count of ``Sam3SegmentationHead``."""
    dim = table["d_model"]
    total = _norm(dim) + _dense(dim, dim) + _dense(dim, 2 * dim) + _dense(
        dim, dim)                                          # prompt cross-attend
    total += stages * (_dense(9 * dim, dim) + _norm(dim))  # conv3x3 + groupnorm
    total += _dense(dim, 1) + _dense(dim, dim)             # semantic + instance
    return total + 2 * _dense(dim, dim) + _dense(dim, dim)  # mask_embed MLP


CLOSED_FORMS = {
    "backbone": backbone_params,
    "neck": neck_params,
    "text_encoder": text_encoder_params,
    "transformer": transformer_params,
    "dot_prod_scoring": scoring_params,
}


def component_closed_forms(table):
    """Every component's closed-form count for one variant table."""
    stages = len(table["scale_factors"]) - table["scalp"] - 1
    counts = {key: fn(table) for key, fn in CLOSED_FORMS.items()}
    counts["segmentation_head"] = segmentation_params(table, stages)
    return counts


#: The SHIPPED variant's six component counts.
#:
#: PROVENANCE, stated because it was previously implied (reviewer W-6). These
#: literals were produced by the closed forms above, so
#: `test_the_shipped_closed_forms_are_self_consistent` compares a closed form to
#: a transcription of itself and is a REGRESSION guard on the arithmetic, not a
#: measurement. The measurement is
#: `test_the_shipped_variant_is_instantiated_and_counted`, which builds the real
#: 821 M-parameter model and is OPT-IN (`SAM3_SHIPPED_AUDIT=1`) because it needs
#: ~3.3 GiB of device memory. Independent cross-checks that DO execute in the
#: default gate: `backbone` matches step 2's instantiated 446,237,696;
#: `text_encoder` matches step 4's 353,202,432; `transformer` matches step 7's
#: 11,575,093; `neck` is exactly half of step 3's dual-neck 15,604,224.
SHIPPED_PARAMS = {
    "backbone": 446_237_696,
    "neck": 7_802_112,
    "text_encoder": 353_202_432,
    "transformer": 11_575_093,
    "dot_prod_scoring": 1_182_976,
    "segmentation_head": 1_708_289,
}
SHIPPED_TOTAL = 821_708_598


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def model():
    built = Sam3Image.from_variant("tiny", supervise_joint_box_scores=True)
    built.build(None)
    return built


@pytest.fixture
def batch():
    rng = np.random.default_rng(7)
    return {
        "image": rng.standard_normal((2, 32, 32, 3)).astype("float32"),
        "token_ids": rng.integers(1, 64, (2, 8)).astype("int32"),
        "token_padding_mask": np.array(
            [[False] * 8, [False] * 5 + [True] * 3]),
    }


def randomize(layer, seed=0):
    """Replace every weight with a non-degenerate draw.

    Keras zero-initializes biases and this package zero-initializes the box
    head's last projection ON PURPOSE, so a fresh model's `pred_boxes` is a
    function of `reference_points` ALONE and is blind to the whole box head.
    A round-trip or liveness assertion on a fresh model would be vacuous there.
    """
    rng = np.random.default_rng(seed)
    for weight in layer.weights:
        weight.assign(rng.standard_normal(weight.shape).astype("float32") * 0.3)


# ---------------------------------------------------------------------
# construction and the output contract
# ---------------------------------------------------------------------


class TestConstruction:

    def test_from_variant_builds_every_component(self, model):
        for key in COMPONENT_KEYS:
            assert isinstance(getattr(model, key), keras.layers.Layer)

    def test_the_class_is_the_ninth_and_last_public_class(self):
        import dl_techniques.models.sam3 as package
        assert "Sam3Image" in package.__all__
        assert len(package.__all__) == 9

    def test_unknown_variant_is_refused(self):
        with pytest.raises(ValueError, match="Unknown SAM 3 variant"):
            Sam3Image.from_variant("huge")

    def test_a_none_override_defers_to_the_table(self):
        # S-3: an explicit `None` must behave exactly like an omitted argument.
        default = Sam3Image.from_variant("tiny")
        deferred = Sam3Image.from_variant("tiny", d_model=None)
        assert deferred.d_model == default.d_model

    def test_an_explicit_override_wins_over_the_table(self):
        overridden = Sam3Image.from_variant("tiny", num_queries=3)
        assert overridden.transformer.num_queries == 3

    def test_a_falsy_override_still_wins(self):
        # The sentinel is `None`, not a truthiness test, so `False` must reach
        # the constructor rather than being read as "defer to the table".
        built = Sam3Image.from_variant("tiny", supervise_joint_box_scores=False)
        assert built.supervise_joint_box_scores is False

    def test_the_tiny_variant_is_declared_not_a_published_size(self):
        # Only ONE variant carries a released geometry, and the table says so in
        # its own source rather than leaving a reader to infer it. Inventing the
        # other published SAM 3 sizes would be fabrication, so they are absent.
        import dl_techniques.models.sam3.sam3_image as module
        assert "NOT a published SAM 3 size" in open(module.__file__).read()
        assert Sam3Image.MODEL_VARIANTS["sam3"]["img_size"] == 1008
        assert set(Sam3Image.MODEL_VARIANTS) == {"sam3", "tiny"}

    def test_a_width_mismatch_is_refused(self):
        from dl_techniques.models.sam3 import Sam3DotProductScoring
        parts = Sam3Image.from_variant("tiny")
        with pytest.raises(ValueError, match="must share one width"):
            Sam3Image(
                backbone=parts.backbone, neck=parts.neck,
                text_encoder=parts.text_encoder, transformer=parts.transformer,
                dot_prod_scoring=Sam3DotProductScoring(d_model=16, d_proj=8,
                                                       prompt_mlp_hidden_dim=8),
                segmentation_head=parts.segmentation_head)

    def test_a_multi_level_memory_is_refused_not_silently_truncated(self):
        parts = Sam3Image.from_variant("tiny")
        with pytest.raises(ValueError, match="num_feature_levels must be 1"):
            Sam3Image(
                backbone=parts.backbone, neck=parts.neck,
                text_encoder=parts.text_encoder, transformer=parts.transformer,
                dot_prod_scoring=parts.dot_prod_scoring,
                segmentation_head=parts.segmentation_head,
                num_feature_levels=2)

    def test_a_stage_count_that_does_not_match_the_pyramid_is_refused(self):
        parts = Sam3Image.from_variant("tiny")
        with pytest.raises(ValueError, match="upsampling_stages"):
            Sam3Image(
                backbone=parts.backbone, neck=parts.neck,
                text_encoder=parts.text_encoder, transformer=parts.transformer,
                dot_prod_scoring=parts.dot_prod_scoring,
                segmentation_head=parts.segmentation_head, scalp=0)

    def test_a_missing_input_key_raises(self, model, batch):
        with pytest.raises(ValueError, match="requires inputs"):
            model({"token_ids": batch["token_ids"]})


# ---------------------------------------------------------------------
# SC-2: the end-to-end forward
# ---------------------------------------------------------------------


class TestSam3ImageForward:

    def test_the_five_declared_outputs_are_present_with_their_shapes(
            self, model, batch):
        out = model(batch, training=False)
        assert set(out) == {"pred_logits", "pred_boxes", "pred_masks",
                            "semantic_seg", "presence_logit"}
        assert tuple(out["pred_logits"].shape) == (2, 5, 1)
        assert tuple(out["pred_boxes"].shape) == (2, 5, 4)
        assert tuple(out["pred_masks"].shape) == (2, 5, 32, 32)
        assert tuple(out["presence_logit"].shape) == (2, 1)

    def test_compute_output_shape_matches_the_forward_pass(self, model, batch):
        out = model(batch, training=False)
        declared = model.compute_output_shape()
        for key, shape in declared.items():
            assert tuple(out[key].shape)[1:] == tuple(shape)[1:], key

    def test_every_output_is_finite(self, model, batch):
        out = model(batch, training=False)
        for key, value in out.items():
            assert np.all(np.isfinite(np.array(value))), key

    def test_boxes_are_normalized_to_the_unit_interval(self, model, batch):
        boxes = np.array(model(batch, training=False)["pred_boxes"])
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0

    def test_the_model_runs_without_a_padding_mask(self, model, batch):
        out = model({k: v for k, v in batch.items()
                     if k != "token_padding_mask"}, training=False)
        assert np.all(np.isfinite(np.array(out["pred_logits"])))

    def test_the_padding_mask_actually_reaches_the_prompt_path(self, model,
                                                               batch):
        # LIVENESS ARM (budgeted before the dead-component probe).
        masked = model(batch, training=False)
        unmasked = model({**batch, "token_padding_mask":
                          np.zeros((2, 8), dtype=bool)}, training=False)
        moved = np.max(np.abs(np.array(masked["pred_logits"])
                              - np.array(unmasked["pred_logits"])))
        assert moved > 1e-4, f"the padding mask changed nothing ({moved})"

    def test_class_logits_vary_across_queries(self, model, batch):
        # LIVENESS ARM: a dead scorer emits one value for every query.
        logits = np.array(model(batch, training=False)["pred_logits"])
        assert np.std(logits, axis=1).min() > 1e-5

    def test_class_logits_respond_to_the_text_prompt(self, model, batch):
        # LIVENESS ARM: the same image with a different prompt must score
        # differently, or the open vocabulary is not open.
        #
        # VACUITY NOTE, MEASURED by this file's dead-component probe: with the
        # fusion ENABLED this arm survives a completely dead scorer, because
        # the fused presence is itself prompt-dependent and resurrects the
        # sensitivity. Its discriminating twin is the `_without_the_fusion`
        # variant below, which observes the scorer directly.
        other = {**batch, "token_ids": (batch["token_ids"] + 11) % 63 + 1}
        moved = np.max(np.abs(
            np.array(model(batch, training=False)["pred_logits"])
            - np.array(model(other, training=False)["pred_logits"])))
        assert moved > 1e-4, f"the prompt changed nothing ({moved})"

    def test_class_logits_respond_to_the_prompt_without_the_fusion(
            self, model, batch):
        # LIVENESS ARM added AFTER the dead-component probe measured the arm
        # above to be blind: with the fusion off, `pred_logits` IS the scorer's
        # output and nothing else can carry the signal.
        model.supervise_joint_box_scores = False
        other = {**batch, "token_ids": (batch["token_ids"] + 11) % 63 + 1}
        moved = np.max(np.abs(
            np.array(model(batch, training=False)["pred_logits"])
            - np.array(model(other, training=False)["pred_logits"])))
        assert moved > 1e-4, f"the scorer ignores the prompt ({moved})"

    def test_class_logits_are_not_constant_without_the_fusion(self, model,
                                                              batch):
        # LIVENESS ARM, same reason: a dead scorer emits exactly one value, and
        # a constant output has a fingerprint you COUNT rather than read off a
        # magnitude.
        model.supervise_joint_box_scores = False
        logits = np.array(model(batch, training=False)["pred_logits"])
        assert len(np.unique(logits)) > 1


# ---------------------------------------------------------------------
# the fusion oracle
# ---------------------------------------------------------------------


class TestFusionOracle:
    """The presence x localization fusion, against a float64 oracle.

    Probe (iii) is MANDATORY and it is the one that separates M9.1. The naive
    expectation -- "at saturated presence the two candidates coincide" -- is
    FALSE for this pair: at presence logit +8 the correct fusion is the identity
    on the class logit while the logit-multiply candidate returns 8x it. The
    real coincidence is a thin CURVE through the interior, whose minimum
    separation over a 7x7 probe grid is 0.056 nats at (class=-0.5,
    presence=+1.5). The probes below deliberately sit away from it.
    """

    @staticmethod
    def fuse(class_logit, presence_logit, clamp=10.0):
        cls = ops.convert_to_tensor(
            np.array(class_logit, dtype="float32").reshape(1, 1, -1, 1))
        pres = ops.convert_to_tensor(
            np.array(presence_logit, dtype="float32").reshape(1, 1, 1))
        return np.array(Sam3Image._fuse(cls, pres, clamp)).reshape(-1)

    def test_probe_i_saturated_presence_is_almost_the_identity(self):
        cls = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        got = self.fuse(cls, 8.0)
        np.testing.assert_allclose(got, np_fusion_correct(cls, 8.0),
                                   atol=ORACLE_TOL)
        assert np.max(np.abs(got - cls)) < 0.02

    def test_probe_ii_absent_presence_drives_the_class_logit_to_a_floor(self):
        cls = np.array([-3.0, 0.0, 3.0])
        got = self.fuse(cls, -12.0)
        np.testing.assert_allclose(got, np_fusion_correct(cls, -12.0),
                                   atol=ORACLE_TOL)
        # MEASURED, and it corrects the naive expectation: the floor is the
        # inverse-sigmoid EPS floor, not the +-10 clamp floor.
        assert np.all(got < -INVERSE_SIGMOID_CEILING + 1e-2)
        assert np.all(got > -10.0)

    def test_probe_iii_the_logit_multiply_candidate_differs_measurably(self):
        # The mandated separating probe. Each pair is off the coincidence curve.
        pairs = [(0.0, 0.0), (-2.0, -1.0), (2.0, 2.0), (-1.0, 0.5)]
        for cls, pres in pairs:
            got = self.fuse([cls], pres)[0]
            correct = float(np_fusion_correct(cls, pres))
            wrong = float(np_fusion_logit_multiply(cls, pres))
            assert abs(got - correct) < ORACLE_TOL
            assert abs(correct - wrong) > 0.5, (
                f"probe ({cls}, {pres}) is a coincidence point, separation "
                f"{abs(correct - wrong)}")

    def test_the_probe_set_separation_is_pinned_so_the_tolerance_cannot_hide_it(
            self):
        # A loosened tolerance is exactly how a defect gets swallowed, so the
        # wrong-candidate margin is asserted to be orders above it.
        separations = [abs(float(np_fusion_correct(c, p))
                           - float(np_fusion_logit_multiply(c, p)))
                       for c, p in [(0.0, 0.0), (-2.0, -1.0), (2.0, 2.0)]]
        assert min(separations) > 200 * ORACLE_TOL

    def test_the_inverse_sigmoid_eps_bounds_the_output(self):
        # The literal that actually sets the saturation floor.
        assert abs(np_inverse_sigmoid(1.0) - INVERSE_SIGMOID_CEILING) < 1e-12
        assert abs(np_inverse_sigmoid(0.0) + INVERSE_SIGMOID_CEILING) < 1e-12

    def test_the_ten_clamp_is_a_provable_no_op_at_the_reference_eps(self):
        # M9.3 measured INERT and this test is the proof, kept permanently so a
        # future eps change is caught the moment the clamp starts to bind.
        grid = np.linspace(-40.0, 40.0, 9)
        cls, pres = np.meshgrid(grid, grid)
        clamped = np_fusion_correct(cls, pres, clamp=10.0)
        unclamped = np_inverse_sigmoid(np_sigmoid(cls) * np_sigmoid(pres))
        assert np.max(np.abs(clamped - unclamped)) == 0.0

    def test_a_smaller_eps_would_make_the_clamp_bind(self):
        # The substituted M9.3b: the discriminating literal is eps, not clamp.
        grid = np.linspace(-40.0, 40.0, 9)
        cls, pres = np.meshgrid(grid, grid)
        tight = np_inverse_sigmoid(np_sigmoid(cls) * np_sigmoid(pres), 1e-7)
        assert np.max(np.abs(tight)) > 10.0
        moved = np.max(np.abs(np_fusion_correct(cls, pres)
                              - np.clip(tight, -10.0, 10.0)))
        assert moved > 3.0

    def test_the_fusion_reaches_the_model_output(self, model, batch):
        # LIVENESS ARM: the expression is not merely present, it runs -- and the
        # SAME model instance is used for both arms, so this cannot pass by two
        # independent initializations differing (a non-discriminating shape
        # measured in this package at step 8).
        model.supervise_joint_box_scores = False
        raw = model(batch, training=False)
        model.supervise_joint_box_scores = True
        fused = model(batch, training=False)
        expected = np_fusion_correct(np.array(raw["pred_logits"]),
                                     np.array(raw["presence_logit"])[:, None, :])
        np.testing.assert_allclose(np.array(fused["pred_logits"]), expected,
                                   atol=ORACLE_TOL)

    def test_disabling_the_fusion_changes_the_class_logits(self, model, batch):
        # LIVENESS ARM, same-instance for the same reason.
        model.supervise_joint_box_scores = False
        raw = np.array(model(batch, training=False)["pred_logits"])
        model.supervise_joint_box_scores = True
        fused = np.array(model(batch, training=False)["pred_logits"])
        assert np.max(np.abs(raw - fused)) > 1e-3


# ---------------------------------------------------------------------
# where the presence comes from (I-7)
# ---------------------------------------------------------------------


class TestPresenceSource:

    def test_the_reported_presence_is_the_decoders_last_layer(self, model,
                                                              batch):
        out = model(batch, training=False)
        neck_out = model.neck(model.backbone(batch["image"], training=False))
        pyramid = model._scalped(neck_out["sam3_features"])
        _, _, presence, _ = model.transformer(
            model._flatten(pyramid[-1]),
            memory_text=model.text_encoder(batch["token_ids"], training=False),
            text_padding_mask=batch["token_padding_mask"],
            memory_pos=model._flatten(
                model._scalped(neck_out["sam3_pos"])[-1]),
            training=False)
        np.testing.assert_allclose(np.array(out["presence_logit"]),
                                   np.array(presence)[-1], atol=1e-6)

    def test_the_segmentation_head_exposes_no_presence_surface(self, model):
        # M9.2's construction-time assertion: there is nothing to read.
        head = model.segmentation_head
        assert not any("presence" in name for name in dir(head))
        assert not any("presence" in key for key in head.get_config())
        out_keys = set(model.segmentation_head.compute_output_shape(
            [(None, 32, 32, 8), (None, 16, 16, 8), (None, 8, 8, 8)],
            (2, 2, 5, 8)))
        assert not any("presence" in key for key in out_keys)

    def test_a_decoder_without_a_presence_token_is_refused(self):
        from dl_techniques.models.sam3 import Sam3TransformerDecoder
        parts = Sam3Image.from_variant("tiny")
        with pytest.raises(ValueError, match="use_presence_token=True"):
            Sam3Image(
                backbone=parts.backbone, neck=parts.neck,
                text_encoder=parts.text_encoder,
                transformer=Sam3TransformerDecoder(
                    d_model=8, num_heads=2, num_layers=2, num_queries=5,
                    feat_size=(8, 8), dim_feedforward=16, dropout_rate=0.0,
                    use_presence_token=False),
                dot_prod_scoring=parts.dot_prod_scoring,
                segmentation_head=parts.segmentation_head)


# ---------------------------------------------------------------------
# box refinement
# ---------------------------------------------------------------------


class TestBoxRefinement:

    def test_a_zero_initialized_box_head_leaves_the_anchor_untouched(
            self, model, batch):
        # VACUITY NOTE: this assertion is satisfied BY CONSTRUCTION by a box
        # head that emits zeros for any reason, including a dead one. Its
        # liveness arm is `test_a_live_box_head_moves_the_boxes` below.
        out = model(batch, training=False)
        anchor = np.array(ops.sigmoid(model.transformer.reference_points))
        np.testing.assert_allclose(np.array(out["pred_boxes"]),
                                   np.broadcast_to(anchor, (2, 5, 4)),
                                   atol=1e-6)

    def test_a_live_box_head_moves_the_boxes(self, model, batch):
        # LIVENESS ARM.
        before = np.array(model(batch, training=False)["pred_boxes"])
        last = model.transformer.bbox_embed[-1]
        last.kernel.assign(np.full(last.kernel.shape, 0.5, dtype="float32"))
        after = np.array(model(batch, training=False)["pred_boxes"])
        assert np.max(np.abs(before - after)) > 1e-3

    def test_the_final_box_is_the_last_layers_refinement(self, model, batch):
        # The decoder returns the anchor each layer CONSUMED, so the last
        # refinement exists only here. Randomize the head so the check is not
        # vacuous against the zero-init identity.
        randomize(model.transformer.bbox_embed[-1], seed=3)
        neck_out = model.neck(model.backbone(batch["image"], training=False))
        pyramid = model._scalped(neck_out["sam3_features"])
        hidden, anchors, _, _ = model.transformer(
            model._flatten(pyramid[-1]),
            memory_text=model.text_encoder(batch["token_ids"], training=False),
            text_padding_mask=batch["token_padding_mask"],
            memory_pos=model._flatten(
                model._scalped(neck_out["sam3_pos"])[-1]),
            training=False)
        from dl_techniques.models.sam3 import Sam3TransformerDecoder as Dec
        delta = Dec._run_mlp(model.transformer.bbox_embed, hidden)
        expected = np.array(ops.sigmoid(
            delta + Dec._inverse_sigmoid(anchors)))[-1]
        np.testing.assert_allclose(
            np.array(model(batch, training=False)["pred_boxes"]), expected,
            atol=1e-5)
        assert np.max(np.abs(expected - np.array(ops.sigmoid(
            model.transformer.reference_points)))) > 1e-4


# ---------------------------------------------------------------------
# SC-3: the parameter audit
# ---------------------------------------------------------------------


class TestParameterAudit:

    def test_every_component_matches_its_closed_form_at_the_tiny_variant(self):
        built = Sam3Image.from_variant("tiny")
        built.build(None)
        expected = component_closed_forms(Sam3Image.MODEL_VARIANTS["tiny"])
        for key in COMPONENT_KEYS:
            assert getattr(built, key).count_params() == expected[key], key
        assert built.count_params() == sum(expected.values())

    def test_the_shipped_closed_forms_are_self_consistent(self):
        """NOT a measurement -- a regression guard on the closed forms.

        `SHIPPED_PARAMS` was produced BY `component_closed_forms`, so this
        comparison cannot detect a wrong closed form; it detects a closed form
        that MOVED. The instantiating arm below is the measurement, and it is
        opt-in. (Reviewer W-6: this was previously named as though it verified
        the shipped variant.)
        """
        expected = component_closed_forms(Sam3Image.MODEL_VARIANTS["sam3"])
        assert expected == SHIPPED_PARAMS
        assert sum(expected.values()) == SHIPPED_TOTAL

    @pytest.mark.skipif(
        os.environ.get("SAM3_SHIPPED_AUDIT") != "1",
        reason="builds the 821M-parameter shipped variant (~3.3 GiB device "
               "memory); run with SAM3_SHIPPED_AUDIT=1",
    )
    def test_the_shipped_variant_is_instantiated_and_counted(self):
        """The measurement W-6 asked for: build it, then count it.

        Executed 2026-08-05 on GPU 1 (RTX 4070): all six components and the
        total match the closed forms exactly.
        """
        built = Sam3Image.from_variant("sam3")
        built.build(None)
        expected = component_closed_forms(Sam3Image.MODEL_VARIANTS["sam3"])
        for key in COMPONENT_KEYS:
            assert getattr(built, key).count_params() == expected[key], key
        assert built.count_params() == SHIPPED_TOTAL
        assert expected == SHIPPED_PARAMS

    def test_the_audit_is_not_vacuous_because_a_shrunk_component_fails_it(self):
        """EXECUTE the non-vacuity claim rather than asserting it.

        Iteration 1's `+-25M` band would have passed with the whole memory
        encoder deleted. This test deletes capacity from each component in turn
        and requires the SAME comparison to fail, so the audit is demonstrated
        to have discriminating power rather than assumed to.
        """
        table = dict(Sam3Image.MODEL_VARIANTS["tiny"])
        expected = component_closed_forms(table)
        shrinks = {
            "backbone": {"depth": 1, "global_att_blocks": (0,)},
            "neck": {"scale_factors": (2.0, 1.0, 0.5)},
            "text_encoder": {"text_depth": 1},
            "transformer": {"decoder_layers": 1},
            "dot_prod_scoring": {"prompt_mlp_hidden_dim": 8},
        }
        for key, override in shrinks.items():
            shrunk = Sam3Image.from_variant("tiny", **override)
            shrunk.build(None)
            assert getattr(shrunk, key).count_params() != expected[key], (
                f"shrinking {key} did not change its parameter count -- the "
                f"audit is vacuous for that component")
            assert shrunk.count_params() != sum(expected.values())

    def test_a_deleted_segmentation_stage_fails_the_audit(self):
        # The seg head's own capacity knob rides on the pyramid, so it gets its
        # own arm rather than being silently skipped above.
        table = dict(Sam3Image.MODEL_VARIANTS["tiny"])
        full = segmentation_params(table, 2)
        assert segmentation_params(table, 1) != full
        built = Sam3Image.from_variant("tiny")
        built.build(None)
        assert built.segmentation_head.count_params() == full

    def test_the_closed_forms_are_not_read_back_from_the_model(self):
        # The forms are pure functions of a config dict; feeding them a table
        # the package never instantiates must still produce a number, which is
        # what makes the tiny-vs-shipped agreement meaningful.
        table = dict(Sam3Image.MODEL_VARIANTS["sam3"])
        table["depth"] = 33
        assert backbone_params(table) > SHIPPED_PARAMS["backbone"]


# ---------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------


class TestSerialization:

    def test_get_config_carries_every_init_parameter(self, model):
        config = model.get_config()
        for key in COMPONENT_KEYS:
            assert key in config
        for key in ("num_feature_levels", "scalp", "supervise_joint_box_scores",
                    "detach_presence_in_joint_score", "joint_score_clamp"):
            assert key in config

    def test_config_round_trip_preserves_every_flag(self, model):
        rebuilt = Sam3Image.from_config(model.get_config())
        assert rebuilt.supervise_joint_box_scores is True
        assert rebuilt.scalp == model.scalp
        assert rebuilt.joint_score_clamp == model.joint_score_clamp

    def test_full_keras_roundtrip_preserves_output_VALUES(self, model, batch):
        """A count/path comparison is NOT enough -- measured in this package.

        A nested sub-layer store restores freshly initialized kernels while
        `count_params()`, the weight COUNT and every weight PATH all match
        (D-098). Only an output comparison sees it, so this test compares
        VALUES, and the box head is randomized first so `pred_boxes` is not
        trivially the zero-init identity on both sides.
        """
        randomize(model.transformer.bbox_embed[-1], seed=11)
        before = model(batch, training=False)
        path = os.path.join(tempfile.mkdtemp(), "sam3_image.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        after = restored(batch, training=False)
        assert restored.count_params() == model.count_params()
        for key in before:
            delta = float(np.max(np.abs(np.array(before[key])
                                        - np.array(after[key]))))
            assert delta == 0.0, f"{key} moved by {delta} across the round trip"

    def test_the_roundtrip_probe_is_not_comparing_a_degenerate_output(
            self, model, batch):
        # The guard that makes the test above non-vacuous: every compared
        # tensor must actually carry information.
        randomize(model.transformer.bbox_embed[-1], seed=11)
        out = model(batch, training=False)
        for key, value in out.items():
            assert len(np.unique(np.array(value))) > 1, key

    def test_weight_paths_alone_would_not_have_caught_a_reinitialization(
            self, model):
        # Permanent record of WHY the test above compares values: the path set
        # survives anything that keeps the structure, including fresh weights.
        paths = {w.path for w in model.weights}
        randomize(model.transformer.bbox_embed[-1], seed=5)
        assert {w.path for w in model.weights} == paths


# ---------------------------------------------------------------------
# the training-flag trap
# ---------------------------------------------------------------------


class TestTrainingFlagTrap:
    """`training=None` is NOT inference for this model, and that is measured.

    The repository's shared `StochasticDepth` short-circuits on
    `training is False` ONLY, so a plain `model(inputs)` -- which passes
    `training=None` down -- drops residual paths at any non-zero drop-path rate.
    The `tiny` variant therefore pins its rate to 0.0; the shipped variant keeps
    the reference's 0.1 and IS stochastic under `model(inputs)`.
    """

    def test_the_tiny_variant_is_deterministic_under_a_plain_call(self, model,
                                                                  batch):
        first = np.array(model(batch)["pred_masks"])
        second = np.array(model(batch)["pred_masks"])
        assert np.max(np.abs(first - second)) == 0.0

    def test_the_tiny_variant_pins_drop_path_to_zero_deliberately(self):
        assert Sam3Image.MODEL_VARIANTS["tiny"]["drop_path_rate"] == 0.0
        assert Sam3Image.MODEL_VARIANTS["sam3"]["drop_path_rate"] == 0.1

    def test_a_nonzero_drop_path_makes_a_plain_call_non_deterministic(self,
                                                                     batch):
        # The trap itself, executed rather than described. This is the arm that
        # would have caught the round-trip "failure" that cost this step an hour.
        stochastic = Sam3Image.from_variant("tiny", drop_path_rate=0.9)
        stochastic.build(None)
        draws = [np.array(stochastic(batch)["pred_masks"]) for _ in range(6)]
        spread = max(float(np.max(np.abs(draws[0] - other)))
                     for other in draws[1:])
        assert spread > 0.0, (
            "StochasticDepth no longer treats training=None as training -- "
            "re-check the drop-path rates in MODEL_VARIANTS")
        pinned = [np.array(stochastic(batch, training=False)["pred_masks"])
                  for _ in range(3)]
        assert max(float(np.max(np.abs(pinned[0] - other)))
                   for other in pinned[1:]) == 0.0


# ---------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------


class TestPackageSurface:

    def test_the_curated_exports_all_resolve(self):
        import dl_techniques.models.sam3 as package
        for name in package.__all__:
            assert isinstance(getattr(package, name), type)

    def test_the_module_is_keras_ops_pure(self):
        import dl_techniques.models.sam3.sam3_image as module
        source = open(module.__file__).read()
        assert "import tensorflow" not in source
        assert "\ntf." not in source and " tf." not in source

    def test_no_defective_loss_module_is_referenced(self):
        import dl_techniques.models.sam3 as package
        directory = os.path.dirname(package.__file__)
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            source = open(os.path.join(directory, name)).read()
            for banned in ("SAMMaskLoss", "SegmentationLosses",
                           "segmentation_loss"):
                assert banned not in source, f"{name} references {banned}"
