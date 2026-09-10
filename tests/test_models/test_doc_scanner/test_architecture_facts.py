"""Single-claim guards for the ``doc_scanner`` facts nothing else can see.

Each test here pins ONE transcribed number against a LITERAL. That is the whole
design: a constant is only as good as the thing that would notice it changing,
and every other instrument in this suite is structurally blind to these.

What the neighbouring instruments cannot see
--------------------------------------------
* **A shape test is blind to every number in this file.** The GRU is
  width-agnostic -- a 128-wide hidden state flows end to end exactly as well as
  a 160-wide one -- so ``(B, 288, 288, 3) -> (B, 288, 288, 2)`` holds under the
  wrong width table. This is the concrete reason the width guard below exists
  and is written against literals.
* **A serialization round trip is blind to them.** It compares the model to
  ITSELF; both sides move together when a constant moves.
* **A parameter count would see a width change only as a number.** It reports
  "8,412,096 != 8,533,378", which names nothing.
* **Nothing at all would see an epsilon change.** ``1e-3`` versus ``1e-5`` in a
  normalization layer is a 100x difference with no shape symptom, no warning
  and no parameter-count effect -- it is only visible in the output values, and
  only against a reference this repo does not have.

Why literals, and never a re-derivation
---------------------------------------
Every assertion below writes the number out. Reading the expected value from
the same table under test moves both sides of the comparison at once and the
test stays green forever -- the exact vacuity mode this repo has recorded
repeatedly. If a number here genuinely must change, this file changes too, in
the same commit, deliberately.
"""

import keras
import pytest

from dl_techniques.models.vision.image_restoration.doc_scanner import components


class TestTranscribedConstants:
    """The seven module-level constants, each pinned against its literal."""

    def test_instance_norm_epsilon_is_the_torch_default_not_the_keras_one(self):
        """``1e-5`` (torch ``nn.InstanceNorm2d``), NOT ``1e-3`` (Keras).

        ``keras.layers.GroupNormalization`` -- the stand-in this port uses for
        per-sample instance norm -- defaults to ``epsilon=1e-3``. Accepting
        that default is a silent 100x with no shape symptom, which is why the
        constant exists at all rather than being left implicit.
        """
        assert components.INSTANCE_NORM_EPSILON == 1e-5
        assert components.INSTANCE_NORM_EPSILON != 1e-3

    def test_spatial_divisor_is_eight(self):
        """The rectifier refines at 1/8 resolution: ``model.py:49-50``, ``:65``."""
        assert components.SPATIAL_DIVISOR == 8

    def test_refine_iterations_is_twelve(self):
        """``model.py:67`` default and ``inference.py:28`` call site agree."""
        assert components.REFINE_ITERATIONS == 12

    def test_sequence_loss_gamma_is_the_paper_value(self):
        """``0.85`` from arXiv:2110.14968v2 Eq. 9 -- there is no code to cite.

        The upstream release ships inference only: no training loop, no loss,
        no optimizer. A reader reconciling this constant against the checkout
        will not find it there.
        """
        assert components.SEQUENCE_LOSS_GAMMA == 0.85

    def test_segmentation_mask_threshold_is_a_half(self):
        """``inference.py:25``: ``msk = (msk > 0.5).float()``."""
        assert components.SEG_MASK_THRESHOLD == 0.5

    def test_backward_map_calibration_constants_are_transcribed_exactly(self):
        """``inference.py:29``: ``bm = (2 * (bm / 286.8) - 1) * 0.99``.

        The divisor is 286.8 and NOT the 288 training resolution, and the scale
        is 0.99 and NOT 1.0. Neither number is explained upstream or in the
        paper. The two ``!=`` arms below are the point of the test: the
        plausible "corrections" are exactly what a future reader would apply.
        """
        assert components.BM_CALIBRATION_DIVISOR == 286.8
        assert components.BM_CALIBRATION_DIVISOR != 288.0
        assert components.BM_CALIBRATION_SCALE == 0.99
        assert components.BM_CALIBRATION_SCALE != 1.0


class TestVariantSpecWidths:
    """The width table -- and the dead-default trap it exists to make unrepeatable.

    See ``decisions.md`` D-006 and the ``# DECISION`` anchor at the
    ``_VARIANT_SPEC`` definition site. ``update.py:18`` and ``update.py:36``
    declare ``hidden_dim=128, input_dim=192+128``; NOTHING constructs either
    with those values. ``update.py:88`` overrides with
    ``SepConvGRU(hidden_dim=hidden_dim, input_dim=160+160)`` and
    ``model.py:35-39`` sets ``hidden_dim = context_dim = 160``,
    ``BasicEncoder(output_dim=320)``.
    """

    def test_there_is_exactly_one_variant_and_it_is_docscanner_l(self):
        """One row, named for the released architecture.

        The paper's Table 1 documents DocScanner-B (encoder ending at 256
        channels), which was never released as code and is specified only for
        the encoder column; DocScanner-T has no published architectural split
        at all. Inventing an interpolated row for either is forbidden here, so
        the table has one key and the README records why.
        """
        assert list(components._VARIANT_SPEC) == ["docscanner-l"]

    def test_the_gru_widths_come_from_the_call_site_not_the_dead_default(self):
        """160 / 160 / 320, and the hidden state is never 128.

        A shape test cannot distinguish these: the GRU is width-agnostic and a
        128-wide hidden state simply carries a different width end to end. This
        assertion is the only thing in the tree that can.

        MEASURED, and the reason the ``gru_input_dim`` arm below is an equality
        and not a ``!=``: the dead default's ``input_dim=192+128`` evaluates to
        **320**, which is exactly the live value from ``update.py:88``'s
        ``160+160``. The two disagree on ``hidden_dim`` ALONE (128 vs 160). So
        a reader who trusts ``update.py:36``'s signature gets the input width
        right by coincidence and the hidden width wrong -- there is no arm that
        can catch the input width, because there is nothing to catch, and
        writing ``!= 192 + 128`` here would be a guard that can never pass.
        """
        spec = components._VARIANT_SPEC["docscanner-l"]

        assert spec["hidden_dim"] == 160
        assert spec["context_dim"] == 160
        assert spec["gru_input_dim"] == 320

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-007: do NOT "complete" this
        # with `assert spec["gru_input_dim"] != 192 + 128`. That is the obvious
        # symmetric arm and it can never fail: `192 + 128 == 320`, the live value.
        # `hidden_dim` is the ONLY field the two signatures disagree on.
        assert spec["hidden_dim"] != 128, "read from update.py:36's dead default"
        assert 192 + 128 == 320, (
            "if this ever fails arithmetic broke; it documents that the dead "
            "default's INPUT width coincides with the live one"
        )

    def test_the_hidden_and_context_split_reconstitutes_the_encoder_output(self):
        """``model.py:74``: ``torch.split(fmap1, [160, 160])``.

        The two halves are not independently chosen -- they are the encoder's
        320 output channels split down the middle, so a change to
        ``fnet_output_dim`` that leaves these alone is incoherent. This is the
        one derivation the table encodes redundantly, and the redundancy is
        checked here rather than trusted.
        """
        spec = components._VARIANT_SPEC["docscanner-l"]
        assert spec["hidden_dim"] + spec["context_dim"] == spec["fnet_output_dim"]

    def test_the_gru_input_is_the_motion_encoder_output_plus_the_hidden_state(self):
        """``update.py:88``: ``input_dim=160+160``.

        The motion encoder emits ``hidden_dim`` channels (``update.py:70``'s
        ``160-2`` convolution concatenated with the 2 flow channels), and the
        separable GRU concatenates its hidden state onto that.
        """
        spec = components._VARIANT_SPEC["docscanner-l"]
        assert spec["gru_input_dim"] == 2 * spec["hidden_dim"]

    def test_the_encoder_stem_is_eighty_channels_not_the_reference_sixty_four(self):
        """``extractor.py:96`` emits 80; ``extractor.py:93`` declares ``norm1(64)``.

        The reference applies a norm declared for 64 channels to an 80-channel
        tensor. That is inert in torch only because ``nn.InstanceNorm2d``
        defaults to ``affine=False`` and allocates no parameters. A Keras
        affine normalization allocates by shape and would crash -- so 64 is a
        latent upstream bug and 80 is the true width. A port that transcribes
        the 64 faithfully does not reproduce the reference, it breaks.
        """
        spec = components._VARIANT_SPEC["docscanner-l"]
        assert spec["encoder_stem_channels"] == 80
        assert spec["encoder_stem_channels"] != 64

    def test_the_encoder_stage_widths_are_eighty_one_sixty_two_forty(self):
        """``extractor.py:98-101``, then ``:104``'s 1x1 to ``output_dim``."""
        spec = components._VARIANT_SPEC["docscanner-l"]
        assert tuple(spec["encoder_stage_channels"]) == (80, 160, 240)
        assert spec["fnet_output_dim"] == 320

    @pytest.mark.parametrize(
        "field",
        [
            "hidden_dim",
            "context_dim",
            "gru_input_dim",
            "fnet_output_dim",
            "encoder_stem_channels",
            "encoder_stage_channels",
        ],
    )
    def test_no_declared_width_field_goes_missing(self, field):
        """The liveness arm.

        Without it, deleting a field makes the guards above disappear rather
        than fail -- a ``KeyError`` in one test is a failure, but a field that
        was never referenced anywhere would simply stop being checked.
        """
        assert field in components._VARIANT_SPEC["docscanner-l"]


class TestRegistrationKeysStripFamilyAndSubfamily:
    """H-3. ``dl_techniques.models.doc_scanner.<module>``, literal.

    The classes live at
    ``src/dl_techniques/models/vision/image_restoration/doc_scanner/``, so a key
    derived from the import path would carry ``vision`` and
    ``image_restoration``. Both are filing decisions that have been reshuffled
    before, and a key built from them breaks every archive when they move again.

    Asserted with a literal ``==``, never through a save/load round trip: the
    process shares ONE registry, so a round trip resolves a typo'd key to the
    typo'd class perfectly happily.
    """

    _PACKAGE = "dl_techniques.models.doc_scanner.components"

    @pytest.mark.parametrize(
        "cls_name,expected",
        [
            ("DocScannerResidualBlock",
             "dl_techniques.models.doc_scanner.components>DocScannerResidualBlock"),
            ("DocScannerFeatureEncoder",
             "dl_techniques.models.doc_scanner.components>DocScannerFeatureEncoder"),
        ],
    )
    def test_the_registered_name_is_exactly_this_string(self, cls_name, expected):
        cls = getattr(components, cls_name)
        assert keras.saving.get_registered_name(cls) == expected

    @pytest.mark.parametrize(
        "cls_name", ["DocScannerResidualBlock", "DocScannerFeatureEncoder"]
    )
    def test_the_shared_registration_contract_holds(
            self, cls_name, registration_contract):
        registration_contract(
            getattr(components, cls_name), expected_package=self._PACKAGE)
