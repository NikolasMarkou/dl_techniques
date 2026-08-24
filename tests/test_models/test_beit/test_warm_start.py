"""The MIM -> classifier warm start (SC-10) -- a CROSS-head property.

Moved verbatim from ``test_model.py`` (class ``TestBeitWarmStart``, section 9) during
the step-8 decomposition of plan-2026-08-24T074054-247151fd.

This is why the two heads carry disjoint weight-name prefixes and why
``BACKBONE_NAME`` is pinned: a name-based transfer must move the pretrained trunk into
a freshly built classifier and must NOT move either head. It belongs in its own file
for the same reason ``test_resnet/test_deep_supervision.py`` does -- it is a
cross-cutting behaviour of the pair, owned by neither head's file.
"""

import os
import tempfile

import numpy as np
from keras import ops

from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.models.vision.beit import (
    BACKBONE_NAME,
)
from tests.test_models.test_beit.beit_test_geometry import (
    IMG,
    _mim,
    _classifier,
)

class TestBeitWarmStart:
    """The property the whole two-head prefix discipline exists to deliver."""

    def test_the_two_heads_use_disjoint_prefixes(self):
        """Assert the property DIRECTLY, not via its consequence.

        Why this can fail if the implementation is wrong: any `head_`-prefixed layer
        inside the MIM model (or `decoder_` inside the classifier) would be silently
        skipped by the OTHER model's transfer, and the symptom would be a partially
        random trunk rather than an error.
        """
        mim = _mim()
        clf = _classifier()
        mim.build((None,) + IMG)
        clf.build((None,) + IMG)

        mim_head = {l.name for l in mim.layers} - {BACKBONE_NAME}
        clf_head = {l.name for l in clf.layers} - {BACKBONE_NAME}
        assert mim_head and clf_head
        assert mim_head.isdisjoint(clf_head)
        assert not any(n.startswith("head_") for n in mim_head)
        assert not any(n.startswith("decoder_") for n in clf_head)
        # And nothing inside the shared trunk claims either prefix.
        trunk_names = {l.name for l in mim.backbone.layers}
        assert not any(
            n.startswith(("head_", "decoder_")) for n in trunk_names
        ), trunk_names

    def test_the_trunks_are_weight_identical_in_structure(self):
        """Including the mask token, which the classifier never calls."""
        mim = _mim()
        clf = _classifier()
        mim.build((None,) + IMG)
        clf.build((None,) + IMG)
        mim_w = [tuple(w.shape) for w in mim.backbone.get_weights()]
        clf_w = [tuple(w.shape) for w in clf.backbone.get_weights()]
        assert mim_w == clf_w
        assert clf.backbone.mask_token.built
        assert clf.backbone.mask_token.mask_token is not None

    def test_mim_to_classifier_transfers_the_trunk_values(self):
        """SC-10. A ZERO-LAYER transfer must FAIL this test, not pass it."""
        mim = _mim()
        mim.build((None,) + IMG)

        clf = _classifier()
        # H-12: the TARGET must be built BEFORE the transfer.
        clf.build((None,) + IMG)
        assert clf.built

        source = [ops.convert_to_numpy(w) for w in mim.backbone.get_weights()]
        before = [ops.convert_to_numpy(w) for w in clf.backbone.get_weights()]
        # Precondition: the two trunks start DIFFERENT, otherwise "equal after" is
        # vacuous (both are randomly initialized, so this is a real check).
        assert any(
            not np.allclose(a, b) for a, b in zip(source, before)
        ), "trunks were already identical -- the transfer assertion would be vacuous"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            report = load_weights_from_checkpoint(
                target=clf,
                ckpt_path=path,
                skip_prefixes=("decoder_", "head_"),
            )

        # (b) the report shows the backbone ACTUALLY loaded -- a zero-layer transfer
        # would leave `loaded` empty and this is what makes the test non-vacuous.
        assert BACKBONE_NAME in report.loaded, report.summary_string()
        assert report.num_loaded >= 1
        assert BACKBONE_NAME not in [name for name, _, _ in report.shape_mismatch]
        assert BACKBONE_NAME not in report.missing_in_source
        assert set(report.skipped_by_prefix) == {"decoder_norm", "decoder_head"}

        # (a) trunk weight VALUES are equal post-transfer.
        after = [ops.convert_to_numpy(w) for w in clf.backbone.get_weights()]
        assert len(after) == len(source)
        for i, (a, b) in enumerate(zip(source, after)):
            np.testing.assert_array_equal(a, b, err_msg=f"trunk weight {i}")

    def test_the_classifier_head_is_not_touched_by_the_transfer(self):
        mim = _mim()
        mim.build((None,) + IMG)
        clf = _classifier()
        clf.build((None,) + IMG)
        head_before = ops.convert_to_numpy(clf.head_classifier.kernel)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            load_weights_from_checkpoint(
                target=clf, ckpt_path=path, skip_prefixes=("decoder_", "head_")
            )
        np.testing.assert_array_equal(
            head_before, ops.convert_to_numpy(clf.head_classifier.kernel)
        )

    def test_a_mismatched_backbone_config_is_reported_not_silently_loaded(self):
        """A trunk of a different width must NOT quietly train from scratch."""
        mim = _mim('tiny')
        mim.build((None,) + IMG)
        clf = _classifier('small')          # 384d trunk vs the checkpoint's 192d
        clf.build((None,) + IMG)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            mim.save(path)
            report = load_weights_from_checkpoint(
                target=clf, ckpt_path=path, skip_prefixes=("decoder_", "head_")
            )
        assert BACKBONE_NAME not in report.loaded
        assert BACKBONE_NAME in [name for name, _, _ in report.shape_mismatch]
