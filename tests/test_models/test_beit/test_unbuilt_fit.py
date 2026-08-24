"""``fit()`` with no preceding ``build()`` -- a framework-trap regression guard.

Moved verbatim from ``test_model.py`` (class ``TestBeitUnbuiltFit``, section 10)
during the step-8 decomposition of plan-2026-08-24T074054-247151fd.

Isolated because the hazard is non-obvious and specific: lazy building inside a traced
train step binds any tensor materialized in ``build()`` to the wrong ``FuncGraph``.
``plans/SYSTEM.md:224`` singles this out, and the plan's invariant I-2 keeps
``BeitAttention``'s relative-position index a numpy buffer built inside ``call()`` for
exactly this reason. The path a user actually takes -- construct, compile, fit -- is
the path no other file in this package exercises.

KNOWN FLAKE (G-07): ``test_classifier_fit_without_an_explicit_build`` is intermittent
under GPU contention. Re-run it ALONE before reading a failure as a regression.
"""

import keras
import numpy as np
from keras import ops

from tests.test_models.test_beit.beit_test_geometry import (
    NUM_PATCHES,
    VOCAB,
    _images,
    _mask,
    _mim,
    _classifier,
)

class TestBeitUnbuiltFit:
    """``fit()`` without a preceding ``model.build(...)`` must train, on both heads.

    Regression guard for the defect measured at step 7 of
    ``plan-2026-08-11T012340-f63796dc``: ``BeitAttention.build()`` materialized the
    relative-position index as a TENSOR, so a lazy build inside the traced train step
    produced a constant belonging to the inner ``one_step_on_data`` FuncGraph and
    ``fit()`` died with ``InaccessibleTensorError`` before the first gradient step.
    Every eager forward pass and every explicitly-built test in this module passed
    while that was true, so nothing below may call ``build()`` first.
    """

    def test_classifier_fit_without_an_explicit_build(self):
        model = _classifier('tiny', num_classes=4)
        assert not model.built
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        y = np.random.default_rng(1).integers(0, 4, size=(4,)).astype('int32')
        model.fit(_images(batch=4), y, batch_size=2, epochs=1, verbose=0)
        assert model.built
        assert int(model.optimizer.iterations.numpy()) == 2

    def test_mim_fit_without_an_explicit_build(self):
        model = _mim('tiny')
        assert not model.built
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        rng = np.random.default_rng(2)
        images = _images(batch=4)
        mask = _mask(batch=4)
        tokens = rng.integers(0, VOCAB, size=(4, NUM_PATCHES)).astype('int32')
        # `sample_weight` is how MIM restricts the loss to the masked patches.
        model.fit(
            (images, mask), tokens,
            sample_weight=mask.astype('float32'),
            batch_size=2, epochs=1, verbose=0,
        )
        assert model.built
        assert int(model.optimizer.iterations.numpy()) == 2

    def test_the_unbuilt_fit_actually_moves_the_relative_position_bias_table(self):
        """Not just 'no raise': the guarded path must reach the bias table.

        A `fit()` that silently skipped the relative-position bias would also not
        raise, so the assertion is on weight MOVEMENT of the table itself.
        """
        model = _classifier('tiny', num_classes=4)
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        y = np.random.default_rng(3).integers(0, 4, size=(4,)).astype('int32')
        model.fit(_images(batch=4), y, batch_size=2, epochs=1, verbose=0)

        tables = [
            v for v in model.trainable_variables
            if v.path.endswith("relative_position_bias_table")
        ]
        assert len(tables) == model.backbone.num_layers, (
            f"expected one bias table per encoder layer, found {len(tables)}"
        )
        for table in tables:
            assert float(np.abs(ops.convert_to_numpy(table)).max()) > 0.0, (
                f"{table.path} is still all-zero after fit()"
            )
