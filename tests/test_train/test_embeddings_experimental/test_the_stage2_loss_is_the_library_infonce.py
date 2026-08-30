"""The stage-2 contrastive objective is the LIBRARY loss, not a trainer-private copy.

Before 2026-08-31 this trainer carried its own ~20-line ``SimCSELoss``. It was replaced by
:class:`dl_techniques.losses.SymmetricInfoNCELoss` after the two were measured bit-identical
(worst delta exactly 0.0 under ``mixed_bfloat16`` at the study's batch sizes).

The guard exists because that swap is silently reversible: someone re-adding a local loss class,
or pointing the compile site at a different objective, would change what every future study cell
optimizes while every existing test stayed green. Nothing else in the suite asserts WHICH loss
stage 2 compiles with.

This file lives under ``tests/test_train/`` deliberately. It imports from BOTH ``src/train`` and
``dl_techniques``; that tree already does, so it introduces no new dependency direction. The same
guard under ``tests/test_losses/`` would make the library's tests depend on a trainer.
"""

import inspect

import keras
import numpy as np
import pytest

from dl_techniques.losses import SymmetricInfoNCELoss
from train.embeddings_experimental import train_embeddings


def test_the_trainer_no_longer_defines_its_own_contrastive_loss():
    """A trainer-private loss class must not reappear.

    A failure means someone re-introduced a local copy of the objective. That is exactly the
    duplication the swap removed, and a local copy can drift from the library loss without any
    test noticing.
    """
    assert not hasattr(train_embeddings, "SimCSELoss"), (
        "train_embeddings defines SimCSELoss again. The stage-2 objective is supposed to come "
        "from dl_techniques.losses.SymmetricInfoNCELoss; a trainer-private copy can drift from it "
        "silently."
    )
    assert "SimCSELoss" not in getattr(train_embeddings, "__all__", ()), (
        "SimCSELoss is back in train_embeddings.__all__."
    )


def test_the_stage_two_compile_site_uses_the_library_loss():
    """The stage-2 `compile` call must name `SymmetricInfoNCELoss`.

    Read from source rather than by running a full training stage, which needs a dataset. A
    failure means the compile site was pointed at a different objective -- every future cell would
    optimize something else while the rest of the suite stayed green.
    """
    src = inspect.getsource(train_embeddings)
    assert "SymmetricInfoNCELoss(temperature=" in src, (
        "no `SymmetricInfoNCELoss(temperature=...)` in train_embeddings.py. The stage-2 compile "
        "site no longer uses the library InfoNCE loss."
    )
    assert "loss=SymmetricInfoNCELoss(" in src, (
        "`SymmetricInfoNCELoss` appears but is not passed as `loss=` to a compile call."
    )


@pytest.mark.parametrize("batch", [4, 16])
def test_the_library_loss_still_reproduces_the_objective_the_study_was_run_with(batch):
    """The objective is unchanged from the trainer-private class the first cells used.

    Cells run before the swap used a local ``SimCSELoss``; the transcription below is that class's
    body. Study numbers are only comparable across the swap if these agree. A failure means the
    library loss has drifted from what those cells optimized, and pre- and post-swap cells are no
    longer measuring the same thing.
    """
    temperature = 0.05
    rng = np.random.default_rng(batch)
    raw = rng.standard_normal((batch, 2, 256)).astype("float32")
    y_pred = (raw / np.linalg.norm(raw, axis=-1, keepdims=True)).astype("float32")
    y_true = keras.ops.zeros((batch,), dtype="float32")

    view_a, view_b = y_pred[:, 0, :], y_pred[:, 1, :]
    logits = keras.ops.matmul(view_a, keras.ops.transpose(view_b)) / temperature
    targets = keras.ops.arange(keras.ops.shape(logits)[0])
    forward = keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    backward = keras.losses.sparse_categorical_crossentropy(
        targets, keras.ops.transpose(logits), from_logits=True
    )
    expected = float(keras.ops.convert_to_numpy(keras.ops.mean(forward + backward) / 2.0))

    actual = float(keras.ops.convert_to_numpy(
        SymmetricInfoNCELoss(temperature=temperature)(y_true, y_pred)
    ))
    assert actual == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"batch={batch}: library loss {actual!r} != the pre-swap objective {expected!r}. "
        f"Cells run before and after the 2026-08-31 swap are no longer comparable."
    )
