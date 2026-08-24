"""Make a compiled-but-untrained model's ``.keras`` archive self-consistent.

Why this module exists
----------------------
``model.compile(optimizer=...)`` does **not** allocate the optimizer's slot
variables. Keras allocates them lazily, on the first gradient application. So a
test that compiles a model, saves it, and reloads it -- without ever running a
train step -- writes an archive whose optimizer section holds only the two
always-present scalars (``iteration`` and the learning rate), while the
*reloaded* model's optimizer is built against the restored trainable variables
and therefore has ``2 + 2*len(trainable_variables)`` of them for Adam. Keras
notices the mismatch, warns, and silently skips the optimizer restore::

    UserWarning: Skipping variable loading for optimizer 'adam', because it has
    106 variables whereas the saved optimizer has 2 variables.

That warning is a true statement about the *sequence the test performed*, and
under ``-W error::UserWarning`` it escalates to
``ValueError: A total of 1 objects could not be loaded.``

Measured on ``SqueezeNetV1`` (variant 1.1, 32x32x3, 52 trainable variables),
2026-08-22, CPU, keras 3.x:

===============================================  ==========  ==========  =========
arm                                              @save vars  @load vars  warning
===============================================  ==========  ==========  =========
compile, forward, save                                    2         106  YES
compile, forward, ONE ``train_on_batch``, save          106         106  no
compile, forward, ``optimizer.build(...)``, save        106         106  no
compile, forward, ``save(include_optimizer=False)``        2         106  **YES**
no ``compile`` at all                                     -           -  no
===============================================  ==========  ==========  =========

Note the fourth row. ``include_optimizer=False`` is accepted by
``keras.saving.save_model`` and then **discarded** for the ``.keras`` path
(``keras/src/saving/saving_api.py:56`` pops the kwarg; the branch at line 105
calls ``saving_lib.save_model(model, filepath)`` without it). It is a silent
no-op, not a fix -- do not reach for it.

What to use instead
-------------------
:func:`build_optimizer_state` is the third row: one call, immediately before
``save()``, that allocates the slots the archive is about to claim to contain.
It is preferred over a throwaway ``train_on_batch`` because it moves no weights,
consumes no RNG draws, and needs neither targets nor a loss that accepts them --
so a round-trip test that asserts bit-identical outputs stays valid.
"""

from typing import Any, Optional

import keras

# DECISION plan-2026-08-22T035419-a11304c8/D-016: this helper is the ONE
# test-side repair for RD-2, applied at 24 call sites in 22 files. Three
# alternatives were measured and rejected, and none of them should be
# reintroduced site-by-site:
#   * `model.save(path, include_optimizer=False)` -- a SILENT NO-OP on the
#     `.keras` path (see the table above). It looks like the fix and is not one.
#   * `keras.models.load_model(path, compile=False)` -- works, but throws the
#     loss and the optimizer away, so the seven `tests/test_losses/` round trips
#     that assert `loaded_model.loss.<param>` would stop testing anything.
#     (It IS the right repair inside `src/train/logic/`'s `roundtrip_check`,
#     which compares predictions only -- that is D-015, a different site with a
#     different contract.)
#   * a throwaway `train_on_batch` -- builds the optimizer, but moves the
#     weights, consumes RNG draws and needs targets, which breaks every
#     round trip that asserts bit-identical outputs.
# Do NOT replace a call site with a bare `pytest.warns(UserWarning)` or a
# `filterwarnings` entry: the warning is TRUE about the sequence the test ran,
# and silencing it would leave the archive still lying about its optimizer.


def build_optimizer_state(model: keras.Model) -> Optional[int]:
    """Allocate a compiled model's optimizer slot variables, in place.

    Call this immediately before ``model.save(...)`` in any test that compiles a
    model but never runs a training step. After it, the saved optimizer state
    and the reloaded one agree, so ``keras.models.load_model`` restores instead
    of warning-and-skipping.

    :param model: A model. It need not be compiled; it need not be built.
    :type model: keras.Model
    :returns: The optimizer's variable count after the call, or ``None`` if the
        model has no optimizer (never compiled, or compiled with
        ``optimizer=None``). ``None`` is not an error: a model with no optimizer
        already round-trips consistently.
    :rtype: Optional[int]
    :raises ValueError: propagated from ``optimizer.build`` if the model has an
        optimizer but no trainable variables -- that means the model was not
        built, which is a different defect (RD-4, ``model.save()`` on an unbuilt
        model) and must not be papered over here.
    """
    optimizer: Any = getattr(model, "optimizer", None)
    if optimizer is None:
        return None
    if not getattr(optimizer, "built", False):
        optimizer.build(model.trainable_variables)
    return len(optimizer.variables)
