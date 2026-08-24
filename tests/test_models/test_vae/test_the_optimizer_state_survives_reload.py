"""``VAE.compile_from_config`` must rebuild the optimizer's slot variables.

D-014 (plan-2026-08-22T035419-a11304c8). ``VAE`` overrides
``compile_from_config`` so the D-009 vmf ``jit_compile=False`` opt-out survives a
reload. The override reproduced Keras' ``Trainer.compile_from_config`` body but
DROPPED its last two lines::

    if hasattr(self, "optimizer") and self.built:
        self.optimizer.build(self.trainable_variables)

Consequence, measured 2026-08-22 on a VAE fitted for one epoch: the archive held
**122** optimizer variables, the reloaded model's optimizer held **2**, and
``BaseOptimizer.load_own_variables`` warned and restored NOTHING. Resuming from
such a checkpoint silently restarts Adam with zeroed moments and a zeroed step
count -- a training-state loss with no error and no shape symptom.

This is the LIBRARY half of RD-2. The other 29 node ids in that family are the
mirror shape (saved 2, reloaded N) and are a property of the test's own
sequence; see ``tests/optimizer_state.py``.
"""

import os
import warnings
from typing import List, Tuple

import keras
import numpy as np

from dl_techniques.models.vae.model import VAE

_OPTIMIZER_SKIP_TEXT = "Skipping variable loading for optimizer"

LATENT_DIM = 4
INPUT_SHAPE = (28, 28, 1)


def _fitted_vae(sampling_type: str = "gaussian") -> Tuple[VAE, np.ndarray]:
    keras.utils.set_random_seed(0)
    vae = VAE(
        latent_dim=LATENT_DIM,
        input_shape=INPUT_SHAPE,
        sampling_type=sampling_type,
    )
    vae.compile(optimizer="adam")
    data = np.random.RandomState(0).rand(4, *INPUT_SHAPE).astype("float32")
    vae.fit(data, epochs=1, verbose=0)
    return vae, data


def _save_reload(vae: VAE, path: str) -> Tuple[keras.Model, List[str]]:
    vae.save(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = keras.models.load_model(path)
    return reloaded, [str(w.message) for w in caught]


def test_the_reloaded_optimizer_has_the_same_variable_count(tmp_path) -> None:
    vae, _ = _fitted_vae()
    n_saved = len(vae.optimizer.variables)

    # Non-vacuity: a never-fitted optimizer has exactly 2 variables, so the
    # assertion below would be satisfiable by a model that was never trained.
    assert n_saved == 2 + 2 * len(vae.trainable_variables) > 2

    reloaded, messages = _save_reload(vae, os.path.join(str(tmp_path), "vae.keras"))

    assert len(reloaded.optimizer.variables) == n_saved
    assert not any(_OPTIMIZER_SKIP_TEXT in m for m in messages), messages


def test_the_reloaded_optimizer_slot_values_are_bit_identical(tmp_path) -> None:
    """The count is necessary, not sufficient: pin the VALUES.

    A `compile_from_config` that built the optimizer and then failed to load into
    it would pass the count assertion with all-zero moments.
    """
    vae, _ = _fitted_vae()
    before = [keras.ops.convert_to_numpy(v) for v in vae.optimizer.variables]
    assert max(float(np.max(np.abs(v))) for v in before) > 0.0

    reloaded, _ = _save_reload(vae, os.path.join(str(tmp_path), "vae_values.keras"))
    after = [keras.ops.convert_to_numpy(v) for v in reloaded.optimizer.variables]

    assert len(after) == len(before)
    for i, (b, a) in enumerate(zip(before, after)):
        assert float(np.max(np.abs(b - a))) == 0.0, i


def test_the_vmf_jit_compile_opt_out_still_survives_the_reload(tmp_path) -> None:
    """The control for D-009, which the D-014 edit must not disturb.

    `VAE.compile_from_config` exists ONLY to force `jit_compile=False` for vmf on
    the reload path. If a future edit deletes the override to "simplify" the
    optimizer build, this fails.
    """
    vae, _ = _fitted_vae(sampling_type="vmf")
    reloaded, _ = _save_reload(vae, os.path.join(str(tmp_path), "vmf.keras"))
    assert reloaded.sampling_type == "vmf"
    assert reloaded.jit_compile is False
