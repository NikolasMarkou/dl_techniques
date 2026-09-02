"""Shared instruments for the ``dit`` suite. Not collected (leading ``_``).

**Why this module exists at all.** A freshly built :class:`DiT` predicts the
EXACT zero tensor: every block's ``adaln/linear`` is zero in kernel and bias (so
all six modulation chunks are 0 and each block is an exact identity) and the
final layer's read-out projection is zero in kernel and bias too. That is
correct adaLN-Zero behaviour, it is what makes a 28-block stack trainable, and
it is a trap for every output-level instrument in the tree:

* an assertion of the form *"these two outputs differ"* is vacuously
  UNSATISFIABLE at initialisation, so a live knob reads dead;
* an assertion of the form *"these two outputs agree"* is vacuously TRUE, so a
  broken path reads healthy;
* a gradient oracle driven by a mean-of-squares loss on the output sits at a
  stationary point -- measured, ``39 of 39`` trainable tensors read dead and an
  optimizer step cannot move the model out of it.

:func:`activate` removes the first two hazards; :func:`ddpm_training_batch`
plus a real :class:`~dl_techniques.losses.ddpm_hybrid_loss.DDPMHybridLoss`
removes the third.

A second, subtler hazard this module also owns: a channel-uniform perturbation
of the token stream is ANNIHILATED by the ``LayerNormalization`` at the head of
every block and of the final layer. Any probe that perturbs a tensor by adding
the same scalar to every channel therefore measures exactly ``0.0`` and proves
nothing about the path it thought it was testing. Perturb per-channel.

The configuration :data:`TINY` and the input helpers were MOVED here from
``test_dit_model.py`` when this module gained a second consumer -- one home for
the geometry the whole directory shares, rather than two dicts kept equal by
hand.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.model import DiT

# ---------------------------------------------------------------------
# The geometry every arm in this directory shares
# ---------------------------------------------------------------------

#: The smallest DiT that still has a real block stack, a 4x4 token grid, a
#: multi-head attention and a null label row. Every field is stated: a helper
#: that inherited a constructor default would stop describing the model the
#: moment the default moved.
TINY: Dict[str, Any] = {
    "input_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "hidden_size": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "class_dropout_rate": 0.1,
    "num_classes": 10,
    "learn_sigma": True,
    "frequency_embedding_size": 16,
}

#: Default batch size for the helpers below.
BATCH: int = 4


def np_(x: Any) -> np.ndarray:
    """Convert any backend tensor to a NumPy array.

    Interface contract: pure, accepts anything ``keras.ops`` can convert
    (including a plain ``np.ndarray``), never mutates its argument.

    :param x: A backend tensor or array.
    :type x: Any
    :return: The equivalent NumPy array.
    :rtype: np.ndarray
    """
    return np.asarray(keras.ops.convert_to_numpy(x))


def dit_config(**overrides: Any) -> Dict[str, Any]:
    """:data:`TINY` with per-call overrides, as a fresh dict.

    Interface contract: returns a NEW dict every call; :data:`TINY` itself is
    never mutated, so a caller that edits the result cannot poison the rest of
    the session.

    :param overrides: Constructor kwargs replacing the corresponding
        :data:`TINY` entries.
    :type overrides: Any
    :return: The merged configuration.
    :rtype: Dict[str, Any]
    """
    config = dict(TINY)
    config.update(overrides)
    return config


def tiny_model(**overrides: Any) -> DiT:
    """Construct (but do NOT build) the shared tiny :class:`DiT`.

    Interface contract: the returned model is UNBUILT -- a subclassed
    ``keras.Model`` has zero weights until its first ``call()``, and a helper
    that hid a warm-up forward pass inside itself would make every
    ``len(model.weights)`` reading in the callers silently depend on it. Callers
    that need weights call the model, or :func:`built_model`.

    :param overrides: Constructor kwargs replacing the :data:`TINY` entries.
    :type overrides: Any
    :return: A fresh, unbuilt model.
    :rtype: DiT
    """
    return DiT(**dit_config(**overrides))


def tiny_inputs(
    seed: int = 0,
    batch: int = BATCH,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deterministic ``(x, t, y)`` triple for a configuration's geometry.

    Interface contract: draws from a LOCAL ``np.random.default_rng(seed)``, so
    the result is independent of the process-global Keras/NumPy RNG and
    therefore of pytest collection order. ``t`` is float32 (the model casts) and
    ``y`` is int32 in ``[0, num_classes)`` -- never the null row, which only
    :meth:`DiT.forward_with_cfg` and the sampler are entitled to index.

    :param seed: Seed for the local generator.
    :type seed: int
    :param batch: Number of samples.
    :type batch: int
    :param config: Geometry to match. Defaults to :data:`TINY`.
    :type config: Optional[Dict[str, Any]]
    :return: ``(x, t, y)`` as NumPy arrays.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    cfg = config or TINY
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(
        (batch, cfg["input_size"], cfg["input_size"], cfg["in_channels"])
    ).astype("float32")
    t = rng.integers(0, 1000, size=(batch,)).astype("float32")
    y = rng.integers(0, cfg["num_classes"], size=(batch,)).astype("int32")
    return x, t, y


def built_model(seed: int = 0, **overrides: Any) -> DiT:
    """A tiny :class:`DiT` seeded, constructed and warmed to build its weights.

    Interface contract: seeds the process-global RNG with
    ``keras.utils.set_random_seed(seed)`` BEFORE construction, so two calls with
    the same seed and the same overrides hold bit-identical weights. It does not
    restore the previous global state itself; ``tests/conftest.py``'s autouse
    ``_restore_process_global_rng_state`` fixture does that per test, which is
    why a helper here is allowed to seed at all.

    :param seed: Global seed applied before construction.
    :type seed: int
    :param overrides: Constructor kwargs replacing the :data:`TINY` entries.
    :type overrides: Any
    :return: A BUILT model.
    :rtype: DiT
    """
    config = dit_config(**overrides)
    keras.utils.set_random_seed(seed)
    model = DiT(**config)
    model(list(tiny_inputs(seed=seed, config=config)), training=False)
    return model


def activate(model: keras.Model, seed: int = 5) -> keras.Model:
    """Replace every all-zero TRAINABLE weight with a random one, in place.

    Interface contract: mutates ``model`` and returns the same object. The
    replacements come from a local ``np.random.default_rng(seed)``, so two
    models of the same weight-SHAPE signature receive bit-identical
    replacements and a downstream difference is attributable to the knob under
    test rather than to the draw. Non-trainable weights -- the frozen sin-cos
    positional table and the timestep frequency ladder -- are NEVER touched: the
    whole point of those is that they hold a specific computed value.

    The model must already be BUILT; an unbuilt subclassed ``keras.Model`` has
    no weights and this is then a silent no-op.

    :param model: A built model.
    :type model: keras.Model
    :param seed: Seed for the replacement generator.
    :type seed: int
    :return: The same model, mutated.
    :rtype: keras.Model
    :raises ValueError: If the model has no weights (i.e. is unbuilt).
    """
    if not model.weights:
        raise ValueError(
            f"{type(model).__name__} has no weights -- activate() on an "
            "unbuilt model is a silent no-op, and every 'the output changed' "
            "claim downstream would be made about the zero tensor"
        )
    rng = np.random.default_rng(seed)
    for weight in model.weights:
        if not weight.trainable:
            continue
        value = np_(weight)
        if np.any(value != 0.0):
            continue
        weight.assign(rng.normal(scale=0.3, size=value.shape).astype(value.dtype))
    return model


def relative_paths(model: keras.Model) -> List[str]:
    """Sorted weight paths with the ROOT segment stripped.

    Interface contract: pure. Keras auto-increments the root name per instance
    (``di_t``, then ``di_t_1``, ...), so two separately constructed models are
    never comparable on the full ``w.path``. Only the first segment is dropped;
    every deeper segment is compared verbatim, which is what makes a
    Keras-generated intermediate name (``dense_3``) a parity FAILURE rather than
    something silently normalized away.

    :param model: A built model or layer.
    :type model: keras.Model
    :return: The sorted relative paths.
    :rtype: List[str]
    """
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def ddpm_training_batch(
    model: DiT,
    loss: DDPMHybridLoss,
    batch: int = BATCH,
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build the ``([x_t, t, y], y_true)`` pair the real objective consumes.

    Interface contract: the forward process is run with ``loss.schedule``'s own
    tables, so ``x_t`` is exactly the ``q_sample`` the loss re-derives
    internally. ``y_true`` is the ``2C + 1`` channel pack the loss's contract
    declares -- ``[0:C]`` noise, ``[C:2C]`` ``x_start``, ``[2C:2C+1]`` the
    per-sample ``t`` broadcast over ``(H, W)``. Everything is drawn from a local
    generator; nothing here touches the global RNG.

    ``model.learn_sigma`` must be ``True``: the loss reads a variance logit out
    of the second channel half and a ``C``-wide prediction cannot supply one.

    :param model: The model whose geometry the batch must match.
    :type model: DiT
    :param loss: The compiled objective; its ``schedule`` drives ``q_sample``.
    :type loss: DDPMHybridLoss
    :param batch: Number of samples.
    :type batch: int
    :param seed: Seed for the local generator.
    :type seed: int
    :return: ``([x_t, t, y], y_true)``.
    :rtype: Tuple[List[np.ndarray], np.ndarray]
    :raises ValueError: If the model does not emit ``2 * in_channels`` channels,
        or its channel count disagrees with the loss's.
    """
    if not model.learn_sigma:
        raise ValueError(
            "DDPMHybridLoss needs a 2*C-wide prediction; this model was built "
            "with learn_sigma=False and emits C"
        )
    if model.in_channels != loss.in_channels:
        raise ValueError(
            f"model.in_channels={model.in_channels} but "
            f"loss.in_channels={loss.in_channels}; the channel split would be "
            "silently misaligned"
        )

    schedule = loss.schedule
    n, c = model.input_size, model.in_channels
    rng = np.random.default_rng(seed)

    x_start = rng.normal(size=(batch, n, n, c)).astype("float32")
    noise = rng.normal(size=(batch, n, n, c)).astype("float32")
    t = rng.integers(0, schedule.num_timesteps, size=(batch,)).astype("int32")
    y = rng.integers(0, model.num_classes, size=(batch,)).astype("int32")

    alpha = schedule.sqrt_alphas_cumprod[t][:, None, None, None]
    sigma = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    x_t = (alpha * x_start + sigma * noise).astype("float32")

    t_plane = np.broadcast_to(
        t[:, None, None, None].astype("float32"), (batch, n, n, 1)
    )
    y_true = np.concatenate([noise, x_start, t_plane], axis=-1).astype("float32")
    return [x_t, t.astype("float32"), y], y_true
