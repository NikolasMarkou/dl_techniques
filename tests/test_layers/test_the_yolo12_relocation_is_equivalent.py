"""Equivalence harness for the yolo12 ``AreaAttention`` / ``AttentionBlock`` relocation.

This module is invariant **I2** of plan `plan-2026-09-01T055648-e6d380a5`: the relocated
``AreaAttention`` (``layers/attention/area_attention.py``) and the relocated block
(``layers/transformers/area_attention_block.py``) must reproduce the *pre-move* classes'
outputs on identical weights and identical inputs, to a tolerance DERIVED from float32
reassociation noise rather than pasted as a literal.

The reference arm is a **pinned pre-move copy** of ``src/dl_techniques/layers/yolo12_blocks.py``
taken from commit ``PINNED_BASE_COMMIT`` with ``git show`` and loaded from a scratch file via
``importlib.util.spec_from_file_location``. It is deliberately NOT loaded by putting the
scratch directory on ``PYTHONPATH``: ``pyproject.toml`` sets ``pythonpath = ["src"]``, which
silently wins, and the harness would then compare the live class against itself and report
``0.0`` forever. ``test_the_pinned_module_is_not_the_live_module`` is the anti-vacuity guard
for exactly that failure, and it asserts a CONTENT difference (the pinned copy predates
``YOLO12_NORM_KWARGS``), not merely a path.

--------------------------------------------------------------------------------
THE SEAM — read this before step 4 / step 6
--------------------------------------------------------------------------------
Two module-level factory functions are the ONLY places that name the "new" arm:

* :func:`build_new_area_attention`      -> repointed by plan step 4
* :func:`build_new_area_attention_block` -> repointed by plan step 6

Each takes the loaded pinned module plus the constructor kwargs and returns an unbuilt
layer. Today both return the *pinned* class, so the harness measures itself and every probe
point must read **exactly 0.0**. To repoint an arm, change only that function's body (each
carries a ``STEP 4``/``STEP 6`` block comment naming the import to use) and flip the matching
``NEW_ARM_IS_RELOCATED_*`` flag to ``True``. Nothing else in this file needs to change.

The two flags are not decoration: while a flag is ``False`` the corresponding equivalence
test demands an EXACT ``0.0`` (self-comparison has no reassociation noise to hide behind);
once it is ``True`` the test demands ``<= _atol(...)`` with ``rtol=0`` and reports the
measured max|delta| in the assertion message so it can be quoted in ``verification.md``.

Weight transfer is by ordered :meth:`set_weights`, so the relocated classes must build their
sub-layers in the SAME order as the pre-move classes. The harness asserts the two weight
SHAPE sequences match before transferring; a reordering shows up there, as a shape mismatch,
rather than as a mysterious numeric delta.

--------------------------------------------------------------------------------
Regime
--------------------------------------------------------------------------------
TF32 is disabled for this module through the shared ``tf32_disabled`` fixture (scoped and
restored -- never an import-time global), and every build and call happens inside
``keras.device(GOLDEN_REFERENCE_DEVICE)``. Both sides of every comparison are pinned to the
same device; pinning only one makes the comparison cross-device, which is a different
measurement.
"""

import hashlib
import importlib.util
import pathlib
import subprocess
from typing import Any, Dict, List, NamedTuple, Sequence

import keras
import numpy as np
import pytest

from tests.conftest import GOLDEN_REFERENCE_DEVICE
from tests.numerics import reassociation_atol

# TF32 truncates the matmul mantissa to 10 bits (~4100x eps_f32), which would swamp both the
# exact-0.0 self-comparison and the derived tolerance. Opt in per module; the fixture
# restores the process-global flag even when a test body raises.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

# ---------------------------------------------------------------------
# The pinned pre-move reference
# ---------------------------------------------------------------------

#: The tree state BEFORE this plan's first edit. `AreaAttention` / `AttentionBlock` /
#: `ConvBlock` as they existed here are the ground truth for invariant I2. A WRONG ref here
#: is the plan's single most dangerous silent failure (it would compare the new class against
#: itself), which is why `test_the_pinned_module_is_not_the_live_module` exists.
PINNED_BASE_COMMIT = "607ffcea9"

_PINNED_REPO_PATH = "src/dl_techniques/layers/yolo12_blocks.py"

#: Loading the scratch copy re-runs `@register_dl_technique`, which OVERWRITES the live
#: entries in `keras.saving.get_custom_objects()`. That is a process-global side effect on
#: every later test in the session, so the dict is snapshotted and restored around the load.
_LIVE_MODULE = "dl_techniques.layers.yolo12_blocks"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _git_show(ref_path: str) -> str:
    """Return the blob content at ``ref_path`` (``<commit>:<path>``) as text.

    A missing object is a MISSING INSTRUMENT, not a failing invariant. This module is a
    permanent member of ``tests/test_layers/``, so it runs in environments the plan that
    wrote it never saw: a ``--depth 1`` clone, an exported source tarball with no ``.git``
    at all, or a tree whose history was rewritten past ``PINNED_BASE_COMMIT``. In every one
    of those the reference arm simply cannot be constructed, and reporting that as an ERROR
    takes all 17 tests in this module down with a ``CalledProcessError`` that says nothing
    about ``AreaAttention``. It is skipped with a named reason instead.

    Note what is NOT softened: once the blob loads, every equivalence assertion is hard, and
    ``test_the_pinned_module_is_not_the_live_module`` still fails loudly if the wrong thing
    was loaded. The skip covers "no reference obtainable", never "reference disagrees".

    :param ref_path: a ``<commit>:<path>`` revision spec.
    :return: the blob content as text.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(_repo_root()), "show", ref_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:  # no `git` executable on PATH
        pytest.skip(
            f"the pinned pre-move reference {ref_path!r} is unobtainable: no `git` "
            "executable on PATH. The relocation equivalence harness needs the base "
            "commit's blob and has no other source for it."
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"the pinned pre-move reference {ref_path!r} is unobtainable "
            f"(git exited {exc.returncode}: {(exc.stderr or '').strip()!r}). This is "
            "expected in a shallow clone, an exported tarball, or after a history "
            "rewrite past PINNED_BASE_COMMIT -- the reference arm cannot be built, so "
            "the equivalence claim is UNMEASURED here, not violated."
        )
    return completed.stdout


@pytest.fixture(scope="module")
def pinned(tmp_path_factory) -> Any:
    """Load the pre-move ``yolo12_blocks`` module from a pinned scratch copy.

    The module is given the spec name of the live module on purpose: the pinned classes'
    ``__module__`` then matches the live ones', so ``register_dl_technique``'s alias
    collision check treats the load as the same definition re-executed (which it is) instead
    of raising. It is NOT inserted into ``sys.modules`` -- the pinned file is self-contained
    and uses absolute imports only, so nothing needs to resolve it by name.

    :param tmp_path_factory: pytest factory for the scratch directory.
    :return: the loaded module object.
    """
    source = _git_show(f"{PINNED_BASE_COMMIT}:{_PINNED_REPO_PATH}")
    scratch = tmp_path_factory.mktemp("pinned_yolo12") / "pinned_yolo12_blocks.py"
    scratch.write_text(source)

    spec = importlib.util.spec_from_file_location(_LIVE_MODULE, scratch)
    module = importlib.util.module_from_spec(spec)

    custom_objects = keras.saving.get_custom_objects()
    snapshot = dict(custom_objects)
    try:
        spec.loader.exec_module(module)
    finally:
        custom_objects.clear()
        custom_objects.update(snapshot)

    module._scratch_path = str(scratch)
    module._pinned_source = source
    return module


# ---------------------------------------------------------------------
# THE SEAM (see the module docstring)
# ---------------------------------------------------------------------

#: Flipped to ``True`` by step 4, together with the body of `build_new_area_attention`.
NEW_ARM_IS_RELOCATED_ATTENTION = True

#: Flipped to ``True`` by step 6, together with the body of `build_new_area_attention_block`.
NEW_ARM_IS_RELOCATED_BLOCK = True


def build_new_area_attention(pinned_module: Any, **kwargs: Any) -> keras.layers.Layer:
    """Construct the "new" ``AreaAttention`` arm -- REPOINTED by step 4.

    Every newly-added knob keeps its default here (``dropout_rate=0.0``,
    ``qk_norm_type=None``, ``probability_type='softmax'``, ``attention_mask=None``,
    ``use_bias=False``) -- I2 is an equivalence claim at the yolo12 constructor arguments
    only.

    ``normalization_kwargs`` is the ONE exception, and it is not a loophole. The pre-move
    ``yolo12_blocks.ConvBlock`` HARDCODED the D-067 pair ``epsilon=1e-3, momentum=0.97``;
    ``standard_blocks.ConvBlock`` reaches the normalization factory, whose defaults are
    ``epsilon=1e-6, momentum=0.99``. Under D-005 that pair keeps exactly ONE home
    (``yolo12_blocks.YOLO12_NORM_KWARGS``) and is threaded to the relocated class as DATA by
    its yolo12 caller, so "the constructor arguments as used by yolo12" INCLUDE it. Passing
    it here is what makes this arm the configuration yolo12 will actually build; omitting it
    would compare a 1e-6 layer against a 1e-3 one and measure the epsilon, not the
    relocation. Measured with it omitted, on this exact probe grid: max|delta|
    1.423168e-02 .. 2.361488e-02 against bounds of 6.47e-05 .. 1.05e-04, i.e. 200x-300x
    over tolerance at every point. The harness SEES the epsilon; threading it is not
    silencing a discriminator, it is aiming the comparison at the configuration under test.

    :param pinned_module: the loaded pre-move module (unused now that the arm is relocated).
    :param kwargs: constructor arguments shared by both arms.
    :return: an unbuilt layer.
    """
    from dl_techniques.layers.attention.area_attention import AreaAttention
    from dl_techniques.layers.yolo12_blocks import YOLO12_NORM_KWARGS

    return AreaAttention(normalization_kwargs=dict(YOLO12_NORM_KWARGS), **kwargs)


def build_new_area_attention_block(pinned_module: Any, **kwargs: Any) -> keras.layers.Layer:
    """Construct the "new" attention-block arm -- REPOINTED by step 6.

    Note the D-006 rename: the relocated class is ``AreaAttentionBlock``, not
    ``AttentionBlock``.

    Every newly-added knob keeps its default here (``use_bias=False``, and every knob the
    attention sub-layer gained in step 4 -- ``dropout_rate=0.0``, ``qk_norm_type=None``,
    ``probability_type='softmax'``, ``attention_mask=None``). I2 is an equivalence claim at
    the yolo12 constructor arguments only.

    ``normalization_kwargs`` is the ONE exception, for exactly the reason spelled out on
    `build_new_area_attention`: the pre-move ``yolo12_blocks.ConvBlock`` HARDCODED the D-067
    pair ``epsilon=1e-3, momentum=0.97`` in the block's ``mlp1``/``mlp2`` as well as inside
    the attention, while ``standard_blocks.ConvBlock`` reaches the normalization factory
    whose defaults are ``epsilon=1e-6, momentum=0.99``. Under D-005 that pair keeps exactly
    ONE home (``yolo12_blocks.YOLO12_NORM_KWARGS``) and is threaded to the relocated block as
    DATA by its yolo12 caller. Measured with it OMITTED, on this exact probe grid: max|delta|
    3.099632e-02 .. 8.306503e-02 against bounds of 1.19e-04 .. 2.98e-04, i.e. 255x-285x over
    tolerance at every point. The harness SEES the epsilon; threading it aims the comparison
    at the configuration under test rather than silencing a discriminator.

    :param pinned_module: the loaded pre-move module (unused now that the arm is relocated).
    :param kwargs: constructor arguments shared by both arms.
    :return: an unbuilt layer.
    """
    from dl_techniques.layers.transformers.area_attention_block import AreaAttentionBlock
    from dl_techniques.layers.yolo12_blocks import YOLO12_NORM_KWARGS

    return AreaAttentionBlock(normalization_kwargs=dict(YOLO12_NORM_KWARGS), **kwargs)


def build_reference_area_attention(pinned_module: Any, **kwargs: Any) -> keras.layers.Layer:
    """Construct the pinned pre-move ``AreaAttention``. NEVER repoint this arm."""
    return pinned_module.AreaAttention(**kwargs)


def build_reference_area_attention_block(pinned_module: Any, **kwargs: Any) -> keras.layers.Layer:
    """Construct the pinned pre-move ``AttentionBlock``. NEVER repoint this arm."""
    return pinned_module.AttentionBlock(**kwargs)


# ---------------------------------------------------------------------
# Probe grid
# ---------------------------------------------------------------------


class Probe(NamedTuple):
    """One point of the equivalence grid.

    :param label: pytest id.
    :param dim: feature width (must be divisible by ``num_heads``).
    :param num_heads: attention heads.
    :param area: number of attention groups; ``1`` means global.
    :param height: input height.
    :param width: input width.
    :param expects_area_branch: whether ``AreaAttention.call`` is expected to take the
        grouped branch (``area > 1 and seq_len % area == 0``) rather than the global one.
    """

    label: str
    dim: int
    num_heads: int
    area: int
    height: int
    width: int
    expects_area_branch: bool


BATCH = 2

PROBES: Sequence[Probe] = (
    # area=1 -> the global branch, one head.
    Probe("global_h1_6x8", dim=16, num_heads=1, area=1, height=6, width=8, expects_area_branch=False),
    # area=1 -> the global branch, eight heads, non-square input.
    Probe("global_h8_6x8", dim=16, num_heads=8, area=1, height=6, width=8, expects_area_branch=False),
    # area=4, seq_len=64 divisible by 4 -> the grouped branch, square input.
    Probe("area4_h8_8x8", dim=16, num_heads=8, area=4, height=8, width=8, expects_area_branch=True),
    # area=4, seq_len=48 divisible by 4 -> the grouped branch, non-square input, one head.
    Probe("area4_h1_6x8", dim=16, num_heads=1, area=4, height=6, width=8, expects_area_branch=True),
    # area=4 but seq_len=35 is NOT divisible by 4 -> silently falls back to the global
    # branch. This is a real branch of the pre-move `call`, and a relocation that "fixed"
    # the fallback would be a behaviour change, not a refactor.
    Probe("area4_fallback_h8_5x7", dim=16, num_heads=8, area=4, height=5, width=7, expects_area_branch=False),
    # ------------------------------------------------------------------
    # PRODUCTION head_dim. The five probes above all use dim=16, giving head_dim 16 or 2 --
    # neither of which any real yolo12 ever builds. Every `AreaAttention` in every scale has
    # head_dim EXACTLY 32 (measured: n `[(32,1,32),(64,2,32),(128,4,32)]` ... x
    # `[(192,6,32),(384,12,32),(768,24,32)]`), with `area` 4 at the first A2C2f stage and 1
    # at the rest. This matters because the step-4 scale substitution
    # (`common.compute_attention_scale`, a Python `math.sqrt`) is NOT bit-identical to the
    # pre-move `1/ops.sqrt(cast(d,'float32'))` spelling at every head_dim: they differ by
    # 1 ulp at head_dim 24, 28 and 96. head_dim 32 is one of the bit-identical ones, but
    # that is a MEASURED fact about 32, not a property of the substitution, and a grid that
    # never visits 32 cannot claim it. See verification.md § "Step 9.1 -- W4".
    # ------------------------------------------------------------------
    Probe("prod_global_h2_8x8_hd32", dim=64, num_heads=2, area=1, height=8, width=8, expects_area_branch=False),
    Probe("prod_area4_h2_8x8_hd32", dim=64, num_heads=2, area=4, height=8, width=8, expects_area_branch=True),
)

MLP_RATIO = 1.2

PE_KERNEL_ELEMENTS = 25  # the 5x5 depthwise positional-encoding conv


def _attention_reduction_lengths(probe: Probe) -> List[int]:
    """Contraction lengths along ``AreaAttention``'s compared path, in order.

    ``qk`` 1x1 (over the input channels), ``v`` 1x1, the 5x5 depthwise positional encoding,
    the score matmul (over ``head_dim``), the attention-weighted value matmul (over the
    attended sequence length), and the output projection 1x1.

    :param probe: the grid point.
    :return: one contraction length per reduction on the path.
    """
    seq_len = probe.height * probe.width
    attended = seq_len // probe.area if probe.expects_area_branch else seq_len
    return [
        probe.dim,                     # qk conv 1x1
        probe.dim,                     # v conv 1x1
        PE_KERNEL_ELEMENTS,            # pe depthwise 5x5
        probe.dim // probe.num_heads,  # scores  = q . k^T
        attended,                      # output  = softmax(scores) . v
        probe.dim,                     # proj conv 1x1
    ]


def _block_reduction_lengths(probe: Probe) -> List[int]:
    """``_attention_reduction_lengths`` plus the block's two MLP 1x1 contractions."""
    hidden = int(probe.dim * MLP_RATIO)
    return _attention_reduction_lengths(probe) + [probe.dim, hidden]


def _atol(reduction_lengths: Sequence[int], scale: float) -> float:
    """Pre-registered bound. ``rtol=0`` at every call site; see ``tests/numerics.py``."""
    return reassociation_atol(reduction_lengths, num_steps=1, scale=scale)


# ---------------------------------------------------------------------
# Comparison machinery
# ---------------------------------------------------------------------


def _probe_seed(probe: Probe) -> int:
    """Per-probe seed derived from the label, so two grid points never share a draw and a
    copy-pasted probe cannot silently duplicate another one's measurement."""
    return int.from_bytes(hashlib.sha256(probe.label.encode()).digest()[:4], "big")


def _probe_input(probe: Probe) -> np.ndarray:
    """Fixed-seed input for one grid point."""
    rng = np.random.default_rng(_probe_seed(probe))
    return rng.standard_normal((BATCH, probe.height, probe.width, probe.dim)).astype(np.float32)


def _build_and_transfer(reference: keras.layers.Layer, other: keras.layers.Layer, x: np.ndarray) -> None:
    """Build both layers on ``x`` and copy ``reference``'s weights into ``other``, in order.

    :raises AssertionError: if the two ordered weight-shape sequences differ, which is what a
        sub-layer creation-order change looks like.
    """
    reference(x, training=False)
    other(x, training=False)

    ref_weights = reference.get_weights()
    other_weights = other.get_weights()
    assert [w.shape for w in ref_weights] == [w.shape for w in other_weights], (
        "ordered weight-shape sequences differ, so `set_weights` would transfer the wrong "
        f"tensors: reference={[w.shape for w in ref_weights]} "
        f"other={[w.shape for w in other_weights]}"
    )
    other.set_weights(ref_weights)


def _max_abs_delta(
    reference: keras.layers.Layer,
    other: keras.layers.Layer,
    x: np.ndarray,
) -> float:
    """Return ``max|reference(x) - other(x)|`` with both sides on the pinned device."""
    expected = np.asarray(reference(x, training=False))
    measured = np.asarray(other(x, training=False))
    assert expected.shape == measured.shape, (
        f"output shapes differ: {expected.shape} vs {measured.shape}"
    )
    return float(np.max(np.abs(expected - measured)))


def _run_arm(
    pinned_module: Any,
    probe: Probe,
    reference_factory,
    new_factory,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, float]:
    """Build the reference and new arms, transfer weights, and measure the delta."""
    kwargs = dict(
        dim=probe.dim,
        num_heads=probe.num_heads,
        area=probe.area,
        **extra_kwargs,
    )
    x = _probe_input(probe)
    with keras.device(GOLDEN_REFERENCE_DEVICE):
        # Seed the REFERENCE's weight draw. The equivalence reading itself does not need
        # this -- weights are transferred, so the delta is 0.0 either way -- but `scale`
        # (and therefore the derived bound, and the max|delta| quoted into
        # `verification.md`) is a function of the drawn weights. Unseeded, those numbers
        # move on every run and cannot be compared across steps 3/4/6. The autouse
        # `_restore_process_global_rng_state` fixture puts the process-global streams back.
        keras.utils.set_random_seed(_probe_seed(probe) % (2**31))
        reference = reference_factory(pinned_module, name="reference", **kwargs)
        new = new_factory(pinned_module, name="new", **kwargs)
        _build_and_transfer(reference, new, x)
        delta = _max_abs_delta(reference, new, x)
        scale = float(np.max(np.abs(np.asarray(reference(x, training=False)))))
    return {"delta": delta, "scale": scale}


# ---------------------------------------------------------------------
# The planted mutation (anti-vacuity control)
# ---------------------------------------------------------------------


class _ScoreTransposeMutation(RuntimeError):
    """Never raised. Exists only to make the mutation greppable by name."""


def _make_mutant(pinned_module: Any):
    """Return a subclass of the pinned ``AreaAttention`` whose score matmul is transposed.

    The mutation is applied to exactly ONE arm. An injection that moves both sides proves
    nothing, and a control that is never observed to have RUN proves nothing either -- hence
    ``calls``, which the control test asserts is non-zero.
    """

    class _MutatedAreaAttention(pinned_module.AreaAttention):
        calls = 0

        def _compute_attention(self, q, k, v):
            batch_size = keras.ops.shape(q)[0]
            num_areas = keras.ops.shape(q)[1]
            seq_len = keras.ops.shape(q)[-2]
            shape = (batch_size, num_areas, seq_len, self.num_heads, self.head_dim)
            q = keras.ops.transpose(keras.ops.reshape(q, shape), (0, 1, 3, 2, 4))
            k = keras.ops.transpose(keras.ops.reshape(k, shape), (0, 1, 3, 2, 4))
            v = keras.ops.transpose(keras.ops.reshape(v, shape), (0, 1, 3, 2, 4))

            scale = keras.ops.cast(
                1.0 / keras.ops.sqrt(keras.ops.cast(self.head_dim, "float32")), q.dtype
            )
            scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 2, 4, 3))) * scale
            # --- THE PLANTED MUTATION: transpose the score matrix's last two axes. ---
            scores = keras.ops.transpose(scores, (0, 1, 2, 4, 3))
            type(self).calls += 1

            attn_weights = keras.ops.nn.softmax(scores, axis=-1)
            attn_output = keras.ops.matmul(attn_weights, v)
            attn_output = keras.ops.transpose(attn_output, (0, 1, 3, 2, 4))
            return keras.ops.reshape(attn_output, (batch_size, num_areas, seq_len, self.dim))

    return _MutatedAreaAttention


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestTheHarnessIsMeasuringWhatItClaims:
    """Guards on the INSTRUMENT. If any of these fail, no verdict below means anything."""

    def test_the_pinned_module_is_not_the_live_module(self, pinned) -> None:
        live = importlib.import_module(_LIVE_MODULE)

        # 1. It came from the scratch file, not from `src/`.
        assert pinned.__file__ == pinned._scratch_path
        assert "src/dl_techniques" not in pinned.__file__

        # 2. CONTENT, not just path: the pinned copy predates step 2, so it does NOT carry
        #    `YOLO12_NORM_KWARGS` -- while the live module, by construction, does. A harness
        #    that had silently loaded the live module would fail here.
        assert hasattr(live, "YOLO12_NORM_KWARGS"), (
            "the live module lost YOLO12_NORM_KWARGS -- this guard's discriminator is gone "
            "and must be replaced before trusting any equivalence result"
        )
        assert not hasattr(pinned, "YOLO12_NORM_KWARGS")

        # 3. The pinned source text differs from the current worktree file.
        worktree = (_repo_root() / _PINNED_REPO_PATH).read_text()
        assert pinned._pinned_source != worktree

        # 4. The pinned classes are distinct objects that build their OWN sub-layers, i.e.
        #    the pre-move `ConvBlock` defined in the same pinned file.
        assert pinned.ConvBlock is not getattr(live, "ConvBlock", None)
        assert pinned.AreaAttention.__init__.__globals__["ConvBlock"] is pinned.ConvBlock

    def test_the_probe_grid_covers_both_branches(self) -> None:
        for probe in PROBES:
            seq_len = probe.height * probe.width
            engaged = probe.area > 1 and seq_len % probe.area == 0
            assert engaged == probe.expects_area_branch, (
                f"{probe.label}: declared expects_area_branch={probe.expects_area_branch} "
                f"but area={probe.area}, seq_len={seq_len}"
            )
            assert probe.dim % probe.num_heads == 0

        assert any(p.expects_area_branch for p in PROBES), "no probe reaches the area branch"
        assert any(not p.expects_area_branch for p in PROBES), "no probe reaches the global branch"
        assert {p.num_heads for p in PROBES} >= {1, 8}
        assert any(p.height != p.width for p in PROBES), "no non-square probe"
        assert len({p.label for p in PROBES}) == len(PROBES)

        # PRODUCTION COVERAGE. Every `AreaAttention` yolo12 actually builds, at every
        # scale, has head_dim 32 -- and the grid spent its first five points at head_dim
        # 16 and 2. A grid that never visits the shipped shape can only claim equivalence
        # for shapes nobody runs, which is the "probe design can manufacture the effect"
        # trap. Both branches must be reachable AT head_dim 32, because the grouped branch
        # is what the first A2C2f stage (`area=4`) takes.
        head_dims = {p.dim // p.num_heads for p in PROBES}
        assert 32 in head_dims, (
            f"the probe grid visits head_dims {sorted(head_dims)} but production yolo12 "
            "builds head_dim 32 everywhere -- add a probe at the shipped shape"
        )
        prod = [p for p in PROBES if p.dim // p.num_heads == 32]
        assert any(p.expects_area_branch for p in prod), "no head_dim-32 probe reaches the area branch"
        assert any(not p.expects_area_branch for p in prod), "no head_dim-32 probe reaches the global branch"


class TestAreaAttentionEquivalence:
    """I2 for ``AreaAttention``. Repointed at the relocated class by step 4."""

    @pytest.mark.parametrize("probe", PROBES, ids=[p.label for p in PROBES])
    def test_the_new_arm_reproduces_the_pinned_reference(self, pinned, probe) -> None:
        result = _run_arm(
            pinned,
            probe,
            build_reference_area_attention,
            build_new_area_attention,
            extra_kwargs={},
        )
        delta, scale = result["delta"], result["scale"]

        if not NEW_ARM_IS_RELOCATED_ATTENTION:
            assert delta == 0.0, (
                f"{probe.label}: both arms are the SAME pinned class, so the only possible "
                f"reading is exactly 0.0; measured {delta!r} (scale {scale!r}). The harness "
                "is non-deterministic and nothing downstream can be trusted."
            )
            return

        bound = _atol(_attention_reduction_lengths(probe), scale)
        assert delta <= bound, (
            f"{probe.label}: max|delta| {delta!r} exceeds the pre-registered bound {bound!r} "
            f"(scale {scale!r}, rtol=0)"
        )


class TestAreaAttentionBlockEquivalence:
    """I2 for the transformer block. Repointed at ``AreaAttentionBlock`` by step 6."""

    @pytest.mark.parametrize("probe", PROBES, ids=[p.label for p in PROBES])
    def test_the_new_arm_reproduces_the_pinned_reference(self, pinned, probe) -> None:
        result = _run_arm(
            pinned,
            probe,
            build_reference_area_attention_block,
            build_new_area_attention_block,
            extra_kwargs={"mlp_ratio": MLP_RATIO},
        )
        delta, scale = result["delta"], result["scale"]

        if not NEW_ARM_IS_RELOCATED_BLOCK:
            assert delta == 0.0, (
                f"{probe.label}: both arms are the SAME pinned class, so the only possible "
                f"reading is exactly 0.0; measured {delta!r} (scale {scale!r})."
            )
            return

        bound = _atol(_block_reduction_lengths(probe), scale)
        assert delta <= bound, (
            f"{probe.label}: max|delta| {delta!r} exceeds the pre-registered bound {bound!r} "
            f"(scale {scale!r}, rtol=0)"
        )


class TestThePlantedMutationIsDetected:
    """Anti-vacuity control: the harness must be able to go RED.

    A comparison that has never been observed failing is not a measurement. The planted
    mutation transposes the score matmul's last two axes in exactly ONE arm; the same
    build/transfer/compare machinery the equivalence tests use must report a delta well
    above the pre-registered bound at EVERY probe point.
    """

    @pytest.mark.parametrize("probe", PROBES, ids=[p.label for p in PROBES])
    def test_the_score_transpose_mutation_moves_the_number(self, pinned, probe) -> None:
        mutant_cls = _make_mutant(pinned)
        before = mutant_cls.calls

        def _mutant_factory(pinned_module, **kwargs):
            return mutant_cls(**kwargs)

        result = _run_arm(
            pinned,
            probe,
            build_reference_area_attention,
            _mutant_factory,
            extra_kwargs={},
        )
        delta, scale = result["delta"], result["scale"]

        assert mutant_cls.calls > before, (
            f"{probe.label}: the mutated `_compute_attention` never ran, so this RED reading "
            "would prove nothing about the mutation"
        )

        bound = _atol(_attention_reduction_lengths(probe), scale)
        assert delta > bound, (
            f"{probe.label}: the planted score-transpose mutation moved max|delta| only to "
            f"{delta!r}, which is within the pre-registered bound {bound!r} -- the harness "
            "cannot see the defect it exists to catch"
        )
        assert delta > 0.0
