"""The whole-model base-vs-HEAD equivalence claim of `plan-2026-09-01T055648-e6d380a5`, as a test.

The claim
---------
That plan deleted three public classes from ``layers/yolo12_blocks.py`` (``ConvBlock``,
``AreaAttention``, ``AttentionBlock``), relocated two of them into ``layers/attention/`` and
``layers/transformers/``, widened the shared ``standard_blocks.ConvBlock``, and rerouted 20
construction sites through ``yolo12_conv_block``. Its single strongest safety claim (D-020)
is that the *assembled model* is unchanged: **682 weights, 2,749,372 parameters, and
``max|delta| == 0.0`` between the pre-plan commit and HEAD on identical weights and inputs.**

That claim was measured once, by an uncommitted one-off script, and existed only as prose in
``progress.md``. An iteration-1 adversarial review called that out: a headline claim nobody
can re-run is not a guard. This file is the fix -- the claim now re-derives on every run.

How the reference arm is obtained
---------------------------------
``git worktree add --detach`` materialises the *whole source tree* at
:data:`PLAN_BASE_COMMIT` and a SUBPROCESS builds the model there with
``PYTHONPATH`` pointed at that tree's ``src``. A subprocess is not fastidiousness: the base
and HEAD trees define the same dotted module names, so importing both into one interpreter
would have ``sys.modules`` silently serve whichever won the race, and the harness would
compare a tree against itself and report ``0.0`` forever. That failure is invisible by
construction, which is why :meth:`TestTheReferenceArmIsReallyThePinnedTree` proves the base
arm by CONTENT -- the base tree predates ``YOLO12_NORM_KWARGS`` and does not define it --
rather than by the path it was loaded from. (The same discipline, and the same
discriminator, as ``tests/test_layers/test_the_yolo12_relocation_is_equivalent.py``, which
covers the two relocated layers in isolation. This file covers the assembled model.)

Both arms are pinned to CPU: the subprocess by ``CUDA_VISIBLE_DEVICES=""`` and this process
by ``keras.device(GOLDEN_REFERENCE_DEVICE)``. Pinning one side only turns the measurement
into a cross-device comparison, which is a different -- and much noisier -- question.

Scope, stated honestly
----------------------
* One configuration: ``scale='n'``, ``tasks=['detection']``, ``input_shape=(64, 64, 3)``.
  The other four scales and the segmentation/classification heads are NOT compared
  numerically here (a whole-model census of all of them is in the plan's
  ``verification.md``); building ``scale='x'`` costs 132M parameters per arm.
* ``training=False`` only. BatchNorm's batch-statistics path and its moving-average updates
  (and therefore ``momentum``) are outside the ``0.0`` claim entirely. ``momentum`` is
  guarded statically instead, by :class:`TestTheD067PairSurvivesIntoTheAssembledModel` below
  and by ``tests/test_layers/test_the_yolo12_conv_block_contract_is_pinned.py``.

If the base commit is unreachable -- a ``--depth 1`` clone, an exported tarball, a rewritten
history -- the reference arm cannot be built and these tests SKIP with a named reason. A
missing instrument is not a violated invariant.
"""

import collections
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from tests.conftest import GOLDEN_REFERENCE_DEVICE

#: The tree state immediately BEFORE the plan's first edit.
PLAN_BASE_COMMIT = "607ffcea9"

#: The measured facts this file pins. Both were re-derived at HEAD, not copied.
EXPECTED_PARAMS = 2749372
EXPECTED_WEIGHTS = 682

#: The one configuration compared numerically. Named once so both arms cannot drift apart.
MODEL_KWARGS: Dict[str, Any] = {
    "num_detection_classes": 4,
    "tasks": ["detection"],
    "input_shape": (64, 64, 3),
    "scale": "n",
}

BATCH = 2

#: Run in the base worktree. Writes weights + outputs to an ``.npz`` and a one-line JSON
#: provenance record to stdout. It reports where it imported ``dl_techniques`` FROM and
#: whether that module carries the post-plan ``YOLO12_NORM_KWARGS``, so the parent can prove
#: the arm rather than assume it.
_BASE_ARM_SOURCE = textwrap.dedent(
    '''
    import json, sys
    import numpy as np
    import keras

    out_npz, in_npy, kwargs_json = sys.argv[1], sys.argv[2], sys.argv[3]

    import dl_techniques
    import dl_techniques.layers.yolo12_blocks as yolo12_blocks
    from dl_techniques.models.vision.yolo12 import create_yolov12_multitask

    kwargs = json.loads(kwargs_json)
    kwargs["input_shape"] = tuple(kwargs["input_shape"])

    keras.utils.set_random_seed(0)
    with keras.device("cpu"):
        model = create_yolov12_multitask(**kwargs)
        x = np.load(in_npy)
        outputs = keras.tree.flatten(model(x, training=False))
        outputs = [np.asarray(keras.ops.convert_to_numpy(t)) for t in outputs]
        weights = model.get_weights()

    np.savez(
        out_npz,
        **{"w%d" % i: w for i, w in enumerate(weights)},
        **{"o%d" % i: o for i, o in enumerate(outputs)},
    )
    sys.stdout.write("<<<META>>>" + json.dumps({
        "package_file": dl_techniques.__file__,
        "yolo12_blocks_file": yolo12_blocks.__file__,
        "has_YOLO12_NORM_KWARGS": hasattr(yolo12_blocks, "YOLO12_NORM_KWARGS"),
        "n_weights": len(weights),
        "n_outputs": len(outputs),
        "params": int(model.count_params()),
    }))
    '''
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


class BaseArm:
    """Everything the base worktree produced, plus its own provenance record."""

    def __init__(self, meta: Dict[str, Any], weights: List[np.ndarray],
                 outputs: List[np.ndarray], tree: pathlib.Path) -> None:
        self.meta = meta
        self.weights = weights
        self.outputs = outputs
        self.tree = tree


@pytest.fixture(scope="module")
def probe_input(tmp_path_factory) -> np.ndarray:
    """One fixed NHWC batch, shared by both arms. Seeded, never regenerated."""
    return np.random.default_rng(0).standard_normal(
        (BATCH, *MODEL_KWARGS["input_shape"])).astype("float32")


@pytest.fixture(scope="module")
def base_arm(tmp_path_factory, probe_input) -> BaseArm:
    """Build the model at :data:`PLAN_BASE_COMMIT` in a detached worktree, out of process."""
    root = tmp_path_factory.mktemp("yolo12_base_arm")
    tree = root / "base_tree"

    try:
        subprocess.run(
            ["git", "-C", str(_repo_root()), "worktree", "add", "--detach",
             str(tree), PLAN_BASE_COMMIT],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        pytest.skip(
            "the pre-plan reference tree is unobtainable: no `git` executable on PATH. "
            "This whole-model equivalence check needs the base commit's ENTIRE source "
            "tree and has no other source for it."
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"the pre-plan reference tree at {PLAN_BASE_COMMIT} is unobtainable "
            f"(git worktree exited {exc.returncode}: {(exc.stderr or '').strip()!r}). "
            "Expected in a shallow clone, an exported tarball, or after a history "
            "rewrite past the base commit -- the equivalence claim is UNMEASURED here, "
            "not violated."
        )

    try:
        script = root / "base_arm.py"
        script.write_text(_BASE_ARM_SOURCE)
        npz = root / "base.npz"
        npy = root / "x.npy"
        np.save(npy, probe_input)

        env = dict(os.environ)
        # PYTHONPATH is placed ahead of the editable-install .pth entry that points at the
        # LIVE src, so the base tree wins the import. The META assertions below verify that
        # it actually did; this is a preference, not a guarantee.
        env["PYTHONPATH"] = str(tree / "src")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["MPLBACKEND"] = "Agg"

        kwargs = dict(MODEL_KWARGS)
        kwargs["input_shape"] = list(kwargs["input_shape"])
        completed = subprocess.run(
            [sys.executable, str(script), str(npz), str(npy), json.dumps(kwargs)],
            capture_output=True, text=True, cwd=str(tree), env=env,
        )
        if completed.returncode != 0 or "<<<META>>>" not in completed.stdout:
            pytest.skip(
                f"the base arm at {PLAN_BASE_COMMIT} could not be built "
                f"(exit {completed.returncode}). This measures the OLD tree, whose "
                "dependencies are not this test's to repair. stderr tail: "
                f"{(completed.stderr or '')[-600:]!r}"
            )

        meta = json.loads(completed.stdout.split("<<<META>>>", 1)[1])
        data = np.load(npz)
        weights = [data[f"w{i}"] for i in range(meta["n_weights"])]
        outputs = [data[f"o{i}"] for i in range(meta["n_outputs"])]
        yield BaseArm(meta, weights, outputs, tree)
    finally:
        subprocess.run(
            ["git", "-C", str(_repo_root()), "worktree", "remove", "--force", str(tree)],
            capture_output=True, text=True,
        )


def _build_head():
    """Build the model at HEAD under the same seed the base arm used."""
    from dl_techniques.models.vision.yolo12 import create_yolov12_multitask

    keras.utils.set_random_seed(0)
    return create_yolov12_multitask(**MODEL_KWARGS)


class TestTheReferenceArmIsReallyThePinnedTree:
    """Anti-vacuity. Without these, a self-comparison would report 0.0 forever."""

    def test_the_base_arm_imported_the_worktree_not_the_live_src(self, base_arm) -> None:
        package_file = pathlib.Path(base_arm.meta["package_file"]).resolve()
        assert base_arm.tree.resolve() in package_file.parents, (
            f"the base arm imported dl_techniques from {package_file}, which is not "
            f"inside the pinned worktree {base_arm.tree}. It measured the LIVE tree "
            "against itself; every 0.0 below would be meaningless."
        )
        assert _repo_root() not in package_file.parents

    def test_the_base_tree_predates_the_plan_by_content_not_by_path(self, base_arm) -> None:
        """A path check can be satisfied by a copy of the CURRENT tree. Content cannot."""
        from dl_techniques.layers import yolo12_blocks as live

        assert hasattr(live, "YOLO12_NORM_KWARGS"), (
            "the LIVE module lost YOLO12_NORM_KWARGS -- this guard's discriminator is "
            "gone and must be replaced before trusting any equivalence result here"
        )
        assert base_arm.meta["has_YOLO12_NORM_KWARGS"] is False, (
            "the base arm's yolo12_blocks DEFINES YOLO12_NORM_KWARGS, which was created "
            f"by this plan's step 2. The tree at {PLAN_BASE_COMMIT} therefore is not the "
            "pre-plan tree, and the comparison below is not base-vs-HEAD."
        )

    def test_the_base_tree_still_defines_the_three_deleted_classes(self, base_arm) -> None:
        """The other half of the discriminator: the base has what HEAD deleted."""
        source = (base_arm.tree / "src/dl_techniques/layers/yolo12_blocks.py").read_text()
        for name in ("class ConvBlock", "class AreaAttention", "class AttentionBlock"):
            assert name in source, (
                f"{name!r} is absent from the base tree's yolo12_blocks.py -- the pinned "
                "commit is not the pre-deletion state this file claims to compare against"
            )
        live = (_repo_root() / "src/dl_techniques/layers/yolo12_blocks.py").read_text()
        for name in ("class ConvBlock", "class AreaAttention", "class AttentionBlock"):
            assert name not in live, (
                f"{name!r} is back in the live yolo12_blocks.py; the relocation this file "
                "measures has been undone"
            )


class TestTheWholeModelIsUnchanged:
    """D-020, re-derived: the entire plan is a no-op on the assembled yolo12."""

    def test_the_parameter_count_is_identical_across_the_plan(self, base_arm) -> None:
        with keras.device(GOLDEN_REFERENCE_DEVICE):
            head = _build_head()
        assert base_arm.meta["params"] == EXPECTED_PARAMS
        assert head.count_params() == EXPECTED_PARAMS, (
            f"yolo12 {MODEL_KWARGS['scale']!r} has {head.count_params()} parameters at "
            f"HEAD against {EXPECTED_PARAMS} before the plan. A `use_bias=True` "
            "regression in `yolo12_conv_block` alone moves this number."
        )

    def test_the_weight_shape_sequence_is_identical(self, base_arm) -> None:
        """Ordered ``set_weights`` is only meaningful if the sequences already agree.

        A reordered sub-layer build shows up HERE, as a shape mismatch with a readable
        index, instead of downstream as a mysterious numeric delta.
        """
        with keras.device(GOLDEN_REFERENCE_DEVICE):
            head = _build_head()
        head_shapes = [tuple(w.shape) for w in head.get_weights()]
        base_shapes = [tuple(w.shape) for w in base_arm.weights]
        assert len(head_shapes) == EXPECTED_WEIGHTS
        assert len(base_shapes) == EXPECTED_WEIGHTS
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(head_shapes, base_shapes)) if a != b), None)
        assert first_diff is None, (
            f"weight #{first_diff} is {head_shapes[first_diff]} at HEAD and "
            f"{base_shapes[first_diff]} at {PLAN_BASE_COMMIT}"
        )

    def test_the_forward_pass_is_bitwise_identical_on_transferred_weights(
            self, base_arm, probe_input) -> None:
        """The headline: ``max|delta| == 0.0``, not "within tolerance"."""
        with keras.device(GOLDEN_REFERENCE_DEVICE):
            head = _build_head()
            head.set_weights(base_arm.weights)
            outputs = keras.tree.flatten(head(probe_input, training=False))
            outputs = [np.asarray(keras.ops.convert_to_numpy(t)) for t in outputs]

        assert len(outputs) == len(base_arm.outputs)
        deltas = []
        for i, (new, old) in enumerate(zip(outputs, base_arm.outputs)):
            assert new.shape == old.shape, f"output #{i}: {new.shape} vs {old.shape}"
            deltas.append(float(np.max(np.abs(new - old))))
        assert max(deltas) == 0.0, (
            f"whole-model max|delta| is {max(deltas)!r} (per output: {deltas}). The plan "
            "claims the relocation is EXACTLY a no-op on this model; anything non-zero "
            "means a real behaviour change, however small, and must be argued rather "
            "than absorbed into a tolerance."
        )

    def test_the_probe_input_is_not_degenerate(self, base_arm, probe_input) -> None:
        """Anti-vacuity: zeros in, zeros out would make ``0.0`` free."""
        assert float(np.max(np.abs(probe_input))) > 0.0
        assert max(float(np.max(np.abs(o))) for o in base_arm.outputs) > 0.0, (
            "the reference outputs are all zero -- a max|delta| of 0.0 would be an "
            "artefact of a dead model, not evidence of equivalence"
        )


class TestTheD067PairSurvivesIntoTheAssembledModel:
    """Both halves of D-067, censused on the built model. No forward pass involved.

    The shared provenance oracle
    (``tests/test_models/test_the_norm_epsilon_provenance_is_stated.py``) buckets EPSILON
    only, across ten packages, and every numeric check in this plan runs ``training=False``
    -- the one regime in which BatchNorm ``momentum`` cannot affect an output. So the
    ``momentum=0.97`` half of the decision-pinned pair had NO guard anywhere. This is it.
    """

    @staticmethod
    def _census(model) -> Dict[str, collections.Counter]:
        eps: collections.Counter = collections.Counter()
        mom: collections.Counter = collections.Counter()
        for layer in model._flatten_layers(include_self=True):
            if isinstance(layer, keras.layers.BatchNormalization):
                eps[f"{float(layer.epsilon):.0e}"] += 1
                mom[f"{float(layer.momentum)!r}"] += 1
        return {"epsilon": eps, "momentum": mom}

    def test_every_batchnorm_carries_momentum_0_97(self) -> None:
        with keras.device(GOLDEN_REFERENCE_DEVICE):
            census = self._census(_build_head())
        assert dict(census["momentum"]) == {"0.97": 134}, (
            f"yolo12 momentum census is {dict(census['momentum'])}, not "
            "{'0.97': 134}. Keras' BatchNormalization default is 0.99 and "
            "`create_normalization_layer` falls back to it SILENTLY when the kwarg is "
            "missing -- no raise, no shape change, and no effect at all under "
            "training=False, so no other test in this repository can see it. 0.97 is "
            "Ultralytics' 0.03 transcribed into Keras' opposite momentum sense "
            "(decisions.md D-067)."
        )

    def test_every_batchnorm_carries_epsilon_1e_3(self) -> None:
        """The epsilon half, censused here too so the pair is read in one place."""
        with keras.device(GOLDEN_REFERENCE_DEVICE):
            census = self._census(_build_head())
        assert dict(census["epsilon"]) == {"1e-03": 134}

    def test_the_momentum_census_can_observe_a_moved_momentum(self) -> None:
        """Liveness. Without this arm an empty census would pass the assertion above."""
        from dl_techniques.layers.yolo12_blocks import yolo12_conv_block

        block = yolo12_conv_block(filters=4)
        moved = keras.Sequential([block])
        moved.build((None, 8, 8, 4))
        assert dict(self._census(moved)["momentum"]) == {"0.97": 1}

        rogue = keras.Sequential([keras.layers.BatchNormalization(momentum=0.5)])
        rogue.build((None, 8, 8, 4))
        assert dict(self._census(rogue)["momentum"]) == {"0.5": 1}, (
            "the census reported the same thing for momentum 0.97 and 0.5 -- it cannot "
            "see a moved momentum, so the assertions above are vacuous"
        )
