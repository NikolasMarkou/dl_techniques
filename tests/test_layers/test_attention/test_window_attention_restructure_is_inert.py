"""The instrument that grades the ``WindowAttention`` restructure.

This file answers exactly one question, for every cell of a pinned matrix: **did the
restructure move a single bit of the SPATIAL path?** It is not a behaviour test. It
has no opinion about whether ``partition_mode='grid'`` / ``'zigzag'`` compute the
right attention -- only about whether they compute the SAME attention they computed
before the restructure started.

Plan ``plan-2026-08-25T053412-0f1fa04f`` steps 2, 3 and 4 all edit
``layers/attention/window_attention.py`` and
``layers/attention/single_window_attention.py`` while claiming the spatial behaviour
is unchanged (invariant I-1). That claim needs an instrument that exists BEFORE the
first edit, or it is unfalsifiable afterwards. This is that instrument.

**The reference is EXTERNAL, and that is the whole design.** A restructure test that
compares the current code against itself passes against ANY breakage -- this repo has
a measured instance of a bit-identity arm passing 7/7 against a deliberately broken
model because both sides went down the same branch. So the reference side is obtained
by ``git show``-ing BOTH modules out of the recorded pre-restructure commit into a
process-local temporary directory and loading them through ``importlib``.

**Which mechanism, and why.** The `importlib` mechanism was used -- NOT a committed
``.npz`` of golden arrays. Both modules import cleanly in isolation once
``__package__`` is set to ``dl_techniques.layers.attention``: their relative imports
(``.common``, ``..ffn.kan_linear``, ``..norms.factory``) then resolve against the real
installed package graph, which is untouched by this plan. The one import that must NOT
resolve that way is ``window_attention``'s own
``from .single_window_attention import SingleWindowAttention`` -- resolved naively it
would pair the golden wrapper with the CURRENT inner layer, silently making half of
every comparison a self-comparison. :func:`golden_modules` therefore shadows
``sys.modules['dl_techniques.layers.attention.single_window_attention']`` with the
golden inner module for the duration of the golden wrapper's ``exec_module``, then
restores it. :func:`test_the_golden_reference_is_external_and_not_a_self_comparison`
asserts that pairing held.

**Initialization is not part of the comparison.** Every cell builds both layers, then
copies the golden layer's weights into the current layer BY WEIGHT PATH (not by
``set_weights`` position, so a reordering cannot silently mis-pair tensors). Without
this the two layers would hold different random kernels and the harness would compare
noise -- passing or failing for the wrong reason.

**Bitwise, not ``allclose``.** The comparison is ``np.array_equal``. This plan is a
pure cost restructure with a CPU-pinned reference; there is no legitimate nonzero
delta to absorb. A failure here is a defect to find, never a tolerance to widen
(plan Pre-Mortem #1: do not widen the tolerance -- escalate with the measured delta).

**Device.** CPU only, via the ``golden_reference_device`` fixture. A GPU digest cannot
answer an inertness question in this repo: TF32 and non-deterministic reductions make
the same forward disagree with itself run to run.

**Window sizes are small ONLY, and that is the defect under repair, not an oversight.**
``window_size=128`` -- the size ModernBERT actually ships -- costs 17.69 GB on the
current code and cannot be run here at all. The matrix covers ``{2, 4, 7, 8}``, which
is exactly the range the four spatial consumers use (``swin_transformer`` 7/8,
``fastvlm`` 8, ``tirex`` 8), so I-1 coverage for those consumers is real. It is NOT
coverage of the large-window regime; plan Pre-Mortem #2 owns that hole.

RED-PROOF
=========
A harness that has never gone red is not known to work. Two DIFFERENT injections were
applied to the CURRENT tree, each run, each observed to fail on the ASSERTION (not on
an exception from upstream -- this repo has a recorded false-RED where an
``InputSpec(ndim=4)`` raised before the recorder ever ran), then reverted.

Injection (i) -- shape-preserving VALUE change. In
``single_window_attention.py:760``, ``x = self.proj(x, training=training)`` became
``x = self.proj(x, training=training) * 1.0000001``. No shape moves, no weight moves,
no RNG draw moves. Observed, verbatim, at cell ``ws2-N16-d32-h4/grid/rpb``::

    E           AssertionError: WindowAttention(window_size=2, partition_mode='grid', use_relative_position_bias=True) on (2, 16, 32) is NOT bitwise identical to the pre-restructure layer at ad27896189876609c006ea6adeea7b4ac6d31d29
    E             mismatching elements: 1024 of 1024
    E             max |delta|: 1.1920928955078125e-07
    E             current head:  [-0.328561 -0.522456 -0.162221  0.141906 -0.354493  0.089452]
    E             golden  head:  [-0.328561 -0.522456 -0.162221  0.141906 -0.354493  0.089452]

**37 failed, 1 passed / 38 collected**: all 36 matrix cells and the ``@tf.function``
arm went red; only :func:`test_the_golden_reference_is_external_and_not_a_self_comparison`
stayed green, which is correct -- it compares module identities, not numbers. Note the
two printed heads are IDENTICAL to six decimals and ``max |delta|`` is one float32 ulp:
this is exactly the class of change an ``allclose`` assertion would have waved
through, which is why the comparison is ``np.array_equal``.

Injection (ii) -- DEAD COMPONENT. In ``single_window_attention.py:644`` the
relative-position bias addition
``attn = attn + keras.ops.expand_dims(relative_position_bias, 0)`` became
``... + keras.ops.expand_dims(relative_position_bias, 0) * 0.0``: the bias contributes
nothing, while its table, its index, its gather, every shape and every weight stay
exactly as they were. Observed, verbatim, at the same cell::

    E           AssertionError: WindowAttention(window_size=2, partition_mode='grid', use_relative_position_bias=True) on (2, 16, 32) is NOT bitwise identical to the pre-restructure layer at ad27896189876609c006ea6adeea7b4ac6d31d29
    E             mismatching elements: 1024 of 1024
    E             max |delta|: 0.01834934577345848
    E             current head:  [-0.331535 -0.522039 -0.166352  0.145628 -0.355914  0.086218]
    E             golden  head:  [-0.328561 -0.522456 -0.162221  0.141906 -0.354493  0.089452]

**19 failed, 19 passed / 38 collected**: exactly the 18 ``use_relative_position_bias=True``
cells plus the ``@tf.function`` arm (which is an ``rpb=True`` cell), while all 18
``rpb=False`` cells stayed green. That clean split is the evidence the RED is caused
by the injected dead component itself and not by some unrelated upstream exception --
in both injections the text above shows the harness's OWN ``AssertionError`` firing,
never a ``ValueError``/``InputSpec`` raise from somewhere earlier in the call.

Both injections were reverted; the tree was confirmed clean before the harness's own
green run was recorded. Injection (i) was additionally RE-RUN against the final version
of this file (after the weight-path normalization fix documented in
:func:`_copy_weights_by_path`) and reproduces the text above character for character --
37 failed, 1 passed.

GREEN, on the unmodified tree
=============================
``CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venv/bin/python -m pytest
tests/test_layers/test_attention/test_window_attention_restructure_is_inert.py -q
-p no:randomly`` -> **38 passed / 38 collected**, 5.5 s.

The whole directory in one process --
``CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg .venv/bin/python -m pytest
tests/test_layers/test_attention -q -p no:randomly`` -> **1871 passed, 32 skipped,
1 xfailed / 1904 collected**, 250 s. That run matters beyond the count: this file
mutates process-global state (``sys.modules``, the Keras custom-object registry), so
"green alone" would not have been evidence that it leaves the other 1833 tests intact.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import keras
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Pinned capture parameters.

#: ``git rev-parse HEAD`` immediately before the first restructure edit. This is a
#: LITERAL HASH and must never become ``HEAD``: once step 2 commits, ``HEAD`` contains
#: the restructured file and this whole file silently degrades into a
#: self-comparison that cannot fail.
PRE_RESTRUCTURE_COMMIT = "ad27896189876609c006ea6adeea7b4ac6d31d29"

WINDOW_ATTENTION_PATH = "src/dl_techniques/layers/attention/window_attention.py"
SINGLE_WINDOW_ATTENTION_PATH = (
    "src/dl_techniques/layers/attention/single_window_attention.py"
)

#: The package the golden modules pretend to live in, so their RELATIVE imports
#: (``.common``, ``..ffn.kan_linear``, ``..norms.factory``, ``..activations``) resolve
#: against the real, un-restructured package graph.
ATTENTION_PACKAGE = "dl_techniques.layers.attention"

SEED = 1234
BATCH = 2

#: ``(window_size, seq_len, dim, num_heads)``.
#:
#: Every ``window_size`` appears with at least one ``N > window_size**2`` (the grid
#: genuinely tiles into several windows) and one ``N <= window_size**2`` (the
#: degenerate single-window regime step 3 rewrites). ``ws=8, N=64`` is the exact
#: boundary ``N == window_size**2``, where invariant I-2's verbatim-mask contract
#: lives and where both branches must agree.
ROWS = [
    (2, 16, 32, 4),   # N=16 >  ws**2=4   -> 4 windows
    (2, 4, 32, 4),    # N=4  == ws**2=4   -> degenerate, boundary
    (4, 64, 32, 4),   # N=64 >  ws**2=16  -> 4 windows
    (4, 9, 24, 3),    # N=9  <  ws**2=16  -> degenerate, ragged (grid 3x3 padded to 4x4)
    (7, 196, 32, 4),  # N=196 > ws**2=49  -> 4 windows
    (7, 25, 32, 4),   # N=25 <  ws**2=49  -> degenerate, ragged
    (8, 256, 24, 3),  # N=256 > ws**2=64  -> 4 windows
    (8, 64, 32, 4),   # N=64 == ws**2=64  -> degenerate, exact boundary
    (8, 50, 32, 4),   # N=50 <  ws**2=64  -> degenerate, ragged prefix
]

#: ``'band'`` is deliberately absent: it does not exist yet (step 4 adds it). This
#: file pins what EXISTS at :data:`PRE_RESTRUCTURE_COMMIT`.
PARTITION_MODES = ["grid", "zigzag"]

MATRIX = [
    (ws, n, dim, heads, mode, rpb)
    for (ws, n, dim, heads) in ROWS
    for mode in PARTITION_MODES
    for rpb in (True, False)
]


def _cell_id(cell) -> str:
    ws, n, dim, heads, mode, rpb = cell
    return f"ws{ws}-N{n}-d{dim}-h{heads}/{mode}/{'rpb' if rpb else 'norpb'}"


def _repo_root() -> str:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# ---------------------------------------------------------------------
# The external reference


def _exec_golden(source_path: str, module_name: str, tmp_dir: str):
    """``git show`` one module out of the pinned commit and import it."""
    source = subprocess.run(
        ["git", "show", f"{PRE_RESTRUCTURE_COMMIT}:{source_path}"],
        cwd=_repo_root(),
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert "class " in source and len(source) > 1000, (
        f"`git show {PRE_RESTRUCTURE_COMMIT}:{source_path}` produced {len(source)} "
        f"characters that do not look like the module -- the reference side is not "
        f"what it claims to be."
    )
    path = os.path.join(tmp_dir, module_name.rsplit(".", 1)[-1] + ".py")
    with open(path, "w") as handle:
        handle.write(source)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Relative imports resolve against __package__, so the golden text runs
    # unmodified -- no source rewriting, the reference really is the committed bytes.
    module.__package__ = ATTENTION_PACKAGE
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden_modules():
    """The two attention modules as of :data:`PRE_RESTRUCTURE_COMMIT`, importable.

    Yields ``(golden_window_attention_module, golden_single_window_module)``.

    Three pieces of process-global state are snapshotted and restored, because each
    one, left dirty, would corrupt the rest of the pytest process rather than fail:

    1. ``sys.modules[<pkg>.single_window_attention]`` is SHADOWED with the golden
       inner module while the golden wrapper is exec'd, so the wrapper's
       ``from .single_window_attention import SingleWindowAttention`` binds the golden
       class. Without the shadow the golden wrapper would drive the CURRENT inner
       layer and every step-2 edit (which lives entirely in the inner layer) would be
       invisible to this file.
    2. The Keras global custom-object registry: ``@register_keras_serializable()``
       writes into a process-global dict at class-definition time, so exec'ing a
       second ``WindowAttention`` re-points ``Custom>WindowAttention`` at the golden
       class. Any later test in the same process that deserializes a ``.keras``
       archive would then silently get the pre-restructure class back.
    3. The two golden module entries in ``sys.modules``.
    """
    from keras.src.saving import object_registration

    saved_objects = dict(object_registration.GLOBAL_CUSTOM_OBJECTS)
    saved_names = dict(object_registration.GLOBAL_CUSTOM_NAMES)

    live_single_name = f"{ATTENTION_PACKAGE}.single_window_attention"
    live_single = importlib.import_module(live_single_name)

    golden_single_name = f"{ATTENTION_PACKAGE}.single_window_attention__golden"
    golden_window_name = f"{ATTENTION_PACKAGE}.window_attention__golden"

    tmp_dir = tempfile.mkdtemp(prefix="window_attention_pre_restructure_")
    try:
        golden_single = _exec_golden(
            SINGLE_WINDOW_ATTENTION_PATH, golden_single_name, tmp_dir
        )
        sys.modules[live_single_name] = golden_single
        try:
            golden_window = _exec_golden(
                WINDOW_ATTENTION_PATH, golden_window_name, tmp_dir
            )
        finally:
            sys.modules[live_single_name] = live_single

        yield golden_window, golden_single
    finally:
        sys.modules.pop(golden_single_name, None)
        sys.modules.pop(golden_window_name, None)
        sys.modules[live_single_name] = live_single
        object_registration.GLOBAL_CUSTOM_OBJECTS.clear()
        object_registration.GLOBAL_CUSTOM_OBJECTS.update(saved_objects)
        object_registration.GLOBAL_CUSTOM_NAMES.clear()
        object_registration.GLOBAL_CUSTOM_NAMES.update(saved_names)


@pytest.fixture(scope="module")
def current_window_attention():
    """The live ``WindowAttention`` -- the side under test."""
    from dl_techniques.layers.attention.window_attention import WindowAttention

    return WindowAttention


# ---------------------------------------------------------------------
# Capture


def _inputs(ws: int, n: int, dim: int) -> np.ndarray:
    """A per-cell deterministic input. The seed is derived from the cell shape so two
    different cells never accidentally share an input (which would let a copy-paste
    error in the matrix go unnoticed)."""
    rng = np.random.default_rng(abs(hash((ws, n, dim))) % (2**32))
    return rng.standard_normal((BATCH, n, dim)).astype("float32")


def _make(cls, ws, n, dim, heads, mode, rpb):
    keras.utils.set_random_seed(SEED)
    layer = cls(
        dim=dim,
        window_size=ws,
        num_heads=heads,
        partition_mode=mode,
        use_relative_position_bias=rpb,
        dropout_rate=0.0,
    )
    layer.build((None, n, dim))
    assert layer.built
    return layer


def _copy_weights_by_path(source, target) -> int:
    """Copy ``source``'s weights into ``target``, matched by weight PATH.

    Path-matched rather than ``set_weights``-positional on purpose: ``set_weights``
    pairs tensors by index, so a restructure that reordered weight creation would be
    silently mis-paired into garbage and this harness would go red for a reason that
    has nothing to do with the numbers it exists to compare. Weight paths are pinned
    by invariant I-3 (D-081), so a path set that moves is itself a finding.
    """

    def by_path(layer):
        # Key on the portion of the path BELOW this layer's own name. The leading
        # component(s) of `weight.path` carry the layer INSTANCE name, which Keras
        # auto-numbers from a PROCESS-GLOBAL counter (`window_attention_118` when this
        # file runs after the rest of the directory, `window_attention` when it runs
        # alone), and the golden and current instances are necessarily two different
        # numbers. MEASURED: stripping a fixed number of leading components instead
        # made every cell of this file fail with "the weight PATH set moved" in a
        # full-directory run while passing when run alone -- an instrument whose
        # verdict depends on what ran before it is not an instrument.
        prefix = layer.name + "/"
        out = {}
        for w in layer.weights:
            assert prefix in w.path, (
                f"weight path {w.path!r} does not contain the layer's own name "
                f"{prefix!r}; the path normalization below would be a no-op"
            )
            out[w.path.split(prefix, 1)[-1]] = w
        return out

    src, dst = by_path(source), by_path(target)
    assert set(src) == set(dst), (
        f"the weight PATH set moved: only-in-golden={sorted(set(src) - set(dst))}, "
        f"only-in-current={sorted(set(dst) - set(src))}. Invariant I-3 (D-081) pins "
        f"these paths; this is a finding, not a harness problem."
    )
    assert src, "no weights -- the weight copy is vacuous and the comparison is noise"
    for name, weight in src.items():
        assert tuple(dst[name].shape) == tuple(weight.shape), (
            f"weight {name} changed shape: {tuple(dst[name].shape)} != "
            f"{tuple(weight.shape)}"
        )
        dst[name].assign(keras.ops.convert_to_numpy(weight))
    return len(src)


def _forward(layer, x) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(layer(x, training=False)))


# ---------------------------------------------------------------------
# The arm


@pytest.mark.parametrize("cell", MATRIX, ids=_cell_id)
def test_the_spatial_forward_is_bitwise_unchanged(
    cell, golden_modules, current_window_attention, golden_reference_device
):
    """One matrix cell: the current layer and the pre-restructure layer, same weights,
    same input, must agree BIT FOR BIT."""
    ws, n, dim, heads, mode, rpb = cell
    golden_window, _ = golden_modules
    x = _inputs(ws, n, dim)

    with keras.device(golden_reference_device):
        golden_layer = _make(golden_window.WindowAttention, ws, n, dim, heads, mode, rpb)
        current_layer = _make(current_window_attention, ws, n, dim, heads, mode, rpb)
        copied = _copy_weights_by_path(golden_layer, current_layer)
        golden_out = _forward(golden_layer, x)
        current_out = _forward(current_layer, x)

    assert copied >= 3, f"only {copied} weights copied -- suspiciously few"
    assert current_out.shape == golden_out.shape == (BATCH, n, dim)
    # Guard against a vacuous comparison: two all-zero (or constant) outputs are
    # bitwise equal no matter how broken the layer is.
    assert np.all(np.isfinite(golden_out)), "the golden output is not finite"
    assert float(np.std(golden_out)) > 1e-6, (
        f"the golden output is effectively constant (std="
        f"{float(np.std(golden_out))}) -- a bitwise comparison against it is vacuous"
    )

    if not np.array_equal(current_out, golden_out):
        delta = np.abs(current_out.astype("float64") - golden_out.astype("float64"))
        raise AssertionError(
            f"WindowAttention(window_size={ws}, partition_mode={mode!r}, "
            f"use_relative_position_bias={rpb}) on {x.shape} is NOT bitwise identical "
            f"to the pre-restructure layer at {PRE_RESTRUCTURE_COMMIT}\n"
            f"  mismatching elements: {int((delta != 0).sum())} of {delta.size}\n"
            f"  max |delta|: {float(delta.max())}\n"
            f"  current head:  "
            f"{np.array2string(current_out.reshape(-1)[:6], precision=6)}\n"
            f"  golden  head:  "
            f"{np.array2string(golden_out.reshape(-1)[:6], precision=6)}"
        )


# ---------------------------------------------------------------------
# The instrument's own guards


def test_the_golden_reference_is_external_and_not_a_self_comparison(
    golden_modules, current_window_attention
):
    """The reference really is DIFFERENT code, on BOTH levels.

    Without this test the whole file rests on a commit hash and an import shadow that
    nobody re-checks. Two distinct ways it could rot into a self-comparison:

    * the golden WRAPPER is the live class (bad hash, failed exec) -- every cell then
      compares a layer with itself;
    * the golden wrapper is golden but its INNER ``SingleWindowAttention`` is the live
      one. Step 2 of this plan edits ONLY the inner layer, so this failure mode would
      leave the harness green through the exact change it was built to grade. It is
      the more dangerous of the two and it is invisible from the wrapper alone.
    """
    from dl_techniques.layers.attention import single_window_attention as live_single

    golden_window, golden_single = golden_modules

    assert golden_window.WindowAttention is not current_window_attention, (
        "the golden and current WindowAttention are the SAME class -- this file is a "
        "self-comparison and cannot fail"
    )
    assert golden_window.__file__ != current_window_attention.__module__
    assert not golden_window.__file__.startswith(_repo_root()), (
        f"the golden module was loaded from inside the repository tree "
        f"({golden_window.__file__}) -- it must come from `git show`, not from a "
        f"working-tree file the restructure could edit"
    )
    assert golden_single.SingleWindowAttention is not (
        live_single.SingleWindowAttention
    ), "the golden inner layer is the live class"
    assert (
        golden_window.SingleWindowAttention is golden_single.SingleWindowAttention
    ), (
        "the golden WindowAttention is wired to the LIVE SingleWindowAttention -- "
        "the module shadow in the `golden_modules` fixture did not take, and every "
        "step-2 edit (which lives entirely in the inner layer) would be invisible "
        "to this file"
    )
    # And the live package is back to normal for every other test in this process.
    assert sys.modules[
        f"{ATTENTION_PACKAGE}.single_window_attention"
    ] is live_single


def test_the_forward_is_bitwise_unchanged_under_tf_function(
    golden_modules, current_window_attention, golden_reference_device
):
    """One cell re-run inside ``@tf.function``.

    Eager bit-identity is not graph bit-identity in this repo: a recorded case read
    exactly 0.0 eagerly and 4.233e-04 under ``@tf.function`` -- the regime ``fit()``
    actually uses. Without this arm every bitwise claim in steps 2, 3 and 7 would be
    scoped to eager only.
    """
    import tensorflow as tf

    ws, n, dim, heads, mode, rpb = 8, 64, 32, 4, "grid", True
    golden_window, _ = golden_modules
    x = _inputs(ws, n, dim)

    with keras.device(golden_reference_device):
        golden_layer = _make(golden_window.WindowAttention, ws, n, dim, heads, mode, rpb)
        current_layer = _make(current_window_attention, ws, n, dim, heads, mode, rpb)
        _copy_weights_by_path(golden_layer, current_layer)

        @tf.function
        def run(layer, data):
            return layer(data, training=False)

        golden_out = np.asarray(run(golden_layer, tf.constant(x)))
        current_out = np.asarray(run(current_layer, tf.constant(x)))

    assert float(np.std(golden_out)) > 1e-6
    assert np.array_equal(current_out, golden_out), (
        f"under @tf.function, WindowAttention(window_size={ws}, "
        f"partition_mode={mode!r}) is NOT bitwise identical to the pre-restructure "
        f"layer at {PRE_RESTRUCTURE_COMMIT}: max |delta| "
        f"{float(np.abs(current_out - golden_out).max())}"
    )
