"""Guards for the `jit_compile=` optimizer-constructor defect.

THE DEFECT. `src/train/bert/wikipedia/pretrain.py` and `pretrain_english.py`
both built their optimizer as::

    keras.optimizers.AdamW(..., jit_compile=True)   # XLA Compilation for speed

Keras 3 optimizers have no ``jit_compile`` parameter, so that raises
``ValueError: Argument(s) not recognized: {'jit_compile': True}``. It sat on an
unconditional path immediately before ``mlm_model.compile(...)``, so BOTH
scripts crashed at model compilation on every run, for their whole existence.

WHY IT SURVIVED. It is a train-time raise, so ``--help`` exits 0 long before it
(the CLI-contract suite passes), and nothing in the test tree ever constructed
the optimizer. An exit-code sweep and a collection count are both blind to it.

WHY THE OBVIOUS FIX IS WRONG. Moving the keyword to
``model.compile(jit_compile=True)`` is syntactically valid and still broken:
``MaskedLanguageModel.train_step`` calls ``optimizer.apply_gradients``, which
under a distribution strategy emits ``CollectiveGatherV2`` -- an op with no
XLA_GPU_JIT kernel -- so tf2xla conversion fails hard at step 1. Both scripts
build ``MirroredStrategy()`` unconditionally. See the DECISION note at
``src/train/bert/wikipedia/pretrain.py`` for the full measurement.

The two guards below are deliberately different in kind:

* :func:`test_no_optimizer_constructor_receives_jit_compile` is an AST sweep
  over ALL of ``src/train/``. It is the durable one -- it catches the defect
  CLASS anywhere in the tree, not just at the two sites that had it.
* :class:`TestMlmCompilesAndTrains` EXECUTES the construct -> compile -> fit
  path the scripts use, under the same ``mixed_float16`` policy, because an AST
  guard cannot tell you the scripts actually run.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pytest
import tensorflow as tf
import keras

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "src" / "train"

# Keyword names that are NOT valid on any Keras 3 optimizer constructor but are
# valid on `Model.compile(...)`, i.e. the exact confusion this defect is made of.
COMPILE_ONLY_KWARGS = frozenset({"jit_compile", "run_eagerly", "steps_per_execution"})

OPTIMIZER_CLASSES = frozenset(
    {
        "Adam", "AdamW", "SGD", "RMSprop", "Adadelta", "Adagrad", "Adamax",
        "Nadam", "Ftrl", "Lion", "LossScaleOptimizer",
    }
)


def _is_optimizer_construction(node: ast.Call) -> bool:
    """True when `node` looks like `keras.optimizers.X(...)` / `optimizers.X(...)`."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in OPTIMIZER_CLASSES:
        return False
    owner = func.value
    # `keras.optimizers.AdamW(...)` -> Attribute(attr='optimizers');
    # `optimizers.AdamW(...)`       -> Name(id='optimizers').
    if isinstance(owner, ast.Attribute):
        return owner.attr == "optimizers"
    if isinstance(owner, ast.Name):
        return owner.id == "optimizers"
    return False


def _offending_calls(path: Path) -> Iterator[Tuple[int, str, str]]:
    """Yield (lineno, optimizer_class, kwarg) for each bad construction."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_optimizer_construction(node):
            continue
        for kw in node.keywords:
            if kw.arg in COMPILE_ONLY_KWARGS:
                yield node.lineno, node.func.attr, kw.arg


def _python_files() -> List[Path]:
    return sorted(p for p in TRAIN_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_ast_detector_sees_a_planted_defect(tmp_path: Path) -> None:
    """Liveness arm: prove the sweep below can actually FAIL.

    A sweep that returns "no offenders" is worthless until it is shown to
    report a known-bad input. Without this, deleting the body of
    `_offending_calls` would leave the repo guard permanently green.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import keras\n"
        "opt = keras.optimizers.AdamW(learning_rate=1e-4, jit_compile=True)\n",
        encoding="utf-8",
    )
    found = list(_offending_calls(planted))
    assert found == [(2, "AdamW", "jit_compile")], (
        "ASSERT-AST-DETECTOR-IS-LIVE: the detector failed to flag a planted "
        f"`jit_compile=` on an AdamW constructor; got {found!r}"
    )

    clean = tmp_path / "clean.py"
    clean.write_text(
        "import keras\n"
        "opt = keras.optimizers.AdamW(learning_rate=1e-4)\n"
        "model.compile(optimizer=opt, jit_compile=True)\n",  # legal: on compile()
        encoding="utf-8",
    )
    assert list(_offending_calls(clean)) == [], (
        "ASSERT-AST-DETECTOR-NO-FALSE-POSITIVE: `jit_compile=` on "
        "`model.compile(...)` is legal and must not be flagged"
    )


def test_no_optimizer_constructor_receives_jit_compile() -> None:
    """No optimizer under `src/train/` may receive a compile-only kwarg.

    `jit_compile` / `run_eagerly` / `steps_per_execution` belong to
    `Model.compile(...)`. Passing one to an optimizer constructor raises
    `ValueError: Argument(s) not recognized` at runtime.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno} -> keras.optimizers.{cls}({kwarg}=...)"
        for path in _python_files()
        for lineno, cls, kwarg in _offending_calls(path)
    ]
    assert not offenders, (
        "ASSERT-NO-COMPILE-KWARG-ON-OPTIMIZER: these optimizer constructions "
        "pass a kwarg that only `Model.compile()` accepts, so they raise "
        "ValueError at runtime:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "script",
    ["src/train/bert/wikipedia/pretrain.py", "src/train/bert/wikipedia/pretrain_english.py"],
)
def test_the_two_repaired_scripts_carry_no_jit_compile_anywhere(script: str) -> None:
    """Belt-and-braces on the two sites that actually had the defect.

    Scoped to the source text rather than the AST because the point here is
    that neither file should mention the keyword as live code at all -- the
    only permitted occurrences are inside the DECISION comment explaining why.
    """
    path = REPO_ROOT / script
    code_lines = [
        (i, line)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "jit_compile" in line and not line.lstrip().startswith("#")
    ]
    assert not code_lines, (
        f"ASSERT-NO-LIVE-JIT-COMPILE-IN-{path.name}: `jit_compile` appears "
        f"outside a comment at {[i for i, _ in code_lines]}"
    )


class TestMlmCompilesAndTrains:
    """Execute the construct -> compile -> fit path the scripts use.

    The AST guards above cannot tell you the script RUNS. These do -- at a tiny
    size, with no dataset download, under the same `mixed_float16` policy the
    scripts set.
    """

    VOCAB = 128
    SEQ = 16
    BATCH = 4
    STEPS = 2

    @staticmethod
    @pytest.fixture(autouse=True)
    def _restore_dtype_policy() -> Iterator[None]:
        """`set_global_policy` is process-wide; leaking it corrupts later tests."""
        previous = keras.mixed_precision.global_policy()
        yield
        keras.mixed_precision.set_global_policy(previous)

    def _dataset(self) -> tf.data.Dataset:
        rng = np.random.default_rng(0)
        ids = rng.integers(5, self.VOCAB, size=(self.BATCH * self.STEPS, self.SEQ))
        ids = ids.astype("int32")
        return tf.data.Dataset.from_tensor_slices(
            {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        ).batch(self.BATCH)

    def _model(self):
        from dl_techniques.models.bert.bert import BERT
        from dl_techniques.models.masked_language_model.mlm import MaskedLanguageModel

        keras.utils.set_random_seed(42)
        encoder = BERT.from_variant(
            variant="tiny",
            vocab_size=self.VOCAB,
            max_position_embeddings=self.SEQ,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        return MaskedLanguageModel(
            encoder=encoder,
            vocab_size=self.VOCAB,
            mask_ratio=0.15,
            mask_token_id=4,
            special_token_ids=[1, 2, 3, 4],
        )

    @staticmethod
    def _optimizer() -> keras.optimizers.Optimizer:
        """Exactly the call shape both repaired scripts now use."""
        return keras.optimizers.AdamW(
            learning_rate=1e-4, weight_decay=0.01, clipnorm=1.0
        )

    def test_the_repaired_optimizer_call_shape_constructs(self) -> None:
        """The precise regression: this construction used to raise."""
        assert self._optimizer() is not None

    def test_the_original_call_shape_still_raises(self) -> None:
        """Pin the defect itself, so the guard cannot rot into a tautology.

        If a future Keras adds `jit_compile` to optimizers this fails loudly and
        someone re-reads the DECISION note -- which is the correct outcome, not
        a nuisance.
        """
        with pytest.raises(ValueError, match="jit_compile"):
            keras.optimizers.AdamW(
                learning_rate=1e-4, weight_decay=0.01, clipnorm=1.0, jit_compile=True
            )

    def test_mlm_compiles_and_takes_a_step_under_mixed_float16(self) -> None:
        """The end-to-end claim: the scripts' model/optimizer pair now trains."""
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = self._model()
        model.compile(optimizer=self._optimizer())

        history = model.fit(self._dataset(), epochs=1, verbose=0)

        loss = history.history["loss"][0]
        assert np.isfinite(loss), (
            f"ASSERT-MLM-FIT-LOSS-FINITE: fit produced a non-finite loss {loss!r}"
        )
        assert loss > 0.0, (
            f"ASSERT-MLM-FIT-LOSS-POSITIVE: cross-entropy must be > 0, got {loss!r}"
        )

    def test_xla_cannot_compile_this_step_under_a_strategy(self) -> None:
        """Pin WHY `model.compile(jit_compile=True)` is not the repair.

        `apply_gradients` inside `train_step` emits `CollectiveGatherV2` under a
        strategy, and that op has no XLA_GPU_JIT kernel. This is the measurement
        the DECISION note cites; if a future TF/Keras makes it work, this test
        fails and the note should be revisited rather than silently trusted.

        Skipped without a GPU: the failure mode is specific to XLA_GPU_JIT.
        """
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("XLA_GPU_JIT-specific; no GPU visible")

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self._model()
            model.compile(optimizer=self._optimizer(), jit_compile=True)

        with pytest.raises(Exception) as excinfo:
            model.fit(self._dataset(), epochs=1, verbose=0)

        message = str(excinfo.value)
        assert ("CollectiveGatherV2" in message) or ("merge_call" in message), (
            "ASSERT-XLA-UNDER-STRATEGY-FAILS-AS-DOCUMENTED: expected the "
            "documented XLA/collective incompatibility, got a different "
            f"failure:\n{message[:600]}"
        )
