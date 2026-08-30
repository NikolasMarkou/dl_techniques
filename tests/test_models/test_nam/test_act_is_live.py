"""RED proofs for C-35 (plan-2026-08-14T233721-d4f9beb2, D-033/D-034).

(a) `NAM.call` used to set `new_halted = is_last_step` whenever `training` was
    not exactly `True`, so at inference every sequence ran the full
    `halt_max_steps` and the learned `q_halt` head was never consulted — flatly
    contradicting the class docstring's "simple expressions like ``1 + 2`` take
    1 step". (D-033)
(b) `src/train/nam/train_nam.py` collected `q_halt_logits` into `all_q_halt` and
    consumed it in NO loss; there was no `L_halt` in the file. The head was
    neither trained nor read. (D-034)
(d) The per-package catalogue filed the package as "Neural additive model"; it
    is a Neural Arithmetic MODULE. The catalogue has since moved out of
    `models/CLAUDE.md` into `models/README.md`, which is why the guard below
    selects its row BY TABLE STRUCTURE and holds no file line number: an
    address-based citation rots silently, the rule does not.

The halting probes drive `q_halt` by writing the cell's `halt_head` weights
directly, so each arm's halting decision is DICTATED rather than inferred from
whatever a randomly initialised head happens to emit.
"""

import pathlib
import re

import numpy as np
import keras
import pytest
import tensorflow as tf

import dl_techniques.models as models_package
from dl_techniques.models.neural_computer import nam as nam_package
from dl_techniques.models.neural_computer.nam import NAM, NAMConfig

HALT_MAX_STEPS = 4


@pytest.fixture
def config():
    return NAMConfig(
        hidden_size=32,
        num_heads=4,
        num_tree_layers=1,
        intermediate_size=64,
        memory_size=8,
        num_read_heads=2,
        max_expression_len=16,
        halt_max_steps=HALT_MAX_STEPS,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )


@pytest.fixture
def batch():
    ids = np.zeros((2, 16), dtype="int32")
    ids[:, :5] = np.array([[1, 5, 12, 7, 2], [1, 9, 13, 3, 2]], dtype="int32")
    return {"input_ids": ids}


def _pin_q_halt(model, q_halt_value):
    """Force `q_halt` to a constant, whatever the hidden state is.

    Zeroing the kernel makes the head input-independent, so the arm's halting
    decision comes from `q_halt_value` alone and nothing else.
    """
    head = model.cell.halt_head
    kernel, bias = head.kernel, head.bias
    kernel.assign(keras.ops.zeros_like(kernel))
    bias.assign(keras.ops.convert_to_tensor([q_halt_value, 0.0], dtype=bias.dtype))


def _run_to_halt(model, batch, training=False):
    """Drive the ACT loop and return the 1-based step each sequence halted on."""
    carry = model.initial_carry(batch)
    halted_at = np.zeros(batch["input_ids"].shape[0], dtype="int32")
    for step in range(1, HALT_MAX_STEPS + 1):
        carry, _ = model(carry, batch, training=training)
        halted = np.asarray(keras.ops.convert_to_numpy(carry["halted"])).reshape(-1)
        halted_at = np.where((halted_at == 0) & halted, step, halted_at)
        if halted_at.all():
            break
    return halted_at


class TestInferenceConsultsTheLearnedHaltSignal:
    """(a) D-033."""

    def test_a_positive_q_halt_halts_on_the_first_step(self, config, batch):
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, +5.0)

        halted_at = _run_to_halt(model, batch, training=False)
        assert np.array_equal(halted_at, np.array([1, 1])), (
            f"with q_halt pinned to +5 at inference the sequences halted at "
            f"{halted_at.tolist()}, not [1, 1] — the learned halt signal is not "
            "consulted outside training (decisions.md D-033)"
        )

    def test_a_negative_q_halt_still_runs_the_full_budget(self, config, batch):
        """ANTI-VACUITY control: halting is not simply 'always step 1'.

        Without this arm, the assertion above is satisfied by any change that
        makes the model halt immediately regardless of `q_halt`.
        """
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, -5.0)

        halted_at = _run_to_halt(model, batch, training=False)
        assert np.array_equal(
            halted_at, np.full(2, HALT_MAX_STEPS, dtype="int32")
        ), (
            f"with q_halt pinned to -5 the sequences halted at "
            f"{halted_at.tolist()}, not at the {HALT_MAX_STEPS}-step ceiling — "
            "the halt signal's SIGN is being ignored"
        )

    def test_training_and_inference_agree_on_the_predicate(self, config, batch):
        """The two branches must halt on the same rule, exploration aside."""
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=False)
        _pin_q_halt(model, +5.0)
        object.__setattr__(model.config, "halt_exploration_prob", 0.0)

        train_halt = _run_to_halt(model, batch, training=True)
        eval_halt = _run_to_halt(model, batch, training=False)
        assert np.array_equal(train_halt, eval_halt), (
            f"training halted at {train_halt.tolist()} but inference at "
            f"{eval_halt.tolist()} for the same pinned q_halt; the head would be "
            "trained under one rule and read under another"
        )


class TestHaltHeadReceivesGradient:
    """(b) D-034 — the halting loss must actually reach `halt_head`."""

    def _halt_loss(self, model, batch, targets):
        """The `L_halt` term of `train_nam.py`, in the same shape."""
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch, training=True)
        rel_error = tf.abs(outputs["result"] - targets) / (tf.abs(targets) + 1e-8)
        step_correct = tf.cast(tf.less(rel_error, 0.01), tf.float32)
        return tf.reduce_mean(
            keras.losses.binary_crossentropy(
                tf.stop_gradient(step_correct),
                tf.expand_dims(outputs["q_halt_logits"], axis=-1),
                from_logits=True,
            )
        )

    def test_l_halt_moves_halt_head_and_the_probe_is_live(self, config, batch):
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=True)
        targets = tf.constant([[7.0], [12.0]])

        with tf.GradientTape() as tape:
            loss = self._halt_loss(model, batch, targets)
        kernel = model.cell.halt_head.kernel
        grad = tape.gradient(loss, kernel)

        assert grad is not None, (
            "L_halt produced NO gradient path to halt_head.kernel — the "
            "halting head is untrainable (decisions.md D-034)"
        )
        magnitude = float(np.max(np.abs(np.asarray(grad))))
        assert magnitude > 1e-8, (
            f"L_halt's gradient w.r.t. halt_head.kernel is {magnitude:.3e}; the "
            "term is present but inert"
        )

    def test_the_result_head_is_shielded_from_the_halting_target(
        self, config, batch
    ):
        """`L_halt` must not become a second objective on `result`.

        HONEST SCOPE: this pins the PROPERTY, not the `stop_gradient` that
        states it. Measured on this stack, `cast(less(...))` is already
        non-differentiable — `tape.gradient` through it returns `None` — so
        removing the explicit `stop_gradient` would not make this assertion
        fire. The barrier is documentation of intent plus insurance against a
        future soft/relaxed correctness target; the assertion guards the
        invariant either way, and would fire the moment someone swaps the hard
        threshold for a differentiable surrogate without re-adding a barrier.
        """
        keras.utils.set_random_seed(0)
        model = NAM(config=config)
        model(model.initial_carry(batch), batch, training=True)
        targets = tf.constant([[7.0], [12.0]])

        with tf.GradientTape() as tape:
            loss = self._halt_loss(model, batch, targets)
        result_kernel = model.result_head.kernel
        grad = tape.gradient(loss, result_kernel)

        magnitude = 0.0 if grad is None else float(
            np.max(np.abs(np.asarray(grad)))
        )
        assert magnitude == 0.0, (
            f"L_halt reached result_head.kernel with magnitude {magnitude:.3e}; "
            "the correctness target is not stop_gradient'ed and the loss can be "
            "minimised by moving the prediction instead of the halt signal"
        )


NAM_CATALOGUE_KEY = "`neural_computer/nam/`"
"""The exact FIRST cell of every catalogue row that describes this package."""

WRONG_NAME_HEADER = re.compile(r"reads as|misattribut|name", re.IGNORECASE)
"""Headers of columns that legitimately quote the WRONG name in order to correct it.

`models/README.md`'s "Names that misattribute" table has a ``Reads as`` column
holding the literal string "Neural *Additive* Model". A guard that asserted
"additive" is absent from every cell would false-fail against the very row that
documents the correction, so cells are classified BY THEIR COLUMN HEADER.
"""


def _split_row(line):
    """Split one Markdown table row into its cells.

    :param line: a single line of Markdown.
    :type line: str
    :return: the row's cells, stripped, or ``None`` if the line is not a row.
    :rtype: list[str] | None
    """
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_separator(cells):
    """Whether ``cells`` came from a Markdown header/body separator row.

    :param cells: the cells of one table row.
    :type cells: list[str]
    :return: ``True`` for rows like ``|---|---|``.
    :rtype: bool
    """
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in cells
    )


def _nam_rows(text):
    """Select every catalogue row describing this package, BY STRUCTURE.

    For each Markdown table row whose first cell is exactly
    :data:`NAM_CATALOGUE_KEY`, walk UPWARD to that table's own header row (the
    line immediately above the table's ``|---|`` separator) and pair each cell
    with the header above it. Selection therefore survives a column reorder, a
    row reorder and any change of line number; only a whole-table rewrite
    breaks it, which is the intent.

    Pure: takes text, touches no filesystem, so the selection logic is testable
    on synthetic input (see :class:`TestTheCatalogueGuardItself`).

    :param text: the full text of a Markdown document.
    :type text: str
    :return: one dict per matching row, with keys ``row`` (the raw line) and
        ``cells`` (a list of ``(header, cell)`` pairs, header ``""`` when the
        table's header cell is blank).
    :rtype: list[dict]
    """
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        cells = _split_row(line)
        if cells is None or _is_separator(cells):
            continue
        if not cells or cells[0] != NAM_CATALOGUE_KEY:
            continue

        header = None
        for above in range(index - 1, -1, -1):
            above_cells = _split_row(lines[above])
            if above_cells is None:
                break
            if _is_separator(above_cells):
                header = _split_row(lines[above - 1]) if above > 0 else None
                break
        if header is None:
            header = []

        padded = list(header) + [""] * max(0, len(cells) - len(header))
        rows.append(
            {"row": line, "cells": list(zip(padded[: len(cells)], cells))}
        )
    return rows


def _assert_rows_name_the_architecture(rows, source):
    """Assert the catalogue rows name NAM's real architecture.

    The rule, stated rather than addressed: in every cell whose column header is
    not a wrong-name column, "additive" must be ABSENT, and "Arithmetic" must
    appear in at least one such cell.

    :param rows: the output of :func:`_nam_rows`.
    :type rows: list[dict]
    :param source: how to name the text in failure messages.
    :type source: str
    :raises Failed: loudly, when ``rows`` is empty. There is deliberately no
        skip-when-absent path: the absence of the row is the defect this guard
        exists to surface.
    """
    if not rows:
        pytest.fail(
            f"{source} holds NO Markdown table row whose first cell is exactly "
            f"{NAM_CATALOGUE_KEY} — the per-package catalogue this guard reads "
            "has moved or been restructured again (it already migrated out of "
            "models/CLAUDE.md once). The claim it pins is that NAM is a Neural "
            "Arithmetic MODULE, not a Neural Additive Model; it is also stated "
            "in the same file's 'Names that misattribute' table and, primarily, "
            "in the first line of the neural_computer/nam package docstring. "
            "Re-point this guard at wherever the catalogue now lives — do not "
            "delete it."
        )

    for row in rows:
        checked = [
            (header, cell)
            for header, cell in row["cells"]
            if not WRONG_NAME_HEADER.search(header)
        ]
        for header, cell in checked:
            assert "additive" not in cell.lower(), (
                f"{source} row {row['row']!r} calls nam/ 'additive' in column "
                f"{header!r}, which is not a wrong-name column; the package is "
                "a Neural Arithmetic MODULE (tree parse + NTM memory + TRM "
                "halting) and contains no per-feature additive model"
            )
        assert any("Arithmetic" in cell for _, cell in checked), (
            f"{source} row {row['row']!r} never names the actual architecture "
            "('Arithmetic') in any column that is not a wrong-name column"
        )


class TestPackageIsFiledUnderItsRealArchitecture:
    """(d) — the catalogue and the package itself must name what the code is."""

    def test_the_models_catalogue_names_nam_a_neural_arithmetic_module(self):
        readme = pathlib.Path(models_package.__file__).parent / "README.md"
        _assert_rows_name_the_architecture(
            _nam_rows(readme.read_text()), "models/README.md"
        )

    def test_the_package_docstring_names_a_neural_arithmetic_module(self):
        """The primary source: it lives beside the code and moves with it."""
        docstring = nam_package.__doc__
        assert docstring, (
            "the neural_computer/nam package has no module docstring; the "
            "primary statement of what this architecture is has been deleted"
        )
        first_line = docstring.splitlines()[0]
        assert "Neural Arithmetic Module" in first_line, (
            f"the nam package docstring opens with {first_line!r}, which does "
            "not name it a Neural Arithmetic Module"
        )


class TestTheCatalogueGuardItself:
    """RED proofs for the guard above, run on synthetic text.

    `_nam_rows` is pure, so both failure directions are exercised directly
    rather than by mutating a tracked document.
    """

    CATALOGUE = "\n".join(
        [
            "### The small families",
            "",
            "| Package | |",
            "|---|---|",
            "| `graph/relgt/` | Relational Graph Transformer |",
            "| `neural_computer/nam/` | Neural Arithmetic **Module**. "
            "Name misattributes |",
            "",
            "## Names that misattribute",
            "",
            "| Package | Reads as | Actually is | Where the evidence sits |",
            "|---|---|---|---|",
            "| `neural_computer/nam/` | Neural *Additive* Model | Neural "
            "Arithmetic **Module** — not a GAM | the package docstring |",
        ]
    )

    def test_the_guard_accepts_the_two_legitimate_shapes(self):
        """ANTI-VACUITY control for both RED proofs below.

        The misattribution row's "Additive" sits under a `Reads as` header, so
        it must pass; without this arm a guard that rejected everything would
        satisfy both RED proofs.
        """
        rows = _nam_rows(self.CATALOGUE)
        assert len(rows) == 2, (
            f"structural selection found {len(rows)} nam rows in the synthetic "
            "catalogue, not the 2 it contains"
        )
        _assert_rows_name_the_architecture(rows, "synthetic catalogue")

    def test_a_corrupted_catalogue_row_goes_red(self):
        corrupted = self.CATALOGUE.replace(
            "| `neural_computer/nam/` | Neural Arithmetic **Module**. "
            "Name misattributes |",
            "| `neural_computer/nam/` | Neural **Additive** Model |",
        )
        with pytest.raises(AssertionError, match="not a wrong-name column"):
            _assert_rows_name_the_architecture(
                _nam_rows(corrupted), "synthetic catalogue"
            )

    def test_a_catalogue_with_no_matching_row_fails_loudly(self):
        without = "\n".join(
            line
            for line in self.CATALOGUE.splitlines()
            if NAM_CATALOGUE_KEY not in line
        )
        assert _nam_rows(without) == []
        with pytest.raises(pytest.fail.Exception, match="holds NO Markdown"):
            _assert_rows_name_the_architecture(
                _nam_rows(without), "synthetic catalogue"
            )
