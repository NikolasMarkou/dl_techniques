"""RED PROOFS: every guard in this package, proven to fail on its own defect.

Run with ``pytest tests/test_layers/test_adapters/test_the_red_proofs.py -v``
to watch each guard go red in turn. This file mutates source or monkeypatches,
so it is excluded from the default run via ``pytest.mark.red_proof`` — the
point is to be run deliberately and read, not to be counted as a passing suite.

Every mutation below restores the original in a ``finally``, and every one of
these has been executed at least once with the expected failure observed.
"""

import ast
import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.red_proof

REPO_ROOT = pathlib.Path(
    '/media/arxwn/data_fast/repositories/dl_techniques'
)
SRC = REPO_ROOT / 'src' / 'dl_techniques' / 'layers'
ADAPTERS = SRC / 'adapters' / 'gated_adapter.py'
LORA = SRC / 'adapters' / 'lora.py'
GATE = SRC / 'statistics' / 'local_support_gate.py'


def _purge_bytecode_cache():
    """Delete every ``__pycache__`` under the two mutated packages.

    Without this the red proof is worthless: Python compares a ``.pyc``'s
    recorded source mtime AND SIZE against the ``.py``, and a mutation that
    happens to preserve the byte length within the filesystem's mtime
    granularity is judged fresh — so the subprocess imports the ORIGINAL module
    and the guard passes on code that is no longer there. That failure mode is
    silent and reads exactly like a guard that does not work.
    """
    for package in (SRC / 'adapters', SRC / 'statistics'):
        for cache in package.rglob('__pycache__'):
            shutil.rmtree(cache, ignore_errors=True)


def _run_guard(test_nodeid):
    """Run one guard in a fresh process and report whether it FAILED.

    A fresh process because these mutations patch module state, and because a
    red guard must be observed in isolation to be trusted (guide §13.6: a suite
    has ordering-dependent failures, and never gate on a shared process).
    """
    _purge_bytecode_cache()
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_nodeid, '-q', '--no-header',
         '-p', 'no:cacheprovider'],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
        env={
            'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND': 'Agg', 'HOME': '/tmp',
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'VIRTUAL_ENV': '/media/arxwn/data_fast/repositories/dl_techniques/.venv',
        },
    )
    return result.returncode != 0, result.stdout + result.stderr


class TestRedProofs:
    """Each test asserts that injecting a defect turns a named guard RED."""

    def test_the_guards_file_exists_and_is_not_empty(self):
        """Anti-vacuity: the proofs below are worthless against no guards."""
        guards = pathlib.Path(__file__).with_name(
            'test_the_guards_actually_hold.py'
        )
        source = guards.read_text()
        assert 'def test_' in source
        assert len(source) > 1000

    def test_red_removing_the_per_slot_initializer_clone(self):
        """RED PROOF 1 — the D-007 clone.

        Defect: `_a_initializer` calls the shared initializer instance once on the
        stacked 3-D shape, so every slot's `A` slice is drawn from the same
        stateless-deterministic sample.
        Expect: `test_the_per_slot_initializer_clone_is_what_makes_slots_differ`
        and `test_no_two_slots_share_a_bit_identical_a_slice` both go red.
        """
        original = LORA.read_text()
        mutated = original.replace(
            "                clone_initializer(a_initializer_fn)(\n"
            "                    shape=(shape[1], shape[2]), dtype=dtype\n"
            "                )\n",
            "                a_initializer_fn(shape=(shape[1], shape[2]),\n"
            "                                 dtype=dtype)\n",
        )
        assert mutated != original, "the mutation did not apply; proof is void"
        try:
            LORA.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_the_per_slot_initializer_clone_is_what_makes_slots_differ'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            LORA.write_text(original)

    def test_red_replacing_the_gate_score_product_with_the_raw_delta(self):
        """RED PROOF 2 — the gate actually gates.

        Defect: `call` returns the summed adapter delta without multiplying by
        any gate decision, i.e. the gate is computed and then thrown away.
        Expect:
        `test_the_gated_adapter_multiplies_by_the_gate_and_does_not_replace_it`
        goes red on its out-of-phase rate assertion.
        """
        original = ADAPTERS.read_text()
        marker = "            deltas = deltas + ops.broadcast_to(score, ops.shape(delta)) * delta\n"
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "            deltas = deltas + delta  # DEFECT: gate not applied\n",
        )
        try:
            ADAPTERS.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_the_gated_adapter_multiplies_by_the_gate_and_does_not_replace_it'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            ADAPTERS.write_text(original)

    def test_red_making_an_unfitted_gate_open(self):
        """RED PROOF 3 — the safe direction on an unfitted gate.

        Defect: the unfitted branch emits ones instead of zeros, so a fresh gate
        routes every token to the adapter.
        Expect: `test_an_unfitted_gate_routes_everything_closed` goes red, and
        so does the in-phase/out-of-phase pair in the gate suite.
        """
        original = GATE.read_text()
        mutated = original.replace(
            "            scores = ops.zeros(ops.shape(flat)[:-1], 'float32')\n",
            "            scores = ops.ones(ops.shape(flat)[:-1], 'float32')\n"
            "            # DEFECT: an unfitted gate routes everything open\n",
        )
        assert mutated != original, "the mutation did not apply; proof is void"
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_an_unfitted_gate_routes_everything_closed'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)

    def test_red_making_the_streaming_path_step_parameters(self):
        """RED PROOF 4 — sufficient statistics, not parameters.

        Defect: `observe` interpolates the mixture means toward each batch's own
        M-step instead of EMA-ing the accumulators, which is the "last batch
        wins" failure measured at variance 0.12 and identical parameters after
        1 pass and after 60.
        Expect:
        `test_the_streaming_path_accumulates_statistics_not_parameters` goes red
        because the accumulators stop moving.
        """
        original = GATE.read_text()
        # Neutralise the accumulator EMA: force the step to zero so the
        # accumulators freeze, which is exactly the observable consequence.
        marker = "        step = 1.0 / step_index\n"
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "        step = 0.0  # DEFECT: accumulators never advance\n",
        )
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_the_streaming_path_accumulates_statistics_not_parameters'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)

    def test_red_removing_the_causal_mask_from_the_decay_matrix(self):
        """RED PROOF 5 — the smoothing is causal.

        Defect: drop the `offsets >= 0` mask, so each position smooths against
        the WHOLE sequence rather than its own prefix.
        Expect: `test_the_causal_smoothing_never_reads_the_future` goes red on
        its first assertion.

        Note what this REPLACED. The previous version of this proof injected a
        signed (unclamped) exponent instead, and stayed GREEN — correctly, so:
        under an ``ops.where`` mask the discarded branch's ``inf`` values are
        never selected, so a signed exponent is benign in the present
        formulation. A red proof that cannot go red is worse than none, because
        it reads as coverage. The clamp survives as its own clearly-labelled
        defensive guard instead.
        """
        original = GATE.read_text()
        marker = "            offsets >= 0,\n"
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "            ops.ones_like(offsets, 'bool'),  # DEFECT: not causal\n",
        )
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_the_causal_smoothing_never_reads_the_future'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)

    def test_red_dropping_the_fitted_flags_from_get_config(self):
        """RED PROOF 6 — the reload guard.

        Defect: `get_config` omits `fitted_pos`/`fitted_neg`, so a reloaded gate
        reports itself unfitted and routes everything closed despite carrying a
        completed fit in its weights.
        Expect: `test_a_reload_restores_the_fitted_flags_not_just_the_weights`
        and `test_a_saved_gate_restores_its_fitted_state_and_scores` go red.
        """
        original = GATE.read_text()
        mutated = original.replace(
            "            'fitted_pos': self.is_fitted('pos'),\n"
            "            'fitted_neg': self.is_fitted('neg'),\n",
            "",
        )
        assert mutated != original, "the mutation did not apply; proof is void"
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/'
                'test_the_guards_actually_hold.py::'
                'test_a_reload_restores_the_fitted_flags_not_just_the_weights'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)

    def test_red_dropping_the_legacy_config_substitution(self):
        """RED PROOF 7 — the extraction rename.

        Defect: `from_config` no longer substitutes `num_occurrences`, so every
        pre-extraction checkpoint raises on load while every test written after
        the extraction passes.
        Expect: `test_a_pre_extraction_config_still_loads` goes red.
        """
        original = LORA.read_text()
        marker = "        legacy_value = config.pop(LEGACY_NUM_OCCURRENCES_KEY, None)\n"
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "        legacy_value = None  # DEFECT: pre-extraction key not handled\n",
        )
        try:
            LORA.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_adapters/test_lora_adapter.py::'
                'TestSerialization::test_a_pre_extraction_config_still_loads'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            LORA.write_text(original)

    def test_red_repeating_the_revival_variance_to_the_dead_count(self):
        """RED PROOF 9 — the partial-revival broadcast.

        Defect: build the replacement variance as ``(num_dead, d)`` and let it
        meet the ``(K, d)`` live variances in an ``ops.where``. That only works
        when ``num_dead == K``. A PARTIAL revival -- the ordinary case -- raised
        ``SelectV2 ... must be broadcastable`` instead of repairing anything,
        and no smoke test caught it because revivals had always been 0 or all.
        Expect:
        `test_the_repair_revives_exactly_the_components_with_no_mass` goes red.
        """
        original = GATE.read_text()
        marker = (
            "        batch_variance = ops.expand_dims(\n"
            "            ops.var(z, axis=0) + self.variance_floor, 0\n"
            "        )\n"
        )
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "        batch_variance = ops.repeat(\n"
            "            ops.expand_dims(ops.var(z, axis=0) + self.variance_floor, 0),\n"
            "            num_dead, axis=0,\n"
            "        )  # DEFECT: (num_dead, d) cannot meet (K, d)\n",
        )
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_statistics/test_local_support_gate.py::'
                'TestStreamingFit::'
                'test_the_repair_revives_exactly_the_components_with_no_mass'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)

    def test_red_recomputing_the_gate_shape_wrongly(self):
        """RED PROOF 8 — declared shape vs actual.

        Defect: `compute_output_shape` claims the feature axis survives, while
        `call` consumes it. The declared and actual shapes then disagree, and
        only a caller comparing both notices.
        Expect: `test_compute_output_shape_agrees_with_the_forward` goes red.
        """
        original = GATE.read_text()
        marker = "        per_token = shape[:-1]\n"
        assert marker in original, "the mutation target moved; proof is void"
        mutated = original.replace(
            marker,
            "        per_token = shape  # DEFECT: claims the feature axis survives\n",
        )
        try:
            GATE.write_text(mutated)
            red, out = _run_guard(
                'tests/test_layers/test_statistics/test_local_support_gate.py::'
                'TestShape::test_compute_output_shape_agrees_with_the_forward'
            )
            assert red, f"guard stayed GREEN on the injected defect:\n{out}"
        finally:
            GATE.write_text(original)


class TestSourceIntegrity:
    """The mutations above rewrite tracked source; confirm it is restored."""

    @pytest.mark.parametrize('path', [ADAPTERS, LORA, GATE])
    def test_the_source_file_still_parses(self, path):
        ast.parse(path.read_text())

    def test_no_defect_marker_survived_in_the_source(self):
        for path in (ADAPTERS, LORA, GATE):
            assert 'DEFECT:' not in path.read_text(), (
                f"{path.name} still carries an injected DEFECT marker"
            )
