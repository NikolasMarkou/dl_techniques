"""F-21: `top_k_schedule` must reach the graph that actually retrieves.

``PhaseScheduler._apply_top_k_schedule`` assigns ``model.read_controller.top_k``
-- a plain Python attribute -- and ``MemoryReadController.call`` consumes it as
a Python int at TRACE time (``ops.top_k(sim_clipped, k=self.top_k)``). An
already-traced ``train_function`` therefore keeps the k that was baked into it
when ``fit()`` traced the graph, so the log printed ``top_k 4 -> 2`` while the
model kept retrieving 4 keys for the rest of training. This is the exact defect
class the same package documents as eliminated (D-016, and the module docstring
of ``phase_scheduler.py``), reintroduced by the one line that flips a Python
attribute.

**Why this file exists next to the one that already "tests" the schedule.**
``test_phase_scheduler.py::TestTopKSchedule`` drives a pure-Python ``_MockModel``
with a ``_ReadControllerStub`` and asserts that an attribute assignment assigned
an attribute. It would pass with ``MemoryReadController`` deleted. Nothing there
runs a traced step across a top_k transition, which is the only regime in which
the defect exists.

The instrument is a spy on ``keras.ops.top_k`` that records the ``k`` of every
call. A call recorded while a ``tf.function`` is tracing is, by construction, the
constant baked into that graph -- so "the new k was never passed to ``ops.top_k``
after the transition" is a direct measurement of the retrieved key count, not a
proxy for it.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.memory_bank.phase_scheduler import PhaseScheduler

from .test_phase_scheduler import (
    _compiled_probe_model,
    _one_batch_dataset,
    _tiny_kwargs,
)


_CONFIGURED_TOP_K = _tiny_kwargs()["top_k"]  # 4
_SCHEDULED_TOP_K = 2


class _TopKSpy:
    """Records the ``k`` of every ``keras.ops.top_k`` call, then delegates."""

    def __init__(self, real):
        self._real = real
        self.ks = []

    def __call__(self, x, k=None, *args, **kwargs):
        self.ks.append(int(k))
        return self._real(x, k, *args, **kwargs)


def _run_across_a_transition(monkeypatch, epochs=3):
    """Fit across a phase 1 -> 2 boundary with a top_k schedule attached.

    ``phase1_steps=1`` (never 0) for the reason spelled out in
    ``TestCurriculumReachesTheTracedGraph``: the transition must land AFTER
    ``make_train_function()`` has traced the graph, which is the only moment
    the defect is observable.
    """
    spy = _TopKSpy(keras.ops.top_k)
    monkeypatch.setattr(keras.ops, "top_k", spy)

    m = _compiled_probe_model(
        top_k_schedule=lambda step: _SCHEDULED_TOP_K,
    )
    m.fit(
        _one_batch_dataset(),
        epochs=epochs,
        verbose=0,
        callbacks=[
            PhaseScheduler(
                phase1_steps=1, phase2_steps=1_000, phase3_steps=1_000,
            ),
        ],
    )
    return m, spy


class TestTopKScheduleReachesTheTracedGraph:

    def test_the_scheduled_k_is_actually_retrieved(self, monkeypatch):
        m, spy = _run_across_a_transition(monkeypatch)

        # Liveness / anti-vacuity: the spy really did observe the traced
        # retrieval, at the configured k, before the transition. Without this
        # arm an empty `spy.ks` would make the real assertion below
        # unfalsifiable in the wrong direction.
        assert _CONFIGURED_TOP_K in spy.ks, (
            f"the spy never saw the configured top_k={_CONFIGURED_TOP_K}; it "
            f"is not observing the retrieval at all. saw: {spy.ks}"
        )

        assert _SCHEDULED_TOP_K in spy.ks, (
            f"top_k_schedule set read_controller.top_k="
            f"{m.read_controller.top_k}, but `ops.top_k` was never traced with "
            f"k={_SCHEDULED_TOP_K}: the already-traced train_function keeps "
            f"retrieving {_CONFIGURED_TOP_K} keys forever and the schedule is "
            f"a log line only (F-21). k values passed to ops.top_k across the "
            f"whole run: {spy.ks}"
        )

    def test_the_python_attribute_moves_too(self, monkeypatch):
        """The half the pre-existing mock test covered. Kept for the contrast:
        this assertion passed at HEAD while the model retrieved the old k."""
        m, _ = _run_across_a_transition(monkeypatch)
        assert m.read_controller.top_k == _SCHEDULED_TOP_K

    def test_no_schedule_means_no_forced_retrace(self, monkeypatch):
        """Control: a model with no `top_k_schedule` must not pay a retrace at
        every phase boundary, and must keep retrieving its configured k."""
        spy = _TopKSpy(keras.ops.top_k)
        monkeypatch.setattr(keras.ops, "top_k", spy)

        m = _compiled_probe_model()
        m.fit(
            _one_batch_dataset(),
            epochs=3,
            verbose=0,
            callbacks=[
                PhaseScheduler(
                    phase1_steps=1, phase2_steps=1_000, phase3_steps=1_000,
                ),
            ],
        )
        assert set(spy.ks) == {_CONFIGURED_TOP_K}, spy.ks
