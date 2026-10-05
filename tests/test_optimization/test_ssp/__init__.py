"""Tests for ``dl_techniques.optimization.ssp``.

The module boundary mirrors the source: one file per module, plus a shared
``ssp_oracle`` instrument that the RED proofs import.

Guards worth naming:

* ``test_signal.py`` pins the max-entropy weight's three closed forms
  (``w(0.5) == 1``, ``w(0) == w(1) == 2**-lam``, ``lam == 0 -> w == 1``), the
  Bernoulli-KL identity ``D == ln 2 - H(p)``, the GRPO-reduction property at
  ``lam == 0``, and that the batch mean of the objective is zero BY CONSTRUCTION
  while its gradient is not.
* ``test_fusion.py`` pins that fusion never mutates its inputs, that uniform fusion
  of identical specialists is the bit-exact identity, and that linear and
  task-arithmetic fusion coincide when the base is shared.
* ``test_spectrum.py`` pins the per-subdomain selection against a global argmax, and
  that the sampling weights reuse the max-entropy criterion rather than a second
  copy of it.
* ``test_config.py`` pins the OFF switch as a real no-op and the get_config
  round-trip.
"""
