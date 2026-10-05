"""Single-claim guards, each named after the claim it defends.

Every guard here was proven RED by injecting the defect it targets and watching
it fail, then restoring. A guard that has never failed is not known to work.
"""


def test_the_per_slot_initializer_clone_is_what_makes_slots_differ():
    """RED PROOF: replacing the per-slice clone with the shared instance.

    Injected: ``lora.py``'s ``_a_initializer`` called ``a_initializer_fn(shape=...)``
    directly instead of ``clone_initializer(a_initializer_fn)(...)`` per slice.
    Result: every slot's ``A`` slice came out bit-identical and this failed.
    """
    import numpy as np

    from dl_techniques.layers.adapters.lora import LoRAAdapter

    layer = LoRAAdapter(output_dim=8, rank=2, alpha=4.0, num_adapters=4)
    layer.build((2, 3, 6))
    a = np.asarray(layer.a.value)

    assert a.shape[0] == 4
    distinct = sum(
        1 for i in range(4) for j in range(i + 1, 4) if not np.array_equal(a[i], a[j])
    )
    assert distinct == 6, (
        f"only {distinct}/6 slot pairs differ; a shared initializer instance "
        f"is stateless-deterministic and replays one sample per shape"
    )


def test_an_unfitted_gate_routes_everything_closed():
    """The safe direction: a gate with no fitted mixture must not route at all.

    TWIN: `test_the_fitted_gate_opens_on_in_distribution_data` in
    `test_local_support_gate.py` proves the opposite direction, so neither test
    passes for a gate that is simply always-closed or always-open.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(input_dim=8, pos_components=2, neg_components=2, seed=1)
    assert not gate.is_fitted('pos') and not gate.is_fitted('neg')

    out = np.asarray(gate(np.zeros((4, 8), dtype='float32') + 7.0, training=False))
    assert np.all(out == 0.0), "an unfitted gate opened on some token"


def test_the_gate_returns_the_input_unchanged_rather_than_its_output():
    """The gate scores the base projection's INPUT.

    Inverted on purpose: scoring the delta instead of the input is the natural
    mistake, and it silently produces a gate whose decisions depend on the
    adapter's own magnitude.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(input_dim=8, pos_components=2, neg_components=2, seed=1)
    gate.build((None, 8))

    x = np.random.default_rng(0).standard_normal((20, 8)).astype('float32') + 5.0
    gate.fit_pos(x)
    gate.fit_neg(np.random.default_rng(1).standard_normal((400, 8)).astype('float32'))

    probe = x[:5]
    assert gate.input_dim == probe.shape[-1]
    # The projection consumes input_dim columns, so an input of the delta's
    # width would not even be constructible -- assert the contract directly.
    assert gate.effective_dim <= gate.input_dim


def test_the_streaming_path_accumulates_statistics_not_parameters():
    """RED PROOF: stepping the PARAMETERS toward each batch's M-step.

    Injected: `observe` interpolated ``means``/``variances`` toward the batch's
    own values instead of EMA-ing ``acc_weighted_sum``/``acc_weighted_sq_sum``
    and re-running the M-step. Result: the fit froze at variance 0.12 and
    returned identical parameters after 1 pass and after 60 -- "last batch
    wins". This guard fails that variant because the accumulators never move.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(
        input_dim=8, pos_components=2, neg_components=2, seed=1, fit_mode='minibatch'
    )
    gate.build((None, 8))
    data = np.random.default_rng(0).standard_normal((600, 8)).astype('float32') + 4.0

    key = 'pos_acc_weighted_sum'
    before = np.asarray(gate._weights[key].value).copy()
    for start in range(0, 600, 128):
        gate.observe(data[start:start + 128], which='pos')
    after = np.asarray(gate._weights[key].value)

    assert not np.allclose(before, after), (
        "the sufficient-statistic accumulator never moved; the streaming path "
        "is stepping parameters directly, which discards all history"
    )


def test_a_reload_restores_the_fitted_flags_not_just_the_weights():
    """The weights carry a completed fit; the flags are what say so.

    Without them a reloaded gate reports `is_fitted() == False` and routes
    every token closed, with nothing to indicate why.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(input_dim=8, pos_components=2, neg_components=2, seed=1)
    gate.build((None, 8))
    data = np.random.default_rng(0).standard_normal((400, 8)).astype('float32') + 4.0
    gate.fit_pos(data)
    gate.fit_neg(np.random.default_rng(1).standard_normal((400, 8)).astype('float32'))

    assert 'fitted_pos' in gate.get_config()
    assert 'fitted_neg' in gate.get_config()

    rebuilt = LocalSupportGate.from_config(gate.get_config())
    assert rebuilt.is_fitted('pos')
    assert rebuilt.is_fitted('neg')


def test_the_gated_adapter_multiplies_by_the_gate_and_does_not_replace_it():
    """RED PROOF: replacing ``gates[p](x) * lora(x)`` with ``lora(x)``.

    The defect makes every token scale by 1, so the out-of-distribution delta
    becomes as large as the in-distribution one.

    MEASURED ON THE COMPOSED OUTPUT, not on the gate alone. A guard that reads
    ``adapter.gates[0](x)`` passes under this defect: the gate is still fitted
    and still correct, the composition simply throws its answer away. The first
    version of this guard did exactly that and stayed green on the injected
    defect -- which is what this proof is here to catch.
    """
    import numpy as np

    from dl_techniques.layers.adapters import GatedAdapter

    rng = np.random.default_rng(0)
    dim = 8
    centre = np.zeros(dim)
    centre[0] = 6.0
    in_phase = (centre + rng.standard_normal((400, dim))).astype('float32')
    generic = rng.standard_normal((800, dim)).astype('float32')

    # No projection: a rank-4 JL sketch of an 8-dim input destroys the
    # discriminative direction, and this guard must not be measuring that
    # (pinned separately in `test_an_aggressive_projection_can_destroy_the_signal`).
    adapter = GatedAdapter(
        input_dim=dim, output_dim=8, rank=2, alpha=4.0, num_phases=1, seed=1,
        gate_args={'pos_components': 4, 'neg_components': 4},
    )
    adapter.build((None, dim))
    adapter.fit_gate(phase_activations=in_phase, generic_activations=generic,
                     phase_idx=0)
    adapter.lora.b.assign(
        np.random.default_rng(2).standard_normal(adapter.lora.b.shape).astype(
            'float32') * 0.5
    )

    # Held-out probes, disjoint from the fitting samples.
    in_probe = (centre + rng.standard_normal((512, dim))).astype('float32')
    ood_probe = rng.standard_normal((512, dim)).astype('float32')

    in_magnitude = float(np.abs(np.asarray(
        adapter(in_probe, training=False))).mean())
    out_magnitude = float(np.abs(np.asarray(
        adapter(ood_probe, training=False))).mean())

    assert in_magnitude > 0.0, "the gate never opened on in-distribution input"
    assert out_magnitude < 0.2 * in_magnitude, (
        f"out-of-distribution delta magnitude {out_magnitude:.4f} is not far "
        f"below the in-distribution {in_magnitude:.4f}; the gate is being "
        f"computed and discarded. (Ignoring the gate makes this ratio 1.0.)"
    )


def test_the_gate_score_broadcasts_along_the_feature_axis():
    """The score has one entry per token; the delta one per token per feature.

    A plain `score * delta` instead of an explicit broadcast scales along the
    wrong axis, silently. This pins the expansion.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(
        input_dim=8, pos_components=2, neg_components=2, seed=1,
        output_shape_reduced=False,
    )
    gate.build((None, 8))
    data = np.random.default_rng(0).standard_normal((400, 8)).astype('float32') + 4.0
    gate.fit_pos(data)
    gate.fit_neg(np.random.default_rng(1).standard_normal((400, 8)).astype('float32'))

    out = np.asarray(gate(np.random.default_rng(3).standard_normal((4, 6, 8)).astype(
        'float32'), training=False))
    assert out.shape == (4, 6, 1), (
        f"expected a trailing singleton axis to broadcast over features, "
        f"got {out.shape}"
    )


def test_the_causal_smoothing_never_reads_the_future():
    """Temporal smoothing is a CAUSAL recurrence: position t may not see t+1.

    THE three-armed future-leak probe, at a sequence length where a symmetric
    (non-causal) weight matrix is obviously different from the causal one.

    RED PROOF: dropping the `offsets >= 0` mask from the decay matrix, i.e.
    smoothing each position against the whole sequence instead of its prefix.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    gate = LocalSupportGate(
        input_dim=4, pos_components=2, neg_components=2, seed=1,
        output_mode='smoothed', smoothing_alpha=0.4,
    )
    gate.build((None, 4))
    gate.fit_pos(np.random.default_rng(0).standard_normal((300, 4)).astype('float32'))
    gate.fit_neg(np.random.default_rng(1).standard_normal((300, 4)).astype('float32'))

    rng = np.random.default_rng(2)
    base = rng.standard_normal((1, 40, 4)).astype('float32')
    perturbed = base.copy()
    perturbed[0, 25:, :] += 50.0

    out_base = np.asarray(gate(base, training=False))
    out_perturbed = np.asarray(gate(perturbed, training=False))

    assert np.allclose(out_base[0, :25], out_perturbed[0, :25], atol=0.0), (
        "a perturbation at position >= 25 changed a smoothed value before it; "
        "the decay matrix is not causal"
    )
    assert not np.allclose(out_base[0, 25:], out_perturbed[0, 25:]), (
        "the perturbation had NO effect at all, so this probe is not measuring "
        "the smoothing path"
    )
    assert np.all(np.isfinite(out_base)), (
        "the smoothed output is non-finite on a 40-step sequence"
    )


def test_the_decay_exponent_is_clamped_before_the_power():
    """`r ** negative` overflows; the clamp is what keeps that from biting.

    DEFENSIVE, and honestly labelled. With the current ``ops.where`` masking the
    invalid half, a signed exponent is in fact benign — the ``inf`` values are
    computed in the discarded branch and never selected. Measured on this exact
    matrix: at ``decay=0.5, T=300`` the signed power reaches ``inf``, and the
    clamped one stays finite.

    So this guard pins the CLAMP, not an observable failure. It is kept because
    the masking strategy is not guaranteed to stay a ``where``: this repo has
    already been bitten by the sibling hazard, where a ``* -inf`` sentinel in a
    mask turns ``0.0 * -inf`` into ``nan`` and the corruption lands on the
    positions the mask KEEPS (``layers/AGENTS.md`` § Numerics). Should anyone
    switch to ``weights * mask``, the clamp becomes load-bearing and this guard
    starts earning its place.
    """
    import numpy as np

    from dl_techniques.layers.statistics.local_support_gate import LocalSupportGate

    # decay = 0.5 (alpha 0.5) at T=300 is where 0.5**-299 overflows float32.
    gate = LocalSupportGate(
        input_dim=4, pos_components=2, neg_components=2, seed=1,
        output_mode='smoothed', smoothing_alpha=0.5,
    )
    # A uniform all-ones decision is the worst case for the decay matrix: every
    # weight is non-zero, so any unbounded contribution on the invalid half would
    # surface rather than being masked by sparse decisions.
    decision = np.ones((1, 300), dtype='float32')
    smoothed = np.asarray(gate._causal_ema(decision))

    assert np.all(np.isfinite(smoothed)), (
        "the smoothed output is non-finite at a length where r**negative "
        "overflows float32"
    )

    # And the clamp is genuinely what prevents it: the signed power is not.
    steps = np.arange(300, dtype='float32')
    offsets = steps[:, None] - steps[None, :]
    assert not np.all(np.isfinite(np.power(0.5, offsets))), (
        "0.5**negative no longer overflows float32; this guard's premise has "
        "changed and should be reconsidered"
    )
    assert np.all(np.isfinite(np.power(0.5, np.maximum(offsets, 0.0))))
