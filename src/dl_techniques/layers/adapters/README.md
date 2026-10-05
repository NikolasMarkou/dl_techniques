# Adapters (`dl_techniques.layers.adapters`)

Low-rank weight updates, and updates confined to a local support.

## What is here

| File | Contents |
| :--- | :--- |
| `lora.py` | `LoRAAdapter` — an additive low-rank delta with `num_adapters` independent `A`/`B` pairs, selected per call site. |
| `factory.py` | `ADAPTER_REGISTRY` + `create_adapter_layer` / `create_adapter_from_config` / `validate_adapter_config` / `assemble_adapter_config` / `list_adapter_types` / `get_adapter_info` / `get_adapter_requirements`. One key: `lora`. |
| `gated_adapter.py` | `GatedAdapter` — `LoRAAdapter` composed with one `LocalSupportGate` per phase, i.e. the Local Support Learning mechanism. |

## `LoRAAdapter`

Computes a **pure additive delta** `(x @ A[i]) @ B[i] * (alpha / rank)` for a
caller-selected slot `i`. It does **not** own or wrap the base projection it
augments — the caller adds the returned delta onto its own output. That seam is
what lets one adapter serve a `Dense` inside a SwiGLU block and a
grouped-query projection inside a decoder without either knowing the other
exists.

```
x  [..., input_dim]
      │
      ▼  select slot i (a plain Python int, fixed per call site)
    A[i]: (input_dim, rank)
      │
      ▼  [..., rank]
    B[i]: (rank, output_dim)
      │
      ▼  * (alpha / rank)
    delta  [..., output_dim]     ← the CALLER adds this onto its base output
```

The slot index is a **plain Python `int`**, never a tensor: which slots exist is
fixed at model construction, so it must stay outside the traced graph. It
defaults to `0`, which is what makes the layer usable where Keras supplies only
the input — `keras.Sequential`, a functional model, a `fit()`-driven call.

Two consumers use the same axis for unrelated reasons, which is why the layer
is not Zamba2-specific:

* **Zamba2** (`models/language/zamba2/`) — the slot is a *depth position*.
  `num_mem_blocks` physical shared mem-blocks are reused round-robin across far
  more depth positions than there are blocks, and each position gets its own
  pair so it can specialize without multiplying the shared block's parameter
  count.
* **Local Support Learning** (below) — the slot is a *learning phase*.

### Initialization, and why it is written the way it is

`B` is zero-initialised (the standard LoRA convention), so every slot's delta is
exactly zero at construction and attaching the adapter does not perturb an
already-trained block.

`A` is **not** initialized by handing the stacked `(num_adapters, input_dim,
rank)` tensor to a Keras initializer in one call. Two reasons, both measured:

1. Keras' built-in initializers read a 3-D shape as a convolution kernel's
   `(receptive_field, fan_in, fan_out)`, so `num_adapters` would be read as a
   spatial extent and every slice scaled wrongly.
2. Each slice gets a **fresh `clone_initializer` clone** per iteration. A single
   seedless initializer *instance* is stateless-deterministic and replays the
   identical sample at every call of the same shape, which made every slot's `A`
   bit-identical — so every slot read the same rank-`r` subspace of `x` at
   initialisation (`plan-2026-09-12T075714-035fd488/D-007`).

Consequence worth knowing: at the zero-init, `grad_B` is nonzero but `grad_A` is
zero, because `A` only enters the product multiplied by `B`. The two come alive
in order. This is not a defect — it is why the zero-init is inert — but it does
mean a loss quadratic in the delta has zero gradient at step 0.

## `GatedAdapter`

Local Support Learning: an additive update whose effect is confined to the input
region that produced it. Applied everywhere, `ΔW` alters the output of *every*
input, which is why conventional adaptation forgets.

```
x  (..., input_dim)        ← the gate scores the base projection's INPUT
      │
      ├──► LocalSupportGate ──► g  (...,)     1[log Φ_pos(x) > log Φ_neg(x)]
      │
      └──► LoRAAdapter ──────► delta  (..., output_dim)

output = x @ W.T + Σ_p  g_p ⊙ LSL_p(x)          Σ over phases
```

`g_p` is phase `p`'s gate decision on *this* token; `LSL_p(x)` is phase `p`'s
delta via `LoRAAdapter(num_adapters=P)` indexed by `adapter_idx=p`. Every phase
keeps its own delta and its own support estimate, so a later phase cannot
overwrite an earlier one's behaviour. Inference cost grows linearly in the phase
count — the price of not being able to merge the deltas into the base weights.

The gate lives at `dl_techniques.layers.statistics.local_support_gate` and is
fitted by an explicit call (`fit_gate`, or `observe` for the streaming path),
never as a side effect of `call`.

**A gate that has not been fitted routes every token closed**, so an unfitted
phase contributes *nothing* rather than its ungated delta. That is the safe
direction and it is silent: the phase's parameters are present and
trained-looking while having no effect. Check `is_phase_fitted` when a phase
appears to be ignored.

### Freezing a phase

`set_active_phases([...])` restricts which phases contribute. A frozen phase's
delta stays in the checkpoint and its gate stays fitted, but stops being applied,
so the optimizer cannot reach it through this layer.

## The factory is strict

`create_adapter_layer` raises on any keyword the chosen type does not declare;
it never filters-and-drops. The message carries
`factory.STRICT_DROPPED_KEY_MARKER` as a stable substring.

The reason is measured damage, not taste. A factory that silently discards an
undeclared keyword converts a typo into a model that trains and behaves
plausibly while missing the setting the caller asked for; this tree has four
recorded instances (a `dropout=` key landing on `dropout_rate` killed dropout
across every vision encoder; `max_seq_len`/`rope_theta` landing on a type
declaring no rotary parameter made a stack exactly permutation-equivariant).

A **wrapper** carrying a superset of knobs pre-filters through
`assemble_adapter_config`, which drops its own noise silently while forwarding
`caller_args` unfiltered — an explicit caller request still raises.

`GatedAdapter` is deliberately **not** a factory key: it is a composition,
reachable only by naming the class, so a caller cannot reach a gated adapter by
passing a string.

## Tests

`tests/test_layers/test_adapters/`:

* `test_lora_adapter.py` — construction, per-slot independence (D-007), slot
  index reconciliation, gradient flow, serialization, `.keras` round trip on
  values.
* `test_adapter_factory.py` — strictness, registry surface, deep-copy
  isolation, `from_config`.
* `test_the_guards_actually_hold.py` — single-claim guards, sentence-named.
* `test_the_red_proofs.py` — **not a routine run.** Rewrites library source to
  prove each guard goes red on its own defect; marked `red_proof`. Run it
  deliberately:
  `pytest tests/test_layers/test_adapters/test_the_red_proofs.py -v`

A note on that last one: two of the original red proofs could not go red and
were replaced. One injected a signed (unclamped) exponent into the smoothing
decay matrix and stayed green — correctly, because the `ops.where` mask discards
the offending branch anyway. A red proof that cannot go red is worse than none,
because it reads as coverage. The clamp survives as a separately-labelled
defensive guard, and the replacement proof breaks causality instead.

## References

* Hu et al., 2021. LoRA. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* Ben-Kish, Kumar, Glass & Giryes, 2026. Local Support Learning.
  [arXiv:2610.02126](https://arxiv.org/abs/2610.02126)
* Glorioso et al., 2024. Zamba2. [arXiv:2411.15242](https://arxiv.org/abs/2411.15242)
