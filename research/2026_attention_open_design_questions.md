# Open Design Questions in `layers/attention/`

**Author:** Nikolas Markou · **Date:** 2026-07-28 · **Status:** Decision brief (no code changes)
**Scope:** `src/dl_techniques/layers/attention/` at commit `58232e63`
**Produced by:** `plan-2026-07-27T183600-b4ef45f0`, step 9 (Tier 4)

> **Read this first.** **None of the 15 questions below were changed by the plan that
> wrote this document.** Every one is live at HEAD. This is a decision brief, not a
> changelog: it records what the code does today, what the concrete options are, who
> would be hit by each option, and which option the author would take. Line numbers were
> re-verified against HEAD while writing — they drifted repeatedly during the fix pass
> that preceded this note, so do not trust any line number quoted in an older plan file.

---

## Table of Contents

1. [What the preceding plan actually shipped](#1-what-the-preceding-plan-actually-shipped)
2. [How to read this brief](#2-how-to-read-this-brief)
3. [Question index](#3-question-index)
4. [Group A — Mask support that does not exist](#group-a--mask-support-that-does-not-exist)
   - [Q1 — `mobile_mqa` accepts `attention_mask` and ignores it](#q1--mobile_mqa-accepts-attention_mask-and-ignores-it)
   - [Q2 — `shared_weights_cross_attention` accepts `attention_mask` and never reads it](#q2--shared_weights_cross_attention-accepts-attention_mask-and-never-reads-it)
   - [Q3 — `anchor_attention` has no mask, so left-padding poisons the anchor set](#q3--anchor_attention-has-no-mask-so-left-padding-poisons-the-anchor-set)
   - [Q4 — `TransformerLayer` silently DROPS a caller's mask for three attention types (NEW)](#q4--transformerlayer-silently-drops-a-callers-mask-for-three-attention-types-new)
5. [Group B — Numerical conditioning and backend limits](#group-b--numerical-conditioning-and-backend-limits)
   - [Q5 — `rpc_attention` cannot run under `mixed_float16` at all](#q5--rpc_attention-cannot-run-under-mixed_float16-at-all)
   - [Q6 — `MASK_BIAS_VALUE = -1e9` costs `rpc_attention` ~6% accuracy](#q6--mask_bias_value---1e9-costs-rpc_attention-6-accuracy)
   - [Q7 — `performer_attention` mis-scales odd `nb_features`](#q7--performer_attention-mis-scales-odd-nb_features)
   - [Q8 — `wave_field_attention.scale` uses the inverse of the package convention](#q8--wave_field_attentionscale-uses-the-inverse-of-the-package-convention)
6. [Group C — Design ambiguities inherited from the source papers](#group-c--design-ambiguities-inherited-from-the-source-papers)
   - [Q9 — TripSE branch combination: sum vs average (2-vs-2, not 1-vs-2)](#q9--tripse-branch-combination-sum-vs-average-2-vs-2-not-1-vs-2)
   - [Q10 — `hopfield_attention` feeds back `A K` but returns `A V`](#q10--hopfield_attention-feeds-back-a-k-but-returns-a-v)
   - [Q11 — `rpc_attention` `return_attention_scores=True` returns weights that did not produce the output](#q11--rpc_attention-return_attention_scorestrue-returns-weights-that-did-not-produce-the-output)
7. [Group D — Framework-level questions](#group-d--framework-level-questions)
   - [Q12 — Should explicit `training=` forwarding be a package-wide rule?](#q12--should-explicit-training-forwarding-be-a-package-wide-rule)
   - [Q13 — `perceiver_attention.build` forwards a container its sibling rejects](#q13--perceiver_attentionbuild-forwards-a-container-its-sibling-rejects)
8. [Group E — Deferred by tier, recorded here so they are not lost](#group-e--deferred-by-tier-recorded-here-so-they-are-not-lost)
   - [Q14 — Dead constructor parameters that are also `get_config()` keys](#q14--dead-constructor-parameters-that-are-also-get_config-keys)
   - [Q15 — `LighthouseAttention` is the package's known-red pair](#q15--lighthouseattention-is-the-packages-known-red-pair)
9. [Corrections to prior findings](#9-corrections-to-prior-findings)

---

## 1. What the preceding plan actually shipped

A future reader needs the baseline, because several questions below only make sense
against it. `plan-2026-07-27T183600-b4ef45f0` did three things.

**Tier 1 — the `mixed_float16` mask-NaN family (10 modules, fixed).** Every masked
attention layer in the package applied its mask bias arithmetically
(`logits + (1 - keep) * -1e9`) in the compute dtype. Under `mixed_float16` the constant
becomes `-inf`, and at every *unmasked* position `0 * -inf = NaN` — so an all-ones mask
that masks nothing NaN'd the entire batch. One shared helper now exists:

```python
# src/dl_techniques/layers/attention/common.py
apply_attention_mask(logits, keep, *, out_dtype=None, rescue_axis=-1)
```

It builds the bias with `ops.where` inside `mask_dtype(...)` (`>= float32`), takes the
**keep predicate** from the caller so polarity is never inferred, and by default rescues
a fully-masked softmax slice ("keeps nothing" is treated as "keeps everything") so a
degenerate row is finite garbage rather than `NaN`. Adopted at `rpc`, `capsule_routing`,
`gated`, `group_query`, `multi_head_cross`, `hopfield`, `single_window`,
`multi_head_latent`, `differential` and `ring`.

**Tier 2 — six non-mask defects (fixed).** `ring_attention` rank-2 mask
(`UnboundLocalError`), `perceiver_attention.compute_output_shape` returning `None`,
`SpatialAttention.build` missing its rank check, `attention_routing_capsule`
initializer/regularizer plumbing, `capsule_routing_attention`'s misplaced static-`seq_len`
guard, and `tripse_attention`'s missing `training=` forwarding.

**In-flight (fixed).** Two dtype defects surfaced during testing:
`differential_attention.get_lambda()` (could not run under `mixed_float16` *or*
`float64`, mask or no mask) and `RotaryPositionEmbedding`'s float32-pinned cos/sin cache
(took `gated`, `group_query` and `multi_head_latent` down under `float64`).

**What it deliberately did NOT do.** Tier 3 (dead `get_config()` parameters — the user
excluded it; see [Q14](#q14--dead-constructor-parameters-that-are-also-get_config-keys)),
Tier 4 (this document — a written deliverable only, zero source edits), and the
`LighthouseAttention` causality defect (see
[Q15](#q15--lighthouseattention-is-the-packages-known-red-pair)).

---

## 2. How to read this brief

Each question carries six fields, in a fixed order:

| Field | What it means |
|---|---|
| **Today** | The current behavior, with a `file.py:line` verified at HEAD. |
| **Why it matters** | A concrete input → wrong output/crash. Not an abstraction. |
| **Options** | At least two, each written as *"X **at the cost of** Y"*. |
| **Blast radius** | Actual consumers, derived by grepping `src/` and `tests/` for the class name **and** for the `factory.py` registry string, because two consumers in this package are reached only through a factory string. |
| **Pinned by a test?** | The test id, or "no". Most of these are *unguarded* — an in-source comment is a comment, not a guard. |
| **Recommendation** | One option, plus what would change the author's mind. |

A note on blast radius that cost real time to learn: a name grep is **not sufficient** in
this package. `create_attention_layer(attention_type='anchor', ...)` and
`TransformerLayer(attention_type='perceiver', ...)` reach `AnchorAttention` and
`PerceiverAttention` through a string, and
`src/dl_techniques/layers/time_series/mixed_sequential_block.py:267` does exactly that —
a consumer absent from every prior blast-radius table in this plan family. Grep the
registry key too.

---

## 3. Question index

| # | Question | Group | Severity | Pinned? |
|---|---|---|---|---|
| Q1 | `mobile_mqa` ignores `attention_mask` | A | medium (production consumer) | no |
| Q2 | `shared_weights_cross_attention` ignores `attention_mask` | A | low (no consumer) | no |
| Q3 | `anchor_attention` left-padding corrupts the anchor set | A | **high** (silent, consumed) | no |
| Q4 | `TransformerLayer` drops the mask for 3 attention types | A | **high** (silent, NEW) | no |
| Q5 | `rpc_attention` has no fp16 path (`Svd[T=DT_HALF]` absent) | B | low (library-only) | **yes** |
| Q6 | `MASK_BIAS_VALUE = -1e9` costs rpc ~6% | B | medium (package-wide constant) | **yes** |
| Q7 | `performer_attention` odd `nb_features` mis-scale | B | low (library-only) | no |
| Q8 | `wave_field_attention.scale` inverse convention | B | low (naming) | no |
| Q9 | TripSE sum vs average | C | low (library-only) | no |
| Q10 | `hopfield` feeds back `A K`, returns `A V` | C | low (library-only) | no |
| Q11 | rpc `return_attention_scores` is ~2x cost AND unmasked | C | medium (correctness of a returned value) | partial |
| Q12 | Package-wide explicit `training=` forwarding | D | policy question | **yes** |
| Q13 | `perceiver.build` forwards a container its sibling rejects | D | low (one consumer) | **yes** |
| Q14 | Dead `get_config()` parameters (Tier 3) | E | low (documentation) | no |
| Q15 | `LighthouseAttention` known-red pair | E | medium (2 red tests) | **yes (red)** |

---

# Group A — Mask support that does not exist

Three layers accept an `attention_mask` argument and do nothing with it, and one
*consumer* silently discards a mask the caller supplied. In all four cases the failure is
the same shape: **a caller who masks gets an unmasked result, with no exception, no
warning, and no shape error.** That is the most dangerous failure class in this package,
which is why it leads the brief.

## Q1 — `mobile_mqa` accepts `attention_mask` and ignores it

**Today.** `MobileMQA.call` takes `attention_mask` at
`mobile_mqa.py:254`, documents it as "Currently **IGNORED**" at `mobile_mqa.py:265-272`,
and the forward path carries only a comment where masking would go
(`mobile_mqa.py:333-335`):

```python
# Note: Masking is typically not used in standard MobileMQA vision contexts,
# but if passed, would need careful handling due to downsampling.
# We omit mask logic here to strictly match the vision use-case or GQA super if needed.
```

The argument is not forwarded anywhere. `MobileMQA` subclasses `GroupedQueryAttention`
(`mobile_mqa.py:63`) but **overrides `call` entirely**, so the parent's now-fixed masking
path is not inherited — the Tier-1 mask fix landed at `group_query_attention.py` and does
not reach this subclass.

**Why it matters.** The hard part is real and is stated in the docstring: `MobileMQA`
optionally **spatially downsamples K and V**, so the key sequence length is
`(H/s)·(W/s)` while the query sequence length is `H·W`. A caller's token mask is indexed
by the *query* length. There is no unambiguous way to project it onto a downsampled key
axis. Concretely: `MobileMQA(dim=64, num_heads=4, downsampling_factor=2)` on a
`(2, 16, 16, 64)` input has 256 queries and 64 keys; a `(2, 256)` padding mask has no
defined restriction to 64 positions.

**Options.**

- **(a) Wire it up with an explicit downsampling contract** — define that the mask is
  max-pooled onto the key grid with the same stride as K/V (a key block is masked only if
  *every* source position in it is masked). *Buys* real masking for the vision-transformer
  use case **at the cost of** inventing a semantic that no paper specifies, and of a
  mask-shape contract that changes meaning when `downsampling_factor` changes — i.e. a
  config key silently reinterprets a caller's mask.
- **(b) Raise on a non-`None` mask** — turn the silent no-op into a loud
  `NotImplementedError`. *Buys* the elimination of the entire silent-wrong-answer class
  **at the cost of** breaking any caller who currently passes a mask harmlessly (e.g. a
  generic wrapper that forwards `attention_mask` unconditionally), and of a signature
  that advertises a capability it refuses to provide.
- **(c) Support masking only when `downsampling_factor == 1`, raise otherwise** —
  *buys* correct masking on the unambiguous path **at the cost of** a capability that
  depends on an unrelated constructor argument, which is exactly the kind of conditional
  API that surprises people.
- **(d) Leave as-is (documented dead)** — *buys* zero risk **at the cost of** keeping a
  silent-wrong-answer path in a production model.

**Blast radius.** `src/dl_techniques/models/mobilenet/mobilenet_v4.py` (the only
non-package consumer). Factory key `'mobile_mqa'` (`factory.py:474`) — no `src/` consumer
reaches it through the string. Tests: `tests/test_layers/test_attention/test_mobile_mqa.py`
(zero `attention_mask` occurrences — nothing to break). Option (b) would be visible to
MobileNetV4 only if that model ever starts passing a mask; it does not today.

**Pinned by a test?** No. `test_mobile_mqa.py` never supplies a mask.

**Recommendation: (c), with (b) as the fallback for the downsampled case.** The
`downsampling_factor == 1` path is ordinary MQA and there is nothing ambiguous about it;
the downsampled path genuinely has no defensible default, and a `NotImplementedError`
naming `downsampling_factor` is a better answer than a silently ignored argument. **What
would change my mind:** if a real consumer needs masked *and* downsampled attention, (a)
with the max-pool contract becomes necessary and should be written up as its own design
note before being implemented — do not let it arrive as a "small fix".

---

## Q2 — `shared_weights_cross_attention` accepts `attention_mask` and never reads it

**Today.** `SharedWeightsCrossAttention.call` takes `attention_mask` at
`shared_weights_cross_attention.py:366`, documents it as "**Accepted and never read**"
at `:378-384`, and carries an explicit do-not-fix-casually note at `:391-395`. Neither
`_two_modality_attention` (`:429`) nor `_anchor_query_attention` (`:506`) has a mask
parameter at all.

**Why it matters.** This layer takes a **concatenated** sequence plus `split_sizes`, and
in 4-way mode splits it into `[mod_a_anchor, mod_a_query, mod_b_anchor, mod_b_query]`.
A caller who right-pads modality A to a fixed length and passes the obvious
`(batch, total_seq_len)` mask gets padding tokens participating at full weight in every
cross-attention score. With a shared-weight bidirectional design that means modality B's
representation is contaminated by modality A's padding, and vice versa — no exception,
plausible-looking outputs, degraded training.

**Options.**

- **(a) Thread the mask through both helpers, slicing it with the same `split_sizes`
  the inputs use.** *Buys* correct masking with the slicing rule already established by
  the layer's own input handling **at the cost of** deciding the 4-way anchor semantics
  (does an anchor sub-slice mask apply to the query sub-slice's read of it? — yes, but
  that is a decision, not a derivation), plus a real behavior change for anyone currently
  passing a mask and getting a no-op.
- **(b) Remove the parameter.** *Buys* an honest signature **at the cost of** a public
  API break for a layer registered in the factory (`'shared_weights_cross'`,
  `factory.py:662`) with `optional_params` that a config might name.
- **(c) Raise `NotImplementedError` on non-`None`.** *Buys* loudness with no semantic
  invention **at the cost of** the same "advertises then refuses" awkwardness as Q1(b).
- **(d) Leave as-is.** *Buys* nothing beyond zero risk.

**Blast radius.** **None outside the package.** Grep for `SharedWeightsCrossAttention`
across `src/` and `tests/` finds only `layers/attention/__init__.py`,
`layers/attention/factory.py`, `tests/test_layers/test_attention/test_attention_factory.py`
and its own test file. No `src/` file constructs it by the factory string either. This is
the *cheapest* of the four Group-A questions to fix.

**Pinned by a test?** No. `test_shared_weights_cross_attention.py` has zero
`attention_mask` occurrences.

**Recommendation: (a).** Zero external consumers means the behavior change costs nothing,
and the slicing rule is not actually ambiguous — `split_sizes` already defines it. The
4-way anchor question has one sensible answer (an anchor masked in the input is masked as
a key for every reader of it) and should be written into the docstring as a decision, not
left implicit. **What would change my mind:** if the 4-way mode turns out to have a
second defensible anchor semantics that a downstream model needs, prefer (c) until a
consumer exists to arbitrate — inventing the wrong semantics is worse than raising.

---

## Q3 — `anchor_attention` has no mask, so left-padding poisons the anchor set

**Today.** `AnchorAttention.call` is `(x, num_anchor_tokens=None, training=None)` — no
mask parameter, deliberately, as a frozen signature carve-out documented at
`anchor_attention.py:213-225`. Anchors are the **first `K` positions, chosen
positionally** (`_hierarchical_attention`, `anchor_attention.py:567`). Every non-anchor
token reads *only* from the anchor keys and values.

**Why it matters.** This is the most concretely dangerous item in the brief, because the
corruption is proportional to the padding and is completely silent. Take a batch of two
sequences of true lengths 40 and 8, **left**-padded to 48, with `num_anchor_tokens=8`:

- Row 0's anchors are 8 real tokens. Correct.
- Row 1's anchors are **8 pad tokens**. Every one of that row's 40 real tokens
  cross-attends exclusively to an all-padding global summary. The row's output is a
  function of the padding embedding and nothing else.

Right-padding is safe, and the docstring says so. But left-padding is the *default* in
most decoder-side tokenizer tooling, so the trap is well placed to catch someone.

**Options.**

- **(a) Add `attention_mask` to `call()`** and use it both to exclude masked keys and to
  select anchors from the first `K` *unmasked* positions. *Buys* correctness under any
  padding convention **at the cost of** breaking the frozen-signature carve-out — the
  factory's dispatch and `TransformerLayer._MASKLESS_ATTENTION_TYPES`
  (`transformer.py:196`) both encode "this layer takes no mask", so the change is not
  local to one file.
- **(b) New layer (`MaskedAnchorAttention`), leave this one frozen.** *Buys* correctness
  for callers who need it, with zero risk to the existing signature contract, **at the
  cost of** two near-duplicate layers and the ongoing question of which one a new caller
  should pick.
- **(c) Detect and raise.** A left-padded batch is not detectable from `x` alone (there is
  no pad token id at the layer's level), so this option is **not available** — recorded
  so nobody re-proposes it.
- **(d) Loud documentation only (status quo).** *Buys* zero risk **at the cost of** a
  guarantee that someone eventually loses a training run to it, since the docstring is
  the only guard.

**Blast radius.** `src/dl_techniques/layers/transformers/transformer.py` (constructs it
at `:441` under `attention_type == 'anchor'`, calls it maskless at `:624` / `:653`);
`src/dl_techniques/layers/time_series/mixed_sequential_block.py:267` (**factory-string
consumer** — reached as `'anchor'`, invisible to a class-name grep);
`tests/test_layers/test_transformers/test_transformer.py`. Factory key `'anchor'`
(`factory.py:154`). Option (a) touches all three plus `factory.py`'s `optional_params`.

**Pinned by a test?** No. `test_anchor_attention.py` contains no padding test at all — a
RED test would compare a left-padded and a right-padded encoding of the same logical
sequence and show they differ.

**Recommendation: (b), plus an immediate cheap guard on (d).** The carve-out exists for a
real reason and (a)'s blast radius reaches the factory dispatch table; a separate layer is
the honest way to add the capability. But regardless of which is chosen, the *first* piece
of work here should be the RED test described above — right now the entire mitigation is a
docstring paragraph, and this package has already learned that an anchor is a comment, not
a guard. **What would change my mind:** if `TransformerLayer` grows a general
"mask-capable attention type" contract for other reasons, folding anchor into it (option
(a)) becomes cheaper than maintaining two layers — and Q4 below is a reason that contract
might be needed anyway.

---

## Q4 — `TransformerLayer` silently DROPS a caller's mask for three attention types (NEW)

> **This question is not in the plan's original Tier-4 list.** It was found while deriving
> Q3's blast radius and is recorded here because it *amplifies* Q1–Q3 from "the layer
> ignores a mask" to "a generic, widely-used wrapper ignores a mask on the layer's
> behalf".

**Today.** `TransformerLayer` maintains a frozen set of attention types whose `call`
signature has no mask parameter (`transformer.py:196`):

```python
_MASKLESS_ATTENTION_TYPES = frozenset({'fnet', 'anchor', 'lighthouse'})
```

and at both call sites (`transformer.py:623-626` for the pre-norm branch,
`transformer.py:652-658` for the post-norm branch) it does:

```python
elif self.attention_type in self._MASKLESS_ATTENTION_TYPES:
    x = self.attention(x, training=training)
else:
    x = self.attention(x, attention_mask=attention_mask, training=training)
```

The `attention_mask` the *caller of `TransformerLayer`* supplied is simply not passed on.
Nothing warns.

**Why it matters.** The mechanism was introduced correctly — it exists to avoid a
`TypeError` from a sub-layer whose `call` genuinely rejects the argument, and the
`# DECISION plan_2026-06-12_0bb1729b/D-001` anchor above it says so. But the *effect* is
that `TransformerLayer(attention_type='anchor')(x, attention_mask=m)` runs unmasked,
identically to `TransformerLayer(attention_type='anchor')(x)`. A caller building an
encoder stack over a padded batch, who correctly threads a mask through every layer, gets
silently unmasked attention for these three types — and for `'anchor'` specifically that
compounds with Q3's positional anchor selection.

The three types are not equivalent cases, and the fix differs per type:

- `'fnet'` — `FNetFourierTransform` is a parameter-free 2D DFT over the token axis. Masking
  a DFT is not a no-op *or* an obvious operation; there is a real design question here.
- `'anchor'` — see Q3. The mask is meaningful and its absence is harmful.
- `'lighthouse'` — a training-only selection attention; masking interacts with the
  top-K selection, and this layer is separately known-red (Q15).

**Options.**

- **(a) Raise when a mask is supplied for a maskless type.** *Buys* the elimination of a
  silent wrong answer with a one-line change in one file **at the cost of** breaking
  existing stacks that pass a mask uniformly to every layer and currently "work" — which
  is precisely the population most likely to be silently wrong today, so the break is
  informative rather than gratuitous.
- **(b) Warn once (via `dl_techniques.utils.logger`) and continue.** *Buys* visibility
  with no hard break **at the cost of** a warning that gets filtered out in training logs,
  i.e. a mitigation that mostly does not mitigate.
- **(c) Give each of the three real mask support** (which requires resolving Q3 for
  anchor, a new design question for fnet, and Q15-adjacent work for lighthouse) and
  delete the frozen set. *Buys* a uniform contract **at the cost of** three separate
  design decisions and a much larger blast radius.
- **(d) Leave as-is, document loudly in `TransformerLayer`'s docstring.** *Buys* zero risk
  **at the cost of** the status quo.

**Blast radius.** `TransformerLayer` is one of the most widely consumed classes in the
library. Option (a) is scoped narrowly — it only fires when a caller passes a **non-`None`**
mask **and** selects one of the three types, a combination that is *by construction*
already broken today, so no currently-correct code path can regress. Options (b) and (c)
are respectively wider in noise and wider in code.

**Pinned by a test?** No. `tests/test_layers/test_transformers/test_transformer.py`
exercises the `'anchor'` type but does not assert that a supplied mask has an effect.

**Recommendation: (a).** The set of callers it breaks is exactly the set of callers it is
warning, and a one-line raise in one file is the cheapest possible conversion of a silent
wrong answer into a loud one. **What would change my mind:** if a real downstream model
passes a mask to an `'fnet'` layer as part of a uniform stack and genuinely does not need
it masked (plausible — an FNet block is often used precisely because it is
token-mixing-without-attention), then (a) breaks a legitimate use and (b) plus a per-type
opt-out flag becomes the right shape.

---

# Group B — Numerical conditioning and backend limits

## Q5 — `rpc_attention` cannot run under `mixed_float16` at all

**Today.** `RPCAttention`'s robust scoring runs a Principal Component Pursuit
decomposition whose inner loop calls `ops.svd` (`_pcp_decomposition`, invoked at
`rpc_attention.py:605`). **There is no float16 `Svd` kernel on this backend outside XLA**
— `Svd[T=DT_HALF]` is registered for `XLA_CPU_JIT` and `XLA_GPU_JIT` only. Measured on
TF 2.18 / CUDA / RTX 4070, an fp16 forward raises `NotFoundError: Could not find device
for node: {{node Svd}} = Svd[T=DT_HALF, ...]`. The full kernel table is quoted in the
class docstring's `.. warning::` block at `rpc_attention.py:107-136`.

**Why it matters — and the part that is genuinely backwards.** A **masked** fp16 forward
*succeeds*. A **no-mask** fp16 forward *raises*. The reason is that
`apply_attention_mask`'s default `out_dtype` promotes the biased scores to
`mask_dtype(...)` (`>= float32`) before the SVD sees them, so masking is what
accidentally rescues the layer. Supplying a mask is therefore a **prerequisite** for the
layer running under mixed precision — the exact inverse of every other layer in the
package, where masking is the risky path.

**Options.**

- **(a) Leave it.** *Buys* unchanged no-mask numerics for every existing caller **at the
  cost of** an API where `mask=None` is the broken case, which nobody will guess.
- **(b) Promote the score tensor to `mask_dtype(...)` unconditionally**, mask or no mask,
  and cast back at the existing D-005 boundary. *Buys* a layer that runs under every
  policy with a single, comprehensible rule ("PCP always runs at `>= float32`") **at the
  cost of** changing **no-mask fp16 numerics for every existing caller** — though note
  that "changing" here means "from raising to producing numbers", since there is no
  working fp16 no-mask path to preserve. The genuine cost is float32-width SVDs on a path
  the user asked to be fp16.
- **(c) Raise a named error in `build()`/`call()` when the compute dtype is float16.**
  *Buys* a diagnostic that names the missing kernel instead of a raw TF `NotFoundError`
  **at the cost of** refusing the masked fp16 path that currently works.
- **(d) Require `jit_compile=True` for fp16** and document it. *Buys* a real fp16 path
  (XLA has the kernel) **at the cost of** a layer whose supported dtypes depend on the
  caller's compilation mode, which is not expressible in the layer's own contract.

**Blast radius.** **None.** `RPCAttention` has no consumer anywhere in `src/` outside
`layers/attention/__init__.py` and `factory.py` (key `'rpc'`, `factory.py:94` import), and
no `src/` file constructs it by the factory string. Library-only.

**Pinned by a test?** **Yes** —
`tests/test_layers/test_attention/test_rpc_attention.py:622::TestRPCAttentionNoMaskFp16IsAMissingBackendKernel`
asserts all three facts: the no-mask fp16 raise, the masked fp16 success, and that
float32/float64 are unaffected. It fails loudly if TF ever ships the kernel.

**Recommendation: (b).** The asymmetry is indefensible as a contract, the "no-mask
numerics must not change" objection does not apply when the no-mask fp16 path does not
execute at all, and zero consumers means the cost is bounded to this file plus its tests.
**What would change my mind:** if someone is running this layer under fp16 in float32
*compute* with a policy where `mask_dtype` widening would materially change the *float32*
result — it would not; `ops.cast` to a tensor's own dtype is the identity, which is what
makes (b) safe for float32 and float64 callers. Verify that identity holds before landing
anything, per the D-005 pattern.

---

## Q6 — `MASK_BIAS_VALUE = -1e9` costs `rpc_attention` ~6%

**Today.** `common.py:140` defines the package-wide additive mask bias:

```python
MASK_BIAS_VALUE = -1e9
```

with a `WHAT NOT TO DO` block above it (`common.py:126-139`) that rules out the
arithmetic form and rules out per-dtype magic constants. At nine of the ten mask sites
the value is inert — it feeds a softmax, `exp(-1e9 - max)` underflows to exactly `0`, and
any sufficiently negative constant gives the same answer.

At `rpc_attention` it is not inert. The biased score matrix feeds `ops.svd`, and a matrix
whose entries span `[-1e9, ~10]` has a condition number nine orders of magnitude worse
than the unmasked one. **Measured** with byte-identical weights and no fp16 involved:
applying a mask makes the float32 and float64 forwards diverge by **0.12–0.16 on outputs
of absmax 1.9–3.2 (~6%)**, versus **0.002** for the same unmasked forward. The divergence
is `1e9 · eps_f32` propagated through a global factorization.

**Why it matters.** Two ways. First, a float32 `RPCAttention` with a mask is measurably
less accurate than the same layer in float64 — a gap that exists *only because of the
constant*, not because of the mask semantics. Second, the constant is a **shared,
package-wide** value: lowering it is a ten-site numerics change, so the decision cannot be
made locally at `rpc`.

**Options.**

- **(a) Leave `-1e9` global.** *Buys* one constant, no per-site reasoning, no change at
  nine sites **at the cost of** rpc's measured ~6% float32 error.
- **(b) Lower the global constant** to something like `-1e4` (still `exp(-1e4) == 0` in
  float32 and float64; still far below any realistic logit). *Buys* the same softmax
  semantics at nine sites with five fewer orders of magnitude of conditioning damage at
  the tenth **at the cost of** a global numerics change at ten sites, all of which are
  currently bit-verified against HEAD — every one of those bit-identity guards would have
  to be re-derived, and the "how negative is negative enough" question re-opens the exact
  headroom argument the `WHAT NOT TO DO` block settled.
- **(c) Make it per-site: keep `-1e9` globally, pass a smaller bias at `rpc` only** via a
  new `bias_value=` keyword on `apply_attention_mask`. *Buys* a fix confined to the one
  site that is provably hurt **at the cost of** a third keyword on a helper that already
  carries two (`out_dtype`, `rescue_axis`), and of the exact "second thing to get wrong"
  that `common.py`'s docstring warns against.
- **(d) Fix it structurally at `rpc`** — do the PCP decomposition on the *unmasked* scores
  and apply the mask bias to the reconstructed `L + S` afterwards. *Buys* an SVD that
  never sees a badly-conditioned matrix at all **at the cost of** a real semantic change
  (the low-rank/sparse split would no longer be computed over masked scores, which is
  arguably *more* correct but is a different algorithm), plus a restructuring of
  `_compute_attention` — the same restructuring Q11 needs.

**Blast radius.** `MASK_BIAS_VALUE` is imported by eleven modules (all ten mask sites plus
`energy_attention.py`, which re-exports it privately as `_MASK_BIAS_VALUE` under the D-007
contract read by `layers/transformers/energy_transformer.py`). Options (a) and (c) touch
one site; option (b) touches all eleven and every bit-identity test written by the
preceding plan; option (d) touches `rpc_attention.py` only.

**Pinned by a test?** **Yes** —
`tests/test_layers/test_attention/test_rpc_attention.py:686::TestRPCAttentionConditioning`.
Note its shape, because it is unusually well built: for masked kinds it asserts the
divergence is both `<= budget` **and `> 0.01`**, i.e. it fails if the loose tolerance ever
becomes *unnecessary*. That means a fix here will turn this test red on purpose, and that
is the correct signal, not a regression.

**Recommendation: (d), and explicitly not (b).** The problem is one site's interaction
with an SVD, not a bad global constant — nine sites are provably indifferent, so paying a
ten-site numerics change to help one is the wrong trade. (d) also composes with Q11, which
wants the same `_compute_attention` restructuring for a different reason; doing them
together is one careful pass instead of two. **What would change my mind:** if a *second*
site ever routes biased logits into something other than a softmax (an eigendecomposition,
a `logsumexp` over a huge axis, a matrix inverse), then the constant really is the problem
and (b) becomes right — treat "how many sites consume the bias non-softmax-wise" as the
deciding measurement.

---

## Q7 — `performer_attention` mis-scales odd `nb_features`

**Today.** Only positivity is validated (`performer_attention.py:207-208`):

```python
if nb_features <= 0:
    raise ValueError(f"nb_features must be positive, got {nb_features}")
```

The projection matrix is built with width `nb_features // 2`
(`performer_attention.py:308`), and `_create_kernel_features` concatenates `cos` and
`sin` of that projection (`performer_attention.py:364-366`) — so an **odd** request builds
`2 * (nb_features // 2) == nb_features - 1` actual features. But the FAVOR+ normalizer is
computed from the nominal count (`performer_attention.py:242`):

```python
self._feature_scale = math.sqrt(2.0 / float(self.nb_features))
```

and applied at `performer_attention.py:369`. **Measured:** `nb_features=255` builds 254
features and scales by `sqrt(2/255)`.

**Why it matters.** No crash, no shape error — the kernel approximation is simply scaled
by `sqrt(254/255) ≈ 0.998` relative to the correct FAVOR+ normalization. At
`nb_features=255` that is a 0.2% error and nobody will ever notice. At `nb_features=3`
(2 actual features, scaled by `sqrt(2/3)`) it is an 18% error. The real defect is not the
magnitude; it is that **the layer silently reinterprets a configuration value**, and the
`get_config()` round-trip preserves the *nominal* 255 while the built model has 254
features. This one has **no in-source comment at all** — it is the single item in this
brief that no prior pass documented.

**Options.**

- **(a) Reject odd values** — `if nb_features % 2 != 0: raise ValueError(...)`. *Buys* a
  configuration that means what it says, with a diagnostic at construction time **at the
  cost of** rejecting configs that "work" today, and of a `.keras` reload failing for any
  model saved with an odd value (none exist in this repo — see blast radius).
- **(b) Fix the scale** — `math.sqrt(2.0 / float(2 * (self.nb_features // 2)))`. *Buys*
  a correct normalization for every input, no rejection **at the cost of** a silent
  numerics change for odd values, and of leaving `get_config()` reporting a feature count
  the model does not have.
- **(c) Round up to even in `__init__`** and store the rounded value, so `get_config()`
  is truthful. *Buys* truthfulness and correctness **at the cost of** silently changing a
  caller's config value — the same class of surprise as (b), just relocated.
- **(d) Leave it, document it.** *Buys* nothing; this is currently the state, minus even
  the documentation.

**Blast radius.** **None.** `PerformerAttention` has no consumer in `src/` outside
`layers/attention/__init__.py` and `factory.py` (key `'performer'`, `factory.py:92`
import), and no `src/` file reaches it by the factory string. `factory.py`'s
`optional_params` for `'performer'` would need the validation mirrored under option (a).
Note also the standing frozen-signature anchor `plan_2026-06-14_0c5d4a21/D-007` at
`factory.py:986-991`, which pins `performer.call`'s non-standard signature — none of the
options above touches `call`, so the anchor is not implicated.

**Pinned by a test?** No. `test_performer_attention.py` never uses an odd `nb_features`
and never inspects `_feature_scale`.

**Recommendation: (a).** `nb_features` is a `get_config()` key; a layer that stores 255
and builds 254 has a configuration that lies, and no amount of scale-fixing repairs that.
Rejecting is the only option that keeps the config honest, and with zero consumers the
rejection costs nothing. **What would change my mind:** if odd `nb_features` is ever
generated programmatically (e.g. from a dimension calculation elsewhere), (c) becomes the
kinder option — but it must log the adjustment, not swallow it.

---

## Q8 — `wave_field_attention.scale` uses the inverse of the package convention

**Today.** `wave_field_attention.py:296` sets

```python
self.scale = math.sqrt(self.head_dim)
```

and `call()` **divides** by it (`wave_field_attention.py:630`):

```python
q_mod = self.query_modulation_activation(q / self.scale, training=training)
```

`q / sqrt(d)` is the standard temperature. **The site is arithmetically self-consistent.
There is no missing reciprocal.** Every sibling in the package instead stores
`common.compute_attention_scale(head_dim)` — which returns `1 / sqrt(d)` — and
multiplies.

**Why it matters.** Not for correctness today; for the trap it sets. `self.scale` in this
package means `1/sqrt(d)` everywhere except here. A future editor "unifying" this line
with the shared helper, and not also flipping the `/` at `:630` to `*`, produces
`q / (1/sqrt(d)) = q * sqrt(d)` — the temperature **squared**, in the wrong direction.
For `head_dim=64` that is a factor of 64 on the query-modulation logits, which will not
crash and will not be caught: **no test pins `layer.scale`, and no wave_field test pins an
absolute output value.** The in-source block at `wave_field_attention.py:624-639` says
this in as many words, including an explicit HONESTY NOTE that the anchor is a comment,
not a guard.

**Options.**

- **(a) Rename and re-base to the package convention** — `self.scale =
  compute_attention_scale(head_dim)` **and** `q * self.scale` at `:630`, in the same
  commit. *Buys* one convention across the package, so `scale` never means two things
  **at the cost of** touching a layer with three production consumers on a path where the
  only guard is a code comment, for a purely cosmetic gain.
- **(b) Rename the attribute instead** — e.g. `self._inv_temperature_denominator` or
  simply `self._sqrt_head_dim` — leaving the arithmetic untouched. *Buys* removal of the
  name collision (which is the whole trap) with **zero numerics risk** **at the cost of**
  a name that is not the package convention either, and one `get_config`-irrelevant
  attribute rename (`scale` is not a config key here).
- **(c) Write the missing test first and defer the rename** — a one-line
  `assert layer.scale == math.sqrt(head_dim)`. *Buys* the guard the HONESTY NOTE asks for
  **at the cost of** leaving the confusing name in place.
- **(d) Leave as-is.** *Buys* nothing beyond zero risk.

**Blast radius.** Three production consumers:
`src/dl_techniques/models/wave_field_llm/wave_field_llm.py`,
`src/dl_techniques/models/memory_bank/wave_field_memory_llm.py`,
`src/train/wave_field_llm/pretrain.py`; plus
`tests/test_models/test_wave_field_llm/test_wave_field_llm.py` and factory key
`'wave_field'` (`factory.py:99` import). `self.scale` is **not** a `get_config()` key, so
no checkpoint schema is at risk from any option — but option (a) is a numerics change on a
trained-model path if botched.

**Pinned by a test?** **No**, and that is the load-bearing fact. `test_wave_field_attention.py`
contains zero references to `.scale`.

**Recommendation: (c) then (b).** Write the pin first — it is one line and it converts
the entire situation from "a comment protects a 64x error" to "a test does". Then rename
the attribute rather than re-basing the arithmetic: the trap is the *name*, and (b)
removes the trap with no numerics risk at all on a layer that has three production
consumers and a trainer. Option (a) buys uniformity that nobody reads for a risk that is
entirely avoidable. **What would change my mind:** if a shared SDPA extraction ever lands
that requires every attention layer to expose a uniformly-defined `scale` attribute, (a)
becomes mandatory — and at that point it should be done with the pin from (c) already in
place, so the flip is verified rather than argued.

---

# Group C — Design ambiguities inherited from the source papers

These three are not bugs with an obviously correct answer. Each is a place where an
implementation had to pick between two defensible readings and the choice was never
settled.

## Q9 — TripSE branch combination: sum vs average (2-vs-2, not 1-vs-2)

**Today.** The four TripSE variants combine their three attention branches two different
ways, and the shipped source contains an **unresolved design comment**
(`tripse_attention.py:479-481`):

```python
combined = ops.add(ops.add(out_hw, out_cw), out_hc)
# Average before SE? Original paper usually implies Sum or Avg.
# We'll use sum as per the prompt's implied logic, SE handles magnitude.
```

| class | class line | combination | line |
|---|---|---|---|
| `TripSE1` | `:340` | **sum** | `:479` |
| `TripSE2` | `:505` | average (`ops.divide(total, 3.0)`) | `:697-698` |
| `TripSE3` | `:720` | average (`ops.divide(total, 3.0)`) | `:897-898` |
| `TripSE4` | `:1092` | **sum** | `:1301` |

> **Correction to the prior findings.** Every earlier write-up of this item describes it
> as "`TripSE1` sums while `TripSE2`/`TripSE3` average" — a 1-vs-2 split. It is **2-vs-2**:
> `TripSE4` (`tripse_attention.py:1301`) also sums, and was missed. That changes the
> framing: this is not one outlier, it is two conventions coexisting in one file.

**Why it matters.** A sum produces branch outputs 3x larger than an average, going into
the SE block. Both `TripSE1` and `TripSE4` feed their combination straight into an SE
module (`self.se_block` / `self.final_se`), and the shipped comment's argument is exactly
that — "SE handles magnitude". That argument is *plausible but unverified*: an SE block
recalibrates channels via a learned gate, and a 3x input scale shifts where that gate's
`sigmoid` sits on its curve at initialization. So the two conventions are not obviously
equivalent even up to learning, and nothing in the file measures which is better.

**Options.**

- **(a) Standardize on average** (divide `TripSE1` and `TripSE4` by 3). *Buys* one
  convention **at the cost of** changing two classes' output magnitude, invalidating any
  externally-trained TripSE1/TripSE4 weights, and picking the side the shipped comment
  argued *against*.
- **(b) Standardize on sum** (drop the `/3` in `TripSE2`/`TripSE3`). Same trade, other
  direction, and hits the two classes whose current behavior nobody has questioned.
- **(c) Make it a constructor argument** (`branch_reduction: str = ...`) with each class
  keeping its current default. *Buys* the ability to A/B the question, and honesty about
  the fact that it is unsettled **at the cost of** a new `get_config()` key on four
  classes — a checkpoint-forward commitment made to resolve a question nobody has
  measured yet.
- **(d) Leave the behavior, delete the unresolved comment, and replace it with a
  measured note.** *Buys* removal of a shipped `# TODO`-shaped comment **at the cost of**
  requiring an actual experiment to write the note honestly.

**Blast radius.** **None outside the package.** No `src/` file references `TripSE1`,
`TripSE2`, `TripSE3`, `TripSE4` or `TripletAttentionBranch`, and none constructs them via
the factory strings (`'tripse1'`, `factory.py:761`, and siblings). Library-only. Also note
these five classes have **zero constructor validation** — a separate, unrelated backlog
item, but the same file.

**Pinned by a test?** No. `test_tripse_attention.py` does not pin either combination
numerically.

**Recommendation: (d), then (c) only if the experiment says it matters.** Nobody should
change a numeric convention in four classes to resolve a question that has never been
measured on a task. The cheap, honest move is to run TripSE1-with-average against
TripSE1-as-shipped on one small classification benchmark, then write the result into the
file and delete the "as per the prompt's implied logic" comment, which is the kind of
provenance note that should never have shipped. **What would change my mind:** if the
measurement shows a real gap, (a) or (b) follows directly from it — and with zero
consumers, whichever wins can be applied to all four classes at no cost.

---

## Q10 — `hopfield_attention` feeds back `A K` but returns `A V`

**Today.** The iteration at `hopfield_attention.py:722` updates the query with the
attention-weighted **keys**:

```python
current_query = ops.matmul(attention_weights, key_proj)
```

while the returned output comes from `attention_weights @ value_proj` inside
`_hopfield_update_step` (`hopfield_attention.py:491`). The class docstring states the
consequence plainly at `hopfield_attention.py:84-87`: "the loop feeds back `A K` (keys),
while the returned output is `A V` (values). The two coincide in the self-attention case
... a heuristic generalization, not the energy descent above."

**Why it matters.** With `K == V` (self-attention) the two are identical and the iteration
*is* the Ramsauer et al. modern-Hopfield energy descent the docstring's ASCII diagram
(`hopfield_attention.py:25-26`) claims. With `K != V` (cross-attention, or any
`key_dim != value_dim` configuration) the fixed point of the iteration is a fixed point of
the *key* dynamics, while the caller receives a *value* readout that is not itself
converging to anything. Concretely: `HopfieldAttention(key_dim=32, value_dim=64,
update_steps_max=3)` called with distinct K and V returns a tensor whose relationship to
the "stored patterns" the layer's docstring describes is undefined — and the layer will
happily report more update steps as if they were converging it.

**Options.**

- **(a) Feed back `A V` too** (and require `key_dim == value_dim` for the iteration, or
  project). *Buys* an iteration whose fixed point matches the returned quantity **at the
  cost of** changing the fixed point for every `K != V` caller, and of a shape constraint
  the layer does not currently impose.
- **(b) Keep the behavior, restrict the claim** — raise (or warn) when
  `update_steps_max > 0` **and** the call is genuinely cross-attention, and rewrite the
  ASCII diagram so it does not describe an energy descent the code does not perform.
  *Buys* an honest contract with no numerics change **at the cost of** either refusing a
  configuration that runs today, or a warning nobody reads.
- **(c) Make it explicit** — a `feedback_source: {'key', 'value'} = 'key'` constructor
  argument. *Buys* both readings, defaulting to today's **at the cost of** a new
  `get_config()` key for a layer with no consumers, and a config surface for a question
  that has never been measured.
- **(d) Leave as-is (documented).** Current state. *Buys* zero risk **at the cost of** a
  docstring whose headline diagram and whose caveat paragraph contradict each other for
  half the layer's configuration space.

**Blast radius.** **None.** `HopfieldAttention` has no consumer in `src/` outside
`layers/attention/__init__.py` and `factory.py` (key `'hopfield'`, `factory.py:400`).
**Do not confuse it with `HopfieldNetwork` in
`src/dl_techniques/layers/transformers/energy_transformer.py`** — a different class,
which *is* consumed (Energy Transformer models) and is not implicated here. A naive
`grep -i hopfield` matches both; the distinction is called out in `energy_transformer.py`
itself.

**Pinned by a test?** No. `test_hopfield_attention.py` has no test that isolates `K == V`
from `K != V` convergence behavior.

**Recommendation: (b).** The behavior is a defensible heuristic; the *documentation* is
what is wrong, because a reader of the diagram at `:25-26` will believe they are getting
an energy descent in a configuration where they are not. Fix the claim, add the missing
`K == V` vs `K != V` convergence test so the distinction is measured rather than asserted,
and leave the numerics alone. **What would change my mind:** if the added convergence test
shows the `K != V` iteration actually *diverges* (rather than merely converging to
something else), this stops being a documentation question and (a) or a hard restriction
becomes necessary.

---

## Q11 — `rpc_attention` `return_attention_scores=True` returns weights that did not produce the output

**Today.** `rpc_attention.py:711-725`, reproduced with its own in-source admission:

```python
if return_attention_scores:
    # KNOWN DEFECT (pre-existing, deliberately NOT fixed here ...): this
    # branch re-runs the ENTIRE PCP decomposition, i.e. `max_pcp_iter` more
    # batched SVDs, roughly doubling the cost of the forward pass. Worse, it
    # recomputes the scores WITHOUT applying `mask`, so the weights returned
    # to the caller do not correspond to the weights that produced `output`
    # whenever a mask is supplied.
    attention_scores = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
    attention_scores = attention_scores * self.attention_scale
    L, S = self._pcp_decomposition(attention_scores)
    attention_weights = self.attn_prob(L + S)
```

The output path computes its own weights at `rpc_attention.py:646`
(`attention_weights = self.attn_prob(robust_attention_scores)`), inside
`_compute_attention`, where the mask **has** been applied. The two are returned together
at `rpc_attention.py:741`.

**Why it matters.** Two independent problems in one branch.

1. **Cost.** PCP is `max_pcp_iter` batched SVDs. Running it twice roughly doubles the
   forward cost of the most expensive attention layer in the package, for a debugging
   flag.
2. **Correctness of the returned value.** Call
   `layer(x, mask=m, return_attention_scores=True)` with `m` masking positions 20–31 of a
   32-token sequence. The returned `output` respects the mask. The returned
   `attention_weights` has **non-zero weight on positions 20–31**. Anyone using the
   weights for attention-map visualization, head-pruning analysis, or an attention-entropy
   regularizer is reading a tensor from a different computation than the one that produced
   the output. This is the failure mode where a plot looks fine and is wrong.

**Options.**

- **(a) Cache and reuse.** Have `_compute_attention` return `(output, attention_weights)`
  and hand the cached, mask-correct weights back. *Buys* both fixes at once — half the
  cost and a weights tensor that actually corresponds to the output — **at the cost of**
  changing an internal method's return signature (not a public one; `_compute_attention`
  is defined at `rpc_attention.py:517` and has exactly one caller, `rpc_attention.py:708`)
  and holding an `(B, H, N, N)` tensor alive on
  every forward, even when the flag is `False`, unless the caching is made conditional.
- **(b) Same, but conditional** — thread `return_attention_scores` into
  `_compute_attention` so the weights are only retained when asked for. *Buys* (a)'s
  benefits with no memory cost on the common path **at the cost of** a flag threaded
  through a private method, which is the mild code smell (a) avoids.
- **(c) Minimal fix: pass `mask` to the recompute.** *Buys* correctness for a two-line
  change **at the cost of** keeping the ~2x cost, and of two code paths that must be kept
  numerically in lockstep by hand — which is exactly the duplication smell that produced
  this defect.
- **(d) Remove the flag.** *Buys* deletion of the whole problem **at the cost of** a
  public `call()` signature change, which is forbidden here: `rpc.call`'s signature is
  frozen by the standing anchor `plan_2026-06-14_0c5d4a21/D-007` at `factory.py:986-991`.
  **Not available.**

**Blast radius.** **None outside the package.** No `src/` consumer for `RPCAttention`.
`_compute_attention` is private (`rpc_attention.py:517`) with exactly one caller
(`rpc_attention.py:708`). Options
(a)–(c) are entirely internal; none touches `call()`'s signature, so the D-007 freeze is
respected.

**Pinned by a test?** **Partially.**
`tests/test_layers/test_attention/test_rpc_attention.py:137::test_return_attention_scores`
exercises the flag but passes **no mask**, so it cannot see the defect. A RED test would
supply a mask and assert the returned weights are ~zero at masked key positions.

**Recommendation: (b).** It is the only option that fixes both halves without paying
memory on the common path, and the D-007 freeze does not reach a private method. Do it in
the same pass as Q6(d), since both want `_compute_attention` restructured and doing them
separately means restructuring the same method twice. Explicitly reject (c): keeping two
hand-synchronized score computations in one file is how this defect was born. **What would
change my mind:** if profiling shows the conditional threading in (b) prevents some XLA
fusion that (a) preserves, take (a) — the `(B, H, N, N)` retention is bounded and this
layer already materializes that tensor.

---

# Group D — Framework-level questions

## Q12 — Should explicit `training=` forwarding be a package-wide rule?

**Today.** Keras 3 propagates `training` through two channels. The explicit kwarg, and an
ambient `CallContext` that `Layer.__call__` fills in when the kwarg is absent
(`keras/src/layers/layer.py:838-853`):

```python
call_context = self._get_call_context()
training = call_spec.user_arguments_dict.get("training", None)
if training is None:
    training = call_context.training          # <- ambient fallback
    ...
call_context.training = training              # <- overwrite, never restored
```

The ambient channel is a **single mutable slot** in global state. Every nested `__call__`
overwrites it (`layer.py:851`) and **nobody restores it on exit** —
`_maybe_reset_call_context` (`layer.py:1501-1504`) clears it only for the *outermost* entry
layer:

```python
def _maybe_reset_call_context(self):
    layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
    if layer_call_ctx is None or layer_call_ctx.entry_layer == self:
        global_state.set_global_attribute("current_call_ctx", None)
```

**Why it matters — and why the naive framing of this is wrong.** Measured during this
plan's step 7: a sub-layer that does *not* receive `training=` explicitly nonetheless sees
the correct value in all six scenarios probed (`True`/`False`/`None` × eager/`fit`). So
"the sub-layer never learns the training mode" is **false**, and a `call()`-only probe
cannot distinguish forwarded from un-forwarded code.

What is actually unsound is the slot. If **any** sibling sub-layer between the parent's
entry and the un-forwarded child forces a different `training` on a child of its own — a
frozen-BN wrapper, a teacher/EMA branch, anything that evaluates a sub-module in inference
mode — it leaves the ambient slot poisoned for every later un-forwarded call in the same
outer call. Injecting exactly that (a numerically transparent context-poisoner wrapped
around `batch_norm`) made an un-forwarded `sigmoid` in `tripse_attention.py` observe
`training=False` inside a parent called with `training=True`.

So explicit forwarding is **defensive hardening against a framework hazard**, not
redundancy — but it is also, today, a provable no-op for every current consumer.

**Options.**

- **(a) Make explicit forwarding a package-wide rule**, enforced by a lint or a review
  checklist item: every sub-layer call that accepts `training` receives it. *Buys*
  immunity to slot poisoning across the package **at the cost of** a large mechanical diff
  touching most layers, all of it currently no-op, and a rule that a reader will reasonably
  mistake for cargo-culting unless the rationale travels with it.
- **(b) Forward only where a poisoner can plausibly sit** — i.e. where a stateful
  sub-layer runs between the parent's entry and the target. *Buys* the same protection
  where it can actually bite, with a much smaller diff **at the cost of** a judgement call
  per site, which is the kind of rule that decays into "nobody forwards anything".
- **(c) Do nothing and rely on Keras.** *Buys* zero diff **at the cost of** a
  correctness property that depends on no sibling in the entire call tree ever forcing a
  `training` value — an invariant this package cannot enforce and does not check.
- **(d) Write the rationale into `CLAUDE.md` / the package guide** and forward
  opportunistically. *Buys* the knowledge surviving **at the cost of** no enforcement.

**Blast radius.** Package-wide by construction. Note the fix already landed at two sites
(`tripse_attention.py:278-299` and `:1058-1065`) with a full injection harness, so the
pattern and its test template already exist.

**Pinned by a test?** **Yes**, and unusually well:
`tests/test_layers/test_attention/test_tripse_attention.py:739::test_the_ambient_training_context_is_a_single_poisonable_slot`
asserts the *poison mechanism itself*, so the entire test class fails loudly (rather than
becoming vacuous) if Keras ever starts restoring the slot. The suite also ships the
un-poisoned control, so nobody mistakes it for evidence that Keras fails to propagate.

**Recommendation: (a), stated as a rule with the rationale attached, applied
opportunistically rather than as a big-bang sweep.** The reason to prefer the rule over
(b)'s judgement call is that the hazard is **non-local** — whether a poisoner exists
depends on layers this file has never heard of, so a per-site judgement is made on
information the author does not have. But a 40-file mechanical diff to land it at once is
not worth it; write the rule down (option (d) is a *prerequisite*, not an alternative) and
apply it as files are touched. **What would change my mind:** if Keras starts restoring
the slot in `_maybe_reset_call_context` (a one-line upstream change), the entire hazard
evaporates and (c) becomes correct — which is exactly why the poison-mechanism test above
exists.

---

## Q13 — `perceiver_attention.build` forwards a container its sibling rejects

**Today.** `perceiver_attention.py` classifies its `input_shape` with a module-level
predicate (`_is_list_of_shapes`, `perceiver_attention.py:56-76`) that accepts **either a
`list` or a `tuple`** container:

```python
return (
    isinstance(s, (list, tuple))
    and len(s) > 0
    and isinstance(s[0], (list, tuple))
)
```

`build` uses it at `perceiver_attention.py:299` and then forwards the **raw container**
to the wrapped layer at `perceiver_attention.py:319`:

```python
self.cross_attention.build(input_shape)
```

`MultiHeadCrossAttention.build`'s own predicate (`multi_head_cross_attention.py:376`)
accepts only a `list`. So a **tuple** of two shapes — which this module's predicate
accepts and classifies correctly — is re-classified by the sibling as a *single* shape and
raises from inside it:

```
ValueError: Query input must be 3D, got shape ((2, 8, 32), (2, 40, 32))
```

**Why it matters.** The error names a shape the caller did not pass, from a class the
caller did not construct. `build(((2, 8, 32), (2, 40, 32)))` — a perfectly ordinary way to
spell two shapes in Python — fails, while `build([(2, 8, 32), (2, 40, 32)])` succeeds. The
difference is invisible from the layer's documented contract.

The reason it is not simply fixed: `perceiver_attention.py:286-292` carries an **R13
prohibition** on unifying its predicate with the two sibling spellings
(`multi_head_cross_attention.py:376`, `multi_head_latent_attention.py:412`), because each
classifies serialized-shape edge cases differently and one shared body would silently
change two other layers' build paths.

**Options.**

- **(a) Normalize at the forwarding boundary** — `self.cross_attention.build(
  list(input_shape) if _is_list_of_shapes(input_shape) else input_shape)`. *Buys* the fix
  with a one-line change confined to `perceiver_attention.py`, honoring R13 completely
  (no sibling predicate is touched) **at the cost of** a normalization that papers over a
  real disagreement between two files rather than resolving it — the next person to add a
  wrapped layer hits the same thing.
- **(b) Widen `MultiHeadCrossAttention`'s predicate to accept tuples.** *Buys* the fix at
  the source **at the cost of** directly violating R13 — and `MultiHeadCrossAttention` has
  production consumers (`layers/transformers/transformer_decoder.py`,
  `tests/test_layers/test_blt_core.py`), so a build-path classification change there is
  genuinely risky.
- **(c) Narrow `perceiver`'s predicate to `list` only**, matching the sibling. *Buys*
  agreement **at the cost of** breaking the `.keras` round-trip case the permissive
  predicate exists for, which is the defect that
  `plan-2026-07-27T183600-b4ef45f0` step 6 just fixed. Regressive; recorded so it is not
  re-proposed.
- **(d) Reject tuple containers in `perceiver.build` with a named error.** *Buys*
  a diagnostic from the right class **at the cost of** refusing an input the layer could
  handle.

**Blast radius.** `PerceiverAttention` → `src/dl_techniques/layers/transformers/perceiver_transformer.py`
(one consumer), plus `src/dl_techniques/layers/time_series/mixed_sequential_block.py:267`
(**factory-string consumer** — reached as `'perceiver'`). Factory key `'perceiver'`
(`factory.py:631`). Option (a) touches one file and cannot reach either consumer's
behavior for `list` inputs. Option (b) additionally reaches
`layers/transformers/transformer_decoder.py` and the BLT stack.

**Pinned by a test?** **Yes, as documentation** —
`tests/test_layers/test_attention/test_perceiver_attention.py:252::test_tuple_of_shapes_is_classified_the_same_way_build_classifies_it`
records the behavior verbatim in its docstring so the next reader does not rediscover it.

**Recommendation: (a).** R13 is a real constraint with real production consumers behind
it, and a one-line normalization at the boundary is the only option that fixes the caller's
experience without touching a predicate two other layers depend on. It is a papering-over,
and the docstring should say so. **What would change my mind:** if a third wrapped layer
appears with a fourth spelling, the right move stops being per-boundary normalization and
becomes a single explicit `normalize_input_shapes()` utility in `common.py` with the three
existing predicates rewritten in terms of it — a much larger job that needs its own plan,
and R13's warning about silently changing build paths applies in full.

---

# Group E — Deferred by tier, recorded here so they are not lost

## Q14 — Dead constructor parameters that are also `get_config()` keys

> **Tier 3, deliberately not executed.** The user did not select this tier;
> invariant I7 of `plan-2026-07-27T183600-b4ef45f0` forbade touching these files. Recorded
> here so the reasoning is not re-derived from scratch.

**Today.** Two parameters are validated, stored, serialized, and **never read**:

| parameter | validated | stored | `get_config()` | read by `build`/`call`? |
|---|---|---|---|---|
| `FNetFourierTransform.epsilon` | `fnet_fourier_transform.py:233-234` | `:239` | `:478` | **no** |
| `HopfieldAttention.update_steps_eps` | `hopfield_attention.py:270-271` | `:312` | `:784` | **no** |

`fnet`'s is documented at `fnet_fourier_transform.py:182-187` and `:228-232`; `hopfield`'s
at `:216-233`, including *why* it went dead — it gated a convergence early-exit that was
removed because a data-dependent Python `if` on a traced tensor crashes under
`@tf.function`/jit (`hopfield_attention.py:697-706`).

**Why deletion is not available.** Both are `get_config()` keys. `keras.saving.load_model`
reconstructs a layer by calling `from_config(config)`, which passes every saved key to
`__init__`. Remove the key and remove the parameter, and **every existing checkpoint of
that layer type raises `TypeError: __init__() got an unexpected keyword argument
'epsilon'`** on load. This is not hypothetical for out-of-repo users; there happen to be
no in-repo checkpoints using either class today (a grep of every `config.json` under
`results/` finds zero matches for any of these class names), but the compatibility
commitment is forward-looking and permanent.

**What "deprecate in place" would concretely look like** — spelled out because it is the
only available remedy and the phrase is otherwise hand-waving:

1. **Keep the `__init__` parameter, keep the `get_config()` key, keep the validation.**
   None of those can move.
2. **Change the default to a sentinel** that means "not supplied": `epsilon: Optional[float] = None`.
   Nothing downstream reads it, so this is safe.
3. **Warn once on an explicitly-supplied value**, via `dl_techniques.utils.logger`
   (not `warnings`, per the package's logging convention):

   ```python
   if epsilon is not None:
       logger.warning(
           "FNetFourierTransform: `epsilon` is inert — it is validated, stored and "
           "serialized for checkpoint compatibility but is never read. It will be "
           "removed no earlier than <version>. Omit it."
       )
   ```

   The warning must **not** fire on `from_config` reload of a checkpoint that stored the
   old default, which is why step 2's sentinel matters: a reloaded config carrying
   `epsilon: 1e-12` *will* fire it, so either the sentinel is written into
   `get_config()` too (changing the serialized value, a bigger commitment) or the warning
   is accepted as firing on legacy reloads. **This is the unresolved sub-question**, and
   it is why "just deprecate it" is not free.
4. **Docstring**: `.. deprecated::` directive naming the version, on both the class
   docstring and the `:param:` entry.
5. **Test**: assert the warning fires for an explicit value and does *not* fire for the
   default — otherwise the deprecation is itself unguarded.

**Options.** (a) Leave documented-dead — current state, zero cost, zero risk. (b) The
deprecate-in-place procedure above — *buys* a migration path **at the cost of** runtime
warning machinery on a hot constructor, the unresolved reload-warning question in step 3,
and a removal date the project must actually honor. (c) Revive `update_steps_eps` for real
by reimplementing the convergence exit jit-safely (a fixed-tolerance `keras.ops.cond` or a
bounded loop with a masked update) — *buys* a parameter that means something **at the cost
of** a genuine behavior change on a layer with no consumers, to restore an optimization
whose removal is documented as numerically neutral.

**Blast radius.** `FNetFourierTransform` → `layers/fnet_encoder_block.py`,
`layers/transformers/transformer.py` (factory key `'fnet'`, `factory.py:344`).
`HopfieldAttention` → none (see Q10's blast radius, including the `HopfieldNetwork`
false-positive warning).

**Pinned by a test?** No.

**Recommendation: (a) for both.** A deprecation that cannot end in a removal is warning
machinery with no payoff, and the reload-warning question in step 3 has no clean answer.
The parameters are already documented as inert in three places each. **What would change
my mind:** if a `major` version bump is ever planned, that release is the moment to remove
both keys outright — and the removal should be listed in the release notes rather than
preceded by a deprecation cycle that nobody will observe.

---

## Q15 — `LighthouseAttention` is the package's known-red pair

**Today.** Two tests fail at HEAD and have failed for the whole of this plan and the one
before it. They are the **entire** known-red baseline for
`tests/test_layers/test_attention/` (`2 failed, 1649 passed, 31 skipped` at `58232e63`):

- `tests/test_layers/test_attention/test_lighthouse_attention.py:42::TestLighthouseAttention::test_initialization_defaults`
- `tests/test_layers/test_attention/test_lighthouse_attention.py:148::TestLighthouseAttention::test_causality`

They are two different problems wearing one name:

1. **`test_initialization_defaults` is a defaults mismatch, not a causality bug.** The
   test asserts `layer.qk_norm_type == "rms_norm"`, while `LighthouseAttention.__init__`
   declares `qk_norm_type: Optional[str] = None`. One of the two is wrong; nothing in the
   file says which. This is a 15-minute question, not a research question, and it is
   currently hiding behind the harder one.
2. **`test_causality` is the real defect.** The test perturbs the input at the last
   position by `+100.0` and asserts that outputs at positions `0..N/2` are unchanged.
   Measured: `Mismatched elements: 64/1024 (6.25%)`, max abs diff **2.58**. Information
   flows **backwards in time** through the pyramid-pool / top-K selection path. For a
   layer whose entire purpose is causal long-context *pretraining*, a causality leak is
   not a tolerance issue — it is the layer being wrong.

**Why it matters.** `LighthouseAttention` is reachable from `TransformerLayer`
(`transformer.py:449`, `attention_type == 'lighthouse'`), so it is one factory string away
from a training run. And per [Q4](#q4--transformerlayer-silently-drops-a-callers-mask-for-three-attention-types-new)
it is also in `_MASKLESS_ATTENTION_TYPES`, so a mask passed to a lighthouse
`TransformerLayer` is silently dropped on top of the causality leak.

A second-order cost: a permanently-red pair *normalizes* red. Every gate comparison in the
two plans that touched this package had to carry "the baseline is 2, not 0" as an explicit
invariant, and had to check in **both directions** (a known-red test unexpectedly turning
green is a stop signal, not a win). That is real, recurring overhead on every future plan.

**Options.**

- **(a) Fix the causality defect.** *Buys* a correct layer and a 0-red baseline **at the
  cost of** a real investigation into the pooled top-K selection and its causal-boundary
  filler — the design note at `research/2026_lighthouse_attention.md` §3 describes the
  intended construction and is the right starting point.
- **(b) Split the two failures now.** Resolve the defaults mismatch (a docstring/test
  disagreement, decidable by reading `research/2026_lighthouse_attention.md` §3 for the
  paper's default) and leave only `test_causality` red. *Buys* a known-red set of **1**
  and removes a distractor from every future gate comparison **at the cost of** nothing —
  this is strictly cheaper than the status quo.
- **(c) `pytest.mark.xfail(strict=True)` on `test_causality`.** *Buys* a green gate and,
  because of `strict=True`, still fails loudly if the defect is ever accidentally fixed
  **at the cost of** the visibility that a red test provides — an xfail is easy to stop
  seeing.
- **(d) Leave both red.** Current state.

**Blast radius.** `LighthouseAttention` → `src/dl_techniques/layers/transformers/transformer.py`
(constructed at `:449`; called maskless at `:624`/`:653`; listed in
`_MASKLESS_ATTENTION_TYPES` at `:196`). Factory key `'lighthouse'` (`factory.py:434`).

**Pinned by a test?** **Yes — the two red tests above are the guard.** That is the one
item in this brief that is already correctly guarded; the guard is simply failing.

**Recommendation: (b) now, (a) as its own plan.** Splitting the pair costs nothing and
removes a permanent distractor: every future plan gets a known-red set of 1 with a single,
well-understood cause instead of 2 with a conflated one. Then fix the causality leak as a
scoped plan of its own, with `research/2026_lighthouse_attention.md` as the specification.
Explicitly reject (c): converting a real, understood correctness defect into a green gate
is how a defect becomes permanent. **What would change my mind:** if `LighthouseAttention`
is to be removed from the package rather than fixed, deleting it (and its
`_MASKLESS_ATTENTION_TYPES` entry, and its factory key) resolves Q15 and one third of Q4
in a single stroke — that is a legitimate answer and should be considered before investing
in (a).

---

## 9. Corrections to prior findings

Three claims carried in `plans/plan-2026-07-27T183600-b4ef45f0/findings/non-mask-defects.md`
did not survive verification against HEAD. Recorded here because the findings files are in
a gitignored directory and this note is the only copy that survives.

1. **TripSE sum-vs-average is 2-vs-2, not 1-vs-2.** The finding (#10) states "`TripSE1`
   sums, `TripSE2`/`TripSE3` average". `TripSE4` also sums
   (`tripse_attention.py:1301`) and was missed. See
   [Q9](#q9--tripse-branch-combination-sum-vs-average-2-vs-2-not-1-vs-2). This matters for
   framing: it is two coexisting conventions, not one outlier.

2. **`mobile_mqa`'s masking gap is not inherited-fixable.** The finding (#7) treats
   `MobileMQA` as a `GroupedQueryAttention` subclass; it is
   (`mobile_mqa.py:63`), but it **overrides `call` entirely**, so the Tier-1 mask fix that
   landed in the parent does not reach it. Anyone reading the blast-radius table and
   assuming "group_query was fixed, so mobile_mqa was too" would be wrong.

3. **Two consumers are reachable only through a factory string.**
   `src/dl_techniques/layers/time_series/mixed_sequential_block.py:267` constructs
   `AnchorAttention` and `PerceiverAttention` as `'anchor'` / `'perceiver'`. It appears in
   no prior blast-radius table in this plan family, because every one of them was built
   from class-name greps. Grep the registry key as well as the class name.

One further item is new rather than a correction:
[Q4](#q4--transformerlayer-silently-drops-a-callers-mask-for-three-attention-types-new),
`TransformerLayer` silently dropping a supplied `attention_mask` for `'fnet'`, `'anchor'`
and `'lighthouse'`, was found while deriving Q3's blast radius and is not in any prior
catalogue.

---

# Group F — Added at step 10 (adversarial review of this plan)

## Q16 — `attention_routing_capsule` shares ONE initializer INSTANCE across three weights

**Current behavior.** `src/dl_techniques/layers/attention/attention_routing_capsule.py`
stores `self.kernel_initializer = keras.initializers.get(kernel_initializer)` — an
*object* — in `__init__`, and then hands that same object to three consumers: the routing
weight `W`, the query vector `q`, and (since step 6 of this plan, which fixed the
`prob_head` `Dense` receiving the default initializer instead of the caller's) the
`prob_head` `Dense`. Keras initializers are callables, not per-variable state, so nothing
raises and nothing is shape-mismatched.

**The question.** A *seeded* initializer is stateful in exactly the way that matters here.
`keras.initializers.GlorotUniform(seed=42)` reproduces the SAME draw on every call, so the
three weights above are no longer independent samples: whatever correlation the shared
seed induces is baked into the layer at build time. For the default `"glorot_uniform"`
string this is invisible (each `get()` result is unseeded and draws fresh), which is why
no test sees it — and it is why this is a design question rather than a defect report.

**Options.**

1. **Leave it and document it** (one sentence in the `kernel_initializer` docstring:
   "a seeded initializer instance is shared across `W`, `q` and `prob_head`, so their
   draws will be identical"). Cheapest; makes the behavior findable by the next reader.
2. **Deep-copy per consumer** — `keras.initializers.get(self.kernel_initializer.get_config())`
   or `copy.deepcopy` at each use site. Restores independence, but a copied *seeded*
   initializer still reproduces the same draw, so this only helps if the copies are also
   re-seeded, which changes what a user asking for `seed=42` gets.
3. **Store the SPEC, not the instance** — keep `kernel_initializer` as the raw argument and
   call `keras.initializers.get(...)` separately at each use. Same outcome as (2) for
   strings, same non-outcome for a passed instance, and it partially undoes the
   `regularizers.get()` canonicalization that step 6 deliberately added.

**Blast radius.** `attention_routing_capsule.py` and its `CapsuleBlockV2` wrapper only;
consumers are the CapsNet v2 model suite (`tests/test_models/test_capsnet/`) and the
`'attention_routing_capsule'` factory key. No `get_config()` key changes under any option
(I5 held), so no checkpoint compatibility question arises.

**Recommendation.** Option 1. The correlated-draw effect requires a caller to pass a
*seeded instance* — an uncommon, deliberate act — and options 2 and 3 both make the
seeding contract *less* predictable than sharing does, while adding a copy at every use
site. Reported by the step-10 adversarial review as NOTE-level (N10) and explicitly
carried here rather than fixed, because it is a call about what a shared seed should
*mean*, not a bug with a right answer.

---

## Q17 — `probability_type`s other than `softmax` have never been exercised WITH a mask

**Current behavior.** `ProbabilityOutput` (`layers/activations/probability_output.py`)
dispatches to `Softmax`, `Sparsemax`, `ThreshMax`, `AdaptiveTemperatureSoftmax` and the
routing layers; the eight mask sites hand it biased logits containing `MASK_BIAS_VALUE`
(and, after the `out_dtype` cast-back, `-inf` under `mixed_float16`). Every measurement in
this plan — and every one in the step-10 review — used the DEFAULT `softmax`.

**The question.** `sparsemax` and `threshmax` do not consume `-inf` the way a softmax
does: sparsemax sorts and thresholds, and a `-inf` participates in a cumulative sum;
threshmax renormalizes after a cut. Whether the `-1e9`/`-inf` bias produces exactly-zero
weight at masked positions under those strategies is UNMEASURED. Step 10 closed the
adjacent hole for the softmax `axis` — `rescue_axis` is now derived from
`probability_config` (D-017) — which makes this the remaining unexplored dimension of the
same public surface.

**Options.** (1) Measure all four types x four mask kinds x three policies at one site and
record the result; (2) restrict `probability_type` at the mask sites the way
`capsule_routing` and `window_attention` already restrict theirs, if a type is found to
break masking; (3) leave unmeasured and documented.

**Blast radius.** Every `probability_type`-carrying attention layer (eight mask sites).
No code change is implied until (1) is done.

**Recommendation.** Option 1, as a small standalone measurement plan. It is cheap
(one parametrized test file), it is the kind of gap that a finiteness-only suite cannot
see, and the plan that introduced `apply_attention_mask` has already demonstrated that
this exact class of question — "is the claim true for the non-default configuration?" —
is where its defects have actually been.

---

## Sources

- `src/dl_techniques/layers/attention/` at commit `58232e63` (every `file.py:line` in this
  brief was re-read at that commit).
- `keras/src/layers/layer.py` (Keras 3, `.venv`) — `:838-853` training propagation,
  `:1489-1504` call-context lifecycle.
- `research/2026_lighthouse_attention.md` — the specification `LighthouseAttention` is
  measured against in Q15.
- `plan-2026-07-27T183600-b4ef45f0` decisions D-002 (why the bias form is what it is),
  D-005 (rpc dtype boundary), D-006/D-008/D-009 (the degenerate-row rescue),
  D-015 (the `training=` measurement), D-016 (the two in-flight dtype defects).
- `plan-2026-07-27T130643-38c5646a` decision D-015 — the original correction establishing
  that `wave_field_attention`'s scale is self-consistent (Q8).
