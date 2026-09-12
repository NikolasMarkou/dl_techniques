# Decision Log
*Plan: plan-2026-09-12T075714-035fd488*

**python-software.md**: consulted — applicable (§A "Deep Modules"/"Design It
Twice" informed the layers.py class-boundary split in D-001/D-005 below;
§B.16 "When NOT to apply these patterns" confirmed none of the DDD/hexagonal
patterns in §B apply — this is a Keras model package, not a service with a
persistence boundary, so no Repository/UoW/Service-Layer scaffolding was
added around it).

## D-001 | EXPLORE → PLAN | 2026-09-12
**Context**: Zamba2's shared mem-block mechanic needs a way for one physical
attention/MLP block, reused at several depths, to behave slightly
differently at each reuse. No LoRA-style additive adapter exists anywhere in
this repo (`findings/existing-layers-for-zamba2.md` #2) — `LowRankFFN` is a
from-scratch low-rank *replacement*, not an adapter, so it cannot be reused.
**Decision**: Build a new `LoRAAdapter` layer indexed by a global "occurrence
index" (0..num_occurrences-1) rather than by which of the `num_mem_blocks`
physical blocks is invoked — occurrences cycle round-robin through the
physical blocks (e.g. ABAB for `num_mem_blocks=2`). Apply it ONLY to the
shared MLP block's up-projection output (per the goal's explicit scope),
not to the shared attention block's projections.
**Trade-off**: Matches the original PyTorch Zamba2 implementation's
`linear_fc1_lora_A_list[forward_layer_idx]` indexing exactly, and keeps the
attention block parameter-count-free of LoRA (simpler, one fewer place to
get the sharing/indexing wrong) **at the cost of** introducing one
repo-wide-novel abstraction with zero existing precedent to calibrate
against (no established convention for where `rank`/`alpha` live in
`get_config()`, confirmed absent in `findings/existing-layers-for-zamba2.md`
Risks & Unknowns) and at the cost of not LoRA-adapting the attention
projections, a minor divergence from the full original architecture that
the goal's own component list already licenses.
**Reasoning**: The attention block's job (concat original embedding + RoPE +
causal MHA) is already the riskiest composition in the plan (v2 guide §12.1
rank-3 mask trap); keeping LoRA out of it isolates the two highest-risk
mechanisms from compounding into one block. The up-projection is the
natural LoRA target because it is the widest, most-specializing matrix in a
gated-SiLU MLP.
**Anchor-Refs**: `src/dl_techniques/models/language/zamba2/layers.py:885`
(the `self.lora = LoRAAdapter(...)` construction inside
`Zamba2SharedMLPBlock.__init__`).

## D-002 | EXPLORE → PLAN | 2026-09-12
**Context**: The Zamba2 paper ships at 2.7B/7B scale (Zyda corpus, Mistral
tokenizer); the goal text explicitly rules this out as a non-goal and asks
for a repo-appropriate small/base size.
**Decision**: Define three variants (`zamba2_mini`/`zamba2_small`/
`zamba2_base`) at d_model 256/512/768 with 8/18/24 Mamba2 layers and
2/6/8 mem-block occurrences, order-of-magnitude matched to this repo's
existing `hnet`/`gpt2` small variants rather than the paper's own table.
**Trade-off**: A model this repo can actually train end-to-end on a single
consumer GPU within the plan's verification budget **at the cost of**
shipping no variant that is numerically comparable to any published Zamba2
benchmark — this package is an architectural port, not a benchmark
reproduction.
**Reasoning**: Matches the goal's explicit non-goal ("no original-scale
2.7B/7B variants") and the repo's own established scale for `hnet`/`gpt2`.

## D-003 | EXPLORE → PLAN | 2026-09-12
**Context**: `findings/training-pipeline-and-dataset.md` confirms only
Wikipedia is staged on `data0_4tb` (fineweb_edu is gated/empty by a prior
plan's explicit decision); Zamba2's own paper trains on Zyda + a Mistral
tokenizer, neither of which exist in this repo.
**Decision**: Train via Pattern 3 (`train.common.nlp`) against Wikipedia
with tiktoken `cl100k_base`, matching `gpt2`/`colbert`'s existing
subword-LM convention.
**Trade-off**: Zero new infrastructure (no new tokenizer dependency, no
28.5GB fineweb_edu download to request) and full reuse of
`load_wikipedia_train_val`/`preprocess_clm_packed_dataset` **at the cost
of** training on a smaller, narrower-domain corpus than the paper's own
recipe — acceptable because this plan's goal is architectural correctness
and an end-to-end smoke run, not a benchmark-grade pretraining run.
**Reasoning**: The goal text itself frames the training script as
"following repo convention," and no infra exists for the alternative.

## D-004 | EXPLORE → PLAN | 2026-09-12
**Context**: No Zamba2 checkpoint exists anywhere (no HF conversion, no
local training run yet). The repo-wide invariant (`plans/SYSTEM.md`
Invariants) requires `pretrained=True` to raise `NotImplementedError`
rather than silently returning a random-init model.
**Decision**: `create_zamba2(..., pretrained=True)` raises
`NotImplementedError` naming the variant and pointing at the
`pretrained=False` + `model.load_weights(path)` warm-start route; no
`hf_utils.py`-style HF-checkpoint loader is built (explicit non-goal).
**Trade-off**: Zero maintenance burden for a loader with nothing to load
**at the cost of** users who want the original Zyphra HF weights getting no
conversion path from this package — acceptable per the goal's explicit
non-goal statement.
**Reasoning**: Matches the repo-wide `pretrained=True` invariant exactly
and the goal's explicit non-goal scoping.

## D-005 | EXPLORE → PLAN | 2026-09-12
**Context**: Zamba2's Mamba2 mixer blocks are NOT shared across depth in
the original architecture (only the attention+MLP mem-blocks are); a wrong
design could accidentally share Mamba2 weights too, or could fail to share
the mem-blocks at all.
**Decision**: `Zamba2MambaBlock` instances are built fresh (independently
weighted) at every `'m'` position; `Zamba2SharedAttentionBlock`/
`Zamba2SharedMLPBlock` instances are built exactly `num_mem_blocks` times
total and reused round-robin across all `'g'` positions.
**Trade-off**: Matches the paper's actual parameter-sharing topology
**at the cost of** needing two different test postures in the same test
file (a positive "these ARE the same weights" guard for mem-blocks and a
negative "these are NOT the same weights" guard for Mamba2 blocks) — both
are written explicitly in steps 2-4 rather than assumed.
**Reasoning**: Getting this backwards would silently change the model's
effective capacity with no shape-level signal (the exact §12.7 failure
shape this plan's Pre-Mortem names).

## D-008 | EXECUTE iter-1/step-6.1 (completion fix) | 2026-09-12
**Context**: `review-iter-1.md` concern 1 measured
`tests/test_models/test_package_api_contract.py::TestNoMutableDefaults::test_no_mutable_default_anywhere`
RED at HEAD: the module-level `MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {...}`
plus the class-level alias `Zamba2Model.MODEL_VARIANTS = MODEL_VARIANTS` is
exactly the "S3: class attribute aliasing a module-level mutable binding"
shape that guard's own `_sweep_mutable_defaults` flags (the comment at
`model.py:651-654` justified the alias by citing `hnet`'s identical-looking
line, but `hnet`'s alias escapes because `HNet.MODEL_VARIANTS` lives in a
separate `config.py` whose values are immutable dataclass instances with
tuple fields, not because of the "matches hnet" framing itself).
**Decision**: Wrap the module-level `MODEL_VARIANTS` table in
`types.MappingProxyType(...)`, the same remedy `fastvit/model.py`'s
`MCI_VARIANTS`/`FastVitImageEncoder.MODEL_VARIANTS` alias already uses
(D-079 in a prior plan) — the guard's AST sweep only flags a module binding
whose value is a literal `{...}`/`[...]`/`dict()`/`list()` call, and
`MappingProxyType(...)` is none of those, so the class-level alias no
longer resolves to a flagged binding.
**Trade-off**: Zero behavioural change to any existing caller (`dict(...)`,
`.keys()`, `in`, subscript access on a `Mapping` all work identically to a
plain `dict`) **at the cost of** the inner per-variant dicts remaining
plain, mutable `dict` objects — accepted because the guard's own docstring
states the S3 shape is about the OUTER class-attribute alias, and the repo's
own precedent (`fastvit`'s `MCI_VARIANTS`) wraps only the outer level too.
**Reasoning**: Moving the table into a sibling `config.py` (the `hnet`-style
fix) would have been a larger, unrequested restructuring for a single
three-line guard failure; the in-place `MappingProxyType` wrap is the
smaller, already-precedented fix for this exact alias shape.
**Anchor-Refs**: `src/dl_techniques/models/language/zamba2/model.py` (the
`# DECISION plan-2026-09-12T075714-035fd488/D-008` comment immediately
above the `MODEL_VARIANTS` module-level binding).

## D-007 | EXECUTE iter-1/step-1.1 (completion fix) | 2026-09-12
**Context**: `review-iter-1.md` concern 3 measured `LoRAAdapter.build()`'s
`_a_initializer` calling one resolved `keras.initializers.Initializer`
instance (`self.kernel_initializer`) once per occurrence inside a Python
loop, all at the same `(input_dim, rank)` shape. Keras 3's own behaviour
(`dl_techniques.initializers.clone.clone_initializer` module docstring) is
that a seedless initializer instance self-assigns a seed at construction and
REPLAYS the same sample at every later call of the same shape — so the loop
produced `num_occurrences` bit-identical `A` slices (`A[0] == A[1] == ...`,
confirmed `np.array_equal`), directly contradicting the class's own
docstring Note, which claimed slice-by-slice initialization made the
occurrences "independent".
**Decision**: Call `clone_initializer(a_initializer_fn)` fresh inside the
per-occurrence loop, so each slice gets its own cloned initializer instance
(a new self-assigned seed for the common seedless case) rather than reusing
one resolved instance across all `num_occurrences` calls.
**Trade-off**: One `clone_initializer` call per occurrence at `build()` time
(negligible, `build()` runs once) **at the cost of** losing the (already
broken) "same seed replays across slices" property for a caller who
explicitly passes a SEEDED `kernel_initializer` — that caller still gets the
`clone_initializer` seeded-replay exemption (identical slices by contract),
which is correct behaviour for that documented case and not a regression
this fix touches.
**Reasoning**: `clone_initializer` already exists in this repo specifically
to give each call site of a shared initializer instance an independent draw
(its own module docstring, "Measured on two `Dense(4)` layers..."); this is
exactly that site, re-derived independently by the reviewer before this fix
looked at the helper.
**Anchor-Refs**: `src/dl_techniques/models/language/zamba2/layers.py` (the
`# DECISION plan-2026-09-12T075714-035fd488/D-007` comment inside
`LoRAAdapter.build()._a_initializer`).

## D-006 | EXECUTE iter-1/step-7 | 2026-09-12
**Context**: `train.common.clm_pretrain.ClmPretrainConfig` plus
`load_train_val_datasets`/`load_hf_clm_datasets` is the obvious-looking
shortcut for any new CLM trainer (3 of 4 existing CLM trainers --
`gpt2`/`wave_field`/`cliffordnet` -- already subclass it) and the findings
file's own Code Patterns section flags it as reusable infrastructure.
Inspecting it (`load_train_val_datasets`'s `# Wrap labels for dict-output
model` comment, `(x, y) -> (x, {"logits": y})`) showed it unconditionally
wraps every label batch for a DICT-output model -- exactly the shape
`GPT2`'s `{"logits": ...}` forward pass returns, and exactly NOT the shape
`Zamba2Model.call` returns (a single `(batch, seq_len, vocab_size)` tensor,
confirmed at `model.py:597-601`'s own return-type annotation, step 5).
**Decision**: Built a local `Zamba2TrainingConfig` dataclass + `build_datasets`
in `src/train/zamba2/common.py` that calls the LEAF helpers directly
(`load_wikipedia_train_val`, `preprocess_clm_packed_dataset`,
`estimate_clm_steps_per_epoch`) -- the same shape `train.hnet.common` uses
for its byte-level pipeline -- rather than subclassing `ClmPretrainConfig`.
`create_clm_loss_fn` IS reused (it reads only three scalar config fields,
no dict-output assumption), and so is `build_clm_metrics`.
**Trade-off**: A ~60-line local dataclass + `build_datasets` function not
shared with the three dict-output trainers, **at the cost of** avoiding
either (a) wrapping `Zamba2Model`'s plain-tensor output in a dict it does
not have just to satisfy a shared wrapper, or (b) a silent label-shape
mismatch at `model.fit()` time that no shape check would catch until the
loss function raised on a dict it was never given.
**Reasoning**: A field that is plausible to reuse is not the same as a field
whose CONTRACT matches. The wrapper's label-shape assumption is the actual
shared concern, not happens to be shared code; diverging from it here keeps
the divergence honest (a new function, not a quietly-bypassed shared one).
Also covers the repeated `decay_steps = total_steps - warmup_steps`
anchor in `build_optimizer` -- the same framework trap `train.hnet.common`
documents, copied as a fresh instance in this file rather than factored out
(no shared call site exists for it across the two trainers' differing
config shapes).
**Anchor-Refs**: `src/train/zamba2/common.py:~420` (the
`# DECISION plan-2026-09-12T075714-035fd488/D-006` comment inside
`build_optimizer`'s `decay_steps` computation); module docstring of
`src/train/zamba2/common.py` (the "Two things this module deliberately
does NOT do" section explaining the `ClmPretrainConfig` divergence).
