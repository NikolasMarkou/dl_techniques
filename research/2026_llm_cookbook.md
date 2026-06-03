# Convergent Recipes for Mid-Size and Large Language Models: A Four-Way Differential Analysis

**Technical Note - v2.0 - June 2026**

## Abstract

We perform a differential analysis of four contemporaneous open training reports that span roughly
two orders of magnitude in scale and four independent organizations: **DeepSeek-V4** (DeepSeek-AI),
a 1.6T-total / 49B-active (Pro) and 284B / 13B (Flash) Mixture-of-Experts (MoE) frontier model with a
1M-token context, trained on 32-33T tokens; **Kimi K2** (Moonshot AI), a 1.04T-total / 32B-active MoE
trained on 15.5T tokens for agentic intelligence; **MAI-Thinking-1 / MAI-Base-1** (Microsoft AI), a 1T-total
/ 35B-active MoE trained from scratch on 30T + 3.55T mid-training tokens with no third-party distillation;
and **Mellum 2** (JetBrains), a 12B-total / 2.5B-active MoE specialized for in-IDE software engineering,
trained on 10.6T tokens under a fixed single-GPU inference budget. The four programs differ in hardware
(GB200/GB300, H800, H200), optimizer (Muon, MuonClip, AdamW), data philosophy (synthetic-heavy vs
strictly human), attention design (MLA, hybrid compressed-sparse, GQA, GQA+SWA), and serving target
(1M-token frontier reasoning vs commodity-GPU code completion).

The methodological premise is that **agreement across independently-motivated recipes is the reliable
signal**, while disagreement localizes the design axes that are genuinely scale-, data-, or
deployment-dependent. Because the four labs share only the public literature (Muon, GRPO, DeepSeekMoE,
MLA, Qwen3-MoE) and not engineering lineage, a choice that all four converge on despite different
constraints is far stronger evidence than any single ablation. We organize the convergent findings by
subsystem, tag each with a confidence tier proportional to how many of the four adopt it, and isolate the
contested axes together with the variable that appears to govern the choice. We close with a consolidated,
actionable cookbook.

This note supersedes the prior two-way analysis (Mellum 2 + MAI-Thinking-1); adding DeepSeek-V4 and
Kimi K2 as a frontier anchor and a second trillion-scale point converts several previously two-of-two
coincidences into four-way or three-of-four signals, and overturns a few that turned out to be lineage
artifacts.

---

## 1. Method and confidence tiers

We treat each recipe as an independent vote. A design choice is assigned a confidence tier by the number
of systems that adopt it and whether the remainder actively contradict it:

- **C4** - all four adopt it. Treat as a default; deviating requires a specific reason.
- **C3** - three adopt it, the fourth is silent or uses a compatible variant. Strong default.
- **C2** - two adopt it, others silent. Suggestive; adopt when the context matches the adopters.
- **Contested** - the four split, or two explicitly contradict. Not a free parameter: there is usually a
  governing variable (scale, attention type, deployment target) that selects the answer.

**Non-independence caveat.** The four are not perfectly independent. Kimi K2 and DeepSeek-V4 both descend
from the DeepSeek-V3 architectural template (MLA, DeepSeekMoE, MTP), so an agreement between only those
two is weaker than an agreement that also includes Mellum 2 (Qwen3 lineage) or MAI (clean-room). Where a
signal rests only on the two DeepSeek-lineage systems we say so explicitly and downgrade it.

### System cards

| Axis | DeepSeek-V4 (Pro) | Kimi K2 | MAI-Thinking-1 | Mellum 2 |
|---|---|---|---|---|
| Total / active params | 1.6T / 49B | 1.04T / 32B | 1T / 35B | 12B / 2.5B |
| Pre-train tokens | 33T | 15.5T | 30T + 3.55T mid | 10.6T |
| Experts (total / active / shared) | 384 / 6 / 1 | 384 / 8 / 1 | 512 / 8 / 0 (interleaved w/ dense FFN) | 64 / 8 / 0 |
| Attention | Hybrid CSA+HCA (compressed + sparse), shared-KV MQA | MLA, 64 heads | GQA-8, 5:1 local(SWA):global(NoPE) | GQA-4, 3:1 SWA:full |
| Attention-logit control | RMSNorm on Q/KV | QK-Clip (MuonClip) | QK-RMSNorm | QK-Norm |
| Optimizer | Muon (+ AdamW for embed/head/norm) | MuonClip | AdamW | Muon (+ Adam for embed/out) |
| Precision | BF16 + FP8 + FP4 (MXFP4 QAT) | BF16 + FP8 (storage) | BF16 + FP8 | BF16 + FP8 |
| LR shape / peak | warmup-const-cosine / 2.0e-4 | WSD / 2e-4 | warmup-cosine / 2e-4 | WHD / 3e-4 |
| Weight decay | 0.1 | 0.1 | 0.1 | 0.1 |
| MTP | yes (depth 1) | no | no | yes (1 head, spec-decode draft) |
| Context extension | 4K->16K->64K->1M | 4K->32K->128K (YaRN) | 16K->64K->256K | 8K->128K (layer-selective YaRN) |
| Post-train | specialists -> on-policy distillation | SFT -> joint RL (K1.5) | specialists -> consolidation SFT -> RL | SFT (instruct/think) -> RLVR (GRPO) |
| Reward for hard-to-verify | Generative Reward Model | self-critique rubric | AI judge + RM | LLM-as-judge |

---

## 2. Convergent architecture

**[C4] Fine-grained MoE is the backbone.** All four are sparse MoE transformers with many small experts.
Top-k is small and stable across the set: 6 (DeepSeek-V4) or 8 (the other three) active experts. Beyond
that, **sparsity scales with model scale**: the trillion-scale systems run 384-512 experts (sparsity
~48-64x), Mellum at 12B runs 64 experts (sparsity 8x) and explicitly reports that very high sparsity hurts
at small scale. Kimi K2 contributes a sparsity scaling law (more experts at fixed active params lowers
loss monotonically; sparsity 48 cuts FLOPs to reach a target loss by 1.15-1.69x vs sparsity 8-32), and MAI
contributes an efficiency-gain ladder showing the same monotonic trend from 256 to 1024 experts. Takeaway:
choose the highest sparsity your serving infrastructure can tolerate, scaled down for small models.

**[C4] KV-cache pressure is the dominant attention constraint, and every system attacks it.** The specific
mechanism differs (and is contested, see Section 8), but all four treat the KV cache, not raw attention
FLOPs, as the thing to minimize: DeepSeek-V4 compresses KV along the sequence (CSA m=4 + sparse top-k,
HCA m'=128) and stores KV in FP8 except RoPE dims; Kimi K2 uses MLA latent compression and cuts attention
heads from 128 to 64 (the heads-equal-layers choice costs only 0.5-1.2% loss but avoids an 83% inference-
FLOP increase at 128K); MAI pairs GQA-8 with a 5:1 local:global layer ratio; Mellum uses GQA-4 with 3:1
SWA. The shared lesson: **attention is selected against long-context / high-concurrency serving cost, not
against training loss.**

**[C4] Attention-logit control is mandatory.** Every system adds an explicit mechanism to bound pre-softmax
attention logits, and three of four use the same one: query/key RMSNorm (DeepSeek-V4 normalizes Q and the
compressed KV "to avoid exploding attention logits"; MAI and Mellum apply QK-Norm). Kimi K2 is the
exception only because MLA does not materialize full K matrices at inference, so QK-Norm is inapplicable;
it instead introduces QK-Clip (rescale Wq/Wk post-update when a head's max logit exceeds tau=100). The
convergence is on the *requirement*, not the implementation: unbounded attention logits are the most
commonly reported source of large-scale instability, and a normalization or clip on the QK path is the
standard fix.

**[C3] Interleave cheap local attention with full/global attention.** DeepSeek-V4 (sliding-window branch +
attention sink), MAI (5 local : 1 global), and Mellum (3 SWA : 1 full) all keep most layers local and only
a minority global. Kimi K2 abstains, relying on MLA compression for the same KV saving. Where SWA is used,
both Mellum and the Gemma lineage MAI cite a window of 1024 outperforming 512.

**[C2, downgraded] MTP as an auxiliary objective and speculative-decoding draft.** DeepSeek-V4 and Mellum
2 both train a Multi-Token-Prediction head (Mellum reuses it as a built-in draft model, +10.4 HumanEval at
7% training-time cost). MAI and Kimi omit it. Because DeepSeek-V4's MTP is inherited from the DeepSeek-V3
template, this is a genuine two-of-four rather than a four-way signal: useful, not universal.

**[C4 baseline] Shared components.** RMSNorm, gated-SiLU/SwiGLU MLPs, and RoPE (partial in the two
compressed-attention systems) are universal and uncontroversial.

---

## 3. Convergent optimization and precision

**[C3, strong] Muon over AdamW, with three shared implementation details.** Three of four (DeepSeek-V4,
Kimi K2 via MuonClip, Mellum 2 via the Moonlight configuration) train the hidden layers with Muon, citing
its token efficiency; only MAI stays on AdamW. The three Muon users independently agree on:

1. **Hybrid optimizer assignment** - Muon for the 2D hidden weights, AdamW/Adam for embeddings, the output
   head, and all norm/gain parameters (DeepSeek-V4 additionally keeps AdamW for the mHC static biases and
   gates).
2. **RMS-matched update rescaling** so AdamW learning rates transfer: rescale factor ~0.2 (Kimi and Mellum
   both 0.2; DeepSeek-V4 0.18).
3. **Newton-Schulz orthogonalization** (5 iterations Mellum/Kimi; DeepSeek-V4 uses a 10-iteration two-stage
   hybrid).

**[C2, mechanistically important] Muon raises the attention-logit-explosion risk, so it ships with a QK
control.** The two Muon systems that analyze stability both tie Muon to exploding logits: Kimi K2 argues
Muon's full-effective-rank updates grow singular values additively and adds QK-Clip; DeepSeek-V4 states it
applies RMSNorm to Q/KV specifically so it "does not employ the QK-Clip technique." Read together: if you
adopt Muon, budget for an attention-logit control as part of the same decision. (MAI, on AdamW, still
zero-inits attention output, but for a different reason - routing balance, Section 6.)

**[C4] Weight decay 0.1.** All four use exactly wd = 0.1. MAI additionally reduces it on attention (0.01)
and embeddings (0.005).

**[C4] Momentum / beta near 0.95.** DeepSeek-V4 Muon momentum 0.95; Kimi Muon momentum (Moonlight); Mellum
Adam betas 0.9/0.95 and Muon momentum 0.95; MAI AdamW beta1 0.95, beta2 0.925. The high-momentum / high-
beta2 regime is shared.

**[C4] FP8 training with BF16 master and FP32 in the sensitive path.** All four run FP8 (DeepSeek-V4 pushes
to FP4/MXFP4 for routed-expert weights and the indexer QK path during QAT; Kimi uses FP8 for activation
*storage* but deliberately not for compute; MAI and Mellum use FP8 in the GEMMs). Three of four keep the
router and pre-softmax logits in FP32 (MAI is most explicit: FP32 for router logits, output logits, MoE
combine, and the full residual stream); gradient reduction / accumulation is FP32 across the set. Two
(MAI, DeepSeek-V4) add stochastic rounding on the high-to-low downcast.

**[C4] Learning-rate shape: warmup -> long stable/hold -> decay at the very end.** Despite different names
(cosine-after-constant, WSD, WHD), all four use a short warmup, a long plateau at peak, and a decay
concentrated in the final ~10-20% of training. Peak LR is tightly clustered at **2-3e-4** (DeepSeek-V4
2.0-2.7e-4, Kimi 2e-4, MAI 2e-4, Mellum 3e-4). The decay *endpoint* carries a real, shared lesson but a
split answer: MAI found decaying only to 0.1x peak (not 0.01x) improved post-RL results; Mellum found
linear decay to zero beat cosine-to-nonzero; DeepSeek-V4 and Kimi decay to ~1/10. The agreement is that
the endpoint matters and should be tuned against the post-training objective, not left at a default.

**[C2] Batch-size ramp.** DeepSeek-V4 (ramp to 75.5M/94.4M tokens) and Mellum (2048->4096 sequences) ramp
batch size early; Kimi and MAI hold it constant (67M, 134M). Adopt when warmup stability is a concern;
not load-bearing.

---

## 4. Convergent data and curriculum

**[C4] A distinct later phase that upweights high-value domains ("web early, curated/code/math late").**
All four separate a broad early phase from a later high-quality phase. Mellum's three-phase curriculum
shifts code 23% -> 42% -> 59% as the LR decays; MAI mid-trains with STEM/math raised to ~35% and code 55%;
DeepSeek-V4 adds agentic, coding, and long-document data in mid-training; Kimi K2 runs an annealing phase
before long-context activation. The principle: spend the high-LR plateau on diversity and the low-LR tail
on the domains you actually care about, because the model is most plastic to quality late.

**[C3] Aggressive deduplication is treated as a quality lever, not hygiene.** MAI runs the most explicit
stack (boilerplate removal, exact, MinHash-LSH fuzzy at 0.8, templated-page skeletonization, semantic/
embedding dedup, and cross-dataset drop-order); Mellum uses MinHash file-level; DeepSeek-V4 filters
batched auto-generated and templated content to avoid model collapse; Kimi follows the K1.5 pipelines.
MAI and Mellum both report that **insufficiently deduplicated, low-diversity data directly causes loss
spikes** (Section 6), making dedup a stability concern as well as a quality one.

**[C3] Cap raw-data repetition; prefer fresh or rephrased tokens over re-epoching.** Mellum caps any
dataset at 4 epochs ("the point where repetition stops yielding gains"); MAI caps at 8 and adds a
memorization-aware per-source cap; Kimi shows rephrasing once then training beats repeating raw data 10x
on SimpleQA, and rephrases each corpus at most twice. The shared belief: token *utility* per pass, not
raw token count, is the binding constraint when high-quality data is scarce.

**[C2->C3] Document packing that minimizes intra-document truncation.** DeepSeek-V4 and Mellum 2 both
adopt and cite the same work (Ding et al. 2024, "Fewer Truncations Improve Language Modeling") for best-fit
packing; MAI and Kimi also pack to fixed length. The two that cite it report the same motivation: spurious
cross-document context induces hallucination.

**[Contested] Synthetic / rephrased data in pre-training.** Kimi K2 makes rephrasing a central token-
efficiency strategy; Mellum finds synthetic code complements raw code, "particularly for smaller-scale MoE
models where data diversity is crucial." MAI takes the opposite stance and *excludes* model-generated
content from pre-training entirely, arguing capabilities must be learned from human data, not imitated.
DeepSeek-V4 sits in between: it filters AI-generated content out of the web crawl (collapse risk) yet
synthesizes heavily in post-training. Governing variable: data-scarcity pressure and a lab's tolerance for
factual-drift risk. Not a free parameter; pick a side deliberately.

---

## 5. Convergent long-context extension

**[C4] Train short, extend in a dedicated late stage; do not train long throughout.** Every system pre-
trains at a short context (4K-16K) and reaches its target window (128K-1M) through a staged extension run.
DeepSeek-V4 goes 4K->16K->64K->1M; MAI 16K->64K->256K; Kimi 4K->32K->128K; Mellum 8K->128K. MAI states the
rationale plainly (low MFU at long sequence length makes full-length training impractical), and both MAI
and Mellum report that **the extension converges fast and cheap** - MAI sees most of the gain in the first
1-10% of extension iterations, Mellum sees a RULER plateau by ~30B tokens. The representations needed for
long context are largely present after pre-training; extension only recalibrates positional behavior.

**[C2] YaRN, applied layer-selectively.** Mellum and Kimi both use YaRN for the frequency remap. Mellum
and MAI both apply the remap *only to the global/full-attention layers* (the Gemma-3 / OLMo-3 recipe),
leaving local/windowed layers untouched, and Mellum ablates this as clearly superior to a uniform RoPE-
base bump. When you combine windowed and global attention, scale only the global layers.

---

## 6. Convergent training-stability findings

This is where four independent large-MoE runs produce the most striking agreement, because instability is
where reality bites hardest.

**[C4] The MoE router is the primary instability surface.** DeepSeek-V4 ties loss spikes to MoE outliers
that "the routing mechanism itself appears to exacerbate"; MAI traces early spikes to high expert
imbalance under dropless routing; Kimi's entire MuonClip contribution exists to tame attention dynamics
that destabilize MoE training; Mellum links router load-balance to throughput and stability. The router,
not the FFN or attention math, is the thing to instrument.

**[C3, strong] Balance load at global/batch scope, not per-sequence; the aggregation scope matters more
than the loss form.** MAI states this most directly ("we found the aggregation strategy to matter much
more than the load-balancing loss type"; GShard-style ~ loss-free as long as aggregation is global).
DeepSeek-V4 uses auxiliary-loss-free balancing plus only a *slight* sequence-wise term to prevent within-
sequence collapse; Mellum chose global-batch over per-sequence despite per-sequence's slightly better
short-run loss; Kimi's smaller EP groups relax balance constraints. The auxiliary-loss-free vs GShard-loss
debate is second-order; the global aggregation is first-order.

**[C2] Dropless routing.** MAI and Mellum both converge to fully dropless MoE (no capacity factor),
reporting no quality difference at capacity 1.0-1.5 and ~15% initial step-time overhead that shrinks as the
router balances. DeepSeek-V4 removes the routing-target-node constraint; Kimi keeps capacity with selective
recompute. Two-of-four, both at non-frontier scale.

**[C2] Low-diversity / duplicate data causes loss spikes.** MAI filtered samples with abnormally low
lexical diversity (a single repeated token spanning the context); Mellum filters samples with fewer than
82 unique tokens (1% of context) for the same reason, and separately diagnosed periodic spikes from hash-
sorted exact-duplicate chunks. A specific, independently-rediscovered failure mode: screen for degenerate
low-entropy sequences before they hit the optimizer.

**[Contested mechanisms, shared goal] Outlier suppression.** The four use different knobs to suppress the
anomalous activations that precede divergence: DeepSeek-V4 clamps SwiGLU (linear branch to [-10,10], gate
capped at 10) and decouples routing updates from backbone updates (Anticipatory Routing, using historical
parameters for routing indices); Kimi clips QK weights; MAI zero-initializes attention output (output-norm
gain = 0) so the network starts as per-token FFNs and grows cross-token mixing gradually, which fixes the
early routing imbalance. Different mechanisms, one objective: keep early-training activations bounded so
the router does not enter a vicious cycle.

---

## 7. Convergent post-training and RL

**[C4] Two-stage post-training: SFT then RL.** Universal. No system relies on SFT alone.

**[C4] Verifiable rewards (RLVR) are the backbone of RL.** DeepSeek-V4 (rule/test verifiers + GRPO), Kimi
(a "Verifiable Rewards Gym" spanning math/STEM/logic, instruction-following with hybrid rule + hack-check,
coding sandboxes, safety), MAI (RLVR across STEM/agentic), and Mellum (RLVR explicitly chosen over RLHF
"because every prompt admits an unambiguous programmatic check") all anchor RL on programmatically
checkable reward. Train on what you can verify first.

**[C4] Group-relative policy optimization.** DeepSeek-V4, MAI, and Mellum use GRPO or GRPO variants; Kimi
uses the closely-related K1.5 objective (sample K per prompt, subtract the group-mean baseline, regularize
toward the old policy). The family - sample a group, baseline against the group, no learned value network -
is universal.

**[C3] Move from scalar reward models toward generative judges / self-critique for hard-to-verify tasks.**
DeepSeek-V4 explicitly "dispenses with conventional scalar reward models" in favor of a Generative Reward
Model that is the actor itself; Kimi makes the actor its own critic via a self-critique rubric reward with
a closed loop that distills RLVR signal back into the critic; MAI combines a reward model with AI judges
and verifiable rewards; Mellum uses LLM-as-judge for free-form outputs. The direction is consistent: scalar
RMs are noise-prone and are being replaced or backstopped by generative evaluation grounded in verifiable
data.

**[C3] Length / inference-budget control inside RL.** RL inflates response length; three of four push back.
MAI uses a difficulty-scaled length penalty (hard problems penalized less, to preserve exploration); Mellum
uses DAPO's soft-overlong penalty plus an ARLCP-style concision penalty on reflection-trigger words; Kimi
enforces a per-task token budget with truncation-and-penalty. The intent is identical: keep test-time
compute proportional to task difficulty rather than letting it grow unbounded.

**[C3] Treat the training/inference numerics gap as a first-order RL-stability problem - and the MoE router
is the named culprit.** MAI and Mellum both identify the router (the same hidden state can dispatch to a
different expert under the inference kernel than under the trainer) as the principal source of per-token
log-prob disagreement; MAI fixes it with bf16 on both engines plus MoE routing replay and top-p mask
replay, Mellum with IcePop truncation (zero out tokens whose train/infer ratio leaves a band) plus KV-cache
recompute after each weight push. Kimi attacks the same consistency requirement from the systems side with
a checkpoint engine that reshards and broadcasts a full 1T-parameter update in under 30s. Asynchronous MoE
RL is unstable unless train and inference are forced into numeric agreement.

**[C3] Build agentic/SWE environments from real repositories plus simulation, graded by executable tests.**
Kimi (3000+ real MCP tools + 20000+ synthesized, a tool-simulator "world model," and real execution
sandboxes for code), DeepSeek-V4 (the DSec sandbox platform; agentic data in mid-training), and MAI (the
SEE sandbox; SWE environments auto-built from 4.87M GitHub PRs, validated by F2P/P2P test transitions)
all converge on the same construction: harvest real PRs/issues, build executable containers, grade with
hidden tests, and supplement with simulation for scale.

**[C3] Mitigate reward hacking explicitly in executable environments.** MAI scrubs post-base-commit git
history, resets agent-modified test files, and network-isolates sandboxes; Kimi adds a hack-check layer
that detects false claims of instruction compliance; Mellum scores un-scoreable responses as zero and
shows the model its error. Executable verification is necessary but not sufficient; the environment itself
must be hardened.

**[C3] Ship explicit reasoning/thinking modes with controllable budget.** DeepSeek-V4 (Non-think / High /
Max effort modes), MAI (thinking variant), and Mellum (separate Instruct and Thinking variants) all expose
a reasoning-effort axis; Kimi ships non-thinking-first but uses budget control to the same end.

**[C2, but striking] Train domain specialists, then consolidate into one model.** DeepSeek-V4 trains per-
domain specialist experts (SFT + GRPO) then merges them via multi-teacher on-policy distillation (reverse
KL, >10 teachers) into a single student; MAI trains three specialists (STEM, agentic, helpfulness/safety),
distills them into one model via SFT, then runs a final consolidation RL. Two independent trillion-scale
labs arrived at the identical "specialize then consolidate" topology, which is notable even at two-of-four.

**[C3] Distill only from your own models, never third-party.** MAI is explicit (no third-party
distillation); DeepSeek-V4's on-policy distillation is from its own specialists; Kimi self-improves via
rejection sampling on its own outputs. The consensus is self-distillation / on-policy distillation, not
teacher-model imitation.

**[C3] When pre-trained with Muon, fine-tune with Muon.** Kimi states this as a finding ("a Muon-pretrained
checkpoint produces the best performance with Muon fine-tuning"); Mellum and DeepSeek-V4 carry Muon into
post-training. Optimizer choice is sticky across stages.

---

## 8. Convergent infrastructure and the efficiency-first meta-principle

**[C4] Inference cost is a first-class architecture-selection criterion, co-equal with quality.** This is
the strongest cross-cutting meta-pattern. Mellum selects every architectural knob (MoE vs dense, 8-of-64
sparsity, 4 KV heads, 3:1 SWA, MTP draft) by ablation under a fixed budget (match Qwen2.5-7B on one H100)
and reports iso-latency maps. Kimi cuts attention heads 128->64 purely to halve inference FLOPs and derives
a sparsity scaling law in FLOPs-to-target-loss terms. DeepSeek-V4 optimizes single-token FLOPs and KV-cache
size as headline metrics (27% FLOPs, 10% KV vs V3.2 at 1M). MAI introduces an explicit time-denominated
efficiency-gain metric (EG_Time) and co-designs across five model generations to hold MFU above 20%. None
of the four selects architecture on training loss / FLOPs alone.

**[C3] Scaling-law / ladder-guided decisions.** MAI (scaling ladders + efficiency gain across a model
family), Kimi (sparsity and attention-head scaling laws), and DeepSeek-V4 (scaling-law compute envelope)
make architecture decisions by fitting curves on small models and validating the trend at scale - with the
MAI caveat that rank-invariance can fail (a data mixture that wins small can lose large), so the *scaling
trend*, not the single-scale winner, is what should be validated.

**[C3] Asynchronous RL with a fast weight-sync path.** MAI (Rocket; a compiled resharding transfer plan),
Mellum (NeMo-RL async; KV recompute on weight push), and Kimi (colocated engines; checkpoint engine, full
1T update < 30s) all decouple rollout generation from the trainer and treat weight transfer between the two
sharding regimes as a named, optimized subsystem.

**[C3] Activation-memory management via recomputation plus host offload.** MAI (transformer-layer
checkpointing + activation offload), Kimi (selective recompute of cheap high-footprint ops + CPU offload +
FP8 activation storage), and DeepSeek-V4 (tensor-level fine-grained recomputation) all combine selective
recompute with offload rather than relying on either alone.

**[C2] Determinism / batch-invariance as an infrastructure property.** DeepSeek-V4 ships batch-invariant
deterministic kernels for bitwise reproducibility across training and inference; MAI treats determinism as
first-class (bitwise-reproducible runs, deterministic reductions, disabled NVLink SHARP). Both argue
determinism is what makes loss-spike forensics and train/inference alignment tractable. Two-of-four, both
trillion-scale, where the payoff is largest.

**[C3] Fault-tolerant, preemptible rollout/checkpointing.** DeepSeek-V4 (token-granular write-ahead log so
interrupted rollouts resume without length bias), MAI (hot-standby in-job restarts), and Kimi (checkpoint-
engine startup robust to single-replica failure) all engineer the long-horizon RL loop to survive frequent
hardware failure.

---

## 9. Contested axes and their governing variables

These are the choices the four split on. Each is governed by a variable, not by taste.

| Axis | Split | Governing variable |
|---|---|---|
| Optimizer | Muon (DeepSeek-V4, Kimi, Mellum) vs AdamW (MAI) | 3:1 toward Muon for token efficiency; MAI's AdamW is the conservative-stability choice. Lean Muon + a QK control. |
| Attention-logit control | QK-Norm (DeepSeek-V4, MAI, Mellum) vs QK-Clip (Kimi) | Attention type: MLA cannot materialize K at inference, forcing QK-Clip. Use QK-Norm unless you run MLA. |
| Shared expert | yes (DeepSeek-V4, Kimi) vs none (Mellum) / conditional (MAI) | MoE layout and scale: MAI finds shared experts help only in every-layer-MoE, not interleaved; Mellum finds no gain and an inference cost at 12B. Add a shared expert at frontier scale with uniform MoE; skip it for small or interleaved designs. |
| Synthetic data in pre-training | embrace (Kimi, Mellum) vs reject (MAI), filter-then-synth-later (DeepSeek-V4) | Data-scarcity pressure vs factual-drift tolerance. |
| Position encoding | partial RoPE (DeepSeek-V4), RoPE (Kimi, Mellum), NoPE on global layers (MAI) | Whether global layers are expected to extrapolate; NoPE-global is a Gemma-lineage bet. |
| MTP | yes (DeepSeek-V4, Mellum) vs no (Kimi, MAI) | Whether you want a speculative-decoding draft for free; useful, not required. |
| Tied embeddings | tied (MAI) vs untied (Mellum) | Small-model parameter economy vs capacity; weak signal. |
| Dropout | 0.15 (MAI) vs none (others) | MAI's AdamW-only, no-synthetic regime; idiosyncratic. |
| Decay endpoint | ~0.1x peak (DeepSeek-V4, MAI, Kimi) vs 0 (Mellum) | Both report it tuned against post-training; tune it, do not default it. |
| DualPipe | used/adjusted (DeepSeek-V4) vs rejected (Kimi) | Parameter scale: Kimi rejects DualPipe's 2x parameter/gradient memory at 1T and uses interleaved 1F1B. |

---

## 10. The consolidated cookbook

The high-confidence intersection, as an actionable default recipe for a new mid-to-large MoE.

**Architecture (C4 unless noted).**
- Fine-grained MoE, top-k 8 (6 acceptable). Sparsity scaled to model size: ~8x at ~10B, 48-64x at
  trillion scale. Add one shared expert only at frontier scale with a uniform-MoE layout (contested).
- Reduce KV cache aggressively and pick the attention design against serving cost: GQA with few KV heads
  (4-8) and a local:global or SWA:full interleave (3:1 to 5:1) for most budgets; MLA or learned KV
  compression at the frontier.
- Mandatory attention-logit control: QK-Norm (default) or QK-Clip (if MLA).
- RMSNorm, gated-SiLU/SwiGLU, RoPE (partial if attention is compressed). MTP head optional (adds a
  spec-decode draft cheaply).

**Optimizer and precision (C4 / C3).**
- Muon for hidden layers; AdamW for embeddings, head, and norms; RMS-match rescale ~0.2; Newton-Schulz
  orthogonalization. (AdamW-throughout is the conservative fallback.)
- Weight decay 0.1; momentum / beta2 ~0.95; gradient clip 1.0.
- BF16 master, FP8 GEMMs, FP32 for the router, pre-softmax logits, and all reductions/accumulation;
  stochastic rounding on downcasts. FP4 only for post-training QAT of expert weights if you have the
  kernels.
- LR: short warmup -> long plateau at peak ~2-3e-4 -> decay over the final 10-20%; tune the endpoint
  (0 to 0.1x) against your post-training objective.

**Data and curriculum (C4 / C3).**
- Phase the mixture: diverse web early, upweight code/math/curated in a late phase aligned with LR decay.
- Deduplicate hard (exact + MinHash-fuzzy + semantic) and screen out low-entropy / degenerate sequences;
  treat both as stability measures.
- Cap raw-data repetition (4-8 epochs); prefer rephrased/fresh tokens over re-epoching when data is scarce.
- Best-fit pack to minimize intra-document truncation.

**Long context (C4 / C2).**
- Pre-train short (4K-16K); reach the target window in a short, dedicated late extension (it converges in
  ~30B tokens / the first ~10% of iterations).
- Use YaRN, applied only to the global/full-attention layers.

**Post-training (C4 / C3).**
- SFT, then RL. Anchor RL on verifiable rewards; use group-relative policy optimization (GRPO family).
- For hard-to-verify tasks, prefer generative judges / self-critique grounded in the verifiable signal
  over scalar reward models.
- Control response length / inference budget inside RL.
- Build agentic/SWE environments from real repos + simulation, grade with hidden executable tests, and
  harden against reward hacking (git scrub, test reset, hack-check, network isolation).
- For multi-domain models, consider specialize-then-consolidate; self-distill, never imitate third-party
  models.

**Infrastructure (C4 / C3).**
- Select architecture against an explicit inference-cost budget, not training FLOPs alone.
- Validate decisions on a scaling ladder, checking the trend (not the single-scale winner).
- For RL: asynchronous rollout with an optimized weight-sync path; force train/inference numeric agreement
  (bf16 both sides, routing replay, or import-side truncation) because the MoE router is the main mismatch
  source; deterministic kernels and fault-tolerant rollout pay off at scale.

---

## 11. Limitations of this analysis

- **Four points, shared literature.** Convergence here means "four 2026 labs reading the same papers chose
  the same thing," not "this is optimal." Two of the four (DeepSeek-V4, Kimi K2) share the DeepSeek-V3
  template, so a signal resting only on those two is weaker than its raw count suggests; we flagged those
  cases.
- **Reports, not replications.** Each recipe is self-reported, often without the negative ablations that
  would falsify it. Convergence reduces but does not eliminate publication and survivorship bias.
- **Deployment skew.** All four are MoE; three of four are trillion-scale reasoning/agentic models. The
  intersection is therefore a recipe for *large sparse models*, and may not transfer to dense, small, or
  non-text settings. Mellum is the only sub-100B point, so any "scales down" claim rests on a single
  witness.
- **Capability-not-controlled.** The four target different capabilities (1M-context reasoning, agentic
  coding, from-scratch STEM, IDE completion). A convergent choice may reflect a shared capability target
  rather than a universal training truth.

Treat the C4/C3 items as strong priors to deviate from only with cause, the C2 items as context-dependent
options, and the contested axes as decisions to make deliberately against the governing variable named in
Section 9.
