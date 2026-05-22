# CCNets — Design Principles of the Paradigm

This document states the **operational design principles** of Causal Cooperative
Networks: the non-obvious rules that decide whether a CCNet trains and cooperates, or
silently degenerates into three disconnected networks.

It is distinct from its siblings:

- `FOUNDATION.md` — the *conceptual* essay (why causation, what X/Y/E mean).
- `CLAUDE.md` — the *practical* how-to (contract, wiring, task adaptation).
- `FIXES.md` — the *diagnostic* post-mortem (what was broken and how it was repaired).
- **This file** — the *principles*: each rule, why it holds, what fails without it.

Every principle below was either confirmed by the audit in `FIXES.md` or is a load-bearing
invariant of the implementation. Each is stated as: **Principle → Rationale → Failure mode
→ Enforcement.**

---

## Part A — Architectural principles

### P1. The Producer is the verifier; verification is the whole paradigm

**Principle.** A CCNet learns causation not by predicting `Y` from `X`, but by checking
that the inferred causes can *regenerate* the observation: `P(X | Y, E)`. The Producer is
not an auxiliary decoder — it is the epistemic core. "I understand this digit" means "I
can redraw it from its label and style."

**Rationale.** Association (`P(Y|X)`) is unfalsifiable from inside the model. Regeneration
is falsifiable: a wrong cause produces a wrong observation, and the discrepancy is a
usable error signal. Causation is what survives the round trip.

**Failure mode.** If the Producer is weak or decoupled, the system collapses to an
ordinary classifier with a decorative decoder — no causal claim is being tested.

**Enforcement.** `compute_losses` derives all three losses from Producer outputs
(`x_generated`, `x_reconstructed`).

### P2. The two causes must be separated by *role*, not just by name

**Principle.** `Y` (explicit cause / label) and `E` (latent cause / style) are kept
independent: `P(Y, E) = P(Y)·P(E)`. They enter the Producer through *different ports* —
`Y` through the label projection, `E` through style modulation (e.g. FiLM).

**Rationale.** Counterfactual generation — the entire payoff — is only meaningful if `Y`
and `E` are independently manipulable. If `E` leaks label information, "draw this style as
a different digit" changes the style too.

**Failure mode.** Entangled causes: swapping `Y` distorts `E`; the counterfactual matrix
becomes incoherent.

**Enforcement.** Separate input pathways in the Producer; the KL term and P5 keep `E` from
absorbing label information.

### P3. The latent cause is variational, not a point estimate

**Principle.** The Explainer outputs a *distribution* `(mu, log_var)`, the orchestrator
samples `E = mu + eps·exp(0.5·log_var)`, and a KL term pulls the latent toward a smooth
prior.

**Rationale.** A smooth, regularized latent space generalizes — nearby `E` produce nearby
observations, which is what makes style interpolation and transfer well-behaved. A raw
deterministic code is free to memorize per-sample noise.

**Failure mode.** A deterministic encoder gives a pitted latent space; style transfer
produces artifacts; the model overfits stylistic noise.

**Enforcement.** `forward_pass` samples with the reparameterization trick (and clips
`log_var` for numerical safety); `compute_model_errors` adds the KL term.

---

## Part B — Gradient and training principles

### P4. Cooperation requires an unbroken differentiable path between modules

**Principle.** For one module to train another cooperatively, gradient must flow *through*
the intermediate module. The Reasoner is trained cooperatively only because
`reconstruction_loss → Producer → y_inferred → Reasoner` is differentiable end to end.

**Rationale.** "Cooperative training" is not a metaphor — it is literally backpropagation
across module boundaries. Any non-differentiable operation on the path (`argmax`,
hard quantization, index lookup, sampling without reparameterization) is a wall the
gradient cannot cross.

**Failure mode.** The original code did `argmax(y_inferred)` to feed an integer-index
`Embedding`. This severed the path: `reconstruction_loss` reached 18/18 Reasoner
gradients as `None`. The cooperative mechanism — the reason the architecture exists — did
not run. (`FIXES.md`, defect H3.)

**Enforcement.** The Producer consumes `Y` as a probability vector through a
differentiable projection (`Dense(use_bias=False)`), never an `argmax` index. Pinned by
`test_reconstruction_gradient_reaches_reasoner`.

### P5. Every module needs a stable supervised anchor; pure cooperative loss is brittle

**Principle.** Each module's error is an anchor term plus a cooperative term. The Reasoner
error is `w_inf·cross_entropy(y_truth, y_inferred) + w_rec·reconstruction_loss`: a direct,
well-conditioned supervised anchor plus the cooperative reconstruction signal.

**Rationale.** Early in training the Producer is useless, so the cooperative
reconstruction signal is noise. Without an anchor the modules chase each other's errors
and never gain traction. The anchor guarantees forward progress; the cooperative term
adds the causal coupling once the partners are competent. The anchor also degrades
gracefully — on hard data where the Reasoner is inaccurate, the anchor dominates so the
Producer is not dragged toward label-insensitivity.

**Failure mode.** Anchor-free cooperative training stalls or oscillates; modules reach a
pathological mutual equilibrium instead of solving the task.

**Enforcement.** `compute_model_errors`: the cross-entropy term in `reasoner_error`;
ground-truth `generation_loss` anchors the Explainer and Producer.

### P6. Disentanglement is enforced by deliberate gradient blocking, not by loss terms alone

**Principle.** `E` is fed to the Reasoner through `tf.stop_gradient`. The Reasoner *uses*
the latent but cannot *reshape* it.

**Rationale.** If the Reasoner could backprop into `E`, the cheapest way to lower its
classification loss is to make `E` encode the label — collapsing `E` into a label
shortcut and destroying the `Y`/`E` separation (P2). Blocking that gradient removes the
incentive. The Explainer is then shaped only by generation, inference, and KL — pressures
that reward *style*, not *identity*.

**Failure mode.** Remove the `stop_gradient` and `E` quietly becomes a second label
channel; counterfactuals stop working even though every loss still decreases.

**Enforcement.** `e_latent_no_grad = tf.stop_gradient(e_latent)` in `forward_pass`. This
is intentional and must not be "optimized away".

### P7. Per-module credit isolation comes from separate gradient tapes, not loss algebra

**Principle.** Each module is updated by *its own* error differentiated w.r.t. *only its
own* variables. Three independent `GradientTape`s wrap one shared forward pass; each
`tape.gradient(error_i, vars_i)` extracts one module's update.

**Rationale.** The three errors share loss terms (e.g. `reconstruction_loss` appears in
both Reasoner and Producer errors). Optimizing a single summed loss would let one module's
objective leak into another's parameters. Tape isolation gives clean causal credit: a loss
term influences a module only if it appears in *that module's* error.

**Failure mode.** A single combined loss + one optimizer entangles the modules; "the
Producer error excludes inference loss" stops being true in practice.

**Enforcement.** `train_step` opens three tapes; `producer_error` deliberately excludes
`inference_loss` so the Producer is never pulled toward label-insensitivity.

### P8. The error landscape stays additive and positive

**Principle.** Module errors are weighted sums of non-negative loss terms. No subtractive
"reward" terms.

**Rationale.** An earlier CCNet formulation used `cost - reward`, which allowed negative
total error, erratic gradients, and pathological equilibria where one module wins at
another's expense. A purely additive, positive-definite error landscape always has a
well-defined "lower energy" direction, so convergence is stable.

**Failure mode.** Subtractive terms reintroduce negative-error states and competitive
(rather than cooperative) dynamics.

**Enforcement.** All terms in `compute_model_errors` are positive losses with positive
weights.

### P9. Convergence-rate asymmetry must be actively throttled

**Principle.** The Reasoner (a classifier) converges far faster than the Explainer and
Producer (generative). A control strategy throttles Reasoner updates once it is
sufficiently accurate, so the slower modules can catch up.

**Rationale.** Cooperative training assumes partners of comparable competence. A Reasoner
that hits ~99% in a few epochs stops producing informative errors while the Producer is
still poor — the cooperation degenerates. Throttling keeps the modules co-evolving.

**Failure mode.** An unthrottled Reasoner saturates; `inference_loss → 0` carries no
signal; the Explainer/Producer plateau early.

**Enforcement.** `control.py` — `StaticThresholdStrategy` (default, threshold 0.98) or
`AdaptiveDivergenceStrategy`; consulted in `train_step`.

---

## Part C — Implementation principles

### P10. Anything read inside the compiled step is frozen; annealable quantities must be variables

**Principle.** A Python scalar read inside an `@tf.function` is captured as a graph
constant at first trace and never re-read. Any quantity that must change during training
(KL weight, schedules) must be a `tf.Variable`.

**Rationale.** `tf.function` retraces only on new input signatures. With a stable
signature the graph is fixed — including every Python float baked into it. Mutating the
source dict afterwards changes nothing.

**Failure mode.** `CCNetTrainer`'s KL annealing mutated a config dict every epoch; the
compiled `train_step` never saw it. KL annealing was a silent no-op for the whole run.
(`FIXES.md`, defect H5.)

**Enforcement.** `orchestrator.kl_weight` is a `tf.Variable`; the trainer `.assign()`s it.
Pinned by `test_kl_weight_is_live_variable`. The static weights stay floats *because* they
are never annealed — freezing them is correct and cheaper.

### P11. The framework is model-agnostic; the contract is the interface

**Principle.** CCNets specifies *behavioral contracts*, not architectures. Any Keras model
works as a module if it honors its call signature: `explainer(x) → (mu, log_var)`,
`reasoner(x, e) → y_probs`, `producer(y, e) → x_hat`.

**Rationale.** The causal mechanism lives in the orchestration (P1–P10), not in the
convnets. Decoupling the contract from the architecture is what lets the same paradigm
serve images, tabular data, audio, or sequences.

**Failure mode.** Baking modality assumptions into the orchestrator would make every new
task a rewrite.

**Enforcement.** The `CCNetModule` protocol in `base.py`; `wrap_keras_model` adapts any
Keras model to it.

### P12. Documentation is a hypothesis; the code is the system

**Principle.** When a conceptual document and the running code disagree, the code is
ground truth until the document is reconciled. The repair of this framework was driven by
what actually trains, not by `FOUNDATION.md` prose.

**Rationale.** `FOUNDATION.md` specified a reconstruction+inference Reasoner error that the
code never implemented; `README.md` documented config fields (`loss_type`, `kl_weight`)
that did not exist. Trusting the doc as the spec would have produced a "fix" for a system
that was not running. (`FIXES.md`, defect H9.)

**Failure mode.** Treating an aspirational document as an authoritative spec.

**Enforcement.** Cultural, not mechanical — but the test suite (`test_orchestrator.py`)
now encodes the real contract, so the code's behavior is itself documented and pinned.

---

## Design checklist

Before training a new CCNet, confirm every principle is satisfied:

- [ ] Producer regenerates `X` from `(Y, E)` and all losses derive from it (P1)
- [ ] `Y` and `E` enter the Producer through separate ports (P2)
- [ ] Explainer is variational — returns `(mu, log_var)` (P3)
- [ ] Producer consumes `Y` differentiably — no `argmax`/`Embedding` (P4)
- [ ] Each module error has a supervised anchor term (P5)
- [ ] `E` is `stop_gradient`-ed into the Reasoner (P6)
- [ ] Per-module tapes; `producer_error` excludes `inference_loss` (P7)
- [ ] All error terms additive and non-negative (P8)
- [ ] A control strategy throttles the Reasoner (P9)
- [ ] Annealable quantities are `tf.Variable`s, not floats (P10)
- [ ] Modules honor the `CCNetModule` call contract (P11)
- [ ] Behavior is pinned by tests, not just prose (P12)
