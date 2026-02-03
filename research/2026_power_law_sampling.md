# Power Sampling: Future-Aware Decision Making in Sequential Generation

## Abstract

Power sampling is an inference-time technique that reweights immediate choices by their estimated long-term consequences. Unlike greedy methods that optimize locally, power sampling uses Monte Carlo rollouts to approximate the quality of future trajectories, then adjusts current decision probabilities accordingly. This document provides a self-contained treatment of power sampling, its theoretical foundations, relationship to other sequential decision methods, and applications across diverse domains.

---

## 1. Introduction

Sequential generation problems share a common structure: at each step, an agent selects from available choices, and these choices compound to form a complete output. Whether generating text, designing molecules, planning robot trajectories, or denoising images, the fundamental challenge remains the same—**locally attractive choices often lead to globally poor outcomes**.

Consider a chess player evaluating moves. A move that captures a pawn might look optimal when considering only the immediate board state, yet could expose the queen three moves later. Expert players evaluate moves not by immediate gain but by the quality of positions they lead to. Power sampling formalizes this intuition for probabilistic generative systems.

### 1.1 The Local-Global Mismatch

Most generative models are trained to predict the next element given context:

$$P(x_t | x_{<t})$$

This local objective does not guarantee globally coherent outputs. During inference, methods like greedy decoding or low-temperature sampling amplify this mismatch by selecting tokens that maximize immediate probability, potentially committing to trajectories that become incoherent or suboptimal downstream.

### 1.2 Document Scope

This document covers:

1. **Foundational concepts**: Markov processes, trajectory optimization, Monte Carlo methods
2. **Related algorithms**: Viterbi, beam search, Monte Carlo Tree Search
3. **Power sampling theory**: Mathematical formulation and algorithmic implementation
4. **Applications**: Language models, diffusion, reinforcement learning, molecular design, and beyond
5. **Practical considerations**: Hyperparameters, compute tradeoffs, implementation strategies

---

## 2. Foundational Concepts

### 2.1 Markov Decision Processes

A **Markov Decision Process (MDP)** provides the formal framework for sequential decision problems. An MDP is defined by the tuple $(S, A, T, R, \gamma)$:

| Symbol | Definition |
|--------|------------|
| $S$ | State space |
| $A$ | Action space |
| $T(s' \mid s, a)$ | Transition probability |
| $R(s, a)$ | Reward function |
| $\gamma$ | Discount factor |

The **Markov property** states that the future depends only on the current state, not the history:

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

### 2.2 Trajectory and Return

A **trajectory** $\tau$ is a sequence of states and actions:

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$$

The **return** (cumulative reward) of a trajectory is:

$$G(\tau) = \sum_{t=0}^{T} \gamma^t R(s_t, a_t)$$

### 2.3 Policy and Value Functions

A **policy** $\pi(a|s)$ defines the probability of taking action $a$ in state $s$.

The **state-value function** estimates expected return from a state:

$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi}\left[G(\tau) | s_0 = s\right]$$

The **action-value function** (Q-function) estimates expected return from a state-action pair:

$$Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}\left[G(\tau) | s_0 = s, a_0 = a\right]$$

### 2.4 The Bellman Equation

Value functions satisfy a recursive relationship:

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} T(s'|s,a) V^\pi(s') \right]$$

This recursion—current value depends on immediate reward plus discounted future value—is central to understanding why lookahead improves decisions.

### 2.5 Monte Carlo Estimation

When transition dynamics are unknown or state spaces are intractable, **Monte Carlo methods** estimate expectations through sampling:

$$\mathbb{E}[f(X)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i), \quad x_i \sim P(X)$$

Monte Carlo estimation is:
- **Unbiased**: Expected value of estimate equals true expectation
- **Consistent**: Converges to true value as $N \to \infty$
- **Variance**: Decreases as $O(1/N)$

---

## 3. Related Algorithms

### 3.1 Greedy Decoding

**Definition**: Select the highest-probability choice at each step.

$$a_t = \arg\max_a P(a | s_t)$$

**Properties**:
- Computationally cheap: $O(|A|)$ per step
- Deterministic output
- No lookahead—purely local optimization

**Failure mode**: Commits to locally optimal choices that preclude globally optimal trajectories.

### 3.2 Beam Search

**Definition**: Maintain top-$k$ partial trajectories, extend each, keep best $k$.

```
beam = [initial_state]
for each step:
    candidates = []
    for state in beam:
        for action in actions:
            candidates.append((state, action, score))
    beam = top_k(candidates, k)
return best(beam)
```

**Properties**:
- Explores multiple paths simultaneously
- Hard pruning: discarded paths cannot be recovered
- Complexity: $O(k \cdot |A|)$ per step

**Limitation**: Beam search optimizes cumulative log-probability, which may not correlate with task quality. Pruning is irreversible—good trajectories with weak prefixes are lost.

### 3.3 Viterbi Algorithm

**Definition**: Dynamic programming algorithm finding the most probable path through a hidden Markov model.

For HMM with states $S$, observations $O$, transition matrix $A$, emission matrix $B$:

$$\delta_t(j) = \max_{i} \left[ \delta_{t-1}(i) \cdot A_{ij} \right] \cdot B_j(o_t)$$

where $\delta_t(j)$ is the probability of the most likely path ending in state $j$ at time $t$.

**Properties**:
- Exact solution for tractable state spaces
- Complexity: $O(T \cdot |S|^2)$
- Requires enumerable state space and known transition probabilities

**Why Viterbi fails for neural generative models**:
1. State space (vocabulary × context) is astronomically large
2. Transition probabilities are context-dependent (no fixed transition matrix)
3. Finds single best path—no diversity

### 3.4 Monte Carlo Tree Search (MCTS)

**Definition**: Tree search algorithm using random rollouts to estimate action values.

MCTS operates in four phases:

```
1. SELECTION: Traverse tree using UCB1 until reaching leaf
   UCB1(node) = Q(node)/N(node) + c * sqrt(ln(N_parent) / N(node))

2. EXPANSION: Add child node for unexplored action

3. SIMULATION: Random rollout from new node to terminal state

4. BACKPROPAGATION: Update value estimates along traversed path
```

**Properties**:
- Balances exploration (UCB1 term) and exploitation (Q-value)
- Builds explicit search tree
- Anytime algorithm: improves with more iterations

**Relationship to power sampling**:

| Aspect | MCTS | Power Sampling |
|--------|------|----------------|
| Structure | Explicit tree | Implicit (no tree stored) |
| Selection | Hard (UCB1 chooses one path) | Soft (reweighted distribution) |
| Output | Best action from root | Sample from adjusted distribution |
| Memory | $O(\text{tree size})$ | $O(1)$ beyond rollouts |

Power sampling can be viewed as "soft MCTS" that maintains a probability distribution rather than building a discrete tree.

---

## 4. Power Sampling Theory

### 4.1 The Power Distribution

Given a base distribution $P(x)$ and temperature parameter $\beta > 0$, the **power distribution** is:

$$P_\beta(x) = \frac{P(x)^\beta}{\sum_{x'} P(x')^\beta}$$

Effect of $\beta$:
- $\beta = 1$: Original distribution
- $\beta > 1$: Sharper (concentrates on high-probability outcomes)
- $\beta < 1$: Flatter (more uniform)
- $\beta \to \infty$: Converges to argmax

### 4.2 Trajectory-Level Power Sampling

For a trajectory $\tau = (x_1, \ldots, x_T)$ with probability:

$$P(\tau) = \prod_{t=1}^{T} P(x_t | x_{<t})$$

The trajectory-level power distribution is:

$$P_\beta(\tau) \propto P(\tau)^\beta$$

**Problem**: Sampling from $P_\beta(\tau)$ requires evaluating all possible trajectories—computationally intractable.

### 4.3 The Decomposition Insight

The key insight enabling practical power sampling is that trajectory-level reweighting can be decomposed into local corrections.

Define the **future value** of choosing token $x_t$ as:

$$V(x_t) = \mathbb{E}_{\tau_{>t} \sim P}\left[ P(\tau)^{\beta-1} | x_t \right]$$

Then the power distribution can be expressed as:

$$P_\beta(x_t | x_{<t}) \propto P(x_t | x_{<t}) \cdot V(x_t)$$

This factorization means: **reweight the base probability by estimated future trajectory quality**.

### 4.4 Monte Carlo Estimation of Future Value

Since $V(x_t)$ involves an expectation over future trajectories, we estimate it via Monte Carlo:

$$\hat{V}(x_t) = \frac{1}{N} \sum_{i=1}^{N} \text{score}(\tau^{(i)}), \quad \tau^{(i)} \sim P(\cdot | x_{\leq t})$$

where $\tau^{(i)}$ is a rollout starting from context $x_{\leq t}$.

### 4.5 Bias Correction with Jackknife Estimator

Ratio estimators (which arise when normalizing scores) exhibit bias. The **jackknife estimator** reduces this bias:

Given samples $\{y_1, \ldots, y_n\}$:

1. Compute full estimate: $\hat{\theta} = g(y_1, \ldots, y_n)$
2. Compute leave-one-out estimates: $\hat{\theta}_{-i} = g(y_1, \ldots, y_{i-1}, y_{i+1}, \ldots, y_n)$
3. Jackknife estimate: $\hat{\theta}_{\text{jack}} = n\hat{\theta} - (n-1)\bar{\theta}_{-i}$

where $\bar{\theta}_{-i} = \frac{1}{n}\sum_i \hat{\theta}_{-i}$.

---

## 5. Algorithm Specification

### 5.1 Single-Step Power Sampling

```
FUNCTION power_sample_step(model, context, config):
    INPUT:
        model: generative model with logit access
        context: current sequence [x_1, ..., x_t]
        config: {top_k, num_rollouts, rollout_length, temperature, scorer}
    
    OUTPUT:
        selected_token: next token sampled from power distribution
    
    PROCEDURE:
        # Step 1: Identify candidate tokens
        logits = model.forward(context)
        candidates = top_k_indices(logits, config.top_k)
        base_probs = softmax(logits[candidates] / config.temperature)
        
        # Step 2: Estimate future value for each candidate
        future_values = []
        FOR token IN candidates:
            rollout_scores = []
            FOR i IN 1..config.num_rollouts:
                trajectory = model.generate(
                    context + [token], 
                    max_length=config.rollout_length,
                    temperature=config.temperature
                )
                score = config.scorer(trajectory)
                rollout_scores.append(score)
            
            # Apply jackknife bias correction
            future_values.append(jackknife_mean(rollout_scores))
        
        # Step 3: Compute adjusted distribution
        scaling_factors = normalize_positive(future_values)
        adjusted_probs = base_probs * scaling_factors
        adjusted_probs = adjusted_probs / sum(adjusted_probs)
        
        # Step 4: Sample
        selected_idx = categorical_sample(adjusted_probs)
        RETURN candidates[selected_idx]
```

### 5.2 Full Sequence Generation

```
FUNCTION power_sample_sequence(model, prompt, max_length, config):
    context = tokenize(prompt)
    
    FOR step IN 1..max_length:
        next_token = power_sample_step(model, context, config)
        context.append(next_token)
        
        IF next_token == EOS_TOKEN:
            BREAK
    
    RETURN detokenize(context)
```

### 5.3 Trajectory Scoring Functions

The scoring function is task-dependent. Common approaches:

**Perplexity-based (general)**:
$$\text{score}(\tau) = \exp\left( \frac{1}{|\tau|} \sum_{t} \log P(x_t | x_{<t}) \right)$$

**Self-consistency (verifiable tasks)**:
$$\text{score}(\tau) = \frac{\text{count}(\text{answer}(\tau))}{\text{total rollouts}}$$

**Task-specific reward**:
$$\text{score}(\tau) = R(\tau)$$

where $R$ is a domain-specific reward function (validity, quality, etc.).

---

## 6. Applications

### 6.1 Language Model Decoding

**Setting**: Autoregressive text generation (GPT-style models).

**Decision point**: Token selection at each position.

**Rollouts**: Continue generation for $L$ additional tokens.

**Scoring strategies**:
- Perplexity (coherence proxy)
- Answer consistency (for QA/math)
- Format compliance (structured output)

**Empirical findings** (from arXiv:2601.21590):
- Matches or exceeds GRPO post-training on reasoning benchmarks
- 10x+ faster than MCMC-based power sampling
- Largest gains on multi-step reasoning tasks

**When to use**:
- Complex reasoning (math, logic, planning)
- Code generation requiring structural coherence
- Tasks where greedy decoding produces incomplete/rushed answers

**When to avoid**:
- Simple factual recall
- Real-time latency requirements
- Tasks where base model already excels

### 6.2 Diffusion Models

**Setting**: Iterative denoising for image/video generation.

**Decision point**: Noise prediction $\epsilon_\theta(x_t, t)$ at each timestep.

**Adaptation**: Rather than discrete token selection, perturb the denoising direction:

```
FOR each denoising step t:
    # Generate candidate directions
    base_epsilon = model.predict_noise(x_t, t)
    candidates = [base_epsilon + perturbation_i for i in 1..k]
    
    # Rollout each to completion
    FOR candidate IN candidates:
        x_t_minus_1 = denoise_step(x_t, candidate)
        final_image = complete_denoising(x_t_minus_1)
        score = quality_metric(final_image)
    
    # Select weighted by scores
    selected_epsilon = weighted_selection(candidates, scores)
    x_t_minus_1 = denoise_step(x_t, selected_epsilon)
```

**Scoring functions**:
- CLIP alignment with prompt
- Aesthetic predictors
- Structural coherence metrics
- Human preference models

**Challenge**: High-dimensional continuous space makes enumeration impractical. Solutions include:
- Discrete perturbation directions
- Learned candidate proposal networks
- Gradient-based approximations

### 6.3 Reinforcement Learning / Planning

**Setting**: Action selection in sequential decision problems.

**Decision point**: Policy action at each timestep.

**Relationship to existing methods**:

| Method | Lookahead | Selection |
|--------|-----------|-----------|
| Policy gradient | None (single-step) | Sample from $\pi(a|s)$ |
| MCTS | Tree search | UCB1 + visit counts |
| Power sampling | Monte Carlo rollouts | Soft reweighting |

**Implementation**:

```
FUNCTION power_sample_action(policy, environment, state, config):
    action_probs = policy(state)
    candidate_actions = top_k(action_probs, config.k)
    
    future_values = []
    FOR action IN candidate_actions:
        returns = []
        FOR i IN 1..config.num_rollouts:
            trajectory = simulate(environment, state, action, policy)
            returns.append(cumulative_reward(trajectory))
        future_values.append(mean(returns))
    
    adjusted_probs = action_probs[candidate_actions] * softmax(future_values)
    adjusted_probs = adjusted_probs / sum(adjusted_probs)
    
    RETURN sample(candidate_actions, adjusted_probs)
```

**Advantages over MCTS**:
- No explicit tree storage
- Softer exploration (probability vs hard selection)
- Natural integration with neural policies

**Use cases**:
- Game playing with neural policies
- Robotics planning
- Resource allocation

### 6.4 Molecular Design

**Setting**: Autoregressive molecule generation (SMILES, graph-based).

**Decision point**: Atom/bond addition at each step.

**Critical importance**: Local validity does not guarantee global validity. A nitrogen atom might validly bond at position 5, but completing the molecule may violate valence rules or create unstable structures.

**Scoring functions**:
- Validity (syntactic correctness, chemical feasibility)
- Drug-likeness (QED score)
- Synthesizability (SA score)
- Binding affinity (docking scores)
- Novelty (distance from training set)

**Example**:

```
FUNCTION generate_molecule(model, config):
    fragment = START_TOKEN
    
    WHILE not complete(fragment):
        candidates = model.get_candidates(fragment)
        
        future_scores = []
        FOR candidate IN candidates:
            completed_molecules = []
            FOR i IN 1..config.num_rollouts:
                molecule = model.complete(fragment + candidate)
                IF valid(molecule):
                    score = drug_likeness(molecule) * binding_affinity(molecule)
                ELSE:
                    score = 0
                completed_molecules.append(score)
            future_scores.append(mean(completed_molecules))
        
        # Strongly penalize choices leading to invalid molecules
        adjusted_probs = compute_adjusted_probs(candidates, future_scores)
        selected = sample(candidates, adjusted_probs)
        fragment = fragment + selected
    
    RETURN fragment
```

**Impact**: Avoids wasted computation on trajectories destined to fail validity checks.

### 6.5 Neural Architecture Search

**Setting**: Sequential construction of neural network architectures.

**Decision point**: Layer type, size, connectivity at each position.

**Scoring**:
- Proxy accuracy (few epochs of training)
- Latency estimates
- Parameter count
- FLOPs

**Challenge**: Rollout evaluation is expensive (requires training). Mitigations:
- Weight sharing across candidates
- Accuracy predictors trained on architecture-performance pairs
- Early stopping in proxy evaluation

### 6.6 Protein Design

**Setting**: Sequence or structure generation for proteins.

**Decision point**: Amino acid selection (sequence) or torsion angle prediction (structure).

**Scoring functions**:
- Predicted structure confidence (pLDDT from AlphaFold)
- Energy functions
- Stability predictors
- Function-specific metrics

**High stakes**: Wet lab validation is expensive. Computational filtering via power sampling reduces wasted synthesis attempts.

### 6.7 Symbolic Reasoning / Theorem Proving

**Setting**: Tactic selection in proof assistants (Lean, Coq, Isabelle).

**Decision point**: Which tactic to apply at each proof state.

**Scoring functions**:
- Proof completion (binary)
- Proof length (shorter = better)
- Subgoal complexity reduction

**Synergy with language models**: LLMs can generate tactic candidates; power sampling selects among them based on provability of resulting subgoals.

### 6.8 Time Series Forecasting

**Setting**: Autoregressive prediction of future values.

**Decision point**: Each timestep's prediction (especially for discretized or categorical time series).

**Scoring functions**:
- Physical plausibility
- Consistency with constraints
- Forecast confidence calibration

**Application**: Trajectory prediction for autonomous vehicles—local predictions must compose into physically feasible paths.

---

## 7. Theoretical Properties

### 7.1 Convergence

As the number of rollouts $N \to \infty$, the estimated future values converge to true expectations:

$$\hat{V}(x_t) \xrightarrow{N \to \infty} V(x_t)$$

The adjusted distribution converges to the true local marginalization of the trajectory-level power distribution.

### 7.2 Variance Reduction

Variance of the Monte Carlo estimate decreases as:

$$\text{Var}(\hat{V}) = O(1/N)$$

Jackknife correction reduces bias without increasing variance order.

### 7.3 Computational Complexity

Per-step complexity:

$$O(k \cdot N \cdot L \cdot C_{\text{model}})$$

where:
- $k$ = number of candidate tokens
- $N$ = number of rollouts per candidate
- $L$ = rollout length
- $C_{\text{model}}$ = cost of single model forward pass

**Total sequence generation**:

$$O(T \cdot k \cdot N \cdot L \cdot C_{\text{model}})$$

### 7.4 Approximation Quality

The quality of power sampling depends on:

1. **Rollout fidelity**: Rollout policy should match deployment policy
2. **Scorer alignment**: Scoring function should correlate with true objective
3. **Candidate coverage**: Top-k must include good options
4. **Rollout length**: Must be sufficient to reveal trajectory quality differences

---

## 8. Implementation Considerations

### 8.1 Hyperparameter Selection

| Parameter | Range | Guidance |
|-----------|-------|----------|
| `top_k` | 5-20 | Higher for diverse/creative tasks |
| `num_rollouts` | 4-16 | More = better estimates, higher cost |
| `rollout_length` | 16-64 | Task-dependent; longer for complex reasoning |
| `temperature` | 0.6-1.0 | Base sampling temperature |

**Configuration presets**:

```
Minimal (latency-sensitive):
    top_k=5, num_rollouts=4, rollout_length=16
    
Balanced (general use):
    top_k=10, num_rollouts=8, rollout_length=32
    
Quality (offline/batch):
    top_k=15, num_rollouts=16, rollout_length=64
```

### 8.2 Parallelization

Rollouts are embarrassingly parallel:
- Batch all rollouts for all candidates
- Single batched forward pass per rollout step
- GPU utilization scales with batch size

```
# Efficient batched implementation
batch_size = top_k * num_rollouts
contexts = repeat_interleave(context, batch_size)
candidate_tokens = tile(candidates, num_rollouts)
contexts = append(contexts, candidate_tokens)

FOR step IN 1..rollout_length:
    logits = model.forward_batch(contexts)  # Single batched call
    next_tokens = sample_batch(logits)
    contexts = append(contexts, next_tokens)

scores = scorer_batch(contexts)
scores = reshape(scores, [top_k, num_rollouts])
future_values = mean(scores, axis=1)
```

### 8.3 KV-Cache Optimization

For transformer models, cache key-value pairs:

1. Compute KV-cache for shared prefix (context) once
2. Fork cache for each candidate token
3. Continue rollouts from forked caches

Memory cost: $O(k \cdot N \cdot L \cdot d_{\text{model}})$

### 8.4 Early Termination

Terminate rollouts early when:
- Confidence in ranking is high (statistically significant differences)
- Rollout reaches terminal state (EOS, invalid state)
- Quality estimate stabilizes

```
FUNCTION adaptive_rollouts(candidate, min_rollouts, max_rollouts, threshold):
    scores = []
    FOR i IN 1..max_rollouts:
        scores.append(rollout_and_score(candidate))
        
        IF i >= min_rollouts:
            confidence_interval = compute_ci(scores)
            IF width(confidence_interval) < threshold:
                BREAK
    
    RETURN jackknife_mean(scores)
```

### 8.5 Caching Across Steps

When generating sequences, subsequent steps share prefixes with prior rollouts. Opportunities:
- Cache completed rollout trajectories
- Reuse rollouts that match current context prefix
- Amortize rollout cost across generation

---

## 9. Comparison Summary

| Method | Lookahead | Selection | Memory | Parallelizable |
|--------|-----------|-----------|--------|----------------|
| Greedy | None | Deterministic (argmax) | O(1) | N/A |
| Sampling | None | Stochastic | O(1) | N/A |
| Beam Search | Implicit (score accumulation) | Hard pruning | O(k) | Limited |
| Viterbi | Exact DP | Deterministic | O(S) | Limited |
| MCTS | Explicit tree | UCB1 | O(tree) | Moderate |
| **Power Sampling** | Monte Carlo rollouts | Soft reweighting | O(k·N·L) | High |

---

## 10. Limitations and Future Directions

### 10.1 Current Limitations

1. **Computational overhead**: Rollouts multiply inference cost by $O(k \cdot N \cdot L)$
2. **Scorer dependence**: Quality limited by scorer's correlation with true objective
3. **Base model ceiling**: Cannot create capabilities absent from base model
4. **Latency**: Unsuitable for real-time applications without approximations

### 10.2 Research Directions

1. **Learned value functions**: Train networks to predict future quality, eliminating rollout cost
2. **Amortized inference**: Distill power sampling behavior into base model
3. **Adaptive computation**: Dynamically allocate rollouts based on decision difficulty
4. **Hierarchical rollouts**: Multi-resolution lookahead (short rollouts for filtering, long for final selection)
5. **Off-policy rollouts**: Use different (cheaper) models for rollouts

### 10.3 Theoretical Questions

1. When does power sampling provably improve over base sampling?
2. Optimal allocation of compute between candidates vs rollout depth
3. Relationship to optimal control and planning under uncertainty

---

## 11. Conclusion

Power sampling provides a principled framework for incorporating future trajectory quality into current decisions. By decomposing global trajectory optimization into local corrections estimated via Monte Carlo rollouts, it achieves the benefits of lookahead planning without the intractability of exact methods.

The technique is broadly applicable: any sequential generation process where local choices impact global quality is a candidate for power sampling. As inference-time compute becomes increasingly important for model capability, methods like power sampling represent a key tool for extracting maximum performance from fixed model weights.

The core insight—**evaluate choices by where they lead, not just how they look locally**—is fundamental to intelligent decision-making across domains.

---

## References

1. arXiv:2601.21590 - Scalable Power Sampling for LLM Reasoning
2. arXiv:2510.14901 - Power Distribution Analysis
3. Coulom (2006) - Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
4. Kocsis & Szepesvári (2006) - Bandit based Monte-Carlo Planning
5. Holtzman et al. (2020) - The Curious Case of Neural Text Degeneration

---

## Appendix A: Mathematical Derivations

### A.1 Power Distribution Normalization

For distribution $P$ and power $\beta$:

$$Z_\beta = \sum_x P(x)^\beta$$

$$P_\beta(x) = \frac{P(x)^\beta}{Z_\beta}$$

### A.2 Trajectory Factorization

For trajectory $\tau = (x_1, \ldots, x_T)$:

$$P(\tau) = \prod_{t=1}^T P(x_t | x_{<t})$$

$$P(\tau)^\beta = \prod_{t=1}^T P(x_t | x_{<t})^\beta$$

The marginal at position $t$:

$$P_\beta(x_t | x_{<t}) = \frac{\sum_{\tau: \tau_t = x_t} P(\tau)^\beta}{\sum_{\tau} P(\tau)^\beta}$$

### A.3 Local-Global Decomposition

Define:
$$V_t(x) = \mathbb{E}_{\tau_{>t}}\left[ P(\tau_{>t} | x_t = x)^{\beta - 1} \right]$$

Then:
$$P_\beta(x_t | x_{<t}) = \frac{P(x_t | x_{<t}) \cdot V_t(x_t)}{\sum_{x'} P(x' | x_{<t}) \cdot V_t(x')}$$

This shows the power distribution equals base probability times future value, normalized.

---

## Appendix B: Pseudocode Reference

### B.1 Complete Generation Loop

```
FUNCTION generate_with_power_sampling(
    model,
    prompt,
    max_tokens,
    top_k,
    num_rollouts,
    rollout_length,
    temperature,
    scorer
):
    tokens = tokenize(prompt)
    
    FOR i IN 1..max_tokens:
        # Get base distribution
        logits = model.get_logits(tokens)
        candidates = argsort(logits, descending=True)[:top_k]
        base_probs = softmax(logits[candidates] / temperature)
        
        # Estimate future values
        future_values = zeros(top_k)
        all_rollouts = []  # For batched scoring
        
        FOR j, token IN enumerate(candidates):
            token_rollouts = []
            FOR r IN 1..num_rollouts:
                context = tokens + [token]
                rollout = model.generate(
                    context,
                    max_new_tokens=rollout_length,
                    temperature=temperature,
                    do_sample=True
                )
                token_rollouts.append(rollout)
            all_rollouts.append(token_rollouts)
        
        # Score all rollouts
        FOR j IN 0..top_k:
            scores = [scorer(r) for r in all_rollouts[j]]
            future_values[j] = jackknife_mean(scores)
        
        # Adjust and sample
        scaling = future_values / sum(future_values)  # Normalize
        adjusted = base_probs * scaling
        adjusted = adjusted / sum(adjusted)
        
        selected_idx = multinomial(adjusted, 1)
        selected_token = candidates[selected_idx]
        
        tokens.append(selected_token)
        
        IF selected_token == EOS:
            BREAK
    
    RETURN detokenize(tokens)
```

### B.2 Jackknife Estimator

```
FUNCTION jackknife_mean(values):
    n = len(values)
    full_mean = sum(values) / n
    
    loo_means = []
    FOR i IN 0..n:
        loo_sum = sum(values) - values[i]
        loo_means.append(loo_sum / (n - 1))
    
    bias_correction = (n - 1) * (mean(loo_means) - full_mean)
    
    RETURN full_mean - bias_correction
```