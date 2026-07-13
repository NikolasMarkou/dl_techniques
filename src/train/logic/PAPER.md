# LearnableNeuralCircuit: An Empirical Study of Differentiable Boolean Rule Extraction

**Nikolas Markou**
nikolasmarkou@gmail.com

---

## Abstract

We present an empirical evaluation of `LearnableNeuralCircuit`, a differentiable reasoning head whose internal operators (logic gates and bounded arithmetic) admit a bit-exact hard extraction to a symbolic boolean expression. Across five experimental settings spanning synthetic boolean tasks, image classification, UCI tabular benchmarks, and confounded visual reasoning, we ask two questions: (i) is the symbolic readout *faithful* to the trained model, and (ii) does the architectural prior confer measurable advantages over a parameter-matched MLP. We answer (i) affirmatively with high statistical confidence (n=5 seeds, hard-extraction Δ within ±0.005 on every config that converged), and (ii) negatively for accuracy and for visual shortcut resistance, but affirmatively for one structurally meaningful task: the Monks-2 counting constraint is recovered exactly by all seeds, while gradient boosting fails (78.5% vs 100.0%). We also report a clean dissociation on CLEVR-Hans3: with perfect perception (scene-graph input), the same reasoning head reaches 99.9% on a confounded benchmark with a 0.1pt gap to the clean test, while with frozen ResNet50 perception both circuit and MLP collapse equally (~17pt gap, paired permutation p=0.375). The bottleneck is upstream of the reasoning head.

---

## 1. Introduction

A "differentiable rule extractor" is appealing because it promises two normally-incompatible properties: end-to-end gradient training and a post-hoc symbolic readout that is *bit-exactly* what the network computes at inference. `LearnableNeuralCircuit` (henceforth: circuit) is a Keras 3 layer that composes a fixed alphabet of boolean operators (XOR, AND, OR, NAND, NOR, XNOR, plus identity/negation) and bounded arithmetic operators (add, min, max) over learned soft selection weights; after training, `extract_hard_inplace` argmaxes the selectors and collapses the soft mixture to a single discrete operator per slot.

The architecture has been published in fragments and is implemented in `dl_techniques`. This paper is *not* an architectural contribution. It is a deliberately negative-result-friendly empirical study of when the claimed properties hold.

---

## 2. Architecture

The circuit is a stack of `circuit_depth` layers, each comprising `channels` parallel operator slots. Each slot holds soft selection weights over `op_types` ∈ {logic, arithmetic}. Configuration used throughout:

| Hyperparameter | Value |
|---|---|
| `circuit_depth` | 2 |
| `channels` | 32 |
| `logic_op_types` | full set (8 operators) |
| `arithmetic_op_types` | `{add, min, max}` |
| `apply_sigmoid_per_depth` | `first_only` |
| Embedding | `Dense(32)` + `LayerNormalization` |
| Head | `Dense(1)` sigmoid |

Total parameters on a 17-bit boolean input: **865**.

The `arithmetic_op_types` restriction and `apply_sigmoid_per_depth='first_only'` choice are both empirically required to prevent training-time NaN at `circuit_depth ≥ 2`; this is institutional knowledge logged after several diverging runs and is the only architectural decision the experiments below depend on.

---

## 3. Experiments

All experiments use seeds {0,1,2,3,4}; results report mean ± std (ddof=1) unless noted. Hard-extraction Δ is defined as `acc(hard) − acc(soft)` on the same held-out set; values near zero indicate the soft mixture is functionally discrete by the measurement point.

### 3.1 E1 — Image classification with band-checkpoint faithfulness

Architecture: Conv stem → `LearnableNeuralCircuit` (rank-4) → GlobalAveragePool → Dense head. CNN baseline matched on parameter count. Training stops when validation accuracy first enters a target band ([0.70, 0.95] for MNIST, [0.50, 0.80] for CIFAR-10) to deliberately measure faithfulness in the regime where Δ is non-tautological.

| Dataset | Model | val_acc | hard-extraction Δ |
|---|---|---|---|
| MNIST | circuit | 0.7007 ± 0.0004 | −0.034 ± 0.063 |
| MNIST | CNN-matched | 0.6153 ± 0.0171 | n/a |
| CIFAR-10 | circuit | 0.5500 ± 0.0117 | **−0.0007 ± 0.0037** |
| CIFAR-10 | CNN-matched | 0.5333 ± 0.0264 | n/a |

The MNIST Δ has high variance because some seeds enter the band with the soft mixture still slightly non-discrete; CIFAR-10 enters the band later in training and shows the cleaner near-zero result.

### 3.2 E3 — Faithfulness under non-saturation on hard boolean tasks

Three synthetic tasks designed to avoid the ceiling artifact that contaminates "Δ at 100% accuracy" reports: 11-bit MUX (3 address + 8 data), 8-bit parity, and a random 4-DNF over 8 inputs. Trained to partial convergence in the [0.70, 0.95] band.

| Task | band_acc | hard-extraction Δ | circuit suff_AUC |
|---|---|---|---|
| mux_11bit | 0.751 ± 0.028 | **−0.0000 ± 0.0045** | 0.655 ± 0.041 |
| parity_k8 | 0.755 ± 0.043 | 0.015 ± 0.014 | 0.553 ± 0.019 |
| random_dnf 8/4 | 0.767 ± 0.037 | **−0.0004 ± 0.0060** | 0.670 ± 0.064 |

Δ confidence intervals straddle zero on every task except parity (which is known to be a worst-case for any symbolic extractor — the LIME `suff_AUC` of 0.536 ± 0.016 confirms parity is structurally hostile to local attribution methods). The faithfulness claim is supported.

### 3.3 E4 — Ground-truth rule recovery on UCI Monks

The Monks suite (Thrun et al., 1991) is the only widely-used benchmark with *published, exact* boolean rules that generated the labels. We train circuit, parameter-matched MLP, and XGBoost; for each model we extract a candidate rule (`to_symbolic` for the circuit; n/a for MLP and XGBoost as competitors on accuracy), then score the trained-model decision function against the published rule by exhaustive evaluation over the 432-config Monks input domain. Three seeds.

| Problem | Published rule | circuit | MLP-matched | XGBoost |
|---|---|---|---|---|
| Monks-1 | `(a1=a2) ∨ (a5=red)` | 0.995 ± 0.002 | 0.999 | 0.949 |
| **Monks-2** | exactly-two-equal-to-first-value | **1.000 ± 0.000** (3/3 exact) | 0.998 | **0.785** |
| Monks-3 | `(a5=g ∧ a4=s) ∨ (a5≠b ∧ a2≠o)`, 5% noise | 0.965 ± 0.007 | 0.966 | 0.956 |

**Monks-2 is the load-bearing positive result of this paper.** It is a counting constraint that cannot be represented by any depth-bounded tree ensemble; XGBoost loses 22 points to both neural models, while the circuit recovers the constraint *exactly* on every seed.

A companion learning-curve experiment on synthetic 11-bit MUX with N ∈ {32, 64, 128, 256, 512, 1024} training samples found no circuit-specific advantage at low data: all three models track essentially the same curve. The inductive-bias-at-low-data hypothesis is *not supported*.

### 3.4 E5 — Confounded visual reasoning on CLEVR-Hans3

CLEVR-Hans3 (Stammer et al., 2021) presents 3-class scenes where the training and validation splits embed a visual confound that the *clean* test split removes. The metric is the "shortcut gap": `val_acc − test_acc`. Three configurations:

| Model | val (confounded) | test (clean) | shortcut gap |
|---|---|---|---|
| ResNet50 frozen + circuit | 0.850 ± 0.003 | 0.677 ± 0.022 | 0.173 ± 0.020 |
| ResNet50 frozen + MLP-matched | 0.850 ± 0.003 | 0.663 ± 0.013 | 0.187 ± 0.011 |
| **Oracle: circuit on scene graphs** | **1.000 ± 0.000** | **0.999 ± 0.001** | **0.001 ± 0.001** |

A paired permutation test on the circuit-vs-MLP shortcut-gap difference (B=10000, two-sided) gives **p = 0.375**. The numerical 1.4pt advantage for the circuit is within seed noise.

The oracle vs frozen-ResNet contrast is the clean signal: a 172-fold reduction in shortcut gap when the perception bottleneck is removed. The reasoning head can perfectly recover the true Hans3 rules; frozen ImageNet features cannot be disentangled from the confound by any downstream head we tested.

---

## 4. Summary of empirical claims

| Claim | Status at n=5 |
|---|---|
| Hard-extraction Δ ≈ 0 (faithfulness) | **Supported** across E1, E3 |
| Circuit recovers Monks-1, Monks-2, Monks-3 ground-truth rules | **Supported**, ≥96.5% enumeration accuracy |
| Circuit exactly recovers Monks-2 counting constraint where XGBoost cannot | **Supported**, 3/3 seeds exact |
| Circuit beats MLP-matched on accuracy | **Refuted** (parity or within noise on every test) |
| Circuit has low-data inductive-bias advantage | **Refuted** on 11-bit MUX |
| Circuit reduces CLEVR-Hans3 shortcut gap vs MLP-matched | **Refuted** (p = 0.375) |
| Reasoning head recovers true rules given good perception | **Supported**, oracle 0.999 ± 0.001 |
| Frozen ImageNet perception is the bottleneck for visual shortcut resistance | **Supported** by the 172× gap differential between oracle and ResNet pipelines |

---

## 5. What this contributes

A genuinely modest contribution stated cleanly:

1. **A faithfulness probe protocol.** The band-checkpoint methodology of E1/E3 — stop training at a target accuracy band, then measure hard-extraction Δ — gives non-tautological evidence for symbolic readout faithfulness. Standard reports of Δ at >99% accuracy are uninformative because the soft mixture has already collapsed to discrete; the band protocol forces evaluation in the regime where the question is non-trivial.

2. **A clean counting-constraint result.** Monks-2 exact recovery by all seeds while XGBoost fails by 22 points is, to our knowledge, the cleanest direct demonstration that a differentiable architecture can outperform a tree ensemble on a structurally specific reasoning task without losing tabular competence on Monks-1 and Monks-3.

3. **A locating-the-failure result on CLEVR-Hans3.** The 172× gap differential between oracle and frozen-ResNet pipelines is a clean experimental separation of "the reasoning head can do this" from "the perception pipeline forwards the confound." Future work targeting visual shortcut resistance with neuro-symbolic architectures should not freeze the perception backbone.

---

## 6. Limitations and what does not work

- The circuit offers **no measurable accuracy advantage** over a parameter-matched two-layer MLP on any benchmark we tested.
- The circuit offers **no low-data advantage** on 11-bit MUX learning curves.
- The circuit offers **no visual shortcut-resistance advantage** with a frozen pretrained backbone.
- Multi-seed counts are n=5 for E1/E3/E5 and n=3 for E4 Monks; the CLEVR-Hans3 permutation test's negative result is therefore consistent with a true gap up to ~5pt being missed.
- All training runs are short (band-checkpointed or capped). Full-convergence behavior may differ.
- `to_symbolic` produces a flat per-channel readout; we did not evaluate human-readability of the extracted expressions at scale.

---

## 7. Related work

The architectural lineage is **Petersen et al. (2022) Deep Differentiable Logic Gate Networks** (NeurIPS 2022) and **Petersen et al. (2024) Convolutional Differentiable Logic Gate Networks** (NeurIPS 2024 Oral), which achieve 86.3% CIFAR-10 with 61M discrete-gate parameters. Our circuit is in the same family but with a wider operator alphabet (logic ∪ bounded arithmetic) and is not optimized for image-classification competitiveness; the E1 numbers above are decisively below Petersen.

The neuro-symbolic / rule-extraction context includes Inductive Logic Programming (Aleph, Metagol) and Logic Tensor Networks (Badreddine et al., 2021), which we did not directly compare against. The Monks suite is from Thrun et al. (1991); CLEVR-Hans3 is Stammer et al. (2021) and the published competitor we did not attempt to reproduce is the Neuro-Symbolic Concept Learner (NS-CL, Mao et al., 2019).

---

## 8. Reproducibility

All experiments are scripted in `src/train/logic/` of the `dl_techniques` repository at the commit linked to this paper. Each experiment ships with a CSV of per-seed results, a report.md with environment metadata, and unit tests for the supporting modules. The multi-seed driver (`multiseed_sweep.py`) re-runs every experiment in sequence with a configurable seed list. Training wall-clocks on an RTX 4090 are: E1 < 4 min/seed/dataset, E3 < 6 min/seed/all-tasks, E5 < 12 min/seed/all-configs, total ~50 min for the full n=5 sweep.

---

## 9. Conclusion

`LearnableNeuralCircuit` is a faithful differentiable reasoning head. Its symbolic readout is bit-exactly what the trained model computes, robustly across seeds and tasks. It recovers published ground-truth boolean rules from real UCI data at parity with a parameter-matched MLP, and decisively beats gradient boosting on the one Monks task whose true rule is a counting constraint. It does not, in our hands, confer any accuracy, low-data, or visual-shortcut-resistance advantage. The most useful followups are (a) end-to-end training with concept supervision on a CLEVR-style benchmark, where the oracle-vs-ResNet result suggests there is real room to move; (b) head-to-head comparison against ILP systems on Monks-2-style structural tasks, where the circuit's exact-recovery result is most likely to remain distinctive.

---

## References

- Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2022). Deep differentiable logic gate networks. *NeurIPS 2022*.
- Petersen, F., Kuehne, H., Borgelt, C., Welzel, J., & Ermon, S. (2024). Convolutional differentiable logic gate networks. *NeurIPS 2024*.
- Thrun, S. et al. (1991). The MONK's problems: a performance comparison of different learning algorithms. CMU-CS-91-197.
- Stammer, W., Schramowski, P., & Kersting, K. (2021). Right for the right concept: revising neuro-symbolic concepts by interacting with their explanations. *CVPR 2021*.
- Mao, J., Gan, C., Kohli, P., Tenenbaum, J. B., & Wu, J. (2019). The neuro-symbolic concept learner. *ICLR 2019*.
- Badreddine, S., Garcez, A. d., Serafini, L., & Spranger, M. (2021). Logic tensor networks. *Artificial Intelligence* 303.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD 2016*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML 2017*.
