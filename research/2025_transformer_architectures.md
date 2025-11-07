# Transformer Architectures: A Comprehensive Taxonomy

## Executive Summary

The Transformer architecture, introduced in "Attention is All You Need" (Vaswani et al., 2017), has spawned three dominant architectural families: **Encoder-Decoder**, **Encoder-Only**, and **Decoder-Only**. Each represents a different trade-off in the computational efficiency vs. representational capacity space. This guide provides a systematic analysis of these architectures, their computational properties, and their optimal deployment contexts.

**The Fundamental Insight**: The choice of architecture is not arbitrary—it is a direct consequence of the inductive biases required by the target task. There is no universal optimal architecture, only architectures optimized for specific computational objectives.

---

## I. The Original Transformer: Encoder-Decoder Architecture

### Historical Context

The original Transformer was designed for **sequence-to-sequence (seq2seq) transduction**—mapping an input sequence to an output sequence of potentially different length. The canonical application was neural machine translation (NMT): translating English text to French.

### Core Architectural Components

The Encoder-Decoder Transformer consists of two distinct modules operating in tandem:

```
INPUT SEQUENCE (Source Language)
         ↓
    ┌─────────────────────────────┐
    │       ENCODER STACK         │
    │  (Bidirectional Attention)  │
    │                             │
    │  - Self-Attention Layers    │
    │  - Feed-Forward Networks    │
    │  - Layer Normalization      │
    │  - Residual Connections     │
    └─────────────────────────────┘
         ↓
    [Encoded Representations]
         ↓
    ┌─────────────────────────────┐
    │       DECODER STACK         │
    │   (Causal Attention)        │
    │                             │
    │  - Masked Self-Attention    │
    │  - Cross-Attention          │
    │  - Feed-Forward Networks    │
    │  - Layer Normalization      │
    │  - Residual Connections     │
    └─────────────────────────────┘
         ↓
OUTPUT SEQUENCE (Target Language)
```

### Detailed Encoder Architecture

The encoder processes the entire input sequence in parallel, allowing each token to attend to all other tokens (bidirectional attention).

```
Input Tokens: [x₁, x₂, x₃, ..., xₙ]
         ↓
    [Embedding Layer + Positional Encoding]
         ↓
    ┌──────────────────────────────────────┐
    │         ENCODER LAYER 1              │
    │                                      │
    │  ┌────────────────────────────────┐  │
    │  │   Multi-Head Self-Attention    │  │
    │  │                                │  │
    │  │   Q ← Linear(x)                │  │
    │  │   K ← Linear(x)                │  │
    │  │   V ← Linear(x)                │  │
    │  │                                │  │
    │  │   Attention(Q,K,V) =           │  │
    │  │   softmax(QK^T/√d_k)V          │  │
    │  │                                │  │
    │  │   Full Attention Matrix:       │  │
    │  │   Each token attends to ALL    │  │
    │  │   other tokens                 │  │
    │  └────────────────────────────────┘  │
    │            ↓                         │
    │      [Add & Norm]                    │
    │            ↓                         │
    │  ┌────────────────────────────────┐  │
    │  │  Position-wise Feed-Forward    │  │
    │  │                                │  │
    │  │  FFN(x) = max(0, xW₁+b₁)W₂+b₂  │  │
    │  │         (ReLU activation)      │  │
    │  └────────────────────────────────┘  │
    │            ↓                         │
    │      [Add & Norm]                    │
    └──────────────────────────────────────┘
         ↓
    [Repeat for L encoder layers]
         ↓
    [Final Encoder Outputs]
```
         
**Attention Pattern Visualization** (Encoder Self-Attention):

```
      x₁   x₂   x₃   x₄   (Input Tokens)
x₁  [ ✓    ✓    ✓    ✓  ]  ← Token 1 attends to all
x₂  [ ✓    ✓    ✓    ✓  ]  ← Token 2 attends to all
x₃  [ ✓    ✓    ✓    ✓  ]  ← Token 3 attends to all
x₄  [ ✓    ✓    ✓    ✓  ]  ← Token 4 attends to all

Legend: ✓ = Attention allowed (no mask)
```

**Key Property**: **Fully Bidirectional Attention**. Every token has access to contextual information from both past and future tokens.

### Detailed Decoder Architecture

The decoder generates the output sequence autoregressively (one token at a time), using three distinct attention mechanisms:

```
Target Tokens: [y₁, y₂, y₃, ..., yₘ]
         ↓
    [Embedding Layer + Positional Encoding]
         ↓
    ┌──────────────────────────────────────────┐
    │         DECODER LAYER 1                  │
    │                                          │
    │  ┌────────────────────────────────────┐  │
    │  │  Masked Multi-Head Self-Attention  │  │
    │  │  (Causal/Autoregressive)           │  │
    │  │                                    │  │
    │  │  Q ← Linear(y)                     │  │
    │  │  K ← Linear(y)                     │  │
    │  │  V ← Linear(y)                     │  │
    │  │                                    │  │
    │  │  Causal Mask Applied:              │  │
    │  │  Token i can only attend to        │  │
    │  │  tokens j where j ≤ i              │  │
    │  └────────────────────────────────────┘  │
    │            ↓                             │
    │      [Add & Norm]                        │
    │            ↓                             │
    │  ┌────────────────────────────────────┐  │
    │  │   Cross-Attention                  │  │
    │  │   (Decoder-to-Encoder)             │  │
    │  │                                    │  │
    │  │  Q ← Linear(decoder_hidden)        │  │
    │  │  K ← Linear(encoder_output)        │  │
    │  │  V ← Linear(encoder_output)        │  │
    │  │                                    │  │
    │  │  Each decoder position attends     │  │
    │  │  to ALL encoder positions          │  │
    │  └────────────────────────────────────┘  │
    │            ↓                             │
    │      [Add & Norm]                        │
    │            ↓                             │
    │  ┌────────────────────────────────────┐  │
    │  │  Position-wise Feed-Forward        │  │
    │  └────────────────────────────────────┘  │
    │            ↓                             │
    │      [Add & Norm]                        │
    └──────────────────────────────────────────┘
         ↓
    [Repeat for L decoder layers]
         ↓
    [Linear + Softmax]
         ↓
    [Output Token Probabilities]
```

**Attention Pattern Visualization** (Decoder Masked Self-Attention):

```
      y₁   y₂   y₃   y₄   (Target Tokens)
y₁  [ ✓    ✗    ✗    ✗  ]  ← Token 1 sees only itself
y₂  [ ✓    ✓    ✗    ✗  ]  ← Token 2 sees 1,2
y₃  [ ✓    ✓    ✓    ✗  ]  ← Token 3 sees 1,2,3
y₄  [ ✓    ✓    ✓    ✓  ]  ← Token 4 sees all previous

Legend: ✓ = Attention allowed
        ✗ = Attention blocked (causal mask)
```

**Attention Pattern Visualization** (Cross-Attention):

```
Decoder Positions (Queries)
      ↓
      y₁   y₂   y₃   y₄
x₁  [ ✓    ✓    ✓    ✓  ]  ← Encoder position 1
x₂  [ ✓    ✓    ✓    ✓  ]  ← Encoder position 2
x₃  [ ✓    ✓    ✓    ✓  ]  ← Encoder position 3
x₄  [ ✓    ✓    ✓    ✓  ]  ← Encoder position 4
      ↑
Encoder Positions (Keys/Values)

Each decoder position can attend to ALL encoder positions.
This is the mechanism for conditioning generation on the input.
```

### Mathematical Formulation

#### Encoder Self-Attention

For input sequence $X = [x_1, ..., x_n]$:

$$
\text{SelfAttn}_{\text{enc}}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- No masking applied (full attention matrix)

#### Decoder Masked Self-Attention

For target sequence $Y = [y_1, ..., y_m]$:

$$
\text{SelfAttn}_{\text{dec}}(Y) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M$ is the causal mask:

$$
M_{ij} = \begin{cases} 
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

#### Cross-Attention

$$
\text{CrossAttn}(Y, X) = \text{softmax}\left(\frac{Q_{\text{dec}}K_{\text{enc}}^T}{\sqrt{d_k}}\right)V_{\text{enc}}
$$

where:
- $Q_{\text{dec}} = Y W_Q$ (queries from decoder)
- $K_{\text{enc}} = X W_K$ (keys from encoder)
- $V_{\text{enc}} = X W_V$ (values from encoder)

### Computational Complexity

**Time Complexity per Layer**:
- Encoder self-attention: $O(n^2 \cdot d)$ where $n$ = input length, $d$ = model dimension
- Decoder self-attention: $O(m^2 \cdot d)$ where $m$ = output length
- Cross-attention: $O(m \cdot n \cdot d)$

**Total Complexity**: $O((n^2 + m^2 + mn) \cdot d \cdot L)$ where $L$ = number of layers

**Space Complexity**: $O(n^2 + m^2 + mn)$ for attention matrices

### Use Cases & Optimal Applications

**Ideal For**:
1. **Machine Translation**: Translating text between languages
2. **Summarization**: Condensing long documents into short summaries
3. **Question Answering** (extractive): Given context, generate answer
4. **Text-to-Text Tasks**: Any task with distinct input/output sequences

**Canonical Models**:
- **T5** (Text-to-Text Transfer Transformer): Frames all NLP tasks as text-to-text
- **BART** (Bidirectional and Auto-Regressive Transformers): Denoising autoencoder
- **mBART**: Multilingual BART
- **mT5**: Multilingual T5

**Advantages**:
- **Separation of concerns**: Encoder handles understanding, decoder handles generation
- **Flexible length mapping**: Input and output can have different lengths
- **Cross-attention**: Explicit mechanism for conditioning on input

**Disadvantages**:
- **Increased parameter count**: Two separate stacks to train
- **Slower inference**: Must run both encoder and decoder
- **Memory intensive**: Stores attention matrices for both stacks

---

## II. Encoder-Only Architecture (BERT-style)

### Philosophical Shift

Encoder-only models abandon generation entirely. They are designed for **discriminative tasks**—understanding and classifying input, not producing new sequences.

### Core Architecture

```
INPUT SEQUENCE
         ↓
    [Special Token Prepending: [CLS] + tokens + [SEP]]
         ↓
    [Embedding Layer + Positional Encoding]
         ↓
    ┌──────────────────────────────────────┐
    │       ENCODER LAYER 1                │
    │  (Identical to original Transformer  │
    │   encoder, fully bidirectional)      │
    └──────────────────────────────────────┘
         ↓
    ┌──────────────────────────────────────┐
    │       ENCODER LAYER 2                │
    └──────────────────────────────────────┘
         ↓
         ...
         ↓
    ┌──────────────────────────────────────┐
    │       ENCODER LAYER L                │
    └──────────────────────────────────────┘
         ↓
    [Contextualized Token Representations]
         ↓
    ┌──────────────────────────────────────┐
    │      TASK-SPECIFIC HEADS             │
    │                                      │
    │  • Classification: Linear([CLS])     │
    │  • Token Classification: Linear(all) │
    │  • Span Extraction: Start/End heads  │
    └──────────────────────────────────────┘
         ↓
    OUTPUT (Labels, not sequences)
```

### Detailed Single Layer Architecture

```
Input: h^(l-1) = [h₁^(l-1), h₂^(l-1), ..., hₙ^(l-1)]
         ↓
    ┌────────────────────────────────────────┐
    │  Multi-Head Self-Attention             │
    │                                        │
    │  For each head i ∈ {1,...,H}:          │
    │    Qᵢ = h^(l-1) Wᵢ^Q                   │
    │    Kᵢ = h^(l-1) Wᵢ^K                   │
    │    Vᵢ = h^(l-1) Wᵢ^V                   │
    │                                        │
    │    headᵢ = softmax(QᵢKᵢ^T/√dₖ)Vᵢ       │
    │                                        │
    │  MultiHead = Concat(head₁,...,headₕ)W^O│
    └────────────────────────────────────────┘
         ↓
    [Add & Norm]: h' = LayerNorm(h^(l-1) + MultiHead)
         ↓
    ┌────────────────────────────────────────┐
    │  Position-wise Feed-Forward Network    │
    │                                        │
    │  FFN(h') = max(0, h'W₁ + b₁)W₂ + b₂    │
    │           (or GELU activation)         │
    │                                        │
    │  Typical dimensions:                   │
    │  d_model → 4·d_model → d_model         │
    │  (768 → 3072 → 768 for BERT-base)      │
    └────────────────────────────────────────┘
         ↓
    [Add & Norm]: h^(l) = LayerNorm(h' + FFN(h'))
         ↓
Output: h^(l) = [h₁^(l), h₂^(l), ..., hₙ^(l)]
```

### Attention Pattern (Fully Bidirectional)

```
      [CLS] The  cat  sat  [SEP]
[CLS]  [ ✓    ✓    ✓    ✓    ✓  ]
The    [ ✓    ✓    ✓    ✓    ✓  ]
cat    [ ✓    ✓    ✓    ✓    ✓  ]
sat    [ ✓    ✓    ✓    ✓    ✓  ]
[SEP]  [ ✓    ✓    ✓    ✓    ✓  ]

Every token attends to every other token.
No causal masking.
Full context in both directions.
```

### Pre-training Objectives

#### 1. Masked Language Modeling (MLM)

**Procedure**:
1. Randomly mask 15% of input tokens
2. Of the masked tokens:
   - 80% replaced with `[MASK]`
   - 10% replaced with random token
   - 10% left unchanged
3. Predict original token at masked positions

**Example**:
```
Original:    "The cat sat on the mat"
Masked:      "The [MASK] sat on the [MASK]"
Objective:   Predict "cat" and "mat"
```

**Loss Function**:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}})
$$

where $\mathcal{M}$ = set of masked positions

#### 2. Next Sentence Prediction (NSP)

**Procedure**:
1. Given two sentences A and B
2. 50% of time: B actually follows A (label: IsNext)
3. 50% of time: B is random sentence (label: NotNext)
4. Predict relationship using `[CLS]` token representation

**Example**:
```
Positive: A = "I went to the store."
          B = "I bought some milk."
          Label: IsNext

Negative: A = "I went to the store."
          B = "The quantum entanglement paradox."
          Label: NotNext
```

**Loss Function**:

$$
\mathcal{L}_{\text{NSP}} = -\log P(y_{\text{NSP}} | [\text{CLS}])
$$

### Fine-tuning Paradigm

```
Pre-trained BERT (on large corpus)
         ↓
    [Frozen or fine-tuned encoder layers]
         ↓
    [Task-specific head added on top]
         ↓
    [Train on labeled data for specific task]

Common Task Adaptations:

1. SEQUENCE CLASSIFICATION (sentiment, topic, etc.)
   Input: [CLS] sentence [SEP]
   Output: Linear(h_[CLS]) → softmax → class label

2. TOKEN CLASSIFICATION (NER, POS tagging)
   Input: [CLS] w₁ w₂ ... wₙ [SEP]
   Output: Linear(hᵢ) for each token → label

3. QUESTION ANSWERING (span extraction)
   Input: [CLS] question [SEP] context [SEP]
   Output: Start span head and End span head
           logits for each context token

4. SENTENCE PAIR CLASSIFICATION (entailment)
   Input: [CLS] sentence₁ [SEP] sentence₂ [SEP]
   Output: Linear(h_[CLS]) → relationship label
```

### Computational Characteristics

**Time Complexity**: $O(n^2 \cdot d \cdot L)$
- Same as encoder-only portion of Enc-Dec
- Quadratic in sequence length (attention bottleneck)

**Inference Characteristics**:
- **Single forward pass**: No autoregressive generation
- **Parallel processing**: All tokens processed simultaneously
- **Fast**: Typically 10-100× faster than decoder-only models for classification

### Use Cases & Optimal Applications

**Ideal For**:
1. **Text Classification**: Sentiment analysis, topic categorization, spam detection
2. **Named Entity Recognition (NER)**: Identifying people, places, organizations
3. **Question Answering**: Extractive QA (finding answer span in context)
4. **Semantic Similarity**: Determining if two sentences are similar
5. **Token-level Tasks**: POS tagging, chunking, slot filling

**Canonical Models**:
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **RoBERTa** (Robustly Optimized BERT): Removes NSP, trains longer
- **ALBERT** (A Lite BERT): Parameter sharing across layers
- **ELECTRA**: Replaces MLM with token-level discrimination
- **DistilBERT**: Distilled, smaller, faster version of BERT
- **DeBERTa**: Disentangled attention + enhanced mask decoder

**Advantages**:
- **Bidirectional context**: Full context for every token
- **Fast inference**: Single forward pass for classification
- **Strong transfer learning**: Pre-training → fine-tuning paradigm is very effective
- **Interpretability**: Attention weights show what model focuses on

**Disadvantages**:
- **No generation capability**: Cannot produce new sequences
- **Pre-training inefficiency**: MLM only predicts 15% of tokens per sample
- **Input length limitations**: Quadratic attention complexity
- **Task adaptation required**: Needs fine-tuning or task-specific heads

---

## III. Decoder-Only Architecture (GPT-style)

### Philosophical Foundation

Decoder-only models embrace **autoregressive generation** as the universal learning objective. They are trained to predict the next token given all previous tokens, making them natural generators.

**Key Insight**: By framing all tasks as generation problems, a single architecture can handle classification, generation, translation, reasoning, and more—without task-specific modifications.

### Core Architecture

```
INPUT SEQUENCE (Prompt/Context)
         ↓
    [Embedding Layer + Positional Encoding]
         ↓
    ┌──────────────────────────────────────┐
    │     DECODER LAYER 1                  │
    │  (Causal Self-Attention)             │
    │  (No Encoder, No Cross-Attention)    │
    │                                      │
    │  ┌────────────────────────────────┐  │
    │  │  Masked Self-Attention         │  │
    │  │  Token i attends to j ≤ i      │  │
    │  └────────────────────────────────┘  │
    │           ↓                          │
    │     [Add & Norm]                     │
    │           ↓                          │
    │  ┌────────────────────────────────┐  │
    │  │  Feed-Forward Network          │  │
    │  └────────────────────────────────┘  │
    │           ↓                          │
    │     [Add & Norm]                     │
    └──────────────────────────────────────┘
         ↓
    [Repeat for L layers]
         ↓
    ┌──────────────────────────────────────┐
    │     LANGUAGE MODEL HEAD              │
    │  Linear(d_model → vocab_size)        │
    │           ↓                          │
    │      [Softmax]                       │
    └──────────────────────────────────────┘
         ↓
    NEXT TOKEN PROBABILITY DISTRIBUTION
```

### Detailed Layer Structure

```
Input: h^(l-1) = [h₁^(l-1), h₂^(l-1), ..., hₙ^(l-1)]
         ↓
    ┌─────────────────────────────────────────────┐
    │  Causal (Masked) Multi-Head Self-Attention  │
    │                                             │
    │  For each head i:                           │
    │    Qᵢ = h^(l-1) Wᵢ^Q                        │
    │    Kᵢ = h^(l-1) Wᵢ^K                        │
    │    Vᵢ = h^(l-1) Wᵢ^V                        │
    │                                             │
    │    Sᵢ = QᵢKᵢ^T/√dₖ + M (causal mask)        │
    │    headᵢ = softmax(Sᵢ)Vᵢ                    │
    │                                             │
    │  MultiHead = Concat(head₁,...,headₕ)W^O     │
    └─────────────────────────────────────────────┘
         ↓
    [Add & Norm]: h' = LayerNorm(h^(l-1) + MultiHead)
         ↓
    ┌─────────────────────────────────────────────┐
    │  Position-wise Feed-Forward Network         │
    │                                             │
    │  Modern variants often use:                 │
    │  • SwiGLU activation: x ⊙ (xW₁)·(xW_gate)   │
    │  • GeGLU activation: x ⊙ GELU(xW₁)          │
    │                                             │
    │  Dimensions: d_model → 4·d_model → d_model  │
    │  (Some models use 8·d_model intermediate)   │
    └─────────────────────────────────────────────┘
         ↓
    [Add & Norm]: h^(l) = LayerNorm(h' + FFN(h'))
         ↓
Output: h^(l) = [h₁^(l), h₂^(l), ..., hₙ^(l)]
```

### Causal Attention Pattern

```
      The  cat  sat  on   the  mat
The [ ✓    ✗    ✗    ✗    ✗    ✗  ]  Position 1
cat [ ✓    ✓    ✗    ✗    ✗    ✗  ]  Position 2
sat [ ✓    ✓    ✓    ✗    ✗    ✗  ]  Position 3
on  [ ✓    ✓    ✓    ✓    ✗    ✗  ]  Position 4
the [ ✓    ✓    ✓    ✓    ✓    ✗  ]  Position 5
mat [ ✓    ✓    ✓    ✓    ✓    ✓  ]  Position 6

✓ = Can attend (past + current)
✗ = Cannot attend (future, blocked by mask)

Lower triangular attention matrix enforces causality.
```

### Training Objective: Next Token Prediction

**Objective**: Maximize likelihood of next token given all previous tokens

$$
\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i | x_{<i})
$$

**Training Example**:
```
Input Sequence: "The cat sat on the mat"

Training creates multiple examples:
Context: "The"           → Predict: "cat"
Context: "The cat"       → Predict: "sat"
Context: "The cat sat"   → Predict: "on"
Context: "The cat sat on"→ Predict: "the"
...

All processed in parallel during training via teacher forcing.
```

**Teacher Forcing**:
```
Training Time (Parallel):
┌────────────────────────────────────┐
│ Input:  [The] [cat] [sat] [on]     │
│ Target: [cat] [sat] [on]  [the]    │
│                                    │
│ Loss computed for all positions    │
│ simultaneously using ground truth  │
└────────────────────────────────────┘

Inference Time (Sequential):
Step 1: Input: [The]        → Generate: "cat"
Step 2: Input: [The cat]    → Generate: "sat"
Step 3: Input: [The cat sat]→ Generate: "on"
...continues until [EOS] or max length
```

### Generation Strategies

#### 1. Greedy Decoding

```
At each step, select highest probability token:

y_t = argmax P(y | y_{<t})
```

**Problem**: Can miss globally optimal sequences, gets stuck in local optima.

#### 2. Beam Search

```
Maintain k highest-scoring partial sequences (beams):

Beam 1: [The, cat, sat] (score: -2.5)
Beam 2: [The, dog, sat] (score: -2.7)
Beam 3: [The, cat, was] (score: -3.1)

At each step, expand all beams and keep top k.
```

**Pros**: Better than greedy, explores multiple hypotheses
**Cons**: Computationally expensive, k× slower than greedy

#### 3. Top-k Sampling

```
At each step:
1. Compute probability distribution over vocab
2. Keep only top k most likely tokens
3. Renormalize distribution
4. Sample from truncated distribution

Example (k=5):
Original: P(cat)=0.3, P(dog)=0.2, P(sat)=0.15, ...
After top-k: Only consider 5 highest, set others to 0
Sample randomly from these 5
```

**Effect**: Introduces randomness, prevents repetition

#### 4. Nucleus (Top-p) Sampling

```
At each step:
1. Sort tokens by probability (descending)
2. Find smallest set with cumulative probability ≥ p
3. Sample from this "nucleus"

Example (p=0.9):
Tokens: cat(0.3), dog(0.25), sat(0.2), ...
Nucleus: {cat, dog, sat} (sum = 0.75 < 0.9)
       {cat, dog, sat, on} (sum = 0.92 ≥ 0.9) ← Use this
```

**Advantage**: Adapts to probability distribution (flexible nucleus size)

#### 5. Temperature Scaling

Modify softmax temperature to control randomness:

$$
P(x_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

- **T < 1**: Sharper distribution (more deterministic)
- **T = 1**: Original distribution
- **T > 1**: Flatter distribution (more random)

```
Original logits: [2.0, 1.5, 1.0, 0.5]

T = 0.5 (sharp):  [0.52, 0.28, 0.14, 0.06] ← Peaky
T = 1.0 (normal): [0.42, 0.28, 0.19, 0.11]
T = 2.0 (flat):   [0.31, 0.27, 0.23, 0.19] ← Uniform-ish
```

### In-Context Learning

**Emergent Capability**: Large decoder-only models can learn new tasks from examples in the prompt, without parameter updates.

```
PROMPT STRUCTURE:

┌─────────────────────────────────────────┐
│ Task Description (optional):            │
│ "Translate English to French:"          │
├─────────────────────────────────────────┤
│ Few-Shot Examples:                      │
│ English: "Hello" → French: "Bonjour"    │
│ English: "Goodbye" → French: "Au revoir"│
│ English: "Thank you" → French: "Merci"  │
├─────────────────────────────────────────┤
│ Query:                                  │
│ English: "Good morning" → French:       │
└─────────────────────────────────────────┘
         ↓
    [Model generates: "Bon matin"]

No gradient updates. Learning happens via attention.
```

**Paradigm Shift**: Models become **universal few-shot learners** rather than task-specific systems.

### Computational Complexity

**Training**: $O(n^2 \cdot d \cdot L)$
- Same as encoder-only
- Quadratic attention still applies

**Inference** (autoregressive generation):
- **Per token**: $O(n \cdot d \cdot L)$ where $n$ is current sequence length
- **Total for m tokens**: $O(m \cdot n \cdot d \cdot L)$ where $n$ grows with each token
- Much slower than encoder-only for equivalent tasks

**KV Cache Optimization**:
```
Without cache:
  Generate token 100: Recompute attention for all 100 tokens

With cache:
  Generate token 100: 
    - Reuse cached K,V for tokens 1-99
    - Only compute K,V for token 100
    - Reduces computation significantly
```

### Use Cases & Optimal Applications

**Ideal For**:
1. **Text Generation**: Creative writing, code generation, completion
2. **Dialogue Systems**: Chatbots, conversational AI
3. **Few-shot Learning**: Tasks with limited labeled data
4. **Instruction Following**: General-purpose assistants
5. **Code Synthesis**: Programming from natural language
6. **Reasoning Tasks**: Chain-of-thought, multi-step problem solving

**Canonical Models**:
- **GPT** series (GPT-2, GPT-3, GPT-4)
- **LLaMA** (1, 2, 3): Open-source, permissive license
- **Mistral/Mixtral**: Efficient, high-quality open models
- **Qwen**: Multilingual, strong coding abilities
- **Gemma**: Google's open decoder-only models
- **Phi**: Microsoft's small but capable models

**Advantages**:
- **Universal architecture**: One model, many tasks
- **Few-shot learning**: No fine-tuning needed for new tasks
- **Generation quality**: Produces fluent, coherent text
- **Emergent abilities**: Capabilities scale with size (reasoning, tool use)

**Disadvantages**:
- **Slow inference**: Autoregressive generation is sequential
- **Unidirectional context**: Each token only sees past, not future
- **Training efficiency**: Predicts one token at a time (vs. BERT's 15% mask)
- **High memory**: KV cache grows linearly with sequence length
- **Hallucination**: Can generate plausible but incorrect information

---

## IV. Hybrid & Advanced Architectures

### A. Prefix LM (Prefix Language Model)

**Concept**: Combine bidirectional encoding of a prefix with causal generation of the suffix.

```
INPUT: [Prefix: fully visible] [Suffix: causal]
         ↓
    ┌─────────────────────────────────────────┐
    │         UNIFIED TRANSFORMER             │
    │                                         │
    │  Attention Pattern:                     │
    │                                         │
    │       Pre₁ Pre₂ │ Suf₁ Suf₂ Suf₃        │
    │  Pre₁  [ ✓   ✓  │  ✗   ✗   ✗  ]         │
    │  Pre₂  [ ✓   ✓  │  ✗   ✗   ✗  ]         │
    │  ────────────────────────────────       │
    │  Suf₁  [ ✓   ✓  │  ✓   ✗   ✗  ]         │
    │  Suf₂  [ ✓   ✓  │  ✓   ✓   ✗  ]         │
    │  Suf₃  [ ✓   ✓  │  ✓   ✓   ✓  ]         │
    │                                         │
    │  Prefix: Bidirectional (like BERT)      │
    │  Suffix: Causal (like GPT)              │
    └─────────────────────────────────────────┘
         ↓
    OUTPUT: Next token after suffix
```

**Use Case**: Question answering where question is prefix (bidirectional) and answer is suffix (causal).

**Example**:
```
Prefix: "What is the capital of France?"
Suffix: "The capital is" [Paris]

The question gets bidirectional context.
The answer is generated autoregressively.
```

**Models**: GLM (General Language Model), LaMDA (early versions)

### B. Encoder-Decoder with Shared Parameters

**Concept**: Use the same parameters for both encoder and decoder to reduce model size.

```
INPUT SEQUENCE
         ↓
    ┌─────────────────────────────────┐
    │   SHARED TRANSFORMER LAYERS     │
    │   (Used as Encoder)             │
    │                                 │
    │   Full bidirectional attention  │
    └─────────────────────────────────┘
         ↓
    [Encoded Representations]
         ↓
    ┌─────────────────────────────────┐
    │   SAME TRANSFORMER LAYERS       │
    │   (Used as Decoder)             │
    │                                 │
    │   Causal self-attention +       │
    │   Cross-attention to encoder    │
    └─────────────────────────────────┘
         ↓
    OUTPUT SEQUENCE

Weight tying reduces parameters by ~50%
```

**Models**: ALBERT (partial sharing), T5 (can be configured for sharing)

### C. Universal Transformer (UL2)

**Concept**: Train a single model on multiple objectives simultaneously to create a truly universal model.

```
UL2 Training Objectives:

┌──────────────────────────────────────────┐
│ 1. R-Denoiser (Regular Denoising)        │
│    Short spans, low corruption (~15%)    │
│    Similar to BERT MLM                   │
├──────────────────────────────────────────┤
│ 2. X-Denoiser (Extreme Denoising)        │
│    Long spans, high corruption (~50%)    │
│    Longer-range dependencies             │
├──────────────────────────────────────────┤
│ 3. S-Denoiser (Sequential Denoising)     │
│    Causal language modeling              │
│    Similar to GPT                        │
└──────────────────────────────────────────┘
         ↓
    [Single Unified Model]
         ↓
Can be prompted at inference to behave as:
- Encoder (via R-denoising mode)
- Decoder (via S-denoising mode)
- Encoder-Decoder (via X-denoising mode)
```

**Mode Token**: Special tokens like `[R]`, `[X]`, `[S]` prepended to input to specify mode.

**Model**: UL2, Flan-UL2

### D. Mixture of Experts (MoE)

**Concept**: Replace dense FFN layers with sparse, expert-routed computation.

```
Standard Transformer Layer:
    Input → Self-Attention → FFN (dense) → Output

MoE Transformer Layer:
    Input → Self-Attention → Router → [Expert Selection] → Output
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
          Expert_1        Expert_2   ...  Expert_N
         (FFN_1)         (FFN_2)          (FFN_N)
              ↓               ↓               ↓
         [Only top-k experts activated per token]
              ↓
         [Weighted combination of expert outputs]
```

**Routing Mechanism**:

$$
G(x) = \text{softmax}(x \cdot W_g)
$$

where $G(x)$ is a vector of routing probabilities over experts.

**Top-k Routing**:
```
For each token:
1. Compute routing scores for all N experts
2. Select top-k experts (typically k=2)
3. Zero out all other experts
4. Renormalize selected expert weights
5. Compute weighted sum of selected expert outputs
```

**Sparse Activation**:
```
If N=8 experts, k=2:
  - 2 experts activated per token (25%)
  - 6 experts dormant per token (75%)
  - Effective parameters: ~25% of total
  - Total capacity: 100% (all experts available)
```

**Advantages**:
- Massive parameter count with constant compute per token
- Specialization: Different experts learn different patterns
- Scalability: Can scale to trillions of parameters

**Challenges**:
- Load balancing: Ensuring all experts are used
- Training instability: Routing can collapse (all traffic to one expert)
- Deployment complexity: Distributed inference required for large MoE

**Models**: Switch Transformer, GLaM, Mixtral 8x7B, Grok-1

### E. Sparse Transformer (Sparse Attention Patterns)

**Problem**: Standard attention is $O(n^2)$, prohibitive for very long sequences.

**Solution**: Constrain attention to sparse patterns.

#### 1. Strided Attention

```
Every token attends to:
- All tokens in its local window (e.g., ±64 positions)
- Every k-th token globally (e.g., stride=128)

       0   1   2   3   4   5   6   7   8   (positions)
   0  [✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗]  Local window
   4  [✗   ✗   ✗   ✓   ✓   ✓   ✗   ✗   ✗]
   8  [✗   ✗   ✗   ✗   ✗   ✗   ✗   ✓   ✓]
```

#### 2. Fixed Patterns (Sparse Transformer)

Alternate between two patterns across layers:

**Strided Pattern** (Layer A):
```
Position i attends to: {i-n, i-2n, i-3n, ...}
(Every n-th previous position)
```

**Local Pattern** (Layer B):
```
Position i attends to: {i-w, i-w+1, ..., i}
(Local window of width w)
```

**Combined**: Alternating layers creates O(n√n) complexity while maintaining long-range connections.

#### 3. Longformer Attention

```
Combines:
- Local windowed attention (most tokens)
- Task-motivated global attention (special tokens)

      w₁  w₂  w₃  w₄  [CLS] w₅  w₆  w₇  w₈
w₁   [✓   ✓   ✗   ✗    ✓    ✗   ✗   ✗   ✗]
w₂   [✓   ✓   ✓   ✗    ✓    ✗   ✗   ✗   ✗]
w₃   [✗   ✓   ✓   ✓    ✓    ✗   ✗   ✗   ✗]
w₄   [✗   ✗   ✓   ✓    ✓    ✓   ✗   ✗   ✗]
[CLS][✓   ✓   ✓   ✓    ✓    ✓   ✓   ✓   ✓]  Global
w₅   [✗   ✗   ✗   ✓    ✓    ✓   ✓   ✗   ✗]
w₆   [✗   ✗   ✗   ✗    ✓    ✓   ✓   ✓   ✗]
w₇   [✗   ✗   ✗   ✗    ✓    ✗   ✓   ✓   ✓]
w₈   [✗   ✗   ✗   ✗    ✓    ✗   ✗   ✓   ✓]

Window width: 2 (adjustable)
[CLS] has global attention (attends to all, all attend to it)
```

**Models**: Longformer, BigBird, Sparse Transformer

### F. Retrieval-Augmented Generation (RAG)

**Concept**: Augment generation with retrieved external knowledge.

```
USER QUERY: "What is the capital of Mars?"
         ↓
    ┌─────────────────────────────────┐
    │   RETRIEVAL SYSTEM              │
    │   (e.g., Dense retriever,       │
    │    BM25, vector database)       │
    └─────────────────────────────────┘
         ↓
    [Retrieved Documents]
    - Doc 1: "Mars is the fourth planet..."
    - Doc 2: "Mars has no capital as it's uninhabited..."
    - Doc 3: "Olympus Mons is on Mars..."
         ↓
    ┌─────────────────────────────────┐
    │   ENCODER (optional)            │
    │   Encode retrieved docs         │
    └─────────────────────────────────┘
         ↓
    [Concatenate: Query + Docs]
    "Query: What is the capital of Mars?
     Context: Mars has no capital as it's uninhabited..."
         ↓
    ┌─────────────────────────────────┐
    │   GENERATOR (Decoder-only)      │
    │   Generate answer conditioned   │
    │   on query + retrieved context  │
    └─────────────────────────────────┘
         ↓
    OUTPUT: "Mars doesn't have a capital because 
             it's an uninhabited planet."
```

**Variants**:
- **RAG-Sequence**: Retrieve once, generate entire response
- **RAG-Token**: Retrieve for each generated token (expensive)
- **REALM**: Retrieval as a pre-training objective
- **RETRO**: Cross-attend to retrieved documents at multiple layers

**Advantage**: Grounds generation in factual knowledge, reduces hallucination

---

## V. Comparative Analysis

### Architectural Trade-offs Summary

```
┌──────────────────┬─────────────┬──────────────┬──────────────┐
│ Property         │ Encoder-Dec │ Encoder-Only │ Decoder-Only │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Attention Type   │ Enc: Bidir  │ Bidirection  │ Causal       │
│                  │ Dec: Causal │              │              │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Primary Task     │ Seq2Seq     │ Understand   │ Generate     │
│                  │ Translation │ Classify     │ Complete     │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Inference Speed  │ Slow        │ Fast         │ Very Slow    │
│                  │ (2 passes)  │ (1 pass)     │ (sequential) │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Training         │ Moderate    │ Inefficient  │ Efficient    │
│ Efficiency       │             │ (15% mask)   │ (all tokens) │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Few-shot         │ Limited     │ No           │ Yes          │
│ Learning         │             │              │ (emergent)   │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Generation       │ Yes         │ No           │ Yes          │
│ Capability       │ (Good)      │              │ (Excellent)  │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Classification   │ Moderate    │ Excellent    │ Moderate     │
│ Performance      │             │              │              │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Memory Usage     │ High        │ Moderate     │ High         │
│                  │ (2 stacks)  │              │ (KV cache)   │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Parameter        │ Highest     │ Moderate     │ Moderate     │
│ Count (same      │ (~2x single)│              │              │
│ performance)     │             │              │              │
└──────────────────┴─────────────┴──────────────┴──────────────┘
```

### Task-to-Architecture Mapping

```
TASK TYPE                    → OPTIMAL ARCHITECTURE

Machine Translation          → Encoder-Decoder (T5, mT5, BART)
Summarization               → Encoder-Decoder or Decoder-Only
Text Classification         → Encoder-Only (BERT, RoBERTa)
Sentiment Analysis          → Encoder-Only
Named Entity Recognition    → Encoder-Only
Question Answering:
  - Extractive              → Encoder-Only
  - Generative              → Decoder-Only or Enc-Dec
Text Generation             → Decoder-Only (GPT-family)
Code Generation             → Decoder-Only
Dialogue/Chat               → Decoder-Only
Few-shot Learning           → Decoder-Only (emergent capability)
Instruction Following       → Decoder-Only
Embedding/Retrieval         → Encoder-Only (remove causal mask)
```

### Computational Efficiency Comparison

**For Classification Task** (1000 examples, 512 tokens each):

```
Encoder-Only (BERT):
  - Forward passes: 1,000 (one per example)
  - Parallel: Yes (batch processing)
  - Time: ~10 seconds (V100 GPU)

Decoder-Only (GPT-3):
  - Forward passes: 1,000 (one per example)
  - Parallel: Yes (if no generation needed)
  - Time: ~15 seconds (same GPU)
  - Note: If generating output, much slower due to autoregression

Encoder-Decoder (T5):
  - Forward passes: 2,000 (encoder + decoder per example)
  - Time: ~20 seconds (same GPU)
```

**For Generation Task** (generate 100 tokens):

```
Decoder-Only (GPT-3):
  - Forward passes: 100 (one per generated token)
  - Sequential: Must wait for each token
  - Time: ~5 seconds

Encoder-Decoder (T5):
  - Forward passes: 1 (encoder) + 100 (decoder)
  - Time: ~6 seconds
  - Slight overhead from encoder, but marginal
```

---

## VI. The Future: Architectural Convergence

### Emerging Trends

#### 1. Universal Architectures

The field is moving toward **single architectures that can handle any task**:

```
Traditional Paradigm:
  BERT for classification
  GPT for generation
  T5 for seq2seq
  → 3 different models, 3 training runs

Emerging Paradigm:
  Single decoder-only model
  All tasks framed as generation
  → 1 model, 1 training run
  
Example: GPT-4, PaLM, LLaMA-3 handle classification,
         generation, reasoning, coding, etc.
```

#### 2. Efficient Attention Mechanisms

**Moving beyond O(n²)**:

```
Linear Attention:
  - Linformer: Low-rank approximation
  - Performer: Random feature attention
  - FNet: Fourier transforms replace attention
  
State Space Models:
  - Mamba: Structured state space models
  - H3: Hybrid models (attention + SSM)
  
Complexity: O(n) instead of O(n²)
```

#### 3. Multi-modal Architectures

**Unified models for text, image, audio, video**:

```
        TEXT ────┐
                 │
       IMAGE ────┼──→ [Shared Transformer] ──→ OUTPUT
                 │
       AUDIO ────┘
       
Each modality has specific tokenizer/encoder
but shares core transformer layers

Examples: GPT-4V, Gemini, Flamingo
```

---

## VII. Decision Framework: Choosing the Right Architecture

### Decision Tree

```
START: What is your primary task?
   ↓
   ├─→ Generation/Completion? ─────→ DECODER-ONLY
   │                                 (GPT, LLaMA, Mistral)
   │
   ├─→ Classification/Tagging? ────→ ENCODER-ONLY
   │                                 (BERT, RoBERTa, DeBERTa)
   │
   ├─→ Seq2Seq (translation)? ─────→ ENCODER-DECODER
   │                                 (T5, BART, mT5)
   │
   └─→ Multiple task types? ───────→ Consider:
                                     • Decoder-only (universal)
                                     • Hybrid (UL2)
                                     • Multiple specialized models

Additional Considerations:
   ↓
   ├─→ Need few-shot learning? ────→ DECODER-ONLY (large scale)
   │
   ├─→ Need bidirectional context? ─→ ENCODER-ONLY or ENC-DEC
   │
   ├─→ Inference speed critical? ───→ ENCODER-ONLY (fastest)
   │
   ├─→ Training data limited? ──────→ ENCODER-ONLY (fine-tune BERT)
   │
   └─→ Need variable length I/O? ───→ ENCODER-DECODER or DECODER-ONLY
```

### Practical Recommendations (2025)

**For Research**:
- **Encoder-Only**: Still valuable for embedding models, retrieval
- **Decoder-Only**: Dominant paradigm, most active research area
- **Encoder-Decoder**: Niche applications, less active development

**For Production**:
- **Classification at scale**: Encoder-only (BERT-family) for speed
- **General-purpose systems**: Decoder-only (GPT-family) for flexibility
- **Translation/Summarization**: Encoder-decoder or large decoder-only

**For Startups/Resource-Constrained**:
- Start with pre-trained decoder-only models (LLaMA, Mistral)
- Fine-tune or use in-context learning
- Avoid training encoder-decoder from scratch (expensive, diminishing returns)

---

## VIII. Conclusion: The Architectural Landscape

### The Evolution

```
2017: Encoder-Decoder Transformer
        ↓
      Bifurcation
        ↓
   ┌────┴────┐
   ↓         ↓
BERT      GPT
(2018)    (2018)
   ↓         ↓
Encoder   Decoder
  Era       Era
   ↓         ↓
(2018-21) (2021-present)
   ↓         ↓
 Plateau  Dominance
```

### The Current State

**Encoder-Only**: Mature, stable, excellent for discriminative tasks. Less active research.

**Decoder-Only**: Dominant paradigm. Scaling laws hold, emergent capabilities, universal task handling.

**Encoder-Decoder**: Specialized applications, outperformed by large decoder-only in many tasks.

### The Fundamental Truth

**There is no universal best architecture**—only architectures optimized for specific computational objectives:

- **Bidirectional context** (encoder-only): When you need to understand entire input before making decisions
- **Autoregressive generation** (decoder-only): When you need to produce sequences token-by-token
- **Separation of encoding/decoding** (encoder-decoder): When input and output are fundamentally different

**The Paradigm Shift**: The field is converging on decoder-only as the **universal architecture** not because it's optimal for every task, but because:

1. **Scalability**: Scaling laws are most favorable for decoder-only
2. **Simplicity**: Single architecture, single training objective
3. **Flexibility**: Can be adapted to any task via prompting
4. **Emergent capabilities**: In-context learning, reasoning, tool use

The future belongs to architectures that are **sufficiently general** to handle any task, **sufficiently efficient** to scale to massive sizes, and **sufficiently flexible** to adapt without retraining.

**The winner**: Decoder-only Transformers, at least until the next architectural revolution.