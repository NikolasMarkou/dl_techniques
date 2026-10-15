# VLM Task Heads: Integration Guide

## 1. Overview & Philosophy

The `dl_techniques.layers.heads.vlm` module provides a unified, model-agnostic system for attaching task-specific heads to any Visual Language Model (VLM) foundation model.

**Core Philosophy:** Decouple the *multi-modal encoders* from the *task-specific head*.
*   **Foundation Model Encoders**: (e.g., CLIP, BLIP, Flamingo) are responsible for producing rich, contextualized feature representations from visual and textual inputs.
*   **VLM Task Head**: Is responsible for fusing these uni-modal features and transforming them into task-specific predictions (captions, answer logits, bounding boxes, similarity scores).

This separation allows you to build complex, multi-task VLMs by combining a shared set of encoders with a collection of lightweight, specialized heads.

### Architecture: Two Types of Heads

The framework provides two distinct categories of heads, offering a balance between specialization and flexibility:

1.  **Standalone Task Architectures**: For tasks like `ImageCaptioning` and `VQA`, the heads are complete, self-contained Keras Layers with their own internal logic (e.g., a full Transformer decoder or a custom MLP classifier).
2.  **Flexible Fusion-Based Heads**: For tasks like `ImageTextMatching` and `VisualGrounding`, the heads are built upon a powerful and highly configurable `MultiModalFusion` base layer. Their behavior is primarily controlled by selecting a `fusion_strategy`.

```
Vision Input → Vision Encoder → Vision Features ↘
                                                  Task Head → Task Output
Text Input   → Text Encoder   → Text Features   ↗
```

## 2. Core Concepts: The Input/Output Contract

All VLM heads expect inputs in a dictionary format. The specific keys required depend on the task.

**Input Contract:**
```python
# A dictionary of feature tensors from the foundation encoders.
head_inputs = {
    # Vision features, can be pooled or a sequence of patches/regions
    # Shape: [Batch, Dim] or [Batch, Num_Patches, Dim]
    'vision_features': vision_tensor,

    # Text features, can be pooled or a sequence of tokens
    # Shape: [Batch, Dim] or [Batch, Seq_Len, Dim]
    'text_features': text_tensor,

    # VQAHead reads this key INSTEAD of 'text_features'; the other three
    # heads read 'text_features'. A MultiTaskVLMHead mixing VQA with another
    # task therefore needs both keys.
    'question_features': text_tensor
}
```

**Output Contract:**
Heads always return a dictionary containing task-specific tensors.

| Task | Output Dictionary |
|:---|:---|
| **Image Captioning** | `{'logits': [B, Seq, Vocab], 'hidden_states': [B, Seq, Dim]}` |
| **VQA** | `{'answer_logits': [B, Num_Answers]}` |
| **Visual Grounding** | `{'bbox': [B, 4], 'confidence': [B, Regions], 'grounded_features': [B, Dim]}` |
| **Matching** | `{'similarity_matrix': [B, B], 'logits': [B, B], 'match_score': [B], 'vision_embeddings': [B, ProjDim], 'text_embeddings': [B, ProjDim]}` |

**No head squashes a class-like score.** Every score above is a pre-activation logit, including the
two whose key names read like probabilities: grounding's `'confidence'` (one logit per region, from a
linear `Dense(1)`) and matching's `'match_score'` (one logit per pair, from a linear `Dense(1)`).
Neither is confined to `[0, 1]`. Train them with `from_logits=True`, or apply `keras.ops.sigmoid`
yourself. The grounding region pick is `argmax` over the logits, which is the same region `argmax`
over their sigmoid would pick. The one squashed output is grounding's `'bbox'`: a range map, passed
through `ops.sigmoid` so the box lies in `(0, 1)`. Archives saved before 2026-10-15, when both
scorers were `Dense(1, activation='sigmoid')`, rebuild with linear scorers; their saved weights load,
but the outputs are then logits, not the sigmoid values they were trained to emit.

---

## 3. Catalog of Task Heads

The factory function `create_vlm_head` automatically selects the correct head class based on the `VLMTaskType`.

### 3.1. Image Captioning
*   **Task Type**: `IMAGE_CAPTIONING`
*   **Head Class**: `ImageCaptioningHead` (Standalone Architecture)
*   **Description**: A full, autoregressive Transformer decoder that generates a text sequence conditioned on vision features.
*   **Inputs**: `vision_features` `[B, Patches, Dim]`, `text_features` (embedded partial caption) `[B, Seq, Dim]`.
*   **Key Config**: Requires `vocab_size`. Configured with `num_layers`, `num_heads`.

```python
config = VLMTaskConfig(name="coco_caption", task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=50257)
head = create_vlm_head(config, num_layers=6, num_heads=12)
```

### 3.2. Visual Question Answering (VQA)
*   **Task Type**: `VISUAL_QUESTION_ANSWERING`
*   **Head Class**: `VQAHead` (Standalone Architecture)
*   **Description**: A custom classification model that pools vision and text features and passes them through an MLP to predict an answer from a fixed vocabulary.
*   **Inputs**: `vision_features` `[B, Patches, Dim]`, `question_features` `[B, Seq, Dim]`.
*   **Key Config**: Requires `num_classes`. Configured with `pooling_strategy`, `hidden_dims`.

```python
config = VLMTaskConfig(name="vqa_v2", task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=3129)
head = create_vlm_head(config, pooling_strategy="attention", hidden_dims=[512, 256])
```

### 3.3. Visual Grounding
*   **Task Type**: `VISUAL_GROUNDING`
*   **Head Class**: `VisualGroundingHead` (Fusion-Based)
*   **Description**: Fuses a text query with each visual region feature, scores each region for relevance (one logit per region under `'confidence'`), and regresses a bounding box from the top-scoring region's fused features (sigmoid range map under `'bbox'`).
*   **Inputs**: `vision_features` (unpooled regions) `[B, Regions, Dim]`, `text_features` `[B, Seq, Dim]`.
*   **Key Config**: Configured with `fusion_strategy` and `fusion_config`. Defaults to `'gated'` fusion. It runs no post-fusion stack, so it takes none of the post-fusion options in §4.2 (each raises `ValueError` as an unrecognized keyword).

```python
config = VLMTaskConfig(name="refcoco", task_type=VLMTaskType.VISUAL_GROUNDING)
head = create_vlm_head(config, fusion_strategy='gated')
```

### 3.4. Image-Text Matching
*   **Task Type**: `IMAGE_TEXT_MATCHING`
*   **Head Class**: `ImageTextMatchingHead` (Fusion-Based)
*   **Description**: A dual-purpose head. It projects features for CLIP-style contrastive alignment and separately fuses features to produce a fine-grained matching logit (`'match_score'`) for a single image-text pair.
*   **Inputs**: `vision_features` and `text_features`, each `[B, Dim]` (used as given) or `[B, Seq, Dim]` (mean-pooled over axis 1 first). Both must be `hidden_size` wide for the fusion.
*   **Key Config**: Configured with `projection_dim`, `temperature`, `fusion_strategy`, and the post-fusion options in §4.2.

```python
config = VLMTaskConfig(name="retrieval", task_type=VLMTaskType.IMAGE_TEXT_MATCHING)
head = create_vlm_head(config, projection_dim=256, fusion_strategy='concatenation')
```
---

## 4. Configuration Reference

### 4.1. Standalone Head Configurations
These heads have unique parameters for their specialized architectures.

| Head Class | Option | Type | Default | Description |
|:---|:---|:---|:---|:---|
| `ImageCaptioningHead`| `num_layers` | `int` | `6` | Number of decoder layers in the Transformer. |
| | `num_heads` | `int` | `12`| Number of attention heads in each decoder layer. |
| | `ffn_type` | `Literal[str]`| `'swiglu'`| FFN type for decoder blocks. See FFN table below. |
| `VQAHead` | `pooling_strategy`| `Literal[str]`| `'attention'`| How to pool vision/text sequences. Options: `'mean'`, `'max'`, `'attention'`. |
| | `hidden_dims` | `List[int]` | `[512, 256]`| A list of hidden layer sizes for the final classifier MLP. |

### 4.2. Fusion-Based Head Configurations (`BaseVLMHead`, `ImageTextMatchingHead`)
`BaseVLMHead` owns the `MultiModalFusion` layer and nothing else, and takes `fusion_strategy` and `fusion_config`; both fusion-based heads accept them. The post-fusion options (`normalization_type`, `use_post_fusion_ffn`, `ffn_type`, `ffn_expansion_factor`) belong to `ImageTextMatchingHead` alone, the one head whose `call` runs a post-fusion norm and FFN. No head takes a declared feature width: every width comes from the input shapes at `build`, and `hidden_size` from the `VLMTaskConfig`.

| Option | Type | Default | Description |
|:---|:---|:---|:---|
| `fusion_strategy` | `FusionStrategy`| `'gated'` (grounding), `'concatenation'` (matching) | The core fusion mechanism. Six strategies work: `'concatenation'`, `'addition'`, `'multiplication'`, `'gated'`, `'bilinear'`, `'tensor_fusion'`. `'cross_attention'` (one output per modality) and `'attention_pooling'` (rank-2 output) raise `ValueError` at build: the heads need one rank-preserving tensor. The `BaseVLMHead` signature default `'cross_attention'` is therefore never usable; both subclasses override it. |
| `fusion_config` | `Dict` | `{}` | A dictionary of fine-grained parameters for the chosen `fusion_strategy`. |
| `normalization_type`| `NormalizationType`| `'layer_norm'`| `ImageTextMatchingHead` only. Normalization after the fusion. Options include:<br>`'layer_norm'`, `'rms_norm'`, `'batch_norm'`, `'zero_centered_rms_norm'`, etc. |
| `use_post_fusion_ffn`| `bool` | `True` | `ImageTextMatchingHead` only. Runs a Transformer-style FFN after the post-fusion norm; `False` creates none. |
| `ffn_type` | `FFNType` | `'mlp'` | `ImageTextMatchingHead` only. Type of post-fusion FFN. Options include:<br>`'mlp'`, `'swiglu'`, `'geglu'`, `'gated_mlp'`, `'differential'`, `'residual'`, etc. |
| `ffn_expansion_factor`| `int`| `4`| `ImageTextMatchingHead` only. Hidden dimension expansion factor for the post-fusion FFN. |

Removed on 2026-10-15, with no replacement and no compatibility path: `vision_dim` and `text_dim` (on every head) and `MultiTaskVLMHead`'s `shared_vision_dim`/`shared_text_dim` (stored and serialized, read by nothing), `activation_type` (read by nothing), and the post-fusion `Dropout` (created on both fusion-based heads, applied by neither `call`). Passing any of these arguments now raises `ValueError`, and an archive whose config carries one no longer loads. Because `VisualGroundingHead` no longer creates the post-fusion norm, Dropout and FFN it never ran, and `ImageTextMatchingHead` no longer creates the Dropout, both heads' fixed-seed initialization differs from before; for the same weights the outputs are unchanged (measured bit-identical, weights copied by path).

---

## 5. Deep Dive: Fusion Strategies

The choice of `fusion_strategy` for fusion-based heads is critical and task-dependent.

*   **`'cross_attention'`**: One modality (e.g., text) queries the other (e.g., vision), token by token. **Not usable by the two fusion-based heads**: it returns one tensor per modality, and both heads raise `ValueError` at build (so does `'attention_pooling'`, which returns a rank-2 tensor). `ImageCaptioningHead` and `VQAHead` do not inherit from `BaseVLMHead` and take no `fusion_strategy`.
*   **`'concatenation'`**: Simple, fast, and effective. It concatenates the pooled vision and text vectors. Excellent for matching and retrieval tasks where the subsequent MLP can learn the interactions.
*   **`'gated'`**: Learns a dynamic "gate" to weigh the importance of each modality before combining them. Useful when one modality might be more informative than the other depending on the sample (e.g., visual grounding).
*   **`'bilinear'`**: Models second-order interactions between features via an explicit outer product of the two modality vectors, flattened and projected. This is the only strategy here that forms a genuine cross-modal product, and it is the **most expensive**: the projection weight is `dim^2 x dim`, i.e. cubic in `dim` (262,336 params at `dim=64`, vs 8,384 for `'concatenation'`).
*   **`'tensor_fusion'`**: **Not an outer product**, despite the name it inherits from Zadeh et al. (2017). `MultiModalFusion._call_tensor_fusion` concatenates the modalities on the feature axis, applies `num_tensor_projections` (default 8) parallel `Dense(dim, activation)` layers to that same concatenation, concatenates their outputs and projects back to `dim` — a one-hidden-layer MLP over the concatenation, with a hidden width of `num_tensor_projections * dim`. Up to that extra width and the final linear map it is the **same model as `'concatenation'`** (which additionally applies normalization and dropout, where `'tensor_fusion'` applies neither), so choose it for capacity, not for a different kind of interaction. Cost is quadratic in `dim` like everything here except `'bilinear'`: MEASURED at 2 modalities, `dim=64`, default settings — `concatenation` 8,384 params, `tensor_fusion` 98,880, `bilinear` 262,336. It is ~12x `'concatenation'` but *cheaper* than `'bilinear'` for any `dim > 24`. It also requires all modalities to share one sequence length (it raises a named `ValueError` otherwise); use `'cross_attention'` or `'attention_pooling'` when they differ.

---

## 6. Multi-Task Integration Example

The `create_multi_task_vlm_head` factory simplifies building complex VLMs with a shared backbone. Crucially, it allows you to pass task-specific configurations for each head type.

```python
from dl_techniques.layers.heads.vlm import create_multi_task_vlm_head, VLMTaskConfig, VLMTaskType

# Step 1: Define multiple VLM task configs
task_configs = {
    'caption': VLMTaskConfig(name='caption', task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=50000),
    'vqa': VLMTaskConfig(name='vqa', task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=3000),
    'matching': VLMTaskConfig(name='matching', task_type=VLMTaskType.IMAGE_TEXT_MATCHING)
}

# Step 2: Create multi-task head with task-specific overrides
multi_head = create_multi_task_vlm_head(
    task_configs=task_configs,
    # Any other keyword is shared, routed to every head that accepts it.
    # Pass parameters for specific head types via this dictionary
    task_specific_kwargs={
        'caption': {'num_layers': 4, 'num_heads': 8}, # Config for ImageCaptioningHead
        'vqa': {'pooling_strategy': 'mean', 'hidden_dims': [512]}, # Config for VQAHead
        'matching': {
            'projection_dim': 256, # Specific to ImageTextMatchingHead
            'fusion_strategy': 'concatenation' # Config for the BaseVLMHead part
        }
    }
)

# Step 3: Integrate into a model
# ... (model definition as in previous examples)
```

---

## 7. Training & Best Practices

### Loss Functions
Each VLM task requires a specific loss function.

*   **Captioning**: `SparseCategoricalCrossentropy` on token logits, often with label smoothing.
*   **VQA**: `SparseCategoricalCrossentropy` on answer logits for classification over a fixed answer set.
*   **Grounding**: A combination of a bounding box regression loss (e.g., GIoU loss) on `'bbox'`, which is already in `(0, 1)`, and a scoring loss on the region logits, e.g. `BinaryCrossentropy(from_logits=True)` on `'confidence'`.
*   **Matching (Contrastive)**: InfoNCE loss applied to the `logits` (temperature-scaled similarity matrix).
*   **Matching (Fine-grained)**: `BinaryCrossentropy(from_logits=True)` on `'match_score'`, which is a logit.

### Best Practices
1.  **Start with Pretrained Encoders**: Always initialize your vision and text encoders with pretrained weights (e.g., from CLIP, ViT, BERT) for faster convergence and better performance.
2.  **Gradual Unfreezing**: For the first few epochs, train only the newly added VLM heads while keeping the foundation encoders frozen. Then, unfreeze the encoders and fine-tune the entire model at a lower learning rate.
3.  **Data Balancing**: In a multi-task setting, carefully balance the data sampling ratio across tasks to prevent one task from dominating the training process.
4.  **Feature Projection**: If your vision and text encoders produce features of different dimensions, add a `keras.layers.Dense` projection layer to map one or both to a common dimension *before* passing them to the VLM head.