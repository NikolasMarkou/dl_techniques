# Core Transformer Blocks

The `dl_techniques.layers.transformers` module provides a collection of high-level, configurable building blocks for creating state-of-the-art Transformer architectures. This includes the foundational `TransformerLayer` as well as complete, modality-specific encoder and decoder stacks for vision and text.

## Overview

This module is designed around a core philosophy of modularity and configurability. Each component utilizes factory patterns for its internal mechanisms (e.g., attention, normalization, FFNs), allowing for the seamless construction of a wide variety of models, from standard ViT and BERT architectures to modern variants with advanced components like RMSNorm, SwiGLU, and Mixture of Experts.

This module provides the essential components to build everything from a standard language model to a custom Vision Transformer, all within a unified and maintainable framework.

The primary components are:

-   **`TransformerLayer`**: The foundational building block for all Transformer models.
-   **`VisionEncoder`**: A complete ViT-style encoder for image processing.
-   **`TextEncoder`**: A complete BERT-style bidirectional encoder for natural language understanding.
-   **`TextDecoder`**: A complete GPT-style autoregressive decoder for natural language generation.
-   **Specialized Blocks**: A collection of advanced layers like `SwinTransformerBlock`, `SwinConvBlock`, `PerceiverTransformerLayer`, and `EomtTransformer` for specific architectural needs.

## TransformerLayer

The `dl_techniques.layers.transformers.transformer.TransformerLayer` provides a highly configurable and foundational building block for constructing Transformer-based architectures. It encapsulates a self-attention mechanism and a feed-forward network, each wrapped with residual connections and normalization, forming a single, reusable processing unit for sequence data.

### Overview

This layer is the workhorse of any Transformer model. Its primary function is to refine a sequence of input embeddings by allowing each element in the sequence to interact with every other element, thereby producing a contextually rich output sequence of the same shape. The core design philosophy is modularity and configurability, enabling the seamless integration of various state-of-the-art components.

### Architectural Highlights

-   **Factory-Based Components**: Utilizes the attention and FFN factories to allow for dynamic selection of mechanisms like standard Multi-Head Attention, Window Attention, SwiGLU, or even a Mixture of Experts (MoE) block.
-   **Flexible Normalization**: Supports multiple normalization strategies (`LayerNorm`, `RMSNorm`, etc.) and positions (pre-norm vs. post-norm), allowing for replication of various architectures and improved training stability.
-   **Stochastic Depth**: Integrates optional stochastic depth for regularization, a technique proven to improve the performance and generalization of deep Transformer models by randomly dropping entire residual blocks during training.
-   **Fine-Grained Control**: Exposes dedicated argument dictionaries (`attention_args`, `norm_args`, `ffn_args`) for passing custom configurations to child components, enabling advanced and precise architectural tuning.

### Usage

#### Standard Pre-Norm Transformer Layer

```python
import keras
from dl_techniques.layers.transformers.transformer import TransformerLayer

inputs = keras.Input(shape=(128, 512))
transformer_block = TransformerLayer(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,
    normalization_position='pre',
    ffn_type='swiglu',
    use_stochastic_depth=True,
    stochastic_depth_rate=0.1
)
outputs = transformer_block(inputs)
```

#### Layer with Mixture of Experts (MoE)

```python
from dl_techniques.layers.moe import MoEConfig

moe_config = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "swiglu",
            "output_dim": 512,
            "ffn_expansion_factor": 4
        }
    ),
    gating_config=GatingConfig(top_k=2)
)

moe_transformer_block = TransformerLayer(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,  # This will be ignored
    moe_config=moe_config
)
outputs_moe = moe_transformer_block(inputs)
```

### Arguments

| Argument                 | Type                | Description                                                                                             | Default        |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------- | -------------- |
| `hidden_size`            | `int`               | **Required.** The dimensionality of the input and output embeddings.                                      |                |
| `num_heads`              | `int`               | **Required.** The number of parallel attention heads.                                                     |                |
| `intermediate_size`      | `int`               | **Required.** The dimensionality of the FFN's hidden layer.                                               |                |
| `attention_type`         | `str`               | The type of attention mechanism to use (e.g., `'multi_head'`, `'window'`).                              | `'multi_head'` |
| `attention_args`         | `dict`              | Optional dictionary of custom arguments for the attention layer.                                        | `None`         |
| `normalization_type`     | `str`               | The type of normalization to use (e.g., `'layer_norm'`, `'rms_norm'`).                                  | `'layer_norm'` |
| `normalization_position` | `str`               | The position of the normalization layer (`'pre'` or `'post'`).                                          | `'post'`       |
| `ffn_type`               | `str`               | The type of feed-forward network to use (e.g., `'mlp'`, `'swiglu'`). Ignored if `moe_config` is set.      | `'mlp'`        |
| `moe_config`             | `dict` or `MoEConfig` | Configuration for a Mixture of Experts layer. If provided, replaces the FFN.                            | `None`         |
| `use_stochastic_depth`   | `bool`              | If `True`, enables stochastic depth regularization.                                                       | `False`        |
| `stochastic_depth_rate`  | `float`             | The drop probability for stochastic depth.                                                              | `0.1`          |

## VisionEncoder

The `dl_techniques.layers.transformers.vision_encoder.VisionEncoder` is a configurable, general-purpose Vision Transformer (ViT) encoder. It provides a modular framework for processing images as sequences of patches, serving as a powerful backbone for a wide range of computer vision tasks.

### Overview

The Vision Encoder implements the core ViT architecture, which adapts the Transformer model for image data. This is achieved by first converting an input image into a sequence of flattened patches. Each patch is treated as a "token," analogous to a word in a sentence. This sequence is then processed by a stack of standard `TransformerLayer` blocks, which allows the model to learn complex spatial relationships and long-range dependencies across the entire image.

### Architectural Highlights

-   **Configurable Patch Embeddings**: Supports multiple strategies for converting image patches into embeddings, including the standard linear projection from ViT, the two-stage approach from SigLIP, and more complex convolutional stems.
-   **Factory-Based Composition**: Leverages the `TransformerLayer` and component factories (attention, ffn, norm) to allow for the construction of diverse ViT variants within a single, unified class.
-   **Optional `[CLS]` Token**: Includes a learnable class token that can be prepended to the patch sequence. The final state of this token is often used as a global image representation for classification tasks.
-   **Flexible Output Modes**: Integrated with a `SequencePooling` layer to provide various output formats, such as the `[CLS]` token features, globally averaged patch features, or the full sequence of patch embeddings for dense prediction tasks.

### Usage

#### Standard ViT-Base Configuration

```python
import keras
from dl_techniques.layers.transformers.vision_encoder import VisionEncoder

# A standard ViT-B/16 model
vit_encoder = VisionEncoder(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    patch_embed_type='linear',
    attention_type='multi_head',
    output_mode='cls'
)
inputs = keras.random.normal(shape=(2, 224, 224, 3))
cls_output = vit_encoder(inputs) # Shape: (2, 768)
```

#### Modern, Efficient Configuration

```python
# A more modern and efficient architecture
efficient_encoder = VisionEncoder(
    img_size=256,
    patch_size=16,
    embed_dim=384,
    depth=8,
    num_heads=6,
    patch_embed_type='siglip',
    attention_type='window',
    normalization_type='rms_norm',
    normalization_position='pre',
    ffn_type='swiglu',
    output_mode='mean',
    attention_args={'window_size': 7}
)
mean_pooled_output = efficient_encoder(keras.random.normal(shape=(2, 256, 256, 3))) # Shape: (2, 384)
```

### Arguments

| Argument           | Type    | Description                                                                              | Default    |
| ------------------ | ------- | ---------------------------------------------------------------------------------------- | ---------- |
| `img_size`         | `int`   | The height and width of the input image.                                                 | `224`      |
| `patch_size`       | `int`   | The size of each square patch. `img_size` must be divisible by `patch_size`.               | `16`       |
| `embed_dim`        | `int`   | The dimensionality of the patch embeddings and hidden states.                            | `768`      |
| `depth`            | `int`   | The number of `TransformerLayer` blocks in the encoder stack.                              | `12`       |
| `num_heads`        | `int`   | The number of attention heads in each `TransformerLayer`.                                  | `12`       |
| `patch_embed_type` | `str`   | The patch embedding strategy: `'linear'`, `'siglip'`, `'conv'`, or `'hybrid'`.             | `'linear'` |
| `use_cls_token`    | `bool`  | If `True`, prepends a learnable `[CLS]` token to the sequence.                             | `True`     |
| `output_mode`      | `str`   | The pooling strategy for the final output: `'cls'`, `'mean'`, `'max'`, or `'none'`.        | `'cls'`    |

## TextEncoder

The `dl_techniques.layers.transformers.text_encoder.TextEncoder` is a configurable, Transformer-based encoder for natural language processing. It is designed to create deep, bidirectional representations of text, making it an ideal backbone for tasks like text classification, named entity recognition, and question answering.

### Overview

This layer implements the encoder-side of the Transformer architecture, famously popularized by models like BERT. Its core principle is to process an entire sequence of text simultaneously, allowing every token to attend to every other token. This bidirectional self-attention mechanism enables the model to build a contextually rich understanding of each token based on its complete surrounding text.

### Architectural Highlights

-   **Configurable Embeddings**: Supports multiple strategies for both word embeddings (`learned`, `shared`, `factorized`) and positional encodings (`learned`, `rope`, `sincos`), enabling the construction of both classic and modern architectures.
-   **Token Type Support**: Includes optional support for token type (segment) embeddings, essential for tasks involving multiple text segments, such as sentence-pair classification or question answering.
-   **Factory-Based Composition**: Built upon a stack of `TransformerLayer` blocks, inheriting their full configurability for attention, normalization, and FFN mechanisms.
-   **Flexible Output Modes**: Integrated with a `SequencePooling` layer to provide various output formats suitable for different downstream tasks, such as using the `[CLS]` token for classification or mean pooling for sentence embeddings.

### Usage

#### Standard BERT-like Configuration

```python
import keras
from dl_techniques.layers.transformers.text_encoder import TextEncoder

# A standard BERT-style model configuration
bert_encoder = TextEncoder(
    vocab_size=30522,
    embed_dim=768,
    depth=12,
    num_heads=12,
    max_seq_len=512,
    positional_type='learned',
    use_token_type_embedding=True,
    use_cls_token=True,
    output_mode='cls'
)
input_ids = keras.random.randint(0, 30522, shape=(2, 128))
token_type_ids = keras.ops.zeros_like(input_ids)
cls_output = bert_encoder({'input_ids': input_ids, 'token_type_ids': token_type_ids}) # Shape: (2, 768)
```

#### Modern Encoder with RoPE and SwiGLU

```python
# A modern encoder with advanced components
modern_encoder = TextEncoder(
    vocab_size=50000,
    embed_dim=512,
    depth=8,
    num_heads=8,
    max_seq_len=1024,
    positional_type='rope',
    attention_type='multi_head',
    normalization_type='rms_norm',
    normalization_position='pre',
    ffn_type='swiglu',
    output_mode='mean'
)
mean_pooled_output = modern_encoder(keras.random.randint(0, 50000, shape=(2, 256))) # Shape: (2, 512)
```

### Arguments

| Argument                   | Type    | Description                                                                              | Default     |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------- | ----------- |
| `vocab_size`               | `int`   | **Required.** The size of the token vocabulary.                                          |             |
| `embed_dim`                | `int`   | **Required.** The dimensionality of the token embeddings and hidden states.              |             |
| `depth`                    | `int`   | The number of `TransformerLayer` blocks in the encoder stack.                              | `12`        |
| `num_heads`                | `int`   | The number of attention heads in each `TransformerLayer`.                                  | `12`        |
| `max_seq_len`              | `int`   | The maximum sequence length the model can process.                                       | `512`       |
| `embedding_type`           | `str`   | The word embedding strategy: `'learned'`, `'shared'`, or `'factorized'`.                   | `'learned'` |
| `positional_type`          | `str`   | The positional encoding strategy: `'learned'`, `'rope'`, `'sincos'`, etc.                  | `'learned'` |
| `use_token_type_embedding` | `bool`  | If `True`, adds token type (segment) embeddings.                                         | `False`     |
| `use_cls_token`            | `bool`  | If `True`, prepends a learnable `[CLS]` token to the sequence.                             | `False`     |
| `output_mode`              | `str`   | The pooling strategy for the final output: `'cls'`, `'mean'`, `'none'`, etc.               | `'none'`    |

## TextDecoder

The `dl_techniques.layers.transformers.text_decoder.TextDecoder` is a configurable, Transformer-based decoder stack designed for autoregressive language modeling. It encapsulates the core components of a decoder-only model, including token embeddings, positional encodings, and a stack of causally-masked self-attention blocks.

### Overview

This layer implements the decoder-side of the Transformer architecture, which forms the basis of modern Large Language Models (LLMs) like GPT. Its primary purpose is to model the probability of the next token in a sequence given all previous tokens. This is achieved through causal self-attention, where a "look-ahead" mask is applied to prevent any token from attending to future tokens.

### Architectural Highlights

-   **Automatic Causal Masking**: The layer automatically generates and applies the necessary causal mask to the self-attention mechanism, ensuring the autoregressive property is maintained. It can also seamlessly combine this with a user-provided padding mask.
-   **Configurable Embeddings**: Supports multiple strategies for both word embeddings (`learned`, `shared`, `factorized`) and positional encodings (`learned`, `sincos`).
-   **Factory-Based Composition**: Built upon a stack of `TransformerLayer` blocks, inheriting their full configurability for attention, normalization, and FFN mechanisms.
-   **Designed for Generation**: The output is the full sequence of contextually-aware token representations, which can be directly fed into a final linear layer to produce logits for next-token prediction.

### Usage

#### Basic GPT-style Decoder

```python
import keras
from dl_techniques.layers.transformers.text_decoder import TextDecoder

# A standard decoder configuration similar to GPT-2
gpt_decoder = TextDecoder(
    vocab_size=50257,
    embed_dim=768,
    depth=12,
    num_heads=12,
    max_seq_len=1024,
    positional_type='learned',
    normalization_position='pre'
)
input_ids = keras.random.randint(0, 50257, shape=(2, 64))
output_features = gpt_decoder(input_ids) # Shape: (2, 64, 768)
```

#### Modern Decoder with Advanced Components

```python
# A modern decoder with RMSNorm, SwiGLU, and sinusoidal positions
modern_decoder = TextDecoder(
    vocab_size=32000,
    embed_dim=512,
    depth=8,
    num_heads=8,
    max_seq_len=2048,
    positional_type='sincos',
    normalization_type='rms_norm',
    normalization_position='pre',
    ffn_type='swiglu',
    stochastic_depth_rate=0.1
)
output_features_modern = modern_decoder(keras.random.randint(0, 32000, shape=(2, 128))) # Shape: (2, 128, 512)
```

### Arguments

| Argument        | Type  | Description                                                                              | Default     |
| --------------- | ----- | ---------------------------------------------------------------------------------------- | ----------- |
| `vocab_size`    | `int` | **Required.** The size of the token vocabulary.                                          |             |
| `embed_dim`     | `int` | **Required.** The dimensionality of the token embeddings and hidden states.              |             |
| `depth`         | `int` | **Required.** The number of `TransformerLayer` blocks in the decoder stack.              |             |
| `num_heads`     | `int` | **Required.** The number of attention heads in each `TransformerLayer`.                    |             |
| `max_seq_len`   | `int` | The maximum sequence length for positional embeddings.                                   | `512`       |
| `embedding_type`| `str` | The word embedding strategy: `'learned'`, `'shared'`, or `'factorized'`.                   | `'learned'` |
| `positional_type`| `str` | The positional encoding strategy: `'learned'` or `'sincos'`.                             | `'learned'` |

## Specialized and Hybrid Blocks

This section covers advanced and specialized Transformer layers for specific architectural needs, including vision-specific blocks and hybrid models.

### SwinTransformerBlock

The `dl_techniques.layers.transformers.swin_transformer_block.SwinTransformerBlock` is the core building block of the Swin Transformer architecture. It introduces an efficient window-based self-attention mechanism to achieve linear complexity with respect to image size.

#### Overview

Unlike standard ViTs that compute global self-attention, the Swin Transformer block performs attention within non-overlapping local windows. To enable cross-window communication, it alternates between regular Windowed Multi-Head Self-Attention (W-MSA) and Shifted Window MSA (SW-MSA) in successive blocks. This hierarchical approach makes it highly effective for dense prediction tasks.

#### Architectural Highlights

-   **Windowed Attention**: Restricts self-attention computation to local windows, significantly reducing computational cost.
-   **Shifted Windows**: Enables information flow between adjacent windows across layers, effectively modeling global context.
-   **Linear Complexity**: Achieves computational complexity that scales linearly with the number of image patches, making it suitable for high-resolution images.

#### Usage

```python
import keras
from dl_techniques.layers.transformers.swin_transformer_block import SwinTransformerBlock

inputs = keras.Input(shape=(56, 56, 96))

# A standard windowed attention block
block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=0,  # No shift
)
x = block(inputs)

# A shifted window attention block
shifted_block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=7 // 2,  # Shift by half window size
    stochastic_depth_rate=0.1
)
x_shifted = shifted_block(x)
```

### Arguments

| Argument        | Type    | Description                                                                    | Default |
| --------------- | ------- | ------------------------------------------------------------------------------ | ------- |
| `dim`           | `int`   | **Required.** The dimensionality of the input channels.                        |         |
| `num_heads`     | `int`   | **Required.** The number of attention heads.                                   |         |
| `window_size`   | `int`   | The size of the attention window.                                              | `8`     |
| `shift_size`    | `int`   | The shift size for SW-MSA. `0` for regular W-MSA.                                | `0`     |
| `mlp_ratio`     | `float` | The expansion ratio for the MLP's hidden dimension.                            | `4.0`   |
| `stochastic_depth_rate` | `float` | The stochastic depth (drop-path) rate.                                  | `0.0`   |

### SwinConvBlock

The `dl_techniques.layers.transformers.swin_conv_block.SwinConvBlock` is a hybrid layer that combines the strengths of Swin Transformers and traditional Convolutional Neural Networks (CNNs).

#### Overview

This block processes features through two parallel pathways: a Swin Transformer path for modeling long-range dependencies and a standard convolutional path for efficient local feature extraction. The outputs are then fused, creating a powerful block that captures both local and global context within a residual framework.

#### Architectural Highlights

-   **Hybrid Design**: Leverages the inductive biases of CNNs and the global context modeling of Transformers in a single block.
-   **Parallel Processing**: Employs a "split-transform-merge" strategy to process features independently in convolutional and transformer streams.
-   **Enhanced Feature Representation**: Aims to learn a richer set of features than either CNNs or Transformers could alone.

#### Usage

```python
import keras
from dl_techniques.layers.transformers.swin_conv_block import SwinConvBlock

# Input channels must be conv_dim + trans_dim
inputs = keras.Input(shape=(56, 56, 128))

block = SwinConvBlock(
    conv_dim=64,
    trans_dim=64,
    head_dim=32,
    window_size=7,
    block_type="SW",  # Use shifted windows
    drop_path_rate=0.1
)
outputs = block(inputs)  # Shape: (None, 56, 56, 128)
```

### Arguments

| Argument       | Type    | Description                                                                    | Default |
| -------------- | ------- | ------------------------------------------------------------------------------ | ------- |
| `conv_dim`     | `int`   | **Required.** The number of channels for the convolutional path.               |         |
| `trans_dim`    | `int`   | **Required.** The number of channels for the transformer path.                 |         |
| `head_dim`     | `int`   | The dimension of each attention head.                                          | `32`    |
| `window_size`  | `int`   | The size of the attention window in the transformer path.                      | `8`     |
| `block_type`   | `str`   | The type of Swin block: `'W'` (Window) or `'SW'` (Shifted Window).             | `'W'`   |
| `drop_path_rate` | `float` | The stochastic depth rate.                                                   | `0.0`   |

### PerceiverTransformerLayer

The `dl_techniques.layers.transformers.perceiver_transformer.PerceiverTransformerLayer` implements a Perceiver-style transformer block that uses decoupled cross-attention to handle very large input sequences efficiently.

#### Overview

Standard transformers suffer from quadratic complexity with respect to sequence length. The Perceiver architecture addresses this by using a small, fixed-size latent array to query a much larger input array. This asymmetric cross-attention mechanism reduces complexity from `O(N²)` to `O(M*N)`, where `M` is the small latent size and `N` is the large input size. This makes it possible to process high-dimensional data like raw pixels or audio.

#### Architectural Highlights

-   **Asymmetric Cross-Attention**: Decouples queries (from a small latent array) from keys and values (from a large input array).
-   **Linear Complexity**: Scales linearly with the input sequence length, enabling processing of massive inputs.
-   **Information Bottleneck**: The latent array acts as a bottleneck, forcing the model to distill the most salient information from the input.

#### Usage

```python
import keras
from dl_techniques.layers.transformers.perceiver_transformer import PerceiverTransformerLayer

latents = keras.random.normal(shape=(2, 256, 512))    # Small latent array (M=256)
byte_array = keras.random.normal(shape=(2, 4096, 512)) # Large input array (N=4096)

perceiver_block = PerceiverTransformerLayer(dim=512, num_heads=8)

# Latents attend to the byte array to distill information
updated_latents = perceiver_block(query_input=latents, kv_input=byte_array)
print(updated_latents.shape) # (2, 256, 512)
```

### Arguments

| Argument        | Type    | Description                                                                    | Default |
| --------------- | ------- | ------------------------------------------------------------------------------ | ------- |
| `dim`           | `int`   | **Required.** The hidden dimension of the block.                               |         |
| `num_heads`     | `int`   | The number of attention heads.                                                 | `8`     |
| `mlp_ratio`     | `float` | The expansion ratio for the MLP's hidden dimension.                            | `4.0`   |
| `dropout`       | `float` | The dropout rate for attention and MLP.                                        | `0.0`   |

### EomtTransformer

The `dl_techniques.layers.transformers.eomt_transformer.EomtTransformer` is a specialized layer from the "Encoder-only Mask Transformer" designed for instance segmentation.

#### Overview

This layer adapts a standard transformer encoder to simultaneously process a concatenated sequence of image patch tokens and learnable object query tokens. Its key feature is a **masked self-attention** mechanism that can be enabled during training. When active, it uses ground-truth segmentation masks to force each object query to attend only to the image patches corresponding to its assigned object instance, providing a strong supervisory signal for learning object-centric representations.

#### Architectural Highlights

-   **Joint Patch-Query Processing**: Processes image patches and object queries in a single sequence, allowing for rich, bidirectional information flow.
-   **Masked Self-Attention**: Provides explicit guidance during training by masking attention scores based on ground-truth object locations.
-   **Probabilistic Masking**: Can apply the attention mask probabilistically, with optional annealing, to create a curriculum learning effect.
-   **Configurable Base**: Built upon the highly flexible `TransformerLayer`, inheriting its configurability.

#### Usage

```python
import keras
from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer

# Joint sequence of 196 patch tokens and 100 query tokens
inputs = keras.random.normal(shape=(2, 296, 768))
# Ground truth masks for the 100 queries
masks = keras.random.uniform(shape=(2, 100, 56, 56))

# Instantiate with masked attention enabled
eomt_layer = EomtTransformer(
    hidden_size=768,
    num_heads=12,
    use_masked_attention=True,
    mask_probability=0.8,
    mask_annealing_steps=10000
)

# During training, provide both inputs and masks
outputs = eomt_layer(inputs={'inputs': inputs, 'mask': masks}, training=True)
```

### Arguments

| Argument                 | Type    | Description                                                                    | Default      |
| ------------------------ | ------- | ------------------------------------------------------------------------------ | ------------ |
| `hidden_size`            | `int`   | **Required.** The dimensionality of the input and output embeddings.           |              |
| `num_heads`              | `int`   | The number of attention heads.                                                 | `8`          |
| `use_masked_attention`   | `bool`  | If `True`, enables the segmentation-specific masked attention mechanism.       | `False`      |
| `mask_probability`       | `float` | The probability of applying the mask during a training step.                   | `1.0`        |
| `mask_annealing_steps`   | `int`   | Linearly anneals the mask probability from 0 to its target value over these steps. | `0`          |
| `...`                    |         | Inherits all other arguments from `TransformerLayer` (e.g., `ffn_type`).       |              |

## AdaLNZeroConditionalBlock

DiT-style adaptive layer-normalization "zero" conditional transformer block, adopted from Peebles & Xie 2023 and adapted in Sobal et al.'s LeWM. Two inputs per call: content `x` of shape `(B, T, D)` and conditioning `c` broadcastable to `x`. The conditioning drives six modulation streams (shift / scale / gate for attention and FFN sub-blocks) through a single SiLU-Linear projection whose final Dense is zero-initialized — so at initialization the block is the identity map in `x`. The four sublayer groups (norms, attention, FFN, AdaLN modulation activation) are factory-configurable; leaving every factory kwarg at default reproduces the original DiT/LeWM construction bit-exactly.

### Architecture

```text
x -----> Norm (no affine) --> modulate(shift_msa, scale_msa)
                                                   |
                                                   v
                         causal MultiHeadAttention (self-attn)
                                                   |
                                          gate_msa * (.)
                                                   |
x = x + gate_msa * attn(...)  <-------------------+

x -----> Norm (no affine) --> modulate(shift_mlp, scale_mlp)
                                                   |
                                                   v
                                      FFN (e.g. MLP)
                                                   |
                                          gate_mlp * (.)
                                                   |
x = x + gate_mlp * mlp(...)   <-------------------+
```

`modulate(h, shift, scale) = h * (1 + scale) + shift`. The six modulation tensors come from one Dense(6*dim) layer with zero-initialized kernel **and** bias.

### Usage (default — bit-exact DiT/LeWM)

```python
import keras
from dl_techniques.layers.transformers import AdaLNZeroConditionalBlock

dim, num_heads, dim_head, mlp_dim = 256, 4, 64, 1024

x_in = keras.Input(shape=(seq_len, dim), name="x")
c_in = keras.Input(shape=(seq_len, dim), name="c")
y = AdaLNZeroConditionalBlock(
    dim=dim, num_heads=num_heads, dim_head=dim_head, mlp_dim=mlp_dim,
    dropout=0.1, use_causal_mask=True, eps=1e-6,
)([x_in, c_in])
model = keras.Model([x_in, c_in], y)
```

### Usage (factory-swapped sublayers)

```python
# Swap LayerNorm for RMSNorm (must disable affine via use_scale=False).
block = AdaLNZeroConditionalBlock(
    dim=256, num_heads=4, dim_head=64, mlp_dim=1024,
    normalization_type="rms_norm",
    normalization_args={"use_scale": False, "epsilon": 1e-6},
)

# Swap MLP FFN for SwiGLU.
block = AdaLNZeroConditionalBlock(
    dim=256, num_heads=4, dim_head=64, mlp_dim=1024,
    ffn_type="swiglu",
    ffn_args={"output_dim": 256, "ffn_expansion_factor": 4, "ffn_multiple_of": 16},
)
```

**AdaLN-Zero affine invariant** (HARD requirement): the two normalization layers MUST have no learnable affine parameters — AdaLN's gate/shift/scale provides all per-channel modulation. The default `normalization_type=None` path enforces this with `center=False, scale=False`. For any custom `normalization_type` you MUST disable affine in `normalization_args` (e.g. RMSNorm: `use_scale=False`). The block does NOT silently override your args.

**Factory attention contract**: when `attention_type` is set, the chosen attention layer is invoked as `self.attn(h, training=...)` — no Q/K/V split and `use_causal_mask` is **not** forwarded (attention APIs vary). If you need causal masking via a factory-swapped attention, pass `use_causal_mask` via `attention_args` or pick an attention type whose internal masking semantics fit your task.

### Arguments

| Argument                  | Type             | Description                                                                                  | Default |
| ------------------------- | ---------------- | -------------------------------------------------------------------------------------------- | ------- |
| `dim`                     | `int`            | **Required.** Model (hidden) dimension.                                                      |         |
| `num_heads`               | `int`            | **Required.** Number of attention heads.                                                     |         |
| `dim_head`                | `int`            | **Required.** Per-head dimension for the default MultiHeadAttention.                         |         |
| `mlp_dim`                 | `int`            | **Required.** Hidden dimension of the FFN sub-block.                                         |         |
| `dropout`                 | `float`          | Dropout rate (default-path attention + FFN).                                                 | `0.0`   |
| `use_causal_mask`         | `bool`           | Apply causal self-attention mask in the default attention path.                              | `True`  |
| `eps`                     | `float`          | Norm epsilon (default `layer_norm` path).                                                    | `1e-6`  |
| `normalization_type`      | `Optional[str]`  | Factory normalization key (e.g. `"rms_norm"`). `None` → default `layer_norm` no-affine.      | `None`  |
| `normalization_args`      | `Optional[Dict]` | Kwargs forwarded to `create_normalization_layer` — must disable affine yourself if needed.   | `None`  |
| `attention_type`          | `Optional[str]`  | Factory attention key. `None` → default `keras.layers.MultiHeadAttention` (bit-exact).       | `None`  |
| `attention_args`          | `Optional[Dict]` | Kwargs forwarded to `create_attention_layer`.                                                | `None`  |
| `ffn_type`                | `Optional[str]`  | Factory FFN key (e.g. `"swiglu"`). `None` → default `mlp` via `MLPBlock`.                    | `None`  |
| `ffn_args`                | `Optional[Dict]` | Kwargs forwarded to `create_ffn_layer`.                                                      | `None`  |
| `adaln_activation_type`   | `Optional[str]`  | Activation identifier for the AdaLN modulation activation. `None` → `Activation("silu")`.    | `None`  |
| `adaln_activation_args`   | `Optional[Dict]` | Kwargs forwarded to `resolve_activation_layer`.                                              | `None`  |

References: Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023; Sobal et al., "Learning the World with Minimal Supervision" (LeWM), 2024.

## TransformerDecoderLayer

The `dl_techniques.layers.transformers.transformer_decoder.TransformerDecoderLayer` is the encoder-decoder counterpart of `TransformerLayer`. It performs masked/causal self-attention over the target sequence, cross-attention to encoder memory, and an FFN, each wrapped with residual connections and factory-configurable normalization (pre/post).

### Usage

```python
import keras
from dl_techniques.layers.transformers.transformer_decoder import TransformerDecoderLayer

decoder_block = TransformerDecoderLayer(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,
    normalization_position='pre',
    ffn_type='swiglu',
)
# target sequence + encoder memory
y = decoder_block(target_embeddings, encoder_output=memory)
```

It shares the foundational lifecycle/serialization contract of `TransformerLayer` (None-sentinel build dims, `if self.built: return` guard, explicit child build, full `get_config`).

## BinaryMapper / FreeTransformerLayer

The `dl_techniques.layers.transformers.free_transformer` module implements the FREE (Faster Resolution Encoder-decoder) latent-variable transformer.

-   **`BinaryMapper`**: maps continuous logits to a discrete latent index via per-bit Bernoulli sampling with a straight-through gradient estimator (Equation 8), producing a one-hot over `2^num_bits` categories.
-   **`FreeTransformerLayer`**: a causal transformer block with an optional non-causal encoder path that infers a latent `Z` from the sequence during training and uniform-samples it at inference.

> **Limitation (documented, not redesigned)**: the encoder path's cross-attention does not currently receive the sequence as separate K/V, so the posterior `Q(Z|S)` is unconditional on `S`. See the in-code `D-002` note. Use with awareness; no production model depends on this layer.

```python
from dl_techniques.layers.transformers.free_transformer import FreeTransformerLayer

block = FreeTransformerLayer(hidden_size=512, num_heads=8, intermediate_size=2048,
                             use_free_transformer=True, num_bits=8)
```

## PFTBlock

The `dl_techniques.layers.transformers.progressive_focused_transformer.PFTBlock` is the core block of the Progressive Focused Transformer (PFT-SR, CVPR 2025) for single-image super-resolution. It applies progressive-focused attention with shared focused-attention statistics across the block stack.

```python
from dl_techniques.layers.transformers.progressive_focused_transformer import PFTBlock

block = PFTBlock(dim=180, num_heads=6, window_size=16, mlp_ratio=2.0)
```

> **Note**: `pft_sr` imports `PFTBlock` from `progressive_focused_transformer` (the module name, not `progressive_focused_transformer_block`).

## EnergyTransformer / HopfieldNetwork

The `dl_techniques.layers.transformers.energy_transformer` module implements the Energy Transformer (ET) block of Hoover et al., NeurIPS 2023 ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253)).

ET **replaces the `attn -> FFN` residual stream with `T` steps of gradient descent on a single scalar energy**. Each step normalizes the token state with an `EnergyLayerNorm` (`g = dL/dx` of a Lagrangian with a PSD Hessian), evaluates the closed-form energy gradient, and takes an `alpha`-scaled step:

```
for t in 1..T:
    g = EnergyLayerNorm(x)
    x = x + step_size * (attn.update(g) + hopfield.update(g))     # update == -dE/dg
```

The energy is `E = E_ATT + E_HN`, contributed by two sub-layers:

-   **`EnergyAttention`** (`layers/attention/energy_attention.py`, factory key `'energy'`): token mixing. No value matrix; its `call()` returns the exact closed-form `-dE_ATT/dg`.
-   **`HopfieldNetwork`**: a per-token associative memory with a **single tied** `(hopfield_dim, dim)` matrix `xi` used in both directions and no bias — it is *not* FFN-shaped, which is why it is deliberately **not** registered in the FFN factory and lives beside the block that consumes it. `activation='relu'` (default, both of the paper's headline configs) or `'softmax'` (modern Hopfield), each with its own matching energy/gradient pair.

Because every `update()` is the analytic negative gradient of the energy the block reports, the energy is **provably non-increasing** across the recurrent steps (for `noise_std=0`, `gamma > 0`, and a small enough `step_size`). Both closed forms are verified against a `tf.GradientTape` autodiff oracle in the tests; the source itself is `keras.ops`-only.

### Usage

```python
from dl_techniques.layers.transformers import EnergyTransformer

# The paper's ImageNet ET-Full block config (Table 15): 3.54M params.
block = EnergyTransformer(
    embed_dim=768,
    num_heads=12,
    head_dim=64,
    hopfield_dim=3072,
    num_steps=12,
    step_size=0.1,
    attn_self=False,          # ET-Full: the attention diagonal is excluded
    hopfield_activation='relu',
)
y = block(tokens)             # (batch, seq_len, 768) -> same shape

# Inspect the descent: returns (x, energies) with energies of shape (batch, num_steps + 1).
probe = EnergyTransformer(
    embed_dim=768, num_heads=12, head_dim=64, hopfield_dim=3072,
    return_energy=True,
)
y, energies = probe(tokens)   # energies is monotonically non-increasing per sample
```

### Arguments

-   **`embed_dim`**, **`num_heads`**, **`head_dim`**, **`hopfield_dim`**: required dimensions.
-   **`num_steps`** (default `12`): `T`, the number of descent steps.
-   **`step_size`** (default `0.1`): `alpha`, the descent step. Too large and the discretized descent may overshoot.
-   **`beta`** (default `None` -> `1/sqrt(head_dim)`): the attention inverse temperature.
-   **`attn_self`** (default `False`): whether a token attends to itself (`False` = the paper's ET-Full).
-   **`hopfield_activation`** (default `'relu'`), **`hopfield_beta`** (default `1.0`, read only by `'softmax'` — it is a *second, independent* temperature over the `hopfield_dim` memories, not `beta`).
-   **`noise_std`** (default `0.0`): Langevin noise injected during training only (eq. 27). The descent guarantee is **not** claimed when this is non-zero.
-   **`return_energy`** (default `False`), **`norm_epsilon`** (default `1e-5`), **`seed`** (default `None`).
-   `call(inputs, attention_mask=None, training=None)`. A rank-2 `(B, N)` mask is a per-token *validity* mask applied symmetrically to the key and query axes (see the `energy` notes in `attention/README.md`).

> **Scope**: this is the ET **block** plus an optional mask and optional noise. There is no image or graph model wrapper (no patchify, no `MASK` token, no decoder).
