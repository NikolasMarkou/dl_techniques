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
    shift_size=0, # No shift
)
x = block(inputs)

# A shifted window attention block
shifted_block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=7 // 2, # Shift by half window size
    drop_path=0.1
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
| `drop_path`     | `float` | The stochastic depth rate.                                                     | `0.0`   |

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
    drop_path=0.1
)
outputs = block(inputs) # Shape: (None, 56, 56, 128)
```

### Arguments

| Argument       | Type    | Description                                                                    | Default |
| -------------- | ------- | ------------------------------------------------------------------------------ | ------- |
| `conv_dim`     | `int`   | **Required.** The number of channels for the convolutional path.               |         |
| `trans_dim`    | `int`   | **Required.** The number of channels for the transformer path.                 |         |
| `head_dim`     | `int`   | The dimension of each attention head.                                          | `32`    |
| `window_size`  | `int`   | The size of the attention window in the transformer path.                      | `8`     |
| `block_type`   | `str`   | The type of Swin block: `'W'` (Window) or `'SW'` (Shifted Window).             | `'W'`   |
| `drop_path`    | `float` | The stochastic depth rate.                                                     | `0.0`   |

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
