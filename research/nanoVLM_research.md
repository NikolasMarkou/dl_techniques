# nanoVLM Implementation Guide for dl-techniques Framework

## Overview

This guide provides a comprehensive implementation plan for nanoVLM (nano Vision-Language Model) within the dl-techniques framework. nanoVLM is a compact 222M parameter vision-language model that achieves 35.3% accuracy on MMStar benchmark while maintaining simplicity and efficiency.

## Architecture Summary

nanoVLM consists of three core components:
1. **Vision Encoder**: SigLIP-based Vision Transformer (85M parameters)
2. **Language Decoder**: SmolLM2-based causal transformer (135M parameters)  
3. **Modality Projection**: Pixel shuffle + linear projection (2M parameters)

Total: 222M parameters, trained in ~750 lines of code.

## Required Implementations

### 1. Core Layers (`dl_techniques/layers/`)

#### 1.1 Pixel Shuffle Layer (`pixel_shuffle.py`)

```python
from typing import Optional, Tuple
import keras
from keras import ops
from dl_techniques.utils.logger import logger

@keras.saving.register_keras_serializable()
class PixelShuffle(keras.layers.Layer):
    """Pixel shuffle operation for reducing spatial tokens in vision_heads transformers.
    
    Implements pixel shuffle to reduce the number of visual tokens by rearranging
    spatial information into channel dimensions, enabling more efficient processing
    in vision_heads-language models.
    
    Args:
        scale_factor: Factor by which to reduce spatial dimensions (default: 2)
        **kwargs: Additional keyword arguments for the Layer base class.
    """
    
    def __init__(
        self, 
        scale_factor: int = 2,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self._build_input_shape = None
        
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer."""
        self._build_input_shape = input_shape
        super().build(input_shape)
        
    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply pixel shuffle operation.
        
        Args:
            inputs: Input tensor of shape [batch, num_tokens, channels]
            
        Returns:
            Shuffled tensor with reduced spatial tokens
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1] 
        channels = inputs.shape[-1]
        
        # Handle CLS token separately
        cls_token = inputs[:, 0:1, :]  # [batch, 1, channels]
        spatial_tokens = inputs[:, 1:, :]  # [batch, H*W, channels]
        
        # Calculate spatial dimensions (assuming square)
        spatial_len = seq_len - 1
        h = w = ops.cast(ops.sqrt(ops.cast(spatial_len, "float32")), "int32")
        
        # Reshape to spatial format
        spatial_tokens = ops.reshape(spatial_tokens, [batch_size, h, w, channels])
        
        # Apply pixel shuffle (space to depth)
        new_h = h // self.scale_factor
        new_w = w // self.scale_factor
        new_c = channels * (self.scale_factor ** 2)
        
        # Rearrange operation
        shuffled = ops.reshape(spatial_tokens, [
            batch_size, new_h, self.scale_factor, new_w, self.scale_factor, channels
        ])
        shuffled = ops.transpose(shuffled, [0, 1, 3, 2, 4, 5])
        shuffled = ops.reshape(shuffled, [batch_size, new_h * new_w, new_c])
        
        # Concatenate CLS token back
        return ops.concatenate([cls_token, shuffled], axis=1)
        
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size, seq_len, channels = input_shape
        if seq_len is None:
            return (batch_size, None, channels * (self.scale_factor ** 2))
        
        spatial_len = seq_len - 1  # Remove CLS token
        new_spatial_len = spatial_len // (self.scale_factor ** 2)
        new_seq_len = new_spatial_len + 1  # Add CLS token back
        new_channels = channels * (self.scale_factor ** 2)
        
        return (batch_size, new_seq_len, new_channels)
        
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "scale_factor": self.scale_factor,
        })
        return config
        
    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}
        
    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

#### 1.2 Vision Transformer Layer (`vision_transformer_siglip.py`)

```python
from typing import Optional, Union, Tuple
import keras
from keras import ops
from dl_techniques.utils.logger import logger

@keras.saving.register_keras_serializable()
class SigLIPVisionTransformer(keras.layers.Layer):
    """SigLIP-based Vision Transformer for nanoVLM.
    
    Implements a vision_heads transformer following SigLIP architecture with 
    patch embedding, positional encoding, and transformer blocks.
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Size of image patches (default: 16)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout rate (default: 0.0)
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Will be initialized in build()
        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.blocks = None
        self.norm = None
        self._build_input_shape = None
        
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision_heads transformer components."""
        self._build_input_shape = input_shape
        
        # Patch embedding using conv layers (following SigLIP)
        self.patch_embed = keras.Sequential([
            keras.layers.Conv2D(
                self.embed_dim // 2, 
                kernel_size=self.patch_size // 2, 
                strides=self.patch_size // 2,
                padding='valid',
                name='patch_embed_conv1'
            ),
            keras.layers.LayerNormalization(name='patch_embed_norm1'),
            keras.layers.Activation('gelu', name='patch_embed_gelu'),
            keras.layers.Conv2D(
                self.embed_dim, 
                kernel_size=2, 
                strides=2,
                padding='valid',
                name='patch_embed_conv2'
            ),
        ], name='patch_embed')
        
        # CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        
        # Positional embedding
        self.pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer='truncated_normal',
            trainable=True,
            name='pos_embed'
        )
        
        # Transformer blocks
        self.blocks = []
        for i in range(self.depth):
            block = keras.Sequential([
                keras.layers.LayerNormalization(name=f'norm1_block_{i}'),
                keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.embed_dim // self.num_heads,
                    dropout=self.dropout,
                    name=f'attn_block_{i}'
                ),
                keras.layers.Add(name=f'add1_block_{i}'),
                keras.layers.LayerNormalization(name=f'norm2_block_{i}'),
                keras.layers.Dense(
                    int(self.embed_dim * self.mlp_ratio),
                    activation='gelu',
                    name=f'mlp1_block_{i}'
                ),
                keras.layers.Dropout(self.dropout, name=f'dropout1_block_{i}'),
                keras.layers.Dense(self.embed_dim, name=f'mlp2_block_{i}'),
                keras.layers.Dropout(self.dropout, name=f'dropout2_block_{i}'),
                keras.layers.Add(name=f'add2_block_{i}'),
            ], name=f'transformer_block_{i}')
            self.blocks.append(block)
        
        # Final norm
        self.norm = keras.layers.LayerNormalization(name='final_norm')
        
        super().build(input_shape)
        
    def call(self, inputs: keras.KerasTensor, training: bool = False) -> keras.KerasTensor:
        """Forward pass through vision_heads transformer.
        
        Args:
            inputs: Input images of shape [batch, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Vision features of shape [batch, num_patches + 1, embed_dim]
        """
        batch_size = ops.shape(inputs)[0]
        
        # Patch embedding
        x = self.patch_embed(inputs)  # [batch, h', w', embed_dim]
        x = ops.reshape(x, [batch_size, self.num_patches, self.embed_dim])
        
        # Add CLS token
        cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
        x = ops.concatenate([cls_tokens, x], axis=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            # Manual attention and MLP implementation for clarity
            residual = x
            x = block.layers[0](x)  # LayerNorm
            
            # Self-attention
            attn_output = block.layers[1](x, x, training=training)
            x = block.layers[2]([residual, attn_output])  # Add
            
            # MLP
            residual = x
            x = block.layers[3](x)  # LayerNorm
            x = block.layers[4](x)  # Dense + GELU
            x = block.layers[5](x, training=training)  # Dropout
            x = block.layers[6](x)  # Dense
            x = block.layers[7](x, training=training)  # Dropout
            x = block.layers[8]([residual, x])  # Add
        
        # Final normalization
        x = self.norm(x)
        
        return x
        
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.num_patches + 1, self.embed_dim)
        
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
        })
        return config
        
    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}
        
    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

#### 1.3 Modality Projection Layer (`modality_projection.py`)

```python
from typing import Optional, Tuple
import keras
from keras import ops
from dl_techniques.layers.pixel_shuffle import PixelShuffle
from dl_techniques.utils.logger import logger

@keras.saving.register_keras_serializable()
class ModalityProjection(keras.layers.Layer):
    """Modality projection layer for nanoVLM.
    
    Projects visual features to language embedding space using pixel shuffle
    for token reduction followed by linear projection.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        scale_factor: Pixel shuffle scale factor (default: 2)
        use_gelu: Whether to use GELU activation (default: True)
        use_layer_norm: Whether to apply layer normalization (default: True)
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scale_factor: int = 2,
        use_gelu: bool = True,
        use_layer_norm: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale_factor = scale_factor
        self.use_gelu = use_gelu
        self.use_layer_norm = use_layer_norm
        
        # Will be initialized in build()
        self.pixel_shuffle = None
        self.projection = None
        self._build_input_shape = None
        
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the modality projection components."""
        self._build_input_shape = input_shape
        
        # Pixel shuffle for token reduction
        self.pixel_shuffle = PixelShuffle(scale_factor=self.scale_factor)
        
        # Calculate expected input dimension after pixel shuffle
        shuffled_dim = self.input_dim * (self.scale_factor ** 2)
        
        # Build projection layers
        projection_layers = [
            keras.layers.Dense(
                self.output_dim,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name='projection_dense'
            )
        ]
        
        if self.use_gelu:
            projection_layers.append(keras.layers.Activation('gelu', name='projection_gelu'))
            
        if self.use_layer_norm:
            projection_layers.append(keras.layers.LayerNormalization(name='projection_norm'))
            
        self.projection = keras.Sequential(projection_layers, name='projection')
        
        super().build(input_shape)
        
    def call(self, inputs: keras.KerasTensor, training: bool = False) -> keras.KerasTensor:
        """Apply modality projection.
        
        Args:
            inputs: Visual features of shape [batch, num_tokens, input_dim]
            training: Whether in training mode
            
        Returns:
            Projected features of shape [batch, reduced_tokens, output_dim]
        """
        # Apply pixel shuffle to reduce tokens
        x = self.pixel_shuffle(inputs)
        
        # Project to target dimension
        x = self.projection(x, training=training)
        
        return x
        
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        # Get pixel shuffle output shape
        shuffled_shape = self.pixel_shuffle.compute_output_shape(input_shape)
        batch_size, seq_len, _ = shuffled_shape
        
        return (batch_size, seq_len, self.output_dim)
        
    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "scale_factor": self.scale_factor,
            "use_gelu": self.use_gelu,
            "use_layer_norm": self.use_layer_norm,
        })
        return config
        
    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}
        
    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
```

### 2. Models (`dl_techniques/models/`)

#### 2.1 Main nanoVLM Model (`nanovlm.py`)

```python
from typing import Dict, Optional, Tuple, Union
import keras
from keras import ops
from dl_techniques.layers.vision_transformer_siglip import SigLIPVisionTransformer
from dl_techniques.layers.modality_projection import ModalityProjection
from dl_techniques.utils.logger import logger

@keras.saving.register_keras_serializable()
class NanoVLM(keras.Model):
    """nanoVLM: Compact Vision-Language Model.
    
    A lightweight vision_heads-language model combining SigLIP vision_heads transformer
    with SmolLM2 language decoder through efficient modality projection.
    
    Args:
        vision_config: Configuration dict for vision_heads transformer
        language_config: Configuration dict for language decoder
        projection_config: Configuration dict for modality projection
        vocab_size: Vocabulary size for text embeddings (default: 32000)
        **kwargs: Additional keyword arguments
    """
    
    def __init__(
        self,
        vision_config: Dict,
        language_config: Dict,
        projection_config: Dict,
        vocab_size: int = 32000,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.language_config = language_config
        self.projection_config = projection_config
        self.vocab_size = vocab_size
        
        # Build components
        self.vision_encoder = SigLIPVisionTransformer(**vision_config)
        self.modality_projection = ModalityProjection(**projection_config)
        
        # Text embedding
        self.text_embedder = keras.layers.Embedding(
            vocab_size,
            language_config['hidden_dim'],
            mask_zero=True,
            name='text_embedder'
        )
        
        # Language decoder (simplified transformer)
        self.language_decoder = self._build_language_decoder(language_config)
        
        # Output projection
        self.output_projection = keras.layers.Dense(
            vocab_size,
            use_bias=False,
            name='output_projection'
        )
        
    def _build_language_decoder(self, config: Dict) -> keras.Sequential:
        """Build language decoder layers."""
        layers = []
        
        for i in range(config['num_layers']):
            # Self-attention block
            layers.extend([
                keras.layers.LayerNormalization(name=f'decoder_norm1_{i}'),
                keras.layers.MultiHeadAttention(
                    num_heads=config['num_heads'],
                    key_dim=config['hidden_dim'] // config['num_heads'],
                    dropout=config.get('dropout', 0.1),
                    use_causal_mask=True,
                    name=f'decoder_attn_{i}'
                ),
                keras.layers.Add(name=f'decoder_add1_{i}'),
                keras.layers.LayerNormalization(name=f'decoder_norm2_{i}'),
                keras.layers.Dense(
                    config['mlp_dim'],
                    activation='gelu',
                    name=f'decoder_mlp1_{i}'
                ),
                keras.layers.Dense(
                    config['hidden_dim'],
                    name=f'decoder_mlp2_{i}'
                ),
                keras.layers.Add(name=f'decoder_add2_{i}'),
            ])
            
        layers.append(keras.layers.LayerNormalization(name='decoder_final_norm'))
        
        return keras.Sequential(layers, name='language_decoder')
        
    def call(
        self, 
        inputs: Dict[str, keras.KerasTensor], 
        training: bool = False
    ) -> keras.KerasTensor:
        """Forward pass through nanoVLM.
        
        Args:
            inputs: Dictionary containing 'images' and 'text_tokens'
            training: Whether in training mode
            
        Returns:
            Language model logits of shape [batch, seq_len, vocab_size]
        """
        images = inputs['images']
        text_tokens = inputs['text_tokens']
        
        # Process images through vision_heads encoder
        vision_features = self.vision_encoder(images, training=training)
        
        # Project visual features to language space
        vision_embeddings = self.modality_projection(vision_features, training=training)
        
        # Get text embeddings
        text_embeddings = self.text_embedder(text_tokens, training=training)
        
        # Combine modalities (concatenate along sequence dimension)
        combined_embeddings = ops.concatenate([vision_embeddings, text_embeddings], axis=1)
        
        # Generate through language decoder
        hidden_states = combined_embeddings
        for i in range(0, len(self.language_decoder.layers), 7):
            # Self-attention block
            residual = hidden_states
            hidden_states = self.language_decoder.layers[i](hidden_states)  # LayerNorm
            attn_output = self.language_decoder.layers[i+1](
                hidden_states, hidden_states, training=training
            )  # MultiHeadAttention
            hidden_states = self.language_decoder.layers[i+2]([residual, attn_output])  # Add
            
            # MLP block  
            residual = hidden_states
            hidden_states = self.language_decoder.layers[i+3](hidden_states)  # LayerNorm
            hidden_states = self.language_decoder.layers[i+4](hidden_states)  # MLP1
            hidden_states = self.language_decoder.layers[i+5](hidden_states)  # MLP2
            hidden_states = self.language_decoder.layers[i+6]([residual, hidden_states])  # Add
        
        # Final normalization
        hidden_states = self.language_decoder.layers[-1](hidden_states)
        
        # Output projection to vocabulary
        logits = self.output_projection(hidden_states)
        
        return logits
        
    def generate(
        self,
        image: keras.KerasTensor,
        prompt_tokens: keras.KerasTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> keras.KerasTensor:
        """Generate text given image and prompt.
        
        Args:
            image: Input image tensor [1, height, width, channels]
            prompt_tokens: Initial prompt tokens [1, prompt_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token sequence
        """
        # Process image once
        vision_features = self.vision_encoder(image, training=False)
        vision_embeddings = self.modality_projection(vision_features, training=False)
        
        # Initialize with prompt
        current_tokens = prompt_tokens
        
        for _ in range(max_length):
            # Get current text embeddings
            text_embeddings = self.text_embedder(current_tokens, training=False)
            
            # Combine with vision_heads
            combined_embeddings = ops.concatenate([vision_embeddings, text_embeddings], axis=1)
            
            # Forward through decoder
            logits = self._decode_step(combined_embeddings)
            
            # Sample next token
            next_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = ops.top_k(next_logits, k=top_k)
                next_token_id = ops.random.categorical(top_k_logits[None, :], 1)[0, 0]
                next_token = top_k_indices[next_token_id]
            else:
                next_token = ops.argmax(next_logits)
                
            # Append to sequence
            current_tokens = ops.concatenate([
                current_tokens, 
                ops.reshape(next_token, [1, 1])
            ], axis=1)
            
            # Check for EOS token (assuming token 2 is EOS)
            if next_token == 2:
                break
                
        return current_tokens
        
    def _decode_step(self, embeddings: keras.KerasTensor) -> keras.KerasTensor:
        """Single decoding step."""
        hidden_states = embeddings
        
        # Apply language decoder
        for i in range(0, len(self.language_decoder.layers), 7):
            residual = hidden_states
            hidden_states = self.language_decoder.layers[i](hidden_states)
            attn_output = self.language_decoder.layers[i+1](
                hidden_states, hidden_states, training=False
            )
            hidden_states = self.language_decoder.layers[i+2]([residual, attn_output])
            
            residual = hidden_states
            hidden_states = self.language_decoder.layers[i+3](hidden_states)
            hidden_states = self.language_decoder.layers[i+4](hidden_states)
            hidden_states = self.language_decoder.layers[i+5](hidden_states)
            hidden_states = self.language_decoder.layers[i+6]([residual, hidden_states])
        
        hidden_states = self.language_decoder.layers[-1](hidden_states)
        logits = self.output_projection(hidden_states)
        
        return logits
        
    def get_config(self) -> Dict:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "vision_config": self.vision_config,
            "language_config": self.language_config,
            "projection_config": self.projection_config,
            "vocab_size": self.vocab_size,
        })
        return config
        
    @classmethod
    def from_config(cls, config: Dict) -> 'NanoVLM':
        """Create model from configuration."""
        return cls(**config)


def create_nanovlm_222m() -> NanoVLM:
    """Create nanoVLM-222M model with standard configuration."""
    vision_config = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.0,
    }
    
    language_config = {
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
    }
    
    projection_config = {
        'input_dim': 768,
        'output_dim': 768,
        'scale_factor': 2,
        'use_gelu': True,
        'use_layer_norm': True,
    }
    
    return NanoVLM(
        vision_config=vision_config,
        language_config=language_config,
        projection_config=projection_config,
        vocab_size=32000
    )
```

### 3. Loss Functions (`dl_techniques/losses/`)

#### 3.1 nanoVLM Loss (`nanovlm_loss.py`)

```python
from typing import Optional
import keras
from keras import ops
from dl_techniques.utils.logger import logger

@keras.saving.register_keras_serializable()
class NanoVLMLoss(keras.losses.Loss):
    """Loss function for nanoVLM training.
    
    Implements autoregressive language modeling loss with proper masking
    for vision_heads-language training.
    
    Args:
        ignore_index: Token index to ignore in loss computation (default: 0)
        label_smoothing: Label smoothing factor (default: 0.0)
        **kwargs: Additional keyword arguments for Loss base class
    """
    
    def __init__(
        self,
        ignore_index: int = 0,
        label_smoothing: float = 0.0,
        name: str = "nanovlm_loss",
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        self.sparse_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
            label_smoothing=label_smoothing
        )
        
    def call(
        self, 
        y_true: keras.KerasTensor, 
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute autoregressive language modeling loss.
        
        Args:
            y_true: Target token ids of shape [batch, seq_len]
            y_pred: Predicted logits of shape [batch, seq_len, vocab_size]
            
        Returns:
            Scalar loss value
        """
        # Shift for autoregressive prediction
        # Predict next token based on previous tokens
        y_pred_shifted = y_pred[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        y_true_shifted = y_true[:, 1:]      # [batch, seq_len-1]
        
        # Flatten for loss computation
        y_pred_flat = ops.reshape(y_pred_shifted, [-1, ops.shape(y_pred_shifted)[-1]])
        y_true_flat = ops.reshape(y_true_shifted, [-1])
        
        # Compute per-token loss
        loss_per_token = self.sparse_ce(y_true_flat, y_pred_flat)
        
        # Create mask to ignore padding tokens
        mask = ops.cast(ops.not_equal(y_true_flat, self.ignore_index), ops.dtype(loss_per_token))
        
        # Apply mask
        masked_loss = loss_per_token * mask
        
        # Return mean loss over non-masked tokens
        total_loss = ops.sum(masked_loss)
        total_tokens = ops.sum(mask)
        
        # Avoid division by zero
        return ops.where(
            ops.greater(total_tokens, 0),
            total_loss / total_tokens,
            ops.zeros_like(total_loss)
        )
        
    def get_config(self) -> dict:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
        })
        return config
```

### 4. Datasets (`dl_techniques/utils/datasets/`)

#### 4.1 VQA Dataset Processor (`vqa_dataset.py`)

```python
from typing import Dict, List, Optional, Tuple, Callable
import keras
from keras import ops
import numpy as np
from dl_techniques.utils.logger import logger

class VQADataProcessor:
    """Data processor for Vision Question Answering datasets.
    
    Handles preprocessing of images and text for nanoVLM training,
    including The Cauldron dataset format.
    
    Args:
        image_size: Target image size (default: 224)
        max_text_length: Maximum text sequence length (default: 512)
        vocab_size: Vocabulary size (default: 32000)
        pad_token_id: Padding token ID (default: 0)
        bos_token_id: Beginning of sequence token ID (default: 1)
        eos_token_id: End of sequence token ID (default: 2)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        max_text_length: int = 512,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> None:
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        logger.info(f"Initialized VQA processor with image_size={image_size}, "
                   f"max_text_length={max_text_length}")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for nanoVLM.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array normalized to [-1, 1]
        """
        # Load image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.image_size, self.image_size)
        )
        
        # Convert to array
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Normalize to [-1, 1] (matching SigLIP preprocessing)
        img_array = (img_array / 127.5) - 1.0
        
        return img_array.astype(np.float32)
        
    def preprocess_text(
        self, 
        question: str, 
        answer: Optional[str] = None,
        tokenizer: Optional[Callable] = None
    ) -> Dict[str, np.ndarray]:
        """Preprocess text for training.
        
        Args:
            question: Question text
            answer: Answer text (None for inference)
            tokenizer: Tokenizer function
            
        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        if tokenizer is None:
            # Simple character-level tokenization for demo
            # In practice, use proper tokenizer like SentencePiece
            tokenizer = self._simple_tokenizer
            
        # Format conversation
        if answer is not None:
            # Training format: Question: ... Answer: ...
            text = f"Question: {question} Answer: {answer}"
        else:
            # Inference format: Question: ... Answer:
            text = f"Question: {question} Answer:"
            
        # Tokenize
        tokens = tokenizer(text)
        
        # Add special tokens
        input_ids = [self.bos_token_id] + tokens[:self.max_text_length-2] + [self.eos_token_id]
        
        # Pad to max length
        input_ids.extend([self.pad_token_id] * (self.max_text_length - len(input_ids)))
        input_ids = input_ids[:self.max_text_length]
        
        # For training, labels are the same as input_ids (teacher forcing)
        labels = input_ids.copy() if answer is not None else None
        
        result = {'input_ids': np.array(input_ids, dtype=np.int32)}
        if labels is not None:
            result['labels'] = np.array(labels, dtype=np.int32)
            
        return result
        
    def _simple_tokenizer(self, text: str) -> List[int]:
        """Simple character-level tokenizer for demonstration."""
        # Convert to lowercase and get character codes
        tokens = [min(ord(c), self.vocab_size - 1) for c in text.lower()]
        return tokens
        
    def create_tensorflow_dataset(
        self,
        data_samples: List[Dict],
        batch_size: int = 32,
        shuffle: bool = True,
        num_parallel_calls: int = 4
    ) -> keras.utils.Sequence:
        """Create TensorFlow dataset for training.
        
        Args:
            data_samples: List of data samples with 'image_path', 'question', 'answer'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_parallel_calls: Number of parallel processing calls
            
        Returns:
            TensorFlow dataset
        """
        def process_sample(sample):
            """Process a single sample."""
            # Load and preprocess image
            image = self.preprocess_image(sample['image_path'])
            
            # Process text
            text_data = self.preprocess_text(
                sample['question'], 
                sample.get('answer')
            )
            
            return {
                'image': image,
                'input_ids': text_data['input_ids'],
                'labels': text_data.get('labels', text_data['input_ids'])
            }
            
        # Create dataset
        dataset = keras.utils.Sequence()
        
        # Process samples
        processed_samples = []
        for sample in data_samples:
            try:
                processed = process_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
                
        logger.info(f"Created dataset with {len(processed_samples)} samples")
        
        # Convert to batched format
        class VQASequence(keras.utils.Sequence):
            def __init__(self, samples, batch_size, shuffle):
                self.samples = samples
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.indices = np.arange(len(samples))
                if shuffle:
                    np.random.shuffle(self.indices)
                    
            def __len__(self):
                return len(self.samples) // self.batch_size
                
            def __getitem__(self, idx):
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_samples = [self.samples[i] for i in batch_indices]
                
                # Stack into batches
                batch_images = np.stack([s['image'] for s in batch_samples])
                batch_input_ids = np.stack([s['input_ids'] for s in batch_samples])
                batch_labels = np.stack([s['labels'] for s in batch_samples])
                
                return (
                    {
                        'images': batch_images,
                        'text_tokens': batch_input_ids
                    },
                    batch_labels
                )
                
            def on_epoch_end(self):
                if self.shuffle:
                    np.random.shuffle(self.indices)
                    
        return VQASequence(processed_samples, batch_size, shuffle)


def load_cauldron_sample() -> List[Dict]:
    """Load sample data in Cauldron format for testing.
    
    Returns:
        List of sample data dictionaries
    """
    # This is a placeholder - in practice, load from HuggingFace datasets
    sample_data = [
        {
            'image_path': 'path/to/image1.jpg',
            'question': 'What is shown in this image?',
            'answer': 'A cat sitting on a chair.'
        },
        {
            'image_path': 'path/to/image2.jpg', 
            'question': 'What color is the car?',
            'answer': 'The car is red.'
        }
    ]
    
    logger.info(f"Loaded {len(sample_data)} sample data points")
    return sample_data
```

### 5. Training Script (`train_nanovlm.py`)

```python
"""Training script for nanoVLM model."""

from typing import Dict, Optional
import keras
from keras import ops
from dl_techniques.models.nano_vlm import create_nanovlm_222m
from dl_techniques.losses.nanovlm_loss import NanoVLMLoss
from dl_techniques.datasets.vqa_dataset import VQADataProcessor, load_cauldron_sample
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.utils.logger import logger


def create_training_setup() -> Dict:
    """Create training configuration for nanoVLM."""

    # Learning rate schedule configuration
    lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 1000,
        "warmup_start_lr": 1e-8,
        "learning_rate": 1e-4,  # Base learning rate
        "decay_steps": 50000,
        "alpha": 0.0001
    }

    # Optimizer configuration
    optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }

    # Build components
    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(optimizer_config, lr_schedule)

    return {
        "optimizer": optimizer,
        "lr_schedule": lr_schedule,
        "loss_fn": NanoVLMLoss(ignore_index=0, label_smoothing=0.1)
    }


def setup_different_learning_rates(model: keras.Model) -> Dict:
    """Setup different learning rates for different model components."""

    # Separate parameters
    vision_params = []
    language_params = []
    projection_params = []

    for layer in model.layers:
        if 'vision_heads' in layer.name.lower():
            vision_params.extend(layer.trainable_variables)
        elif 'projection' in layer.name.lower():
            projection_params.extend(layer.trainable_variables)
        else:
            language_params.extend(layer.trainable_variables)

    # Create optimizers with different learning rates
    vision_optimizer = keras.optimizers.AdamW(learning_rate=1e-5)  # Lower for pre-trained
    language_optimizer = keras.optimizers.AdamW(learning_rate=1e-5)  # Lower for pre-trained
    projection_optimizer = keras.optimizers.AdamW(learning_rate=1e-4)  # Higher for new component

    return {
        "vision_optimizer": vision_optimizer,
        "language_optimizer": language_optimizer,
        "projection_optimizer": projection_optimizer,
        "vision_params": vision_params,
        "language_params": language_params,
        "projection_params": projection_params
    }


@keras.utils.register_keras_serializable()
class NanoVLMTrainer:
    """Custom trainer for nanoVLM with multi-optimizer support."""

    def __init__(self, model: keras.Model, loss_fn: keras.losses.Loss):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizers = setup_different_learning_rates(model)

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @keras.utils.register_keras_serializable()
    def train_step(self, batch_data):
        """Custom training step with multiple optimizers."""
        inputs, labels = batch_data

        with keras.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Split gradients by component
        vision_grads = []
        language_grads = []
        projection_grads = []

        for grad, var in zip(gradients, self.model.trainable_variables):
            if var in self.optimizers['vision_params']:
                vision_grads.append(grad)
            elif var in self.optimizers['projection_params']:
                projection_grads.append(grad)
            else:
                language_grads.append(grad)

        # Apply gradients with respective optimizers
        if vision_grads:
            self.optimizers['vision_optimizer'].apply_gradients(
                zip(vision_grads, self.optimizers['vision_params'])
            )
        if language_grads:
            self.optimizers['language_optimizer'].apply_gradients(
                zip(language_grads, self.optimizers['language_params'])
            )
        if projection_grads:
            self.optimizers['projection_optimizer'].apply_gradients(
                zip(projection_grads, self.optimizers['projection_params'])
            )

        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

        return {
            'loss': self.train_loss.result(),
            'accuracy': self.train_accuracy.result()
        }


def train_nanovlm():
    """Main training function for nanoVLM."""
    logger.info("Starting nanoVLM training")

    # Create model
    model = create_nanovlm_222m()
    logger.info("Created nanoVLM-222M model")

    # Setup training
    training_setup = create_training_setup()
    trainer = NanoVLMTrainer(model, training_setup['loss_fn'])

    # Prepare data
    data_processor = VQADataProcessor(
        image_size=224,
        max_text_length=512,
        vocab_size=32000
    )

    # Load sample data (replace with real dataset)
    sample_data = load_cauldron_sample()
    train_dataset = data_processor.create_tensorflow_dataset(
        sample_data,
        batch_size=8,  # Small batch size for demo
        shuffle=True
    )

    # Training configuration
    epochs = 10
    steps_per_epoch = len(train_dataset)

    logger.info(f"Training for {epochs} epochs with {steps_per_epoch} steps per epoch")

    # Training loop
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # Reset metrics
        trainer.train_loss.reset_states()
        trainer.train_accuracy.reset_states()

        # Train for one epoch
        for step, batch in enumerate(train_dataset):
            metrics = trainer.train_step(batch)

            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}, Step {step}: "
                    f"Loss = {metrics['loss']:.4f}, "
                    f"Accuracy = {metrics['accuracy']:.4f}"
                )

        # End of epoch logging
        logger.info(
            f"Epoch {epoch + 1} completed: "
            f"Loss = {metrics['loss']:.4f}, "
            f"Accuracy = {metrics['accuracy']:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"nanovlm_checkpoint_epoch_{epoch + 1}.keras"
            model.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = "nanovlm_final.keras"
    model.save(final_model_path)
    logger.info(f"Training completed. Final model saved: {final_model_path}")


if __name__ == "__main__":
    # Enable mixed precision for better performance
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    train_nanovlm()
```

## Memory Optimization Features

### Gradient Accumulation

```python
class GradientAccumulationTrainer(NanoVLMTrainer):
    """Trainer with gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, model, loss_fn, accumulation_steps=4):
        super().__init__(model, loss_fn)
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = None
        
    def accumulate_gradients(self, batch_data):
        """Accumulate gradients over multiple batches."""
        inputs, labels = batch_data
        
        with keras.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions) / self.accumulation_steps
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradients
        else:
            self.accumulated_gradients = [
                acc + grad for acc, grad in zip(self.accumulated_gradients, gradients)
            ]
            
        return loss * self.accumulation_steps
        
    def apply_accumulated_gradients(self):
        """Apply accumulated gradients and reset."""
        # Split and apply gradients as in parent class
        # Implementation similar to train_step but using accumulated gradients
        
        # Reset accumulated gradients
        self.accumulated_gradients = None
```

## Testing Infrastructure

### Unit Tests for Layers

```python
import unittest
import numpy as np
import keras
from dl_techniques.layers.pixel_shuffle import PixelShuffle
from dl_techniques.layers.modality_projection import ModalityProjection

class TestNanoVLMLayers(unittest.TestCase):
    """Test suite for nanoVLM layers."""
    
    def test_pixel_shuffle_output_shape(self):
        """Test pixel shuffle output shape computation."""
        layer = PixelShuffle(scale_factor=2)
        input_shape = (32, 197, 768)  # [batch, tokens, channels]
        
        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (32, 50, 3072)  # Reduced tokens, increased channels
        
        self.assertEqual(output_shape, expected_shape)
        
    def test_modality_projection_forward(self):
        """Test modality projection forward pass."""
        layer = ModalityProjection(input_dim=768, output_dim=768, scale_factor=2)
        
        # Build layer
        layer.build((None, 197, 768))
        
        # Test forward pass
        inputs = keras.random.normal((2, 197, 768))
        outputs = layer(inputs)
        
        self.assertEqual(outputs.shape, (2, 50, 768))
        
    def test_vision_transformer_serialization(self):
        """Test vision_heads transformer serialization."""
        from dl_techniques.layers.vision_transformer_siglip import SigLIPVisionTransformer
        
        # Create layer
        vit = SigLIPVisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, depth=6
        )
        
        # Build layer
        vit.build((None, 224, 224, 3))
        
        # Test serialization
        config = vit.get_config()
        build_config = vit.get_build_config()
        
        # Recreate layer
        new_vit = SigLIPVisionTransformer.from_config(config)
        new_vit.build_from_config(build_config)
        
        # Test forward pass
        inputs = keras.random.normal((1, 224, 224, 3))
        output1 = vit(inputs)
        output2 = new_vit(inputs)
        
        self.assertEqual(output1.shape, output2.shape)

if __name__ == '__main__':
    unittest.main()
```

## Performance Benchmarking

### Model Size Analysis

```python
def analyze_model_size():
    """Analyze nanoVLM model size and memory requirements."""
    from dl_techniques.models.nano_vlm import create_nanovlm_222m

    model = create_nanovlm_222m()

    # Build model with dummy input
    dummy_inputs = {
        'images': keras.random.normal((1, 224, 224, 3)),
        'text_tokens': keras.random.uniform((1, 50), minval=0, maxval=32000, dtype='int32')
    }

    _ = model(dummy_inputs)

    # Analyze model
    total_params = sum(np.prod(var.shape) for var in model.trainable_variables)

    print(f"Total parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

    # Component breakdown
    vision_params = sum(np.prod(var.shape) for var in model.vision_encoder.trainable_variables)
    projection_params = sum(np.prod(var.shape) for var in model.modality_projection.trainable_variables)
    language_params = total_params - vision_params - projection_params

    print(f"\nComponent breakdown:")
    print(f"Vision encoder: {vision_params:,} ({vision_params / total_params * 100:.1f}%)")
    print(f"Modality projection: {projection_params:,} ({projection_params / total_params * 100:.1f}%)")
    print(f"Language decoder: {language_params:,} ({language_params / total_params * 100:.1f}%)")


if __name__ == "__main__":
    analyze_model_size()
```

## Configuration Templates

### Model Configurations

```python
# nanovlm_configs.py

NANOVLM_CONFIGS = {
    "nanovlm_mini": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16, 
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        "language_config": {
            'hidden_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'mlp_dim': 1536,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 384,
            'output_dim': 384,
            'scale_factor': 2,
        },
        "total_params": "16M"
    },
    
    "nanovlm_base": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 512, 
            'depth': 8,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        "language_config": {
            'hidden_dim': 512,
            'num_layers': 8, 
            'num_heads': 8,
            'mlp_dim': 2048,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 512,
            'output_dim': 512,
            'scale_factor': 2,
        },
        "total_params": "64M"
    },
    
    "nanovlm_222m": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12, 
            'mlp_ratio': 4.0,
            'dropout': 0.0,
        },
        "language_config": {
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_dim': 3072,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 768,
            'output_dim': 768,
            'scale_factor': 2,
        },
        "total_params": "222M"
    }
}
```

## Implementation Checklist

### Phase 1: Core Layers
- [ ] `PixelShuffle` layer with proper Keras 3.x serialization
- [ ] `SigLIPVisionTransformer` layer with multi-head attention
- [ ] `ModalityProjection` layer combining pixel shuffle and projection  
- [ ] Unit tests for all layers
- [ ] Serialization tests

### Phase 2: Model Architecture  
- [ ] `NanoVLM` main model class
- [ ] Multi-optimizer training support
- [ ] Text generation capabilities
- [ ] Model configuration system
- [ ] Memory optimization features

### Phase 3: Training Infrastructure
- [ ] `NanoVLMLoss` with proper masking
- [ ] `VQADataProcessor` for dataset handling
- [ ] Training script with gradient accumulation
- [ ] Checkpoint saving/loading
- [ ] Mixed precision support

### Phase 4: Evaluation and Optimization
- [ ] Inference optimization
- [ ] Benchmarking tools
- [ ] Memory profiling
- [ ] Performance analysis
- [ ] Edge deployment preparation

### Phase 5: Integration
- [ ] Integration with dl-techniques framework
- [ ] Documentation and examples
- [ ] Model hub compatibility
- [ ] Testing on real datasets
- [ ] Performance validation

## Key Technical Decisions

1. **Keras 3.x Compatibility**: All layers follow the new Keras 3.x patterns with proper `build()`, `call()`, and serialization methods.

2. **Backend Agnostic**: Using `keras.ops` for all operations to ensure compatibility across TensorFlow, JAX, and PyTorch backends.

3. **Memory Efficiency**: Implementing gradient accumulation and mixed precision to handle large models on limited hardware.

4. **Modular Design**: Separating vision encoder, language decoder, and projection components for easy experimentation and fine-tuning.

5. **Training Flexibility**: Supporting different learning rates for different model components and gradient accumulation for effective large batch training.

This implementation guide provides a complete roadmap for building nanoVLM within the dl-techniques framework, following best practices for modern deep learning model development.