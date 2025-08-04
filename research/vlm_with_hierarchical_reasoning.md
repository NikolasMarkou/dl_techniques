# Vision Language Model with Hierarchical Reasoning (VLM-HRM) Implementation Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Implementation Steps](#implementation-steps)
4. [Multi-Modal Input Processing](#multi-modal-input-processing)
5. [Hierarchical Reasoning Modules](#hierarchical-reasoning-modules)
6. [Output Heads and Task Routing](#output-heads-and-task-routing)
7. [Training Strategy](#training-strategy)
8. [Usage Patterns](#usage-patterns)
9. [Advanced Features](#advanced-features)
10. [Performance Optimization](#performance-optimization)
11. [Complete Implementation](#complete-implementation)

---

## Architecture Overview

The VLM-HRM combines the adaptive reasoning capabilities of Hierarchical Reasoning Models with multi-modal vision-language understanding. The architecture features:

### Core Design Principles

1. **🧠 Dual-Level Reasoning**
   - **High-Level**: Cross-modal semantics, global context, conceptual relationships
   - **Low-Level**: Fine-grained visual-textual alignment, specific details, precise grounding

2. **🔄 Adaptive Computation Time (ACT)**
   - Simple queries get 2-3 reasoning cycles
   - Complex multi-modal reasoning gets 8-12+ cycles
   - Learned halting mechanism via Q-learning

3. **🎯 Multi-Task Unified Architecture**
   - Single model handles VQA, captioning, grounding, retrieval
   - Task-specific output heads with intelligent routing

### High-Level Architecture Flow

```
Vision Input + Text Input → Multi-Modal Embeddings
                ↓
    ┌─────────────────────────────────────────────────┐
    │  High-Level Reasoning (Cross-Modal Semantics)   │
    │  • Scene-text conceptual alignment              │
    │  • Global context understanding                 │
    │  • Abstract semantic relationships              │
    └─────────────────┬───────────────────────────────┘
                      ↓ (semantic context injection)
    ┌─────────────────────────────────────────────────┐
    │  Low-Level Reasoning (Fine-Grained Grounding)   │
    │  • Pixel-word alignment                         │
    │  • Spatial-textual correspondences              │
    │  • Detailed visual feature extraction           │
    └─────────────────┬───────────────────────────────┘
                      ↓ (iterative refinement)
              Multi-Modal Task Outputs
```

---

## Core Components

### 1. Vision Language HRM Core
```python
@keras.saving.register_keras_serializable()
class VisionLanguageHRMCore(keras.layers.Layer):
    """
    Core multi-modal hierarchical reasoning engine.
    
    Extends the base HRM architecture to handle vision-language inputs
    with cross-modal fusion and task-adaptive processing.
    """
```

### 2. Multi-Modal Input Processor
```python
class MultiModalInputProcessor(keras.layers.Layer):
    """
    Processes and fuses vision and text inputs into unified embeddings.
    
    Components:
    - Vision encoder (SigLIP-based)
    - Text embeddings with positional encoding
    - Cross-modal fusion layer
    - Modality type embeddings
    """
```

### 3. Cross-Modal Reasoning Modules
```python
class CrossModalReasoningModule(ReasoningModule):
    """
    Enhanced reasoning module with cross-modal attention.
    
    Features:
    - Within-modality self-attention
    - Cross-modality attention
    - Hierarchical information flow
    """
```

### 4. Task-Adaptive Output System
```python
class VLMOutputSystem(keras.layers.Layer):
    """
    Multi-task output heads with intelligent routing.
    
    Supports:
    - Visual Question Answering
    - Image Captioning
    - Visual Grounding
    - Image-Text Retrieval
    - Visual Reasoning
    """
```

---

## Implementation Steps

### Step 1: Set Up Base Dependencies

```python
import keras
from keras import ops, layers, initializers, regularizers
import tensorflow as tf
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum

# Import HRM base components
from dl_techniques.layers.hrm_reasoning_core import HierarchicalReasoningCore
from dl_techniques.layers.hrm_reasoning_module import HierarchicalReasoningModule
from dl_techniques.layers.hrm_block import HierarchicalReasoningBlock
from dl_techniques.layers.vision_transformer_siglip import SigLIPVisionTransformer
from dl_techniques.layers.modality_projection import ModalityProjection
from dl_techniques.layers.positional_embedding import PositionalEmbedding
from dl_techniques.layers.rope import RotaryPositionEmbedding
```

### Step 2: Define Task Types and Constants

```python
class VLMTaskType(Enum):
    """Supported VLM task types."""
    VQA = 0
    CAPTIONING = 1
    GROUNDING = 2
    RETRIEVAL = 3
    REASONING = 4

class VLMConfig:
    """Configuration for VLM-HRM model."""
    
    # Vision settings
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    VISION_EMBED_DIM = 768
    
    # Text settings
    VOCAB_SIZE = 32000
    MAX_TEXT_LENGTH = 512
    TEXT_EMBED_DIM = 512
    
    # HRM settings
    EMBED_DIM = 512
    NUM_HEADS = 8
    H_LAYERS = 4
    L_LAYERS = 4
    H_CYCLES = 2
    L_CYCLES = 2
    FFN_EXPANSION = 4
    
    # ACT settings
    MAX_REASONING_STEPS = 16
    HALT_EXPLORATION_PROB = 0.1
    
    # Training settings
    DROPOUT_RATE = 0.1
    USE_BIAS = False
```

### Step 3: Implement Multi-Modal Input Processor

```python
@keras.saving.register_keras_serializable()
class MultiModalInputProcessor(keras.layers.Layer):
    """
    Multi-modal input processing for VLM-HRM.
    
    Handles vision and text inputs, creates unified embeddings with
    modality-specific tokens and cross-modal fusion.
    """
    
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        text_embed_dim: int = 512,
        embed_dim: int = 512,
        max_text_length: int = 512,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.embed_dim = embed_dim
        self.max_text_length = max_text_length
        self.dropout_rate = dropout_rate
        
        # Calculate dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.total_vision_tokens = self.num_patches + 1  # +1 for CLS token
        
        # Components (built in build())
        self.vision_encoder = None
        self.vision_projector = None
        self.text_embedding = None
        self.text_pos_embedding = None
        self.modality_embeddings = None
        self.cross_modal_fusion = None
        self.task_embedding = None
        
    def build(self, input_shape):
        """Build multi-modal input processor components."""
        
        # Vision encoder (SigLIP-based)
        self.vision_encoder = SigLIPVisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.vision_embed_dim,
            num_layers=12,
            num_heads=12,
            name="vision_encoder"
        )
        
        # Vision-to-text projection
        self.vision_projector = ModalityProjection(
            input_dim=self.vision_embed_dim,
            output_dim=self.embed_dim,
            name="vision_projector"
        )
        
        # Text embeddings
        self.text_embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.text_embed_dim,
            mask_zero=True,
            name="text_embedding"
        )
        
        # Text positional embeddings
        self.text_pos_embedding = PositionalEmbedding(
            max_seq_len=self.max_text_length,
            dim=self.text_embed_dim,
            dropout=0.0,
            name="text_pos_embedding"
        )
        
        # Text-to-unified projection
        if self.text_embed_dim != self.embed_dim:
            self.text_projector = layers.Dense(
                self.embed_dim,
                use_bias=False,
                name="text_projector"
            )
        else:
            self.text_projector = None
            
        # Modality type embeddings
        self.modality_embeddings = layers.Embedding(
            input_dim=4,  # VISION, TEXT, FUSION, TASK
            output_dim=self.embed_dim,
            name="modality_embeddings"
        )
        
        # Cross-modal fusion layer
        self.cross_modal_fusion = CrossModalFusionLayer(
            embed_dim=self.embed_dim,
            num_heads=8,
            name="cross_modal_fusion"
        )
        
        # Task type embedding
        self.task_embedding = layers.Embedding(
            input_dim=len(VLMTaskType),
            output_dim=self.embed_dim,
            name="task_embedding"
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Process multi-modal inputs.
        
        Args:
            inputs: Dict with keys:
                - 'images': [batch_size, height, width, channels]
                - 'text_ids': [batch_size, max_text_length]  
                - 'task_type': [batch_size,]
                
        Returns:
            Multi-modal embeddings: [batch_size, total_seq_len, embed_dim]
        """
        images = inputs['images']
        text_ids = inputs['text_ids']
        task_type = inputs['task_type']
        
        batch_size = ops.shape(images)[0]
        
        # Process vision
        vision_features = self.vision_encoder(images, training=training)
        vision_emb = self.vision_projector(vision_features, training=training)
        
        # Add vision modality embedding
        vision_mod_token = self.modality_embeddings(0)  # VISION = 0
        vision_emb = vision_emb + ops.broadcast_to(
            ops.expand_dims(vision_mod_token, 0),
            ops.shape(vision_emb)
        )
        
        # Process text
        text_emb = self.text_embedding(text_ids)
        text_emb = text_emb + self.text_pos_embedding(text_emb)
        
        # Project text if needed
        if self.text_projector is not None:
            text_emb = self.text_projector(text_emb)
            
        # Add text modality embedding
        text_mod_token = self.modality_embeddings(1)  # TEXT = 1
        text_emb = text_emb + ops.broadcast_to(
            ops.expand_dims(text_mod_token, 0),
            ops.shape(text_emb)
        )
        
        # Cross-modal fusion
        fusion_tokens = self.cross_modal_fusion(
            vision_emb, text_emb, training=training
        )
        
        # Add fusion modality embedding
        fusion_mod_token = self.modality_embeddings(2)  # FUSION = 2
        fusion_tokens = fusion_tokens + ops.broadcast_to(
            ops.expand_dims(fusion_mod_token, 0),
            ops.shape(fusion_tokens)
        )
        
        # Task embedding (prepended as special token)
        task_emb = self.task_embedding(task_type)  # [batch_size, embed_dim]
        task_emb = ops.expand_dims(task_emb, 1)    # [batch_size, 1, embed_dim]
        
        # Add task modality embedding
        task_mod_token = self.modality_embeddings(3)  # TASK = 3
        task_emb = task_emb + ops.expand_dims(task_mod_token, 0)
        
        # Concatenate all modalities: [TASK] + [VISION] + [TEXT] + [FUSION]
        multimodal_emb = ops.concatenate([
            task_emb,       # [batch_size, 1, embed_dim]
            vision_emb,     # [batch_size, num_vision_tokens, embed_dim]
            text_emb,       # [batch_size, max_text_length, embed_dim]
            fusion_tokens   # [batch_size, num_fusion_tokens, embed_dim]
        ], axis=1)
        
        return multimodal_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vision_embed_dim": self.vision_embed_dim,
            "text_embed_dim": self.text_embed_dim,
            "embed_dim": self.embed_dim,
            "max_text_length": self.max_text_length,
            "dropout_rate": self.dropout_rate,
        })
        return config
```

### Step 4: Implement Cross-Modal Fusion Layer

```python
@keras.saving.register_keras_serializable()
class CrossModalFusionLayer(keras.layers.Layer):
    """
    Cross-modal fusion layer for vision-text interaction.
    
    Creates fusion tokens that capture cross-modal relationships
    between vision and text representations.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_fusion_tokens: int = 32,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_fusion_tokens = num_fusion_tokens
        self.dropout_rate = dropout_rate
        
        # Components (built in build())
        self.fusion_tokens = None
        self.vision_to_fusion = None
        self.text_to_fusion = None
        self.fusion_norm = None
        self.dropout = None
        
    def build(self, input_shape):
        """Build cross-modal fusion components."""
        
        # Learnable fusion tokens
        self.fusion_tokens = self.add_weight(
            name="fusion_tokens",
            shape=(1, self.num_fusion_tokens, self.embed_dim),
            initializer="truncated_normal",
            trainable=True
        )
        
        # Cross-attention from vision to fusion tokens
        self.vision_to_fusion = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name="vision_to_fusion"
        )
        
        # Cross-attention from text to fusion tokens  
        self.text_to_fusion = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name="text_to_fusion"
        )
        
        # Layer normalization
        self.fusion_norm = layers.LayerNormalization(name="fusion_norm")
        
        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
    
    def call(self, vision_emb, text_emb, training=None):
        """
        Perform cross-modal fusion.
        
        Args:
            vision_emb: [batch_size, num_vision_tokens, embed_dim]
            text_emb: [batch_size, num_text_tokens, embed_dim]
            training: Training mode flag
            
        Returns:
            Fusion tokens: [batch_size, num_fusion_tokens, embed_dim]
        """
        batch_size = ops.shape(vision_emb)[0]
        
        # Broadcast fusion tokens for batch
        fusion_queries = ops.broadcast_to(
            self.fusion_tokens,
            [batch_size, self.num_fusion_tokens, self.embed_dim]
        )
        
        # Vision -> Fusion cross-attention
        vision_fusion = self.vision_to_fusion(
            query=fusion_queries,
            key=vision_emb,
            value=vision_emb,
            training=training
        )
        
        # Text -> Fusion cross-attention
        text_fusion = self.text_to_fusion(
            query=fusion_queries,
            key=text_emb,
            value=text_emb,
            training=training
        )
        
        # Combine and normalize
        fusion_output = self.fusion_norm(
            fusion_queries + vision_fusion + text_fusion
        )
        fusion_output = self.dropout(fusion_output, training=training)
        
        return fusion_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_fusion_tokens": self.num_fusion_tokens,
            "dropout_rate": self.dropout_rate,
        })
        return config
```

### Step 5: Implement Cross-Modal Reasoning Module

```python
@keras.saving.register_keras_serializable()
class CrossModalReasoningModule(ReasoningModule):
    """
    Enhanced reasoning module with cross-modal attention capabilities.
    
    Extends the base ReasoningModule to handle multi-modal interactions
    during the hierarchical reasoning process.
    """
    
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        enable_cross_modal: bool = True,
        **kwargs
    ):
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            **kwargs
        )
        
        self.enable_cross_modal = enable_cross_modal
        
        # Cross-modal attention blocks (built in build())
        self.cross_modal_blocks = []
        
    def build(self, input_shape):
        """Build reasoning blocks with cross-modal attention."""
        super().build(input_shape)
        
        if self.enable_cross_modal:
            # Add cross-modal attention after each reasoning block
            for i in range(self.num_layers):
                cross_block = CrossModalAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name=f"cross_modal_block_{i}"
                )
                self.cross_modal_blocks.append(cross_block)
    
    def call(self, hidden_states, input_injection, training=None, mask=None):
        """
        Forward pass with cross-modal reasoning.
        
        Args:
            hidden_states: Current hidden states [batch_size, seq_len, embed_dim]
            input_injection: Input to inject [batch_size, seq_len, embed_dim]
            training: Training mode flag
            mask: Optional attention mask
            
        Returns:
            Updated hidden states with cross-modal reasoning
        """
        # Input injection
        x = hidden_states + input_injection
        
        # Process through reasoning blocks with cross-modal attention
        for i, block in enumerate(self.blocks):
            # Self-attention within current modality
            x = block(x, training=training, mask=mask)
            
            # Cross-modal attention if enabled
            if self.enable_cross_modal and i < len(self.cross_modal_blocks):
                x = self.cross_modal_blocks[i](x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "enable_cross_modal": self.enable_cross_modal,
        })
        return config


@keras.saving.register_keras_serializable()
class CrossModalAttentionBlock(keras.layers.Layer):
    """
    Cross-modal attention block for inter-modality communication.
    
    Enables information exchange between different modalities
    (vision, text, fusion) during reasoning.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Components (built in build())
        self.cross_attention = None
        self.layer_norm = None
        self.dropout = None
        
    def build(self, input_shape):
        """Build cross-modal attention components."""
        
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
            name="cross_attention"
        )
        
        self.layer_norm = layers.LayerNormalization(name="layer_norm")
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        """
        Apply cross-modal attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            training: Training mode flag
            
        Returns:
            Output with cross-modal attention applied
        """
        # Self-attention across all tokens (cross-modal interaction)
        attn_output = self.cross_attention(
            query=x,
            key=x,
            value=x,
            training=training
        )
        
        # Residual connection and normalization
        output = self.layer_norm(x + attn_output)
        output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config
```

### Step 6: Implement VLM-HRM Core

```python
@keras.saving.register_keras_serializable()
class VisionLanguageHRMCore(keras.layers.Layer):
    """
    Core Vision Language Hierarchical Reasoning Model.
    
    Integrates multi-modal input processing with hierarchical reasoning
    for adaptive vision-language understanding.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        image_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        text_embed_dim: int = 512,
        max_text_length: int = 512,
        h_layers: int = 4,
        l_layers: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 2,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        dropout_rate: float = 0.1,
        use_bias: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.max_text_length = max_text_length
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Calculate sequence dimensions
        num_patches = (image_size // patch_size) ** 2
        self.num_vision_tokens = num_patches + 1  # +1 for CLS
        self.num_fusion_tokens = 32
        self.total_seq_len = (
            1 +                          # Task token
            self.num_vision_tokens +     # Vision tokens
            max_text_length +            # Text tokens
            self.num_fusion_tokens       # Fusion tokens
        )
        
        # Components (built in build())
        self.input_processor = None
        self.h_reasoning = None
        self.l_reasoning = None
        self.h_init = None
        self.l_init = None
        
    def build(self, input_shape):
        """Build VLM-HRM core components."""
        
        # Multi-modal input processor
        self.input_processor = MultiModalInputProcessor(
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            vision_embed_dim=self.vision_embed_dim,
            text_embed_dim=self.text_embed_dim,
            embed_dim=self.embed_dim,
            max_text_length=self.max_text_length,
            dropout_rate=self.dropout_rate,
            name="input_processor"
        )
        
        # High-level reasoning (cross-modal semantics)
        self.h_reasoning = CrossModalReasoningModule(
            num_layers=self.h_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion_factor=self.ffn_expansion_factor,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            enable_cross_modal=True,
            name="h_reasoning"
        )
        
        # Low-level reasoning (fine-grained alignment)
        self.l_reasoning = CrossModalReasoningModule(
            num_layers=self.l_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_expansion_factor=self.ffn_expansion_factor,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            enable_cross_modal=True,
            name="l_reasoning"
        )
        
        # Initial state vectors
        self.h_init = self.add_weight(
            name="h_init",
            shape=(self.embed_dim,),
            initializer="truncated_normal",
            trainable=True
        )
        
        self.l_init = self.add_weight(
            name="l_init", 
            shape=(self.embed_dim,),
            initializer="truncated_normal",
            trainable=True
        )
        
        super().build(input_shape)
    
    def empty_carry(self, batch_size):
        """Create empty carry state for VLM-HRM."""
        return {
            "z_h": ops.zeros((batch_size, self.total_seq_len, self.embed_dim)),
            "z_l": ops.zeros((batch_size, self.total_seq_len, self.embed_dim))
        }
    
    def reset_carry(self, reset_flag, carry):
        """Reset carry state for halted sequences."""
        batch_size = ops.shape(reset_flag)[0]
        
        # Broadcast initial states
        h_init_broadcast = ops.broadcast_to(
            ops.reshape(self.h_init, [1, 1, self.embed_dim]),
            [batch_size, self.total_seq_len, self.embed_dim]
        )
        l_init_broadcast = ops.broadcast_to(
            ops.reshape(self.l_init, [1, 1, self.embed_dim]),
            [batch_size, self.total_seq_len, self.embed_dim]
        )
        
        # Reset based on halt flag
        reset_flag = ops.reshape(reset_flag, [-1, 1, 1])
        new_z_h = ops.where(reset_flag, h_init_broadcast, carry["z_h"])
        new_z_l = ops.where(reset_flag, l_init_broadcast, carry["z_l"])
        
        return {"z_h": new_z_h, "z_l": new_z_l}
    
    def call(self, carry, inputs, training=None):
        """
        Forward pass through VLM-HRM core.
        
        Args:
            carry: Current carry state dict with "z_h" and "z_l"
            inputs: Dict with 'images', 'text_ids', 'task_type'
            training: Training mode flag
            
        Returns:
            Tuple of (new_carry, embeddings_dict)
        """
        # Get multi-modal input embeddings
        input_emb = self.input_processor(inputs, training=training)
        
        z_h, z_l = carry["z_h"], carry["z_l"]
        
        # Hierarchical reasoning cycles (detached for efficiency)
        with ops.stop_gradient():
            for h_step in range(self.h_cycles):
                for l_step in range(self.l_cycles):
                    # Skip last L step of last H cycle
                    if not (h_step == self.h_cycles - 1 and l_step == self.l_cycles - 1):
                        z_l = self.l_reasoning(
                            z_l, z_h + input_emb, training=training
                        )
                
                # Skip last H step
                if h_step != self.h_cycles - 1:
                    z_h = self.h_reasoning(z_h, z_l, training=training)
        
        # Final step with gradients
        z_l = self.l_reasoning(z_l, z_h + input_emb, training=training)
        z_h = self.h_reasoning(z_h, z_l, training=training)
        
        # Return new carry and embeddings for output heads
        new_carry = {
            "z_h": ops.stop_gradient(z_h),
            "z_l": ops.stop_gradient(z_l)
        }
        
        embeddings = {
            "z_h": z_h,
            "z_l": z_l,
            "input_emb": input_emb
        }
        
        return new_carry, embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vision_embed_dim": self.vision_embed_dim,
            "text_embed_dim": self.text_embed_dim,
            "max_text_length": self.max_text_length,
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "h_cycles": self.h_cycles,
            "l_cycles": self.l_cycles,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
        })
        return config
```

### Step 7: Implement Task-Adaptive Output System

```python
@keras.saving.register_keras_serializable()
class VLMOutputSystem(keras.layers.Layer):
    """
    Multi-task output system for VLM-HRM.
    
    Handles different VLM tasks with specialized output heads
    and intelligent task routing.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int = 1000,
        max_caption_length: int = 128,
        num_detection_classes: int = 80,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.max_caption_length = max_caption_length
        self.num_detection_classes = num_detection_classes
        
        # Output heads (built in build())
        self.vqa_head = None
        self.caption_head = None
        self.grounding_head = None
        self.similarity_head = None
        self.classification_head = None
        self.q_head = None
        
    def build(self, input_shape):
        """Build task-specific output heads."""
        
        # VQA head (answer classification)
        self.vqa_head = layers.Dense(
            self.vocab_size,
            use_bias=True,
            name="vqa_head"
        )
        
        # Caption generation head
        self.caption_head = layers.Dense(
            self.vocab_size,
            use_bias=True,
            name="caption_head"
        )
        
        # Visual grounding head (bounding box regression)
        self.grounding_head = layers.Dense(
            4,  # [x1, y1, x2, y2]
            use_bias=True,
            activation="sigmoid",
            name="grounding_head"
        )
        
        # Image-text similarity head
        self.similarity_head = layers.Dense(
            1,
            use_bias=True,
            activation="sigmoid",
            name="similarity_head"
        )
        
        # Classification head
        self.classification_head = layers.Dense(
            self.num_classes,
            use_bias=True,
            name="classification_head"
        )
        
        # Q-learning head for ACT (halt/continue decisions)
        self.q_head = layers.Dense(
            2,  # [halt, continue]
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer=initializers.Constant(-5.0),
            name="q_head"
        )
        
        super().build(input_shape)
    
    def call(self, embeddings, task_type, training=None):
        """
        Generate task-specific outputs.
        
        Args:
            embeddings: Dict with 'z_h', 'z_l', 'input_emb'
            task_type: Task type tensor [batch_size,]
            training: Training mode flag
            
        Returns:
            Dict with task-specific outputs and Q-values
        """
        z_h = embeddings["z_h"]
        z_l = embeddings["z_l"]
        
        # Use task token (first position) for classification tasks
        task_token = z_h[:, 0]  # [batch_size, embed_dim]
        
        # Generate all possible outputs (task routing handled externally)
        outputs = {}
        
        # VQA (use task token)
        outputs["vqa_logits"] = self.vqa_head(task_token)
        
        # Captioning (use all text positions, skip task/vision/fusion tokens)
        # Assuming structure: [TASK] + [VISION] + [TEXT] + [FUSION]
        text_start = 1 + (224//16)**2 + 1  # Skip task + vision tokens  
        text_end = text_start + 512         # Text sequence length
        text_embeddings = z_h[:, text_start:text_end]
        outputs["caption_logits"] = self.caption_head(text_embeddings)
        
        # Grounding (use task token)
        outputs["grounding_boxes"] = self.grounding_head(task_token)
        
        # Similarity (use task token)
        outputs["similarity_score"] = self.similarity_head(task_token)
        
        # Classification (use task token)
        outputs["classification_logits"] = self.classification_head(task_token)
        
        # Q-learning for ACT (use task token)
        q_logits = self.q_head(task_token)
        outputs["q_halt_logits"] = q_logits[..., 0]
        outputs["q_continue_logits"] = q_logits[..., 1]
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
            "max_caption_length": self.max_caption_length,
            "num_detection_classes": self.num_detection_classes,
        })
        return config
```

### Step 8: Implement Complete VLM-HRM Model

```python
@keras.saving.register_keras_serializable()
class VisionLanguageHRM(keras.Model):
    """
    Complete Vision Language Hierarchical Reasoning Model.
    
    Integrates all components into a unified model with adaptive
    computation time and multi-task capabilities.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        text_embed_dim: int = 512,
        max_text_length: int = 512,
        h_layers: int = 4,
        l_layers: int = 4,
        h_cycles: int = 2,
        l_cycles: int = 2,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        dropout_rate: float = 0.1,
        use_bias: bool = False,
        num_classes: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.max_text_length = max_text_length
        self.h_layers = h_layers
        self.l_layers = l_layers
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.num_classes = num_classes
        
        # Core components
        self.core = VisionLanguageHRMCore(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            vision_embed_dim=vision_embed_dim,
            text_embed_dim=text_embed_dim,
            max_text_length=max_text_length,
            h_layers=h_layers,
            l_layers=l_layers,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name="core"
        )
        
        self.output_system = VLMOutputSystem(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            name="output_system"
        )
    
    def initial_carry(self, batch):
        """Initialize carry state for a batch."""
        batch_size = ops.shape(batch["images"])[0]
        
        return {
            # Core reasoning state
            "inner_carry": self.core.empty_carry(batch_size),
            
            # ACT state
            "steps": ops.zeros((batch_size,), dtype="int32"),
            "halted": ops.ones((batch_size,), dtype="bool"),
            
            # Current data cache
            "current_data": {k: ops.zeros_like(v) for k, v in batch.items()}
        }
    
    def call(self, inputs, training=None):
        """
        Forward pass through VLM-HRM.
        
        Args:
            inputs: Either batch dict or (carry, batch) tuple
            training: Training mode flag
            
        Returns:
            Task-specific outputs or (carry, outputs, finished) for step mode
        """
        if isinstance(inputs, dict):
            # Standard call - run until convergence
            return self._forward_complete(inputs, training=training)
        else:
            # Step call
            carry, batch = inputs
            return self._forward_step(carry, batch, training=training)
    
    def _forward_complete(self, batch, training=None):
        """Run complete forward pass until convergence."""
        carry = self.initial_carry(batch)
        
        # Run steps until all sequences halt
        max_iterations = self.halt_max_steps * 2
        for _ in range(max_iterations):
            carry, outputs, all_finished = self._forward_step(
                carry, batch, training=training
            )
            if all_finished:
                break
                
        return outputs
    
    def _forward_step(self, carry, batch, training=None):
        """Single reasoning step with ACT logic."""
        # Update carry for new sequences
        new_inner_carry = self.core.reset_carry(
            carry["halted"], carry["inner_carry"]
        )
        
        # Reset steps for halted sequences
        new_steps = ops.where(carry["halted"], 0, carry["steps"])
        
        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry["current_data"].items():
            reset_mask = ops.reshape(carry["halted"], [-1] + [1] * (len(v.shape) - 1))
            new_current_data[k] = ops.where(reset_mask, batch[k], v)
        
        # Forward pass through core
        new_inner_carry, embeddings = self.core(
            new_inner_carry, new_current_data, training=training
        )
        
        # Generate outputs
        outputs = self.output_system(
            embeddings, new_current_data["task_type"], training=training
        )
        
        # Update steps
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.halt_max_steps
        
        # Determine halting
        halted = is_last_step
        
        if training and self.halt_max_steps > 1:
            # Q-learning based halting
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"] 
            
            halted = halted | (q_halt > q_continue)
            
            # Exploration
            if self.halt_exploration_prob > 0:
                explore_mask = ops.random.uniform(ops.shape(q_halt)) < self.halt_exploration_prob
                min_steps = ops.random.uniform(
                    ops.shape(new_steps),
                    minval=2,
                    maxval=self.halt_max_steps + 1,
                    dtype="int32"
                )
                min_halt_steps = ops.where(explore_mask, min_steps, 1)
                halted = halted & (new_steps >= min_halt_steps)
        
        # Create new carry
        new_carry = {
            "inner_carry": new_inner_carry,
            "steps": new_steps,
            "halted": halted,
            "current_data": new_current_data
        }
        
        # Check if all sequences are finished
        all_finished = ops.all(halted)
        
        return new_carry, outputs, all_finished
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vision_embed_dim": self.vision_embed_dim,
            "text_embed_dim": self.text_embed_dim,
            "max_text_length": self.max_text_length,
            "h_layers": self.h_layers,
            "l_layers": self.l_layers,
            "h_cycles": self.h_cycles,
            "l_cycles": self.l_cycles,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "halt_max_steps": self.halt_max_steps,
            "halt_exploration_prob": self.halt_exploration_prob,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "num_classes": self.num_classes,
        })
        return config
```

---

## Training Strategy

### Multi-Task Loss Function

```python
class VLMMultiTaskLoss(keras.losses.Loss):
    """
    Multi-task loss function for VLM-HRM training.
    
    Combines losses from different VLM tasks with adaptive weighting
    and Q-learning loss for ACT training.
    """
    
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        q_learning_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.task_weights = task_weights or {
            "vqa": 1.0,
            "captioning": 1.0,
            "grounding": 1.0,
            "similarity": 1.0,
            "classification": 1.0
        }
        self.q_learning_weight = q_learning_weight
        
        # Task-specific loss functions
        self.vqa_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.caption_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.grounding_loss = keras.losses.MeanSquaredError()
        self.similarity_loss = keras.losses.BinaryCrossentropy()
        self.classification_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.q_loss = keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        """
        Compute multi-task loss.
        
        Args:
            y_true: Dict with ground truth for each task
            y_pred: Dict with predictions from model
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # VQA loss
        if "vqa_answers" in y_true and "vqa_logits" in y_pred:
            vqa_loss = self.vqa_loss(y_true["vqa_answers"], y_pred["vqa_logits"])
            total_loss += self.task_weights["vqa"] * vqa_loss
        
        # Captioning loss
        if "caption_tokens" in y_true and "caption_logits" in y_pred:
            caption_loss = self.caption_loss(y_true["caption_tokens"], y_pred["caption_logits"])
            total_loss += self.task_weights["captioning"] * caption_loss
        
        # Grounding loss
        if "bounding_boxes" in y_true and "grounding_boxes" in y_pred:
            grounding_loss = self.grounding_loss(y_true["bounding_boxes"], y_pred["grounding_boxes"])
            total_loss += self.task_weights["grounding"] * grounding_loss
        
        # Similarity loss
        if "similarity_labels" in y_true and "similarity_score" in y_pred:
            similarity_loss = self.similarity_loss(y_true["similarity_labels"], y_pred["similarity_score"])
            total_loss += self.task_weights["similarity"] * similarity_loss
        
        # Classification loss
        if "class_labels" in y_true and "classification_logits" in y_pred:
            class_loss = self.classification_loss(y_true["class_labels"], y_pred["classification_logits"])
            total_loss += self.task_weights["classification"] * class_loss
        
        # Q-learning loss for ACT
        if "target_q_continue" in y_true and "q_continue_logits" in y_pred:
            q_loss = self.q_loss(y_true["target_q_continue"], ops.sigmoid(y_pred["q_continue_logits"]))
            total_loss += self.q_learning_weight * q_loss
        
        return total_loss
```

### Training Loop

```python
def train_vlm_hrm(
    model: VisionLanguageHRM,
    train_dataset,
    val_dataset,
    epochs: int = 100,
    initial_lr: float = 1e-4,
    warmup_steps: int = 10000
):
    """
    Training loop for VLM-HRM with multi-task learning.
    
    Args:
        model: VLM-HRM model instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        initial_lr: Initial learning rate
        warmup_steps: Warmup steps for learning rate schedule
    """
    
    # Loss function
    loss_fn = VLMMultiTaskLoss()
    
    # Optimizer with warmup
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=warmup_steps,
        t_mul=2.0,
        m_mul=0.9
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    # Metrics
    metrics = {
        "vqa_accuracy": keras.metrics.SparseCategoricalAccuracy(),
        "caption_perplexity": keras.metrics.SparseCategoricalCrossentropy(),
        "grounding_iou": keras.metrics.MeanSquaredError(),
        "similarity_auc": keras.metrics.BinaryAccuracy(),
    }
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=list(metrics.values())
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="vlm_hrm_best.keras",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

---

## Usage Patterns

### 1. Visual Question Answering

```python
def vqa_inference(model, image, question, tokenizer):
    """
    Perform VQA inference with adaptive reasoning.
    
    Args:
        model: Trained VLM-HRM model
        image: Input image [height, width, channels]
        question: Text question string
        tokenizer: Text tokenizer
        
    Returns:
        Predicted answer string
    """
    # Prepare inputs
    image_batch = ops.expand_dims(image, 0)  # Add batch dimension
    question_tokens = tokenizer.encode(question, max_length=512, padding=True)
    question_batch = ops.expand_dims(question_tokens, 0)
    task_type_batch = ops.constant([VLMTaskType.VQA.value])
    
    # Run inference
    inputs = {
        "images": image_batch,
        "text_ids": question_batch,
        "task_type": task_type_batch
    }
    
    outputs = model(inputs, training=False)
    
    # Decode answer
    answer_logits = outputs["vqa_logits"]
    answer_token = ops.argmax(answer_logits, axis=-1)[0]
    answer = tokenizer.decode(answer_token)
    
    return answer

# Example usage
image = load_image("path/to/image.jpg")
question = "What color is the car in the image?"
answer = vqa_inference(model, image, question, tokenizer)
print(f"Q: {question}")
print(f"A: {answer}")
```

### 2. Image Captioning with Iterative Refinement

```python
def generate_caption_iterative(model, image, tokenizer, max_length=50):
    """
    Generate image caption with iterative refinement.
    
    Args:
        model: Trained VLM-HRM model
        image: Input image
        tokenizer: Text tokenizer
        max_length: Maximum caption length
        
    Returns:
        Generated caption string
    """
    # Initialize
    batch_size = 1
    image_batch = ops.expand_dims(image, 0)
    task_type_batch = ops.constant([VLMTaskType.CAPTIONING.value])
    
    # Start with [START] token
    caption_tokens = [tokenizer.start_token_id]
    
    # Generate iteratively
    for _ in range(max_length):
        # Pad caption tokens
        padded_tokens = tokenizer.pad_sequence(caption_tokens, max_length=512)
        text_batch = ops.expand_dims(padded_tokens, 0)
        
        # Forward pass
        inputs = {
            "images": image_batch,
            "text_ids": text_batch,
            "task_type": task_type_batch
        }
        
        outputs = model(inputs, training=False)
        
        # Get next token prediction
        caption_logits = outputs["caption_logits"]
        next_token_logits = caption_logits[0, len(caption_tokens)-1, :]
        next_token = ops.argmax(next_token_logits, axis=-1)
        
        # Add to caption
        caption_tokens.append(int(next_token))
        
        # Stop if end token
        if next_token == tokenizer.end_token_id:
            break
    
    # Decode caption
    caption = tokenizer.decode(caption_tokens[1:-1])  # Remove start/end tokens
    return caption

# Example usage
image = load_image("path/to/image.jpg")
caption = generate_caption_iterative(model, image, tokenizer)
print(f"Caption: {caption}")
```

### 3. Multi-Modal Chain-of-Thought Reasoning

```python
def multimodal_chain_of_thought(model, image, complex_question, tokenizer):
    """
    Perform multi-modal chain-of-thought reasoning.
    
    Args:
        model: Trained VLM-HRM model
        image: Input image
        complex_question: Complex reasoning question
        tokenizer: Text tokenizer
        
    Returns:
        Reasoning chain and final answer
    """
    # Prepare inputs
    image_batch = ops.expand_dims(image, 0)
    question_tokens = tokenizer.encode(complex_question, max_length=512, padding=True)
    question_batch = ops.expand_dims(question_tokens, 0)
    task_type_batch = ops.constant([VLMTaskType.REASONING.value])
    
    # Initialize carry state
    batch = {
        "images": image_batch,
        "text_ids": question_batch,
        "task_type": task_type_batch
    }
    carry = model.initial_carry(batch)
    
    reasoning_steps = []
    
    # Step through reasoning process
    for step in range(model.halt_max_steps):
        carry, outputs, finished = model._forward_step(carry, batch, training=False)
        
        # Extract intermediate reasoning (simplified)
        # In practice, you'd have a separate head for thought extraction
        step_reasoning = f"Step {step + 1}: Processing visual and textual information..."
        reasoning_steps.append(step_reasoning)
        
        if finished:
            break
    
    # Final answer
    final_answer_logits = outputs["vqa_logits"]
    final_answer_token = ops.argmax(final_answer_logits, axis=-1)[0]
    final_answer = tokenizer.decode(final_answer_token)
    
    return reasoning_steps, final_answer

# Example usage
image = load_image("path/to/complex_scene.jpg")
question = "Why might the person in the image be looking worried, and what should they do?"
reasoning_steps, answer = multimodal_chain_of_thought(model, image, question, tokenizer)

print("Reasoning Process:")
for i, step in enumerate(reasoning_steps):
    print(f"{i+1}. {step}")
print(f"\nFinal Answer: {answer}")
```

---

## Advanced Features

### 1. Attention Visualization

```python
def visualize_cross_modal_attention(model, image, text, tokenizer):
    """
    Visualize cross-modal attention patterns.
    
    Args:
        model: VLM-HRM model
        image: Input image
        text: Input text
        tokenizer: Text tokenizer
        
    Returns:
        Attention maps for visualization
    """
    # Prepare inputs
    image_batch = ops.expand_dims(image, 0)
    text_tokens = tokenizer.encode(text, max_length=512, padding=True)
    text_batch = ops.expand_dims(text_tokens, 0)
    task_type_batch = ops.constant([VLMTaskType.VQA.value])
    
    inputs = {
        "images": image_batch,
        "text_ids": text_batch,
        "task_type": task_type_batch
    }
    
    # Extract attention weights (requires model modification)
    # This would need hooks or explicit attention output in the model
    with keras.utils.custom_object_scope({"attention_outputs": True}):
        outputs = model(inputs, training=False)
    
    # Process attention maps for visualization
    attention_maps = {}
    
    # Vision-to-text attention
    if "vision_text_attention" in outputs:
        attention_maps["vision_to_text"] = outputs["vision_text_attention"]
    
    # Text-to-vision attention  
    if "text_vision_attention" in outputs:
        attention_maps["text_to_vision"] = outputs["text_vision_attention"]
    
    return attention_maps
```

### 2. Adaptive Computation Analysis

```python
def analyze_reasoning_depth(model, dataset, num_samples=1000):
    """
    Analyze reasoning depth patterns across different inputs.
    
    Args:
        model: VLM-HRM model
        dataset: Evaluation dataset
        num_samples: Number of samples to analyze
        
    Returns:
        Statistics about reasoning patterns
    """
    reasoning_stats = {
        "steps_per_task": {task.name: [] for task in VLMTaskType},
        "complexity_analysis": [],
        "halt_decisions": []
    }
    
    for i, batch in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break
            
        # Track reasoning steps
        carry = model.initial_carry(batch)
        steps_taken = 0
        
        for step in range(model.halt_max_steps):
            carry, outputs, finished = model._forward_step(
                carry, batch, training=False
            )
            steps_taken += 1
            
            # Record halt decisions
            q_halt = outputs["q_halt_logits"]
            q_continue = outputs["q_continue_logits"]
            halt_confidence = ops.sigmoid(q_halt - q_continue)
            
            reasoning_stats["halt_decisions"].append({
                "step": step,
                "halt_confidence": float(halt_confidence[0]),
                "task_type": int(batch["task_type"][0])
            })
            
            if finished:
                break
        
        # Record steps per task
        task_type = VLMTaskType(int(batch["task_type"][0]))
        reasoning_stats["steps_per_task"][task_type.name].append(steps_taken)
    
    # Compute statistics
    summary = {}
    for task_name, steps_list in reasoning_stats["steps_per_task"].items():
        if steps_list:
            summary[task_name] = {
                "mean_steps": np.mean(steps_list),
                "std_steps": np.std(steps_list),
                "min_steps": np.min(steps_list),
                "max_steps": np.max(steps_list)
            }
    
    return summary, reasoning_stats
```

### 3. Task-Specific Fine-Tuning

```python
def fine_tune_for_specific_task(
    model: VisionLanguageHRM,
    task_type: VLMTaskType,
    task_dataset,
    epochs: int = 10,
    learning_rate: float = 1e-5
):
    """
    Fine-tune VLM-HRM for a specific task.
    
    Args:
        model: Pre-trained VLM-HRM model
        task_type: Specific task to fine-tune for
        task_dataset: Task-specific dataset
        epochs: Number of fine-tuning epochs
        learning_rate: Fine-tuning learning rate
        
    Returns:
        Fine-tuned model
    """
    # Freeze core reasoning modules (optional)
    model.core.h_reasoning.trainable = False
    model.core.l_reasoning.trainable = False
    
    # Only train task-specific output head
    if task_type == VLMTaskType.VQA:
        # Unfreeze only VQA head
        model.output_system.vqa_head.trainable = True
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        
    elif task_type == VLMTaskType.CAPTIONING:
        # Unfreeze only caption head
        model.output_system.caption_head.trainable = True
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
        
    # Configure optimizer for fine-tuning
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.001
    )
    
    # Compile for specific task
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    # Fine-tune
    history = model.fit(
        task_dataset,
        epochs=epochs,
        verbose=1
    )
    
    return model, history
```

---

## Performance Optimization

### 1. Memory Optimization

```python
def optimize_memory_usage(model_config):
    """
    Optimize VLM-HRM configuration for memory efficiency.
    
    Args:
        model_config: Base model configuration
        
    Returns:
        Memory-optimized configuration
    """
    optimized_config = model_config.copy()
    
    # Reduce dimensions for memory efficiency
    optimized_config.update({
        "embed_dim": 384,           # Reduced from 512
        "vision_embed_dim": 576,    # Reduced from 768
        "text_embed_dim": 384,      # Reduced from 512
        "h_layers": 3,              # Reduced from 4
        "l_layers": 3,              # Reduced from 4
        "num_heads": 6,             # Reduced from 8
        "max_text_length": 256,     # Reduced from 512
        "halt_max_steps": 8,        # Reduced from 16
    })
    
    return optimized_config

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for memory savings during training.
    
    Args:
        model: VLM-HRM model instance
    """
    # Enable gradient checkpointing in reasoning modules
    for layer in model.core.h_reasoning.blocks:
        layer.activation_checkpointing = True
        
    for layer in model.core.l_reasoning.blocks:
        layer.activation_checkpointing = True
```

### 2. Computational Optimization

```python
def optimize_for_inference(model):
    """
    Optimize model for inference speed.
    
    Args:
        model: Trained VLM-HRM model
        
    Returns:
        Optimized model for inference
    """
    # Reduce reasoning cycles for faster inference
    model.core.h_cycles = 1
    model.core.l_cycles = 1
    
    # Disable dropout
    model.core.dropout_rate = 0.0
    
    # Set to evaluation mode
    model.trainable = False
    
    return model

def batch_inference_optimization(batch_size: int = 8):
    """
    Optimize inference for batch processing.
    
    Args:
        batch_size: Optimal batch size for inference
        
    Returns:
        Optimized inference function
    """
    @tf.function
    def optimized_inference(model, image_batch, text_batch, task_batch):
        """Optimized batch inference function."""
        inputs = {
            "images": image_batch,
            "text_ids": text_batch,
            "task_type": task_batch
        }
        return model(inputs, training=False)
    
    return optimized_inference
```

---

## Complete Implementation

### Factory Function

```python
def create_vlm_hrm(
    vocab_size: int,
    config: VLMConfig = None,
    **kwargs
) -> VisionLanguageHRM:
    """
    Factory function to create VLM-HRM model.
    
    Args:
        vocab_size: Vocabulary size for text processing
        config: VLM configuration object
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured VisionLanguageHRM model
    """
    if config is None:
        config = VLMConfig()
    
    # Override config with kwargs
    config_dict = {
        "vocab_size": vocab_size,
        "embed_dim": config.EMBED_DIM,
        "image_size": config.IMAGE_SIZE,
        "patch_size": config.PATCH_SIZE,
        "vision_embed_dim": config.VISION_EMBED_DIM,
        "text_embed_dim": config.TEXT_EMBED_DIM,
        "max_text_length": config.MAX_TEXT_LENGTH,
        "h_layers": config.H_LAYERS,
        "l_layers": config.L_LAYERS,
        "h_cycles": config.H_CYCLES,
        "l_cycles": config.L_CYCLES,
        "num_heads": config.NUM_HEADS,
        "ffn_expansion_factor": config.FFN_EXPANSION,
        "halt_max_steps": config.MAX_REASONING_STEPS,
        "halt_exploration_prob": config.HALT_EXPLORATION_PROB,
        "dropout_rate": config.DROPOUT_RATE,
        "use_bias": config.USE_BIAS,
    }
    
    config_dict.update(kwargs)
    
    return VisionLanguageHRM(**config_dict)

# Example usage
def main():
    """Example usage of VLM-HRM."""
    
    # Create model
    model = create_vlm_hrm(
        vocab_size=32000,
        embed_dim=512,
        h_layers=4,
        l_layers=4,
        halt_max_steps=12
    )
    
    # Example inference
    batch_size = 2
    image_batch = tf.random.normal((batch_size, 224, 224, 3))
    text_batch = tf.random.uniform((batch_size, 512), 0, 32000, dtype=tf.int32)
    task_batch = tf.constant([VLMTaskType.VQA.value, VLMTaskType.CAPTIONING.value])
    
    inputs = {
        "images": image_batch,
        "text_ids": text_batch,
        "task_type": task_batch
    }
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    print("VLM-HRM Model created successfully!")
    print(f"VQA outputs shape: {outputs['vqa_logits'].shape}")
    print(f"Caption outputs shape: {outputs['caption_logits'].shape}")
    print(f"Halt decisions: {outputs['q_halt_logits']}")

if __name__ == "__main__":
    main()
```

---

## Conclusion

This implementation guide provides a complete framework for building Vision Language Models with Hierarchical Reasoning capabilities. The architecture combines:

- **🧠 Adaptive Reasoning**: Variable computation depth based on input complexity
- **🔄 Multi-Modal Fusion**: Sophisticated vision-text integration
- **🎯 Multi-Task Learning**: Unified architecture for diverse VLM tasks
- **⚡ Efficient Training**: Gradient checkpointing and optimized computation
- **🔍 Interpretability**: Attention visualization and reasoning analysis

The VLM-HRM represents a significant advancement in multi-modal AI, enabling more intelligent and adaptive vision-language understanding through hierarchical reasoning processes.

### Next Steps

1. **Dataset Preparation**: Create multi-task datasets with proper annotations
2. **Distributed Training**: Implement multi-GPU/multi-node training
3. **Model Scaling**: Experiment with larger model sizes and datasets
4. **Task-Specific Optimization**: Fine-tune for specialized applications
5. **Deployment**: Optimize for production inference scenarios

The architecture is designed to be modular and extensible, allowing for easy customization and improvement based on specific use cases and research directions.