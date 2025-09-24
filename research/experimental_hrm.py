# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import keras
from keras import ops, layers, initializers, regularizers, activations
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import numpy as np
import tensorflow as tf  # Used for specific ops like where and tape in training

# DL-Techniques framework imports (assuming they exist in the project)
from dl_techniques.utils.logger import logger


# ==============================================================================
# HELPER COMPONENTS & CONFIGURATIONS
# ==============================================================================

class VLMConfig:
    """Configuration for the Vision Language Model with Hierarchical Reasoning."""
    # Vision settings
    IMAGE_SIZE: int = 224
    PATCH_SIZE: int = 16
    VISION_EMBED_DIM: int = 768

    # Text settings
    VOCAB_SIZE: int = 32000
    MAX_TEXT_LENGTH: int = 256
    TEXT_EMBED_DIM: int = 768

    # Shared Model settings
    MODEL_DIM: int = 768
    NUM_HEADS: int = 12
    DROPOUT_RATE: float = 0.1

    # HRM settings
    H_LAYERS: int = 4  # High-level reasoning layers
    L_LAYERS: int = 4  # Low-level reasoning layers
    H_CYCLES: int = 1  # High-level reasoning cycles per step
    L_CYCLES: int = 2  # Low-level reasoning cycles per H-cycle
    FFN_EXPANSION: int = 4

    # ACT (Adaptive Computation Time) settings
    MAX_REASONING_STEPS: int = 12
    HALT_THRESHOLD: float = 0.9
    PONDERING_PENALTY: float = 0.01

    # Output heads
    NUM_VQA_CLASSES: int = 3129  # Example: VQAv2 answer space


# ==============================================================================
# HIERARCHICAL REASONING MODULE (CORE BUILDING BLOCK)
# ==============================================================================

@keras.saving.register_keras_serializable()
class HierarchicalReasoningModule(keras.layers.Layer):
    """
    A stack of transformer-style reasoning blocks for hierarchical processing.

    This layer implements a sequence of transformer encoder blocks, each
    consisting of self-attention and a feed-forward network. It serves as the
    core computational unit for both high-level and low-level reasoning in the
    VLM-HRM, demonstrating the composite layer pattern with sub-layers.

    **Intent**: Provide a modular, reusable block for deep reasoning that can
    be stacked to form powerful hierarchical architectures, while adhering to
    Keras 3 best practices for serialization.

    **Architecture**:
    ```
    Input(shape=[..., seq_len, model_dim])
           ↓
    [ For each layer in num_layers: ]
        ↓
    Residual Connection_1
        ↓
    LayerNorm → MultiHeadAttention → Add
        ↓
    Residual Connection_2
        ↓
    LayerNorm → FeedForwardNetwork → Add
        ↓
    Output(shape=[..., seq_len, model_dim])
    ```

    Args:
        num_layers (int): The number of transformer blocks to stack.
        model_dim (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads.
        ffn_dim (int): The inner dimension of the feed-forward network.
        dropout_rate (float): Dropout rate for attention and FFN layers.
        **kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            num_layers: int,
            model_dim: int,
            num_heads: int,
            ffn_dim: int,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        # CREATE all sub-layers in __init__
        self.blocks = []
        for i in range(num_layers):
            block = keras.Sequential([
                layers.LayerNormalization(epsilon=1e-6, name=f"norm1_block_{i}"),
                layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=model_dim // num_heads,
                    dropout=dropout_rate,
                    name=f"attn_block_{i}"
                ),
                layers.Add(name=f"add1_block_{i}"),
                layers.LayerNormalization(epsilon=1e-6, name=f"norm2_block_{i}"),
                layers.Dense(ffn_dim, activation="gelu", name=f"ffn1_block_{i}"),
                layers.Dropout(dropout_rate, name=f"dropout_block_{i}"),
                layers.Dense(model_dim, name=f"ffn2_block_{i}"),
                layers.Add(name=f"add2_block_{i}"),
            ], name=f"reasoning_block_{i}")
            self.blocks.append(block)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build sub-layers for robust serialization."""
        for block in self.blocks:
            # Manually implement the forward pass logic to correctly build each sub-layer.
            # This ensures Keras knows the shapes and can create weights.
            x_shape = input_shape

            # Sub-layer 0: Norm1
            block.layers[0].build(x_shape)

            # Sub-layer 1: MultiHeadAttention
            block.layers[1].build(query_shape=x_shape, value_shape=x_shape)

            # Sub-layer 2: Add1
            block.layers[2].build([x_shape, x_shape])

            # Sub-layer 3: Norm2
            block.layers[3].build(x_shape)

            # Sub-layer 4: Dense (FFN1)
            block.layers[4].build(x_shape)
            ffn1_shape = block.layers[4].compute_output_shape(x_shape)

            # Sub-layer 5: Dropout
            block.layers[5].build(ffn1_shape)

            # Sub-layer 6: Dense (FFN2)
            block.layers[6].build(ffn1_shape)

            # Sub-layer 7: Add2
            block.layers[7].build([x_shape, x_shape])

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None,
             attention_mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        """Forward pass through the reasoning module."""
        for block in self.blocks:
            # Attention block with residual connection
            residual_1 = x
            normed_x = block.layers[0](x, training=training)
            attn_output = block.layers[1](
                query=normed_x,
                value=normed_x,
                key=normed_x,
                attention_mask=attention_mask,
                training=training
            )
            x = block.layers[2]([residual_1, attn_output])

            # FFN block with residual connection
            residual_2 = x
            normed_x = block.layers[3](x, training=training)
            ffn_output = block.layers[4](normed_x)
            ffn_output = block.layers[5](ffn_output, training=training)
            ffn_output = block.layers[6](ffn_output)
            x = block.layers[7]([residual_2, ffn_output])

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


# ==============================================================================
# HIERARCHICAL REASONING CORE MODEL
# ==============================================================================

@keras.saving.register_keras_serializable()
class VisionLanguageHRM(keras.Model):
    """
    Vision Language Model with Hierarchical Reasoning (VLM-HRM).

    This model integrates a vision_heads encoder and a text encoder with a dual-level
    hierarchical reasoning core. It uses Adaptive Computation Time (ACT) to
    dynamically adjust the number of reasoning steps based on input complexity,
    and supports multiple vision_heads-language tasks through a unified architecture.

    **Intent**: To create a state-of-the-art VLM that explicitly models
    hierarchical reasoning, balancing computational efficiency with deep
    semantic understanding for complex multi-modal tasks.

    **Architecture**:
    ```
    ┌──────────────────────────┐   ┌──────────────────────────┐
    │       Image Input        │   │        Text Input        │
    └────────────┬─────────────┘   └────────────┬─────────────┘
                 │                              │
           Vision Encoder                  Text Embedder
                 │                              │
                 └──────────────┬───────────────┘
                              ↓
                  [ Multi-Modal Input Processor ]
                              ↓
    ┌───────────────────────[ HRM Core with ACT ]─────────────────────────┐
    │                                                                     │
    │  Initial State -> [High-Level Reasoner] -> [Low-Level Reasoner] ... │
    │       ↑                  │                  │                       │
    │       └──────────────────┴──────────────────┘ (Iterative Loop)      │
    │                                                                     │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       ↓
                        [ Task-Adaptive Output System ]
                                       │
                ┌──────────┬───────────┼───────────┬──────────┐
                ↓          ↓           ↓           ↓          ↓
              VQA      Captioning   Grounding   Retrieval   ...
    ```

    Args:
        config (VLMConfig): A configuration object containing all model parameters.
        **kwargs: Additional arguments for the Model base class.
    """

    def __init__(self, config: VLMConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

        # 1. CREATE Multi-Modal Input Processor
        # Vision Encoder (assuming a pre-trained or standard ViT)
        self.vision_encoder = self._create_vision_encoder()
        # Projector to match language model dimension
        self.vision_projector = layers.Dense(config.MODEL_DIM, name="vision_projector")
        # Text Embeddings
        self.text_embedder = layers.Embedding(
            config.VOCAB_SIZE, config.MODEL_DIM, mask_zero=True, name="text_embedder"
        )
        self.text_pos_embedder = layers.Embedding(
            config.MAX_TEXT_LENGTH, config.MODEL_DIM, name="text_pos_embedder"
        )
        # Modality Type Embeddings
        self.modality_embedder = layers.Embedding(3, config.MODEL_DIM,
                                                  name="modality_embedder")  # 0: Vision, 1: Text, 2: Task

        # 2. CREATE Hierarchical Reasoning Modules
        self.h_reasoner = HierarchicalReasoningModule(
            num_layers=config.H_LAYERS,
            model_dim=config.MODEL_DIM,
            num_heads=config.NUM_HEADS,
            ffn_dim=config.MODEL_DIM * config.FFN_EXPANSION,
            dropout_rate=config.DROPOUT_RATE,
            name="high_level_reasoner"
        )
        self.l_reasoner = HierarchicalReasoningModule(
            num_layers=config.L_LAYERS,
            model_dim=config.MODEL_DIM,
            num_heads=config.NUM_HEADS,
            ffn_dim=config.MODEL_DIM * config.FFN_EXPANSION,
            dropout_rate=config.DROPOUT_RATE,
            name="low_level_reasoner"
        )

        # 3. CREATE Task-Adaptive Output System
        self.vqa_head = layers.Dense(config.NUM_VQA_CLASSES, name="vqa_head")
        self.captioning_head = layers.Dense(config.VOCAB_SIZE, name="captioning_head")
        # Halting probability head for ACT
        self.halt_head = layers.Dense(1, activation='sigmoid', name="halt_head")

        # Initial state vectors (will be created in build)
        self.h_init_state = None
        self.l_init_state = None
        self.task_token = None

    def _create_vision_encoder(self):
        # A simplified ViT-like encoder for demonstration. In a real scenario,
        # you would load a pre-trained model like SigLIP or DINOv2.
        config = self.config
        vision_input = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        patches = layers.Conv2D(
            config.VISION_EMBED_DIM,
            kernel_size=config.PATCH_SIZE,
            strides=config.PATCH_SIZE,
            padding="valid",
            name="patch_projection"
        )(vision_input)
        patches_reshaped = layers.Reshape((-1, config.VISION_EMBED_DIM))(patches)

        # Simple Transformer Encoder for vision_heads
        encoded_patches = HierarchicalReasoningModule(
            num_layers=4,  # A shallow encoder for this example
            model_dim=config.VISION_EMBED_DIM,
            num_heads=config.NUM_HEADS,
            ffn_dim=config.VISION_EMBED_DIM * config.FFN_EXPANSION,
            dropout_rate=config.DROPOUT_RATE,
            name="vision_transformer_encoder"
        )(patches_reshaped)

        return keras.Model(vision_input, encoded_patches, name="vision_encoder")

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """Create the model's own weights and build sub-layers."""
        config = self.config

        # Build sub-layers with their expected input shapes
        self.vision_encoder.build((None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
        num_patches = (config.IMAGE_SIZE // config.PATCH_SIZE) ** 2
        self.vision_projector.build((None, num_patches, config.VISION_EMBED_DIM))
        self.text_embedder.build((None, config.MAX_TEXT_LENGTH))
        self.text_pos_embedder.build((None, config.MAX_TEXT_LENGTH))
        self.modality_embedder.build((None,))

        # Total sequence length after fusion
        total_seq_len = 1 + num_patches + config.MAX_TEXT_LENGTH  # Task + Vision + Text
        reasoning_input_shape = (None, total_seq_len, config.MODEL_DIM)

        self.h_reasoner.build(reasoning_input_shape)
        self.l_reasoner.build(reasoning_input_shape)

        # Build output heads
        self.vqa_head.build((None, config.MODEL_DIM))
        self.captioning_head.build(reasoning_input_shape)
        self.halt_head.build((None, config.MODEL_DIM))

        # CREATE model's own weights
        self.h_init_state = self.add_weight(
            name="h_init_state", shape=(config.MODEL_DIM,), initializer="zeros"
        )
        self.l_init_state = self.add_weight(
            name="l_init_state", shape=(config.MODEL_DIM,), initializer="zeros"
        )
        self.task_token = self.add_weight(
            name="task_token", shape=(1, 1, config.MODEL_DIM), initializer="random_normal"
        )

        super().build(input_shape)

    def _prepare_inputs(self, images: keras.KerasTensor, text_ids: keras.KerasTensor) -> Tuple[
        keras.KerasTensor, keras.KerasTensor]:
        """Processes and fuses vision_heads and text inputs."""
        config = self.config
        batch_size = ops.shape(images)[0]

        # 1. Process Vision Input
        vision_features = self.vision_encoder(images, training=False)  # Assume frozen encoder
        vision_embeddings = self.vision_projector(vision_features)
        vision_embeddings += self.modality_embedder(
            ops.zeros(ops.shape(vision_embeddings)[:-1], dtype='int32'))  # Modality 0

        # 2. Process Text Input
        text_embeddings = self.text_embedder(text_ids)
        text_positions = ops.arange(0, ops.shape(text_ids)[1])
        text_embeddings += self.text_pos_embedder(text_positions)
        text_embeddings += self.modality_embedder(
            ops.ones(ops.shape(text_embeddings)[:-1], dtype='int32'))  # Modality 1

        # 3. Create Task Token
        task_token_batch = ops.tile(self.task_token, [batch_size, 1, 1])
        task_token_batch += self.modality_embedder(ops.ones((batch_size, 1), dtype='int32') * 2)  # Modality 2

        # 4. Combine inputs
        combined_embeddings = ops.concatenate([task_token_batch, vision_embeddings, text_embeddings], axis=1)

        # Create attention mask (masking padding in text)
        text_mask = ops.cast(ops.not_equal(text_ids, 0), dtype=combined_embeddings.dtype)
        vision_mask = ops.ones((batch_size, ops.shape(vision_embeddings)[1]), dtype=text_mask.dtype)
        task_mask = ops.ones((batch_size, 1), dtype=text_mask.dtype)
        attention_mask = ops.concatenate([task_mask, vision_mask, text_mask], axis=1)

        return combined_embeddings, attention_mask

    def call(self, inputs: Dict[str, keras.KerasTensor], training: Optional[bool] = None) -> Dict[
        str, keras.KerasTensor]:
        """
        Forward pass with Adaptive Computation Time (ACT).
        """
        images = inputs["images"]
        text_ids = inputs["text_ids"]
        batch_size = ops.shape(images)[0]

        # Prepare initial multi-modal embeddings
        input_embeddings, attention_mask = self._prepare_inputs(images, text_ids)

        # Initialize HRM states
        h_state = ops.tile(ops.expand_dims(self.h_init_state, 0), [batch_size, ops.shape(input_embeddings)[1], 1])
        l_state = ops.tile(ops.expand_dims(self.l_init_state, 0), [batch_size, ops.shape(input_embeddings)[1], 1])

        # Initialize ACT variables
        halting_probability = ops.zeros((batch_size, 1))
        remainders = ops.ones((batch_size, 1))
        n_updates = ops.zeros((batch_size, 1))

        # Main ACT loop (symbolic loop for graph compilation)
        for step in range(self.config.MAX_REASONING_STEPS):
            # Calculate halting probability from high-level state
            task_state = h_state[:, 0, :]  # Use task token state for halting
            p = self.halt_head(task_state)

            # Update halting state
            still_running = ops.cast(ops.less(halting_probability, self.config.HALT_THRESHOLD), 'float32')
            new_halted = ops.cast(
                ops.greater_equal(halting_probability + p * still_running, self.config.HALT_THRESHOLD), 'float32')
            still_running = ops.cast(ops.less(halting_probability + p * still_running, self.config.HALT_THRESHOLD),
                                     'float32')

            halting_probability += p * still_running
            remainders -= p * still_running

            # Update states for running sequences
            update_weights = p * still_running + new_halted * remainders
            update_weights_expanded = ops.expand_dims(update_weights, axis=-1)

            # Hierarchical Reasoning Step
            new_h_state, new_l_state = self._reasoning_step(
                h_state, l_state, input_embeddings, attention_mask, training
            )

            h_state = h_state * (1.0 - update_weights_expanded) + new_h_state * update_weights_expanded
            l_state = l_state * (1.0 - update_weights_expanded) + new_l_state * update_weights_expanded

            n_updates += still_running + new_halted

            # Early exit if all sequences have halted
            if ops.all(ops.greater_equal(halting_probability, self.config.HALT_THRESHOLD)):
                break

        # Final pondering penalty for ACT
        ponder_cost = ops.mean(n_updates + remainders)
        self.add_loss(self.config.PONDERING_PENALTY * ponder_cost)

        # Generate outputs from final high-level state
        final_task_state = h_state[:, 0, :]
        vqa_logits = self.vqa_head(final_task_state)
        captioning_logits = self.captioning_head(h_state)

        return {
            "vqa_logits": vqa_logits,
            "captioning_logits": captioning_logits,
            "ponder_cost": ponder_cost
        }

    def _reasoning_step(self, h_state, l_state, input_embeddings, attention_mask, training):
        """A single step of hierarchical reasoning."""
        # High-level reasoner takes low-level state as input
        for _ in range(self.config.H_CYCLES):
            h_state = self.h_reasoner(
                h_state + l_state,  # Inject low-level context
                training=training,
                attention_mask=attention_mask
            )

        # Low-level reasoner takes high-level state and original input
        for _ in range(self.config.L_CYCLES):
            l_state = self.l_reasoner(
                l_state + h_state + input_embeddings,  # Inject high-level context and sensory input
                training=training,
                attention_mask=attention_mask
            )

        return h_state, l_state

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        # Serialize the config object
        config.update({
            "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VisionLanguageHRM":
        """Create model from configuration."""
        # Recreate the config object
        vlm_config_dict = config.pop("config")
        vlm_config_obj = VLMConfig()
        for k, v in vlm_config_dict.items():
            setattr(vlm_config_obj, k, v)

        return cls(config=vlm_config_obj, **config)


# ==============================================================================
# MULTI-TASK LOSS FUNCTION
# ==============================================================================

@keras.saving.register_keras_serializable()
class VLMMultiTaskLoss(keras.losses.Loss):
    """Multi-task loss for VLM-HRM training."""

    def __init__(self, vqa_weight=1.0, captioning_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vqa_weight = vqa_weight
        self.captioning_weight = captioning_weight
        self.vqa_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.captioning_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                             ignore_class=0)  # Ignore padding

    def call(self, y_true: Dict[str, keras.KerasTensor], y_pred: Dict[str, keras.KerasTensor]) -> keras.KerasTensor:
        """Compute the combined loss."""
        vqa_loss = 0.0
        if "vqa_labels" in y_true:
            vqa_loss = self.vqa_loss_fn(y_true["vqa_labels"], y_pred["vqa_logits"])

        captioning_loss = 0.0
        if "captioning_labels" in y_true:
            # Shift for autoregressive loss
            labels = y_true["captioning_labels"][:, 1:]
            logits = y_pred["captioning_logits"][:, :-1, :]
            captioning_loss = self.captioning_loss_fn(labels, logits)

        total_loss = (self.vqa_weight * vqa_loss) + (self.captioning_weight * captioning_loss)
        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vqa_weight": self.vqa_weight,
            "captioning_weight": self.captioning_weight
        })
        return config


# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

def main():
    """Example usage and demonstration of the VLM-HRM model."""
    logger.info("🚀 Initializing Vision Language Model with Hierarchical Reasoning")

    config = VLMConfig()
    model = VisionLanguageHRM(config)

    # Prepare dummy inputs
    batch_size = 2
    dummy_images = ops.random.uniform(shape=(batch_size, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    dummy_text_ids = ops.random.randint(minval=1, maxval=config.VOCAB_SIZE, shape=(batch_size, config.MAX_TEXT_LENGTH))

    inputs = {
        "images": dummy_images,
        "text_ids": dummy_text_ids
    }

    # Build the model
    model(inputs)
    logger.info("✅ Model built successfully.")
    model.summary()

    # Prepare dummy labels
    dummy_vqa_labels = ops.random.randint(minval=0, maxval=config.NUM_VQA_CLASSES, shape=(batch_size,))
    dummy_caption_labels = ops.random.randint(minval=0, maxval=config.VOCAB_SIZE,
                                              shape=(batch_size, config.MAX_TEXT_LENGTH))

    labels = {
        "vqa_labels": dummy_vqa_labels,
        "captioning_labels": dummy_caption_labels
    }

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=VLMMultiTaskLoss()
    )
    logger.info("✅ Model compiled successfully.")

    # Test a single training step
    loss = model.train_on_batch(x=inputs, y=labels)
    logger.info(f"✅ Single training step successful. Loss: {loss}")

    # Test serialization cycle
    try:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'vlm_hrm.keras')
            model.save(filepath)
            logger.info(f"✅ Model saved to {filepath}")

            loaded_model = keras.models.load_model(filepath)
            logger.info("✅ Model loaded successfully.")

            # Verify predictions match
            pred_original = model(inputs)
            pred_loaded = loaded_model(inputs)

            np.testing.assert_allclose(
                ops.convert_to_numpy(pred_original['vqa_logits']),
                ops.convert_to_numpy(pred_loaded['vqa_logits']),
                rtol=1e-5, atol=1e-5
            )
            logger.info("✅ Serialization cycle passed successfully!")

    except Exception as e:
        logger.error(f"❌ Serialization test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()