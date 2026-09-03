"""
Multi Level Feature Compilation (MLFC) layer for cross-scale feature fusion.

Implements the ACC-UNet MLFC block, ``MLFCLayer``, which enriches each of 4
encoder-level feature maps with information from the other 3. A standard
U-Net skip connection carries only a single encoder level to its matching
decoder level; MLFC instead resizes all 4 levels to each level's own spatial
size, concatenates them, and compiles the result back down through a 1x1
convolution before adding it as a residual. This runs ``num_iterations``
times, each iteration mixing further using the previous iteration's output.

The layer takes and returns a list of exactly 4 tensors, one per encoder
level, and ``channels_list`` must give their 4 channel counts in that order.
A trailing squeeze-excitation recalibration is applied to each level once,
after the last iteration.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, List, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .squeeze_excitation import SqueezeExcitation
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.multi_level_feature_compilation")
class MLFCLayer(keras.layers.Layer):
    """Multi Level Feature Compilation (MLFC) Layer.

    This layer implements multi-level feature fusion from the ACC-UNet
    architecture by resizing feature maps from 4 encoder levels to common
    dimensions, concatenating them along the channel axis, processing through
    1x1 convolutions, and applying residual connections with squeeze-excitation
    recalibration. For each level ``i`` at iteration ``t``:
    ``F_concat = Concat([Resize(F_j, size_i) for j in 1..4])``,
    ``F_compiled = Conv1x1(F_concat)``,
    ``F_merged = Conv1x1(Concat([F_compiled, F_i])) + F_i``.

    Architecture:

    .. code-block:: text

        ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Level 1 │  │ Level 2 │  │ Level 3 │  │ Level 4 │
        │ [H,W,c1]│  │[H/2,c2] │  │[H/4,c3] │  │[H/8,c4] │
        └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │            │
             ▼            ▼            ▼            ▼
        ┌──────────────────────────────────────────────────┐
        │  For each level i (repeat num_iterations times): │
        │    1. Resize all 4 levels to level i dimensions  │
        │    2. Concatenate → Conv1x1 → BN → Activation    │
        │    3. Concat with original → Conv1x1 → + residual│
        └──────────────────────────────────────────────────┘
             │            │            │            │
             ▼            ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
        │   SE    │  │   SE    │  │   SE    │  │   SE    │
        └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │            │
             ▼            ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Output 1│  │ Output 2│  │ Output 3│  │ Output 4│
        └─────────┘  └─────────┘  └─────────┘  └─────────┘

    :param channels_list: Channel counts for each of the 4 levels ``[c1, c2, c3, c4]``.
        Must contain exactly 4 positive integers.
    :type channels_list: List[int]
    :param num_iterations: Number of compilation iterations. Must be positive.
        Defaults to 1.
    :type num_iterations: int
    :param kernel_initializer: Initializer for convolution kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for convolution kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If channels_list doesn't have exactly 4 elements.
    :raises ValueError: If any channel count is not positive.
    :raises ValueError: If num_iterations is not positive.
    """

    def __init__(
        self,
        channels_list: List[int],
        num_iterations: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if len(channels_list) != 4:
            raise ValueError(f"channels_list must have exactly 4 elements, got {len(channels_list)}")

        if any(c <= 0 for c in channels_list):
            raise ValueError(f"All channel counts must be positive, got {channels_list}")

        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")

        # Store configuration parameters.
        self.channels_list = channels_list
        self.num_iterations = num_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.total_channels = sum(channels_list)

        # Create all sub-layers in __init__.
        # These layers are created but not built yet

        # Use flat lists for sub-layers for robust serialization
        self.compilation_convs: List[keras.layers.Layer] = []
        self.merge_convs: List[keras.layers.Layer] = []
        self.batch_norms: List[keras.layers.Layer] = []
        self.merge_batch_norms: List[keras.layers.Layer] = []

        # Create layers for each iteration and level
        for iter_idx in range(self.num_iterations):
            for level_idx in range(4):
                channels = self.channels_list[level_idx]

                # Compilation convolution (total_channels -> level_channels)
                comp_conv = keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'comp_conv_iter{iter_idx}_level{level_idx}'
                )
                self.compilation_convs.append(comp_conv)

                # Compilation batch normalization
                comp_bn = keras.layers.BatchNormalization(
                    name=f'comp_bn_iter{iter_idx}_level{level_idx}'
                )
                self.batch_norms.append(comp_bn)

                # Merge convolution (2*channels -> channels)
                merge_conv = keras.layers.Conv2D(
                    filters=channels,
                    kernel_size=1,
                    padding='same',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'merge_conv_iter{iter_idx}_level{level_idx}'
                )
                self.merge_convs.append(merge_conv)

                # Merge batch normalization
                merge_bn = keras.layers.BatchNormalization(
                    name=f'merge_bn_iter{iter_idx}_level{level_idx}'
                )
                self.merge_batch_norms.append(merge_bn)

        # Squeeze-excitation for each level (applied once at the end)
        self.squeeze_excitations: List[keras.layers.Layer] = []
        for level_idx in range(4):
            se = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'se_level{level_idx}'
            )
            self.squeeze_excitations.append(se)

        # Activation layer
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer and all its sub-layers.

        :param input_shape: List of 4 input shapes for the 4 encoder levels.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If input_shape is not a list of 4 shapes.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError(
                f"input_shape must be a list of 4 shapes, got {type(input_shape)} "
                f"with length {len(input_shape) if isinstance(input_shape, list) else 'N/A'}"
            )

        # Build all sub-layers explicitly for robust serialization
        # This ensures all weight variables exist before weight restoration during loading

        # Build compilation and merge layers
        for iter_idx in range(self.num_iterations):
            for level_idx in range(4):
                idx = iter_idx * 4 + level_idx

                # Build compilation conv on the concatenated (all channels) shape.
                concat_shape = list(input_shape[level_idx])
                concat_shape[-1] = self.total_channels
                self.compilation_convs[idx].build(tuple(concat_shape))

                # Build compilation batch norm
                comp_output_shape = list(concat_shape)
                comp_output_shape[-1] = self.channels_list[level_idx]
                self.batch_norms[idx].build(tuple(comp_output_shape))

                # Build merge conv with 2x channels input
                merge_input_shape = list(input_shape[level_idx])
                merge_input_shape[-1] = 2 * self.channels_list[level_idx]
                self.merge_convs[idx].build(tuple(merge_input_shape))

                # Build merge batch norm
                merge_output_shape = list(input_shape[level_idx])
                merge_output_shape[-1] = self.channels_list[level_idx]
                self.merge_batch_norms[idx].build(tuple(merge_output_shape))

        # Build squeeze-excitation layers
        for level_idx in range(4):
            self.squeeze_excitations[level_idx].build(input_shape[level_idx])

        # LeakyReLU needs no explicit build; any shape works.
        self.activation.build(input_shape[0])

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Forward pass performing iterative cross-level feature compilation.

        :param inputs: List of 4 input tensors from different encoder levels.
        :type inputs: List[keras.KerasTensor]
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: List of 4 output tensors with enriched features.
        :rtype: List[keras.KerasTensor]
        :raises ValueError: If inputs doesn't contain exactly 4 tensors.
        """
        if len(inputs) != 4:
            raise ValueError(f"Expected 4 input tensors, got {len(inputs)}")

        x1, x2, x3, x4 = inputs

        # Apply multiple compilation iterations
        for iter_idx in range(self.num_iterations):
            # Get current feature maps
            current_features = [x1, x2, x3, x4]
            new_features = []

            # Process each level
            for level_idx in range(4):
                idx = iter_idx * 4 + level_idx
                target_shape = ops.shape(current_features[level_idx])
                target_height = target_shape[1]
                target_width = target_shape[2]

                # Resize all features to current level's spatial dimensions using ops
                resized_features = []
                for feat_idx, feat in enumerate(current_features):
                    if feat_idx == level_idx:
                        # Same level, no resizing needed
                        resized_features.append(feat)
                    else:
                        # Use keras.ops for resizing to ensure proper serialization
                        feat_resized = keras.ops.image.resize(
                            feat,
                            size=(target_height, target_width),
                            interpolation='bilinear'
                        )
                        resized_features.append(feat_resized)

                # Concatenate all resized features using ops
                concatenated = ops.concatenate(resized_features, axis=-1)

                # Apply compilation convolution
                compiled_feat = self.compilation_convs[idx](concatenated)
                compiled_feat = self.batch_norms[idx](compiled_feat, training=training)
                compiled_feat = self.activation(compiled_feat)

                # Merge with original features using residual connection
                original_feat = current_features[level_idx]
                merged_input = ops.concatenate([compiled_feat, original_feat], axis=-1)

                # Apply merge convolution with residual
                merged_feat = self.merge_convs[idx](merged_input)
                merged_feat = self.merge_batch_norms[idx](merged_feat, training=training)
                # Residual connection.
                merged_feat = merged_feat + original_feat
                merged_feat = self.activation(merged_feat)

                new_features.append(merged_feat)

            # Update features for next iteration
            x1, x2, x3, x4 = new_features

        # Apply final squeeze-excitation to each level
        final_features = []
        for level_idx, feat in enumerate([x1, x2, x3, x4]):
            feat = self.squeeze_excitations[level_idx](feat)
            final_features.append(feat)

        return final_features

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> List[Tuple[Optional[int], ...]]:
        """Compute output shapes.

        :param input_shape: List of input shapes for the 4 encoder levels.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: List of output shapes (identical to input shapes).
        :rtype: List[Tuple[Optional[int], ...]]
        """
        return input_shape  # Shapes remain unchanged

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'channels_list': self.channels_list,
            'num_iterations': self.num_iterations,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
