"""
Kolmogorov-Arnold Networks, which put the learnable nonlinearity on the edges
of the graph rather than on its nodes.

This model embodies a different answer to where a network's expressive power
should live. An MLP fixes the activation function -- ReLU, GELU, whatever --
and learns only the linear maps between layers, so every unit in a layer
applies the identical nonlinearity and all adaptation happens in the weights. A
KAN inverts that: there are no weight matrices and no fixed activations, only a
learnable univariate function on each connection, and nodes do nothing but sum
what arrives. The motivating result is the Kolmogorov-Arnold representation
theorem, which states that any continuous multivariate function can be written
as a finite composition of continuous univariate functions and addition. That
theorem is an existence statement about a two-layer construction, not a recipe
for a deep network, so it should be read as the intuition behind the design
rather than a guarantee about it.

Each edge function is a B-spline of order `spline_order` over a grid of
`grid_size` intervals, added to a fixed base activation. The spline is what
makes the function learnable and locally adjustable: moving one control point
changes the function only near that knot, so different regions of an edge's
input range can be shaped independently without the global interference a
single parameterized activation would suffer. The additive base activation
matters more than it looks -- it keeps a well-conditioned gradient path through
the edge in regions where the spline coefficients are still near zero, which is
what makes the layer trainable from initialization rather than only after the
splines have found signal.

The grid is the part of this architecture a caller can silently get wrong.
Splines are only defined over their knot range, and the default grid is set at
construction time from nothing but a guess about input scale. If the data
occupies a different range, every edge spends its capacity on the wrong
interval and extrapolates outside it. `update_kan_grids(x)` re-fits the knot
positions to the empirical distribution of a data sample and should be run
before training on any new dataset; it is not optional tuning, it is part of
setup.

Structurally the model is a stack of `KANLinear` layers driven by a list of
per-layer config dicts, with an optional final activation. Five preset variants
span micro (`[16, 8]`, grid 3) through xlarge (`[512, 256, 128, 64]`, grid 12),
trading grid resolution and width together, since a fine grid on a narrow layer
tends to overfit and a coarse grid on a wide one wastes it. The variant table is
named `VARIANT_CONFIGS` here rather than the house `MODEL_VARIANTS`; trainers
and tests reference the existing spelling.

No pretrained weights are distributed with this package. `pretrained=True`
raises `NotImplementedError` rather than warning and returning a randomly
initialized model, which is a deliberate choice: the previous behaviour held a
table of unreachable weight URLs and swallowed the download failure, making an
unavailable checkpoint silently indistinguishable from a successful load. Pass
a local `.keras` path to `pretrained` instead.

References:
    - Liu et al., 2024. KAN: Kolmogorov-Arnold Networks.
      (https://arxiv.org/abs/2404.19756)
    - Kolmogorov, 1957. On the representation of continuous functions of many
      variables by superposition of continuous functions of one variable and
      addition. Dokl. Akad. Nauk SSSR 114.
    - Girosi and Poggio, 1989. Representation Properties of Networks:
      Kolmogorov's Theorem Is Irrelevant. Neural Computation 1(4).
    - de Boor, 1978. A Practical Guide to Splines. Springer.
"""


import os
import keras
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.ffn.kan_linear import KANLinear

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KAN(keras.Model):
    """Modern Kolmogorov-Arnold Network model using Keras 3 functional API patterns.

    KAN stacks multiple KANLinear layers to create a deep network that can
    approximate complex multivariate functions using learnable spline-based
    activation functions on edges.

    **Architecture**:
    ```
    Input(shape=[input_features])
           ↓
    KANLinear(layer_configs[0])
           ↓
         ...
           ↓
    KANLinear(layer_configs[-1]) -> Produces logits/raw output
           ↓
    Activation(final_activation) -> Optional final transform
           ↓
    Output
    ```

    Args:
        layer_configs: List of dictionaries, each containing KANLinear configuration.
        input_features: Integer, number of input features. Must be positive.
        name: Optional string name for the model.
        **kwargs: Additional arguments passed to the Model base class.
    """

    VARIANT_CONFIGS = {
        "micro":  {"hidden_features": [16, 8], "grid_size": 3, "spline_order": 3, "activation": "swish"},
        "small":  {"hidden_features": [64, 32, 16], "grid_size": 5, "spline_order": 3, "activation": "swish"},
        "medium": {"hidden_features": [128, 64, 32], "grid_size": 7, "spline_order": 3, "activation": "gelu"},
        "large":  {"hidden_features": [256, 128, 64, 32], "grid_size": 10, "spline_order": 3, "activation": "gelu"},
        "xlarge": {"hidden_features": [512, 256, 128, 64], "grid_size": 12, "spline_order": 3, "activation": "gelu"},
    }

    #: Canonical alias of ``VARIANT_CONFIGS`` (models/CLAUDE.md Axis 2: "where one
    #: of those is the package's only variant table, add MODEL_VARIANTS as a
    #: class-level alias to the same dict"). An ALIAS, never a rename -- the same
    #: object under both names, so ``from_variant`` and every existing reader stay
    #: on one table. ``src/train/kan/`` and this package's own tests reference the
    #: ``VARIANT_CONFIGS`` spelling, and the module docstring above explains why
    #: that spelling stays; adding the alias is what the rule actually asks for.
    MODEL_VARIANTS = VARIANT_CONFIGS

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_features: int,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if not isinstance(layer_configs, list) or not layer_configs:
            raise ValueError("layer_configs must be a non-empty list")
        if not isinstance(input_features, int) or input_features <= 0:
            raise ValueError(f"input_features must be positive integer, got {input_features}")

        self.layer_configs = self._validate_and_copy_configs(layer_configs)
        self.input_features = input_features
        self.num_layers = len(self.layer_configs)

        # Build the functional graph
        inputs, outputs = self._build_functional_model()

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name or "kan_model",
            **kwargs
        )

        self._log_model_creation()

    def _log_model_creation(self):
        structure = [str(self.input_features)] + [str(cfg['features']) for cfg in self.layer_configs]
        logger.info(f"Created KAN model: {' -> '.join(structure)} ({self.num_layers} layers)")

    def _validate_and_copy_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_configs = []
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} config must be a dict, got {type(config)}")
            if 'features' not in config:
                raise ValueError(f"Layer {i} config missing required 'features' key")
            validated_configs.append(config.copy())
        return validated_configs

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Construct the Functional API graph."""
        inputs = keras.Input(shape=(self.input_features,), name="kan_input")
        x = inputs
        final_activation_fn = None

        for i, config in enumerate(self.layer_configs):
            layer_name = f"kan_layer_{i}"
            # Use a local copy to avoid modifying self.layer_configs during build
            kan_args = config.copy()
            is_last_layer = (i == self.num_layers - 1)

            if is_last_layer:
                # Extract the final network activation (e.g., 'softmax')
                # and force the KAN layer itself to be linear to avoid double activation.
                final_activation_fn = kan_args.pop('activation', 'linear')
                kan_args['activation'] = 'linear'

            kan_layer = KANLinear(name=layer_name, **kan_args)
            x = kan_layer(x)

        # Apply the distinct final activation (Softmax, Sigmoid, etc.) separately
        if final_activation_fn and final_activation_fn != 'linear':
            x = keras.layers.Activation(final_activation_fn, name="final_activation")(x)

        return inputs, x

    def update_kan_grids(self, x_data: Union[keras.KerasTensor, np.ndarray, Any]) -> None:
        """
        Update the B-spline grids of all KANLinear layers using the provided data.

        This is a critical step for KAN training. It performs a forward pass to
        collect the input distribution seen by each hidden layer, then adapts
        that layer's grid to match the distribution (quantile matching).

        Args:
            x_data: Batch of input data (numpy array or tensor). Should be a
                representative sample of the training data (e.g., 100-1000 samples).
        """
        kan_layers = [layer for layer in self.layers if isinstance(layer, KANLinear)]
        if not kan_layers:
            logger.warning("No KANLinear layers found to update.")
            return

        # To update hidden layers, we need their inputs.
        # We build a temporary model to extract intermediate activations.
        # For a functional model, layer.input gives the symbolic tensor feeding the layer.
        layer_inputs = [layer.input for layer in kan_layers]
        
        # Create a temporary extraction model
        # Note: self.input corresponds to the model's main input
        extraction_model = keras.Model(inputs=self.input, outputs=layer_inputs)
        
        # Run inference to get actual values
        # verbose=0 prevents progress bars for this utility op
        intermediate_values = extraction_model.predict(x_data, verbose=0)
        
        # Handle singleton case (predict returns array instead of list if 1 output)
        if len(kan_layers) == 1:
            intermediate_values = [intermediate_values]
            
        # Update each layer with its corresponding input distribution
        for layer, data in zip(kan_layers, intermediate_values):
            layer.update_grid_from_samples(data)
            
        logger.info(f"Updated grids for {len(kan_layers)} KAN layers.")

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        Weights are transferred layer-by-layer via
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        the canonical replacement for ``self.load_weights(by_name=True)`` (which
        raises on ``.keras`` files in Keras 3.8+).

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
                Maps to ``strict=not skip_mismatch``.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            logger.info(f"Loading pretrained weights from {weights_path}")

            report = load_weights_from_checkpoint(
                target=self,
                ckpt_path=weights_path,
                skip_prefixes=(),
                strict=not skip_mismatch,
            )

            logger.info(report.summary_string())
            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., output layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs on
    # a non-existent host; `from_variant` caught the download failure, logged a
    # warning and continued with random initialization, so `pretrained=True`
    # silently produced untrained weights. Do NOT reinstate a warn-and-return
    # branch here or in `from_variant`. No public KAN weights are distributed
    # with dl_techniques; pass a local path via
    # `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "mnist",
        cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public KAN weights ship with dl_techniques. Always
        raises, so an unavailable checkpoint is never silently indistinguishable
        from a successful load.

        Args:
            variant: Model variant name (unused).
            dataset: Dataset identifier (unused).
            cache_dir: Cache directory (unused).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained KAN weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: KAN.from_variant('{variant}', ..., "
            f"pretrained='/path/to/weights.keras')."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        input_features: int,
        output_features: int,
        output_activation: Optional[str] = None,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "mnist",
        weights_input_features: Optional[int] = None,
        cache_dir: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "KAN":
        """Factory method to create KAN models from standard presets.

        Args:
            variant: "micro", "small", "medium", "large", or "xlarge".
            input_features: Dimension of input data.
            output_features: Dimension of output.
            output_activation: Final activation (e.g., "softmax", "sigmoid").
            pretrained: If a string, a path to a local weights file. If True,
                        raises NotImplementedError -- no public KAN weights ship
                        with dl_techniques. If False (default), random init.
            weights_dataset: Dataset the weights were trained on (e.g., "mnist").
                             Only used if pretrained=True and not a local path.
            weights_input_features: Input dimension of the pretrained model.
                                    Used to detect mismatches.
            cache_dir: Directory to cache downloaded weights.
            override_config: Dictionary to override variant defaults.
            **kwargs: Arguments passed to KAN constructor.

        Raises:
            ValueError: If variant is not recognized.
            NotImplementedError: If pretrained is True.
        """
        if variant not in cls.VARIANT_CONFIGS:
            available = list(cls.VARIANT_CONFIGS.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

        config_base = cls.VARIANT_CONFIGS[variant].copy()
        if override_config:
            config_base.update(override_config)

        hidden_features = config_base.pop("hidden_features")
        layer_configs = []

        for features in hidden_features:
            config = config_base.copy()
            config["features"] = features
            layer_configs.append(config)

        output_config = config_base.copy()
        output_config["features"] = output_features

        if output_activation:
            output_config["activation"] = output_activation
        elif output_features > 1:
            output_config["activation"] = "softmax"
        else:
            output_config["activation"] = "linear"

        layer_configs.append(output_config)

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )

            if weights_input_features and weights_input_features != input_features:
                logger.info(
                    f"Pretrained input features ({weights_input_features}) differ from "
                    f"current ({input_features}). Input layer weights may be skipped."
                )
                skip_mismatch = True

            pretrained_classes = 10
            if weights_dataset == "cifar100":
                pretrained_classes = 100
            elif weights_dataset in ["mnist", "cifar10"]:
                pretrained_classes = 10

            if output_features != pretrained_classes:
                logger.info(
                    f"Output features ({output_features}) differ from pretrained "
                    f"({pretrained_classes}). Output layer weights will be skipped."
                )
                skip_mismatch = True

        model = cls(layer_configs=layer_configs, input_features=input_features, **kwargs)

        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    @classmethod
    def from_layer_sizes(
        cls,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = "swish",
        final_activation: Optional[str] = None,
        **kan_layer_kwargs: Any
    ) -> "KAN":
        """Create a KAN by defining a list of node counts per layer."""
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input -> output)")

        input_features = layer_sizes[0]
        output_feature_sizes = layer_sizes[1:]
        layer_configs = []

        for i, features in enumerate(output_feature_sizes):
            config = {
                "features": features,
                "grid_size": grid_size,
                "spline_order": spline_order,
                "activation": activation,
                **kan_layer_kwargs
            }

            # Last layer logic
            if i == len(output_feature_sizes) - 1:
                if final_activation:
                    config["activation"] = final_activation
                elif features > 1:
                    config["activation"] = "softmax"
                else:
                    config["activation"] = "linear"

            layer_configs.append(config)

        return cls(layer_configs=layer_configs, input_features=input_features)

    def get_architecture_summary(self) -> str:
        """Returns a formatted string summarizing the KAN architecture details."""
        lines = ["KAN Model Architecture Summary"]
        lines.append("=" * 50)

        total_features = [self.input_features] + [cfg["features"] for cfg in self.layer_configs]
        lines.append(f"Flow: {' -> '.join(map(str, total_features))}")
        lines.append(f"Total layers: {self.num_layers}")
        lines.append("-" * 50)

        for i, config in enumerate(self.layer_configs):
            # Determine what the activation effectively is for display
            is_last = (i == self.num_layers - 1)
            if is_last:
                act_display = config.get('activation', 'linear')
            else:
                act_display = config.get('activation', 'swish')

            lines.append(
                f"Layer {i:2d}: "
                f"Units={config['features']:<4d} | "
                f"Grid={config.get('grid_size', 'def'):<2} | "
                f"Order={config.get('spline_order', 'def'):<1} | "
                f"Act='{act_display}'"
            )

        lines.append("=" * 50)
        lines.append(f"Est. Parameters: ~{self._estimate_parameters():,}")
        return "\n".join(lines)

    def _estimate_parameters(self) -> int:
        """Estimate number of trainable parameters."""
        total_params = 0
        curr_in = self.input_features

        for config in self.layer_configs:
            curr_out = config["features"]
            grid = config.get("grid_size", 5)
            order = config.get("spline_order", 3)
            num_basis = grid + order

            # Param count logic:
            # 1. Spline weights: in * out * basis
            # 2. Spline scalers: in * out
            # 3. Base scalers:   in * out
            # Note: KANLinear does not currently implement a bias vector.
            
            layer_params = (curr_in * curr_out * num_basis) + (2 * curr_in * curr_out)

            total_params += layer_params
            curr_in = curr_out

        return total_params

    def summary(self, **kwargs: Any) -> None:
        super().summary(**kwargs)
        logger.info("\n" + self.get_architecture_summary())

    def get_config(self) -> Dict[str, Any]:
        # Functional model only needs construction args to be serializable
        return {
            "layer_configs": self.layer_configs,
            "input_features": self.input_features,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        return cls(**config)

# ---------------------------------------------------------------------
# factory method
# ---------------------------------------------------------------------

def create_kan_model(
    variant: str = "small",
    input_features: int = 784,
    output_features: int = 10,
    output_activation: Optional[str] = None,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "mnist",
    weights_input_features: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **model_kwargs: Any
) -> KAN:
    """Helper to create a standard KAN model configuration.

    Args:
        variant: "micro", "small", "medium", "large", or "xlarge".
        input_features: Input dimension.
        output_features: Output dimension.
        output_activation: Final activation.
        pretrained: Path string to a local weights file, or False (default).
            True raises NotImplementedError -- no public KAN weights ship with
            dl_techniques.
        weights_dataset: Dataset for pretrained weights (e.g. "mnist").
        weights_input_features: Original input features of pretrained model.
        cache_dir: Download cache location.
        **model_kwargs: Additional model arguments.

    Returns:
        Uncompiled KAN model.

    Raises:
        NotImplementedError: If pretrained is True.
    """
    return KAN.from_variant(
        variant=variant,
        input_features=input_features,
        output_features=output_features,
        output_activation=output_activation,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_features=weights_input_features,
        cache_dir=cache_dir,
        **model_kwargs
    )
