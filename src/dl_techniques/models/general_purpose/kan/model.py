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
setup. The failure mode when it is skipped is silence rather than an error:
`KANLinear` sums over the input axis, activations grow roughly 30x per layer
and leave `grid_range=(-2.0, 2.0)` after layer 0, the spline basis is then
identically zero, and with `base_scaler` initialized to a constant the whole
model collapses to a constant function. `grids_adapted` exposes that state on
the object so it is readable rather than inferred from a flat loss curve.

Structurally the model is a stack of `KANLinear` layers driven by a list of
per-layer config dicts, with an optional final activation. The last layer is
forced linear and the network-level activation is applied as its own
`Activation` layer, so a `softmax` in a variant preset cannot be applied twice.
Five preset variants span micro (`[16, 8]`, grid 3) through xlarge
(`[512, 256, 128, 64]`, grid 12), trading grid resolution and width together,
since a fine grid on a narrow layer tends to overfit and a coarse grid on a wide
one wastes it. The variant table is named `VARIANT_CONFIGS` here rather than the
house `MODEL_VARIANTS`; trainers and tests reference the existing spelling, and
`MODEL_VARIANTS` is a class-level alias to the same object.

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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.kan.model")
class KAN(keras.Model):
    """
    Kolmogorov-Arnold Network built as a Keras 3 Functional model.

    Stacks ``KANLinear`` layers, each of which places a learnable univariate
    function on every connection -- a B-spline of order ``spline_order`` over
    ``grid_size`` intervals, added to a fixed base activation -- while nodes only
    sum what arrives. The layer list is given as ``layer_configs``, one dict per
    layer, and the final layer's ``activation`` is lifted out into a separate
    ``Activation`` layer with the ``KANLinear`` itself forced to ``'linear'`` so
    the transform is never applied twice.

    Knot grids are a training precondition, not a tuning knob:
    :meth:`update_kan_grids` must be run on a representative data sample before
    training, and :attr:`grids_adapted` reports whether it has been.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │     Input [B, input_features]        │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  KANLinear 0  (features₀)            │──┐
        └───────────────┬──────────────────────┘  │
                        ▼                         │
        ┌──────────────────────────────────────┐  │
        │  KANLinear 1  (features₁)            │  │  edge, not node:
        └───────────────┬──────────────────────┘  │
                        ▼                         │  x ──► base_act(x)·wᵦ ──┐
                       ...                        │  │                      ▼
                        ▼                         │  └► Σ cᵢ·Bᵢ(x)·wₛ ───► (+)
        ┌──────────────────────────────────────┐  │                         │
        │  KANLinear N-1 (output features)     │──┘   node = Σ over edges ◄─┘
        │  activation forced to 'linear'       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Activation(final_activation)        │
        │  (omitted when 'linear')             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, features₍N-1₎]           │
        └──────────────────────────────────────┘

        grids: knots span grid_range; update_kan_grids(x) re-fits them
               per layer by quantile matching on that layer's own inputs

    **Variants:**

    .. code-block:: text

        micro    [16, 8]                grid  3   order 3   swish
        small    [64, 32, 16]           grid  5   order 3   swish
        medium   [128, 64, 32]          grid  7   order 3   gelu
        large    [256, 128, 64, 32]     grid 10   order 3   gelu
        xlarge   [512, 256, 128, 64]    grid 12   order 3   gelu

    :param layer_configs: One dict of ``KANLinear`` keyword arguments per layer,
        ordered input-side first. Each must carry a ``'features'`` key; other
        keys (``grid_size``, ``spline_order``, ``activation``, ...) are forwarded
        verbatim. Copied on entry, so the caller's dicts are never mutated. The
        last entry's ``activation`` becomes the model-level final activation.
    :type layer_configs: List[Dict[str, Any]]
    :param input_features: Number of input features. Must be a positive integer.
    :type input_features: int
    :param name: Optional model name. Defaults to ``'kan_model'``.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: If ``layer_configs`` is not a non-empty list, if any
        entry is not a dict or lacks ``'features'``, or if ``input_features`` is
        not a positive integer.

    Input shape:
        2D tensor with shape ``(batch_size, input_features)``.

    Output shape:
        2D tensor with shape ``(batch_size, layer_configs[-1]['features'])``.

    Example:
        >>> # From a preset variant
        >>> model = KAN.from_variant("small", input_features=784, output_features=10)
        >>> model.update_kan_grids(x_sample)   # required before training
        >>>
        >>> # From bare layer sizes
        >>> model = KAN.from_layer_sizes([784, 64, 32, 10], grid_size=5)
        >>>
        >>> # From explicit per-layer configs
        >>> model = KAN(
        ...     layer_configs=[{"features": 32, "grid_size": 5},
        ...                    {"features": 10, "activation": "softmax"}],
        ...     input_features=784,
        ... )

    Note:
        No pretrained KAN weights are distributed with ``dl_techniques``.
        ``pretrained=True`` raises ``NotImplementedError`` rather than warning
        and returning a randomly-initialized model; pass a local checkpoint via
        ``pretrained='/path/to/weights.keras'`` instead.

    Warning:
        A freshly constructed model **cannot be trained as-is at the documented
        defaults**. Measured: the output is exactly ``1 / output_features`` for
        every input with ``std == 0.0``, and 0 of 12 trainable weights receive a
        non-zero gradient. After :meth:`update_kan_grids` the same model has
        12 of 12 live gradients.
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-052
        # `grids_adapted` is REAL, INSPECTABLE STATE, not decoration. A KAN built
        # at the documented defaults and trained without a grid pass is a CONSTANT
        # FUNCTION — measured: output exactly `1/output_features` for every input,
        # `std == 0.0` over the batch, and 0 of 12 trainable weights receiving a
        # non-zero gradient; the same model after `update_kan_grids` is 12 of 12.
        # That state is pinned by `tests/test_models/test_kan/test_model.py`'s
        # `xfail(strict=True)` pair and is NOT re-litigated here. What this flag
        # closes is the SILENCE: the failure mode is a flat loss curve with no
        # error, so the untrainable state is now readable from the object and
        # announced once at construction. Do NOT set this to `True` anywhere but
        # `update_kan_grids`, and do NOT "fix" the constant-function finding by
        # flipping an initializer: the spline basis is identically zero after
        # layer 0 because the activations leave `grid_range`, so symmetry
        # breaking alone would leave the spline weights just as dead.
        # See decisions.md D-052.
        self._grids_adapted = False

        self._log_model_creation()

    @property
    def grids_adapted(self) -> bool:
        """Whether ``update_kan_grids`` has been run on this instance.

        ``False`` on a freshly constructed model. A KAN whose knot grids have not
        been fitted to the data is not merely under-tuned — at the documented
        defaults it is a constant function with identically-zero gradients — so
        this is a training precondition, not a tuning knob.

        :return: ``True`` once :meth:`update_kan_grids` has completed.
        :rtype: bool
        """
        return self._grids_adapted

    def _log_model_creation(self) -> None:
        """Log the layer widths and warn, once, that the grids are unadapted.

        The warning is emitted at construction rather than at ``fit`` time
        because the unadapted state has no error surface of its own: it presents
        as a flat loss curve, so the only chance to say so is before training
        starts.
        """
        structure = [str(self.input_features)] + [str(cfg['features']) for cfg in self.layer_configs]
        logger.info(f"Created KAN model: {' -> '.join(structure)} ({self.num_layers} layers)")
        logger.warning(
            "KAN knot grids are NOT yet adapted to your data. Call "
            "`model.update_kan_grids(x_sample)` before training: at the "
            "documented defaults an unadapted KAN is a CONSTANT FUNCTION "
            "(output exactly 1/output_features, 0 of 12 trainable weights "
            "receiving a non-zero gradient), and it fails as a flat loss curve "
            "with no error. `model.grids_adapted` reports this state."
        )

    def _validate_and_copy_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate every layer config and return shallow copies of them.

        The copy is not cosmetic: ``_build_functional_model`` and
        ``from_variant`` both mutate per-layer dicts (popping ``activation``,
        inserting ``features``), and without copying here those edits would
        propagate back into the caller's list -- and, via ``from_variant``, into
        the class-level ``VARIANT_CONFIGS`` table.

        :param configs: Raw per-layer ``KANLinear`` config dicts.
        :type configs: List[Dict[str, Any]]
        :return: Validated copies, in the same order.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If an entry is not a dict or lacks ``'features'``.
        """
        validated_configs = []
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} config must be a dict, got {type(config)}")
            if 'features' not in config:
                raise ValueError(f"Layer {i} config missing required 'features' key")
            validated_configs.append(config.copy())
        return validated_configs

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Construct the Functional API graph.

        The final layer's ``activation`` is popped off and applied as its own
        ``Activation`` layer while the ``KANLinear`` is forced to ``'linear'``.
        Leaving it in place would apply the transform twice -- a variant preset
        carrying ``activation='softmax'`` would softmax the edge outputs and then
        softmax the sum.

        :return: The ``(inputs, outputs)`` pair for ``keras.Model``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
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
        """Re-fit every ``KANLinear`` layer's B-spline knot grid to real data.

        A forward pass collects the input distribution each layer actually sees
        -- which is not the model input for anything past layer 0 -- and each
        layer's knots are then quantile-matched to its own distribution. Hidden
        activations are pulled out through a temporary extraction model built on
        the symbolic ``layer.input`` tensors, which is available because this is
        a Functional model.

        This is setup, not tuning: run it on a representative sample (roughly
        100-1000 rows) before training on any new dataset. Sets
        :attr:`grids_adapted`.

        :param x_data: Representative batch of model inputs.
        :type x_data: Union[keras.KerasTensor, np.ndarray, Any]
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

        self._grids_adapted = True
        logger.info(f"Updated grids for {len(kan_layers)} KAN layers.")

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True
    ) -> None:
        """Load pretrained weights into the model from a local checkpoint.

        Transfer is layer-by-layer via
        :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`,
        the canonical replacement for ``self.load_weights(..., by_name=True)``,
        which raises on ``.keras`` files in Keras 3.8+.

        :param weights_path: Path to the weights file (``.keras`` format).
        :type weights_path: str
        :param skip_mismatch: Whether to skip layers whose shapes do not match
            the checkpoint's (typically the output layer under a changed
            ``output_features``). Maps to the transfer helper's ``strict=not
            skip_mismatch``.
        :type skip_mismatch: bool

        :raises FileNotFoundError: If ``weights_path`` does not exist.
        :raises ValueError: If the weights cannot be loaded.
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

    # `_download_weights` raises instead of falling back to random init, so an
    # unavailable checkpoint is never silently indistinguishable from a
    # successful load. Do NOT reinstate a warn-and-return branch here or in
    # `from_variant`. No public KAN weights are distributed with dl_techniques;
    # pass a local path via `pretrained="/path/to/file.keras"` or use
    # `pretrained=False` (default).
    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "mnist",
        cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public KAN weights ship with ``dl_techniques``.
        Always raises. Kept to mirror the house factory recipe and to give an
        explicit failure mode instead of a silent random-init fallback.

        :param variant: Model variant name (unused).
        :type variant: str
        :param dataset: Dataset identifier (unused).
        :type dataset: str
        :param cache_dir: Cache directory (unused).
        :type cache_dir: Optional[str]

        :raises NotImplementedError: Always.
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
        """Create a KAN model from a predefined variant.

        The variant's ``hidden_features`` list expands into one layer config per
        width, each inheriting the preset's ``grid_size`` / ``spline_order`` /
        ``activation``, followed by an output layer of ``output_features``. The
        preset dict is copied before any of that, so the class-level
        ``VARIANT_CONFIGS`` table is never mutated for later callers.

        :param variant: One of ``"micro"``, ``"small"``, ``"medium"``,
            ``"large"``, ``"xlarge"``.
        :type variant: str
        :param input_features: Dimension of the input data.
        :type input_features: int
        :param output_features: Dimension of the output.
        :type output_features: int
        :param output_activation: Final activation. When omitted, defaults to
            ``'softmax'`` for ``output_features > 1`` and ``'linear'`` otherwise.
        :type output_activation: Optional[str]
        :param pretrained: A path to a local weights file, or ``False`` (default)
            for random initialization. ``True`` raises ``NotImplementedError`` --
            no public KAN weights ship with ``dl_techniques``.
        :type pretrained: Union[bool, str]
        :param weights_dataset: Dataset the checkpoint was trained on, used to
            infer its class count. Only consulted when ``pretrained`` is set.
        :type weights_dataset: str
        :param weights_input_features: Input dimension of the pretrained model,
            used to detect a mismatch against ``input_features``.
        :type weights_input_features: Optional[int]
        :param cache_dir: Directory to cache downloaded weights.
        :type cache_dir: Optional[str]
        :param override_config: Overrides merged over the variant preset before
            the layer configs are expanded.
        :type override_config: Optional[Dict[str, Any]]
        :param kwargs: Additional arguments passed to the constructor.

        :return: A KAN instance whose knot grids are NOT yet adapted.
        :rtype: KAN

        :raises ValueError: If ``variant`` is not recognized.
        :raises NotImplementedError: If ``pretrained`` is ``True``.

        Example:
            >>> model = KAN.from_variant("medium", input_features=784,
            ...                          output_features=10)
            >>> model.update_kan_grids(x_sample)
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

            # A checkpoint's head width follows its dataset; a different
            # `output_features` means the affected layer must be skipped rather
            # than refused.
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
        """Create a KAN from a flat list of node counts.

        ``layer_sizes[0]`` is the input width and every later entry is one
        ``KANLinear`` layer, all sharing the same grid, order and activation.

        :param layer_sizes: Node counts, input first, output last. At least two.
        :type layer_sizes: List[int]
        :param grid_size: Number of B-spline intervals per edge.
        :type grid_size: int
        :param spline_order: B-spline order per edge.
        :type spline_order: int
        :param activation: Base activation for every layer.
        :type activation: str
        :param final_activation: Final activation. When omitted, defaults to
            ``'softmax'`` for a multi-unit output and ``'linear'`` otherwise.
        :type final_activation: Optional[str]
        :param kan_layer_kwargs: Additional keyword arguments forwarded to every
            ``KANLinear``.

        :return: A KAN instance whose knot grids are NOT yet adapted.
        :rtype: KAN

        :raises ValueError: If ``layer_sizes`` has fewer than two elements.
        """
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
        """Render a per-layer summary of widths, grids, orders and activations.

        :return: A formatted, multi-line summary string.
        :rtype: str
        """
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
        """Estimate the trainable parameter count from the layer configs alone.

        Per layer and per edge there are ``grid_size + spline_order`` spline
        coefficients plus a spline scaler and a base scaler, giving
        ``in * out * (grid + order) + 2 * in * out``.

        :return: Estimated number of trainable parameters.
        :rtype: int
        """
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
        """Print the Keras summary, followed by the KAN-specific summary.

        :param kwargs: Forwarded to ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("\n" + self.get_architecture_summary())

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "layer_configs": self.layer_configs,
            "input_features": self.input_features,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        """Create a model from its configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: A KAN instance.
        :rtype: KAN
        """
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
    """Convenience function to create KAN models.

    :param variant: One of ``"micro"``, ``"small"``, ``"medium"``, ``"large"``,
        ``"xlarge"``.
    :type variant: str
    :param input_features: Input dimension.
    :type input_features: int
    :param output_features: Output dimension.
    :type output_features: int
    :param output_activation: Final activation.
    :type output_activation: Optional[str]
    :param pretrained: Path to a local weights file, or ``False`` (default).
        ``True`` raises ``NotImplementedError`` -- no public KAN weights ship
        with ``dl_techniques``.
    :type pretrained: Union[bool, str]
    :param weights_dataset: Dataset for the pretrained weights (e.g. ``"mnist"``).
    :type weights_dataset: str
    :param weights_input_features: Original input dimension of the pretrained
        model.
    :type weights_input_features: Optional[int]
    :param cache_dir: Download cache location.
    :type cache_dir: Optional[str]
    :param model_kwargs: Additional arguments passed to the model constructor.

    :return: Uncompiled KAN model whose knot grids are NOT yet adapted
        (``model.grids_adapted is False``).
    :rtype: KAN

    :raises NotImplementedError: If ``pretrained`` is ``True``.

    Example:
        >>> # MNIST-shaped classifier
        >>> model = create_kan_model("small", input_features=784, output_features=10)
        >>> model.update_kan_grids(x_sample)
        >>>
        >>> # Scalar regression
        >>> model = create_kan_model("micro", input_features=8, output_features=1,
        ...                          output_activation="linear")

    Warning:
        The returned model **cannot be trained as-is at these defaults.**
        ``KANLinear`` sums over the input axis, so activations grow roughly 30x
        per layer and leave ``grid_range=(-2.0, 2.0)`` after the first layer; the
        B-spline basis is then identically zero and, with ``base_scaler``
        initialized to the constant 1.0, every output unit computes the same
        value. Measured on the documented defaults: the output is exactly
        ``1 / output_features`` for every input with ``std == 0.0``, and 0 of 12
        trainable weights receive a non-zero gradient. Call
        :meth:`KAN.update_kan_grids` with a representative sample first — after
        it, the same model has 12 of 12 live gradients. This is setup, not
        tuning.
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

# ---------------------------------------------------------------------