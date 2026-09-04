"""
TabM ensemble model for tabular data. ``TabMModel`` batches `k` MLP members
that share almost all of their weights, and one-hot encodes categorical
features inside the model before concatenating them to the numerical ones.

Each backbone layer applies a rank-1 perturbation to one shared weight matrix
`W`: member `i` scales the layer's input by a learned vector `r_i` and its
output by `s_i`, so `y_i = s_i * ((r_i * x) W) + b_i` is a distinct effective
weight matrix per member, in `d_in + d_out` parameters instead of
`d_in * d_out`. `arch_type` selects how far this is pushed, from `'plain'`
(no ensemble) through `'tabm'` (per-layer rank-1 scaling) to `'tabm-packed'`
(a fully independent kernel per member); the six variants are listed in the
`ARCH_SPECS` table.

Tensors carry an explicit member axis, `(batch, k, features)`, evaluated in
one pass rather than a Python loop over members. `call` returns this axis
un-aggregated, shape `(batch, k, d_out)`; `predict_with_uncertainty` and the
module-level `ensemble_predict` reduce it. `share_training_batches=False`
expects the caller to pass a batch already sized `k` times the nominal
batch, reshaped into disjoint per-member slices rather than tiled.

References:
    - Gorishniy et al., 2024. TabM: Advancing Tabular Deep Learning with Parameter-
      Efficient Ensembling. (https://arxiv.org/abs/2410.24210)
    - Lakshminarayanan et al., 2017. Simple and Scalable Predictive Uncertainty
      Estimation using Deep Ensembles. (https://arxiv.org/abs/1612.01474)
    - Wen et al., 2020. BatchEnsemble: An Alternative Approach to Efficient Ensemble
      and Lifelong Learning. (https://arxiv.org/abs/2002.06715)
    - Gorishniy et al., 2021. Revisiting Deep Learning Models for Tabular Data.
      (https://arxiv.org/abs/2106.11959)
    - Grinsztajn et al., 2022. Why do tree-based models still outperform deep
      learning on typical tabular data? (https://arxiv.org/abs/2207.08815)
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, List, Literal, Optional, Tuple, Union, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.one_hot_encoding import OneHotEncoding
from dl_techniques.layers.tabular.tabm_blocks import ScaleEnsemble, NLinear, TabMBackbone
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.tabm.model")
class TabMModel(keras.Model):
    """TabM: batched multi-head ensemble for tabular data.

    Combines numerical and categorical feature preprocessing, a configurable
    MLP backbone, and one of six ensembling strategies selected by
    ``arch_type``.

    Architecture:

    .. code-block:: text

        ┌───────────────────────────┐  ┌────────────────────────────┐
        │ x_num [B, n_num_features] │  │ x_cat [B, n_cat_features]  │
        └─────────────┬─────────────┘  └──────────────┬─────────────┘
                      │                              (optional)
                      │                                ▼
                      │                 ┌────────────────────────────┐
                      │                 │ OneHotEncoding per column  │
                      │                 └──────────────┬─────────────┘
                      └────────────►(concat, −1)◄───────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │ minimal_ensemble_adapter        │
                       │ ScaleEnsemble ('tabm-mini' only)│
                       └────────────────┬─────────────────┘
                                       ▼
                       ┌────────────────────────────────┐
                       │ backbone: TabMBackbone          │
                       │ hidden_dims stack, wired per     │
                       │ ARCH_SPECS[arch_type]            │
                       └────────────────┬─────────────────┘
                                       ▼
                       ┌────────────────────────────────┐
                       │ output_layer: Dense ('plain') or │
                       │ NLinear (ensemble variants)      │
                       └────────────────┬─────────────────┘
                                       ▼
                       ┌────────────────────────────────┐
                       │ Output [B, k, n_classes_or_1]    │
                       │ k=1 for 'plain'                  │
                       └────────────────────────────────┘

    Variants:

    .. code-block:: text

        arch_type          ensemble_type  backbone_scaling  backbone_init  adapter_init
        plain              efficient      False             ones           none
        tabm               efficient      True              random-signs   none
        tabm-normal        efficient      True              normal         none
        tabm-packed        packed         False             ones           none
        tabm-mini          efficient      False             ones           random-signs
        tabm-mini-normal   efficient      False             ones           normal

    :param n_num_features: Number of numerical input features. 0 if only
        categorical features are present.
    :type n_num_features: int
    :param cat_cardinalities: Cardinality of each categorical feature. Empty
        if there are none.
    :type cat_cardinalities: List[int]
    :param n_classes: Number of output classes, or ``None`` for regression.
    :type n_classes: Optional[int]
    :param hidden_dims: Width of each backbone hidden layer.
    :type hidden_dims: List[int]
    :param arch_type: One of ``'plain'``, ``'tabm'``, ``'tabm-normal'``,
        ``'tabm-packed'``, ``'tabm-mini'``, ``'tabm-mini-normal'`` — see
        Variants above.
    :type arch_type: str
    :param k: Number of ensemble members. Required for every variant except
        ``'plain'``, where it must be ``None``.
    :type k: Optional[int]
    :param activation: Activation function for hidden layers.
    :type activation: str
    :param dropout_rate: Dropout probability applied after each hidden layer.
    :type dropout_rate: float
    :param use_bias: Whether linear layers use a bias term.
    :type use_bias: bool
    :param share_training_batches: If ``True``, every member sees the same
        batch, tiled ``k`` ways. If ``False``, an incoming batch of
        ``B * k`` rows is reshaped into disjoint per-member slices; the
        caller must supply a batch of that size.
    :type share_training_batches: bool
    :param kernel_initializer: Initializer for linear-layer weights.
    :param bias_initializer: Initializer for bias terms.
    :param kernel_regularizer: Optional regularizer for linear-layer kernels.
    :param bias_regularizer: Optional regularizer for bias terms.
    :param name: Optional model name.
    :param kwargs: Additional arguments for the ``keras.Model`` base class.

    :raises AssertionError: If ``n_num_features < 0``, or if ``k`` does not
        match what ``arch_type`` requires.
    :raises ValueError: If no features are provided, or the input format is
        invalid.

    Input shape:
        A ``(x_num, x_cat)`` tuple, a ``{'x_num': ..., 'x_cat': ...}`` dict,
        or a single tensor ``x_num`` when there are no categorical features.

    Output shape:
        ``(batch_size, k, n_classes_or_1)``; ``k=1`` for ``'plain'``.

    Example:
        >>> model = TabMModel(n_num_features=10, cat_cardinalities=[5, 3, 12],
        ...                   n_classes=4, hidden_dims=[256, 128],
        ...                   arch_type='tabm', k=8)

    Note:
        All sub-layers are created in ``__init__`` and materialized by
        ``build()``, which traces ``call()`` on symbolic inputs. ``call``
        returns the raw member axis; use ``predict_with_uncertainty`` (or the
        module-level ``ensemble_predict``) to reduce it to a point estimate
        and a spread.
    """

    # Every ARCH_SPECS field is read by `_create_layers`; no two rows share
    # every value, so each arch_type builds a genuinely different model.
    ARCH_SPECS: Dict[str, Dict[str, Any]] = {
        'plain': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': None,
        },
        'tabm': {
            'ensemble_type': 'efficient',
            'backbone_scaling': True,
            'backbone_init': 'random-signs',
            'adapter_init': None,
        },
        'tabm-normal': {
            'ensemble_type': 'efficient',
            'backbone_scaling': True,
            'backbone_init': 'normal',
            'adapter_init': None,
        },
        'tabm-packed': {
            'ensemble_type': 'packed',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': None,
        },
        'tabm-mini': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': 'random-signs',
        },
        'tabm-mini-normal': {
            'ensemble_type': 'efficient',
            'backbone_scaling': False,
            'backbone_init': 'ones',
            'adapter_init': 'normal',
        },
    }

    # Model variant configurations optimized for different dataset scales
    MODEL_VARIANTS = {
        "micro": {
            "hidden_dims": [64, 32],
            "k": 4,
            "arch_type": "tabm-mini",
            "description": "Minimal tabular model - 8K params, small datasets"
        },
        "tiny": {
            "hidden_dims": [128, 64],
            "k": 8,
            "arch_type": "tabm-mini",
            "description": "Lightweight ensemble - 32K params, medium datasets"
        },
        "small": {
            "hidden_dims": [256, 128],
            "k": 8,
            "arch_type": "tabm",
            "description": "Standard configuration - 128K params, most datasets"
        },
        "base": {
            "hidden_dims": [512, 256, 128],
            "k": 8,
            "arch_type": "tabm",
            "description": "High-performance default - 512K params, large datasets"
        },
        "large": {
            "hidden_dims": [1024, 512, 256],
            "k": 16,
            "arch_type": "tabm",
            "description": "Large ensemble - 2M params, very large datasets"
        },
        "xlarge": {
            "hidden_dims": [2048, 1024, 512, 256],
            "k": 32,
            "arch_type": "tabm",
            "description": "XL ensemble - 8M params, massive datasets"
        }
    }

    def __init__(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            n_classes: Optional[int],
            hidden_dims: List[int],
            arch_type: Literal[
                'plain', 'tabm', 'tabm-mini', 'tabm-packed',
                'tabm-normal', 'tabm-mini-normal'
            ] = 'plain',
            k: Optional[int] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            share_training_batches: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            name: Optional[str] = "tabm_model",
            **kwargs: Any
    ) -> None:
        """Validate the config and create the preprocessing, backbone and head layers.

        :raises ValueError: If the configuration is invalid.
        """
        self._validate_parameters(n_num_features, cat_cardinalities, hidden_dims, arch_type, k, share_training_batches, dropout_rate)

        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities.copy() if cat_cardinalities else []
        self.n_classes = n_classes
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: use list(), not
        # .copy() -- the default is a tuple, which has no .copy(). See decisions.md.
        self.hidden_dims = list(hidden_dims)
        self.arch_type = arch_type
        self.k = k
        self.activation = deserialize_activation(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.share_training_batches = share_training_batches
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.d_num = n_num_features
        self.d_cat = sum(cat_cardinalities) if cat_cardinalities else 0
        self.d_flat = self.d_num + self.d_cat

        self._create_layers()

        super().__init__(name=name, **kwargs)

        logger.info(
            f"Initialized TabM model: arch_type={arch_type}, "
            f"k={k}, dims={hidden_dims}, features=({n_num_features} num + {len(cat_cardinalities)} cat)"
        )

    def _validate_parameters(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            hidden_dims: List[int],
            arch_type: str,
            k: Optional[int],
            share_training_batches: bool,
            dropout_rate: float
    ) -> None:
        """Validate the constructor arguments.

        :raises ValueError: If any argument is invalid.
        """
        if n_num_features < 0:
            raise ValueError(
                f"n_num_features must be non-negative, got {n_num_features}"
            )
        if not (n_num_features or cat_cardinalities):
            raise ValueError(
                "Must have either numerical or categorical features"
            )

        if cat_cardinalities and not all(c > 0 for c in cat_cardinalities):
            raise ValueError("All cardinalities must be positive")

        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if not all(d > 0 for d in hidden_dims):
            raise ValueError("All hidden dimensions must be positive")

        if arch_type not in self.ARCH_SPECS:
            raise ValueError(
                f"Unknown arch_type {arch_type!r}. Available: "
                f"{sorted(self.ARCH_SPECS)}"
            )

        if arch_type == 'plain':
            if k is not None:
                raise ValueError("Plain architecture must have k=None")
            if not share_training_batches:
                raise ValueError(
                    "Plain architecture must use share_training_batches=True"
                )
        else:
            if k is None:
                raise ValueError(
                    f"Ensemble architecture {arch_type} requires k to be specified"
                )
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def _create_layers(self) -> None:
        """Create the categorical encoder, ensemble adapter, backbone and output layer."""
        if self.cat_cardinalities:
            self.cat_encoder = OneHotEncoding(self.cat_cardinalities)
        else:
            self.cat_encoder = None

        spec = self.ARCH_SPECS[self.arch_type]

        if spec['adapter_init'] is not None:
            self.minimal_ensemble_adapter = ScaleEnsemble(
                k=self.k,
                input_dim=self.d_flat,
                init_distribution=spec['adapter_init'],
                kernel_regularizer=self.kernel_regularizer
            )
        else:
            self.minimal_ensemble_adapter = None

        backbone_k = None if self.arch_type == 'plain' else self.k
        self.backbone = TabMBackbone(
            hidden_dims=self.hidden_dims,
            k=backbone_k,
            ensemble_type=spec['ensemble_type'],
            ensemble_scaling_in=spec['backbone_scaling'],
            ensemble_scaling_out=spec['backbone_scaling'],
            init_distribution=spec['backbone_init'],
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        d_out = 1 if self.n_classes is None else self.n_classes

        if self.arch_type == 'plain':
            self.output_layer = keras.layers.Dense(
                d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='output'
            )
        else:
            self.output_layer = NLinear(
                n=self.k,
                input_dim=self.hidden_dims[-1],
                output_dim=d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method the model inherits ``Layer.build``, which marks
        the model built while every sub-layer is still unbuilt. The shared
        helper traces ``call()`` on symbolic inputs instead, so what gets
        built cannot drift from what gets called.

        :param input_shape: Shape, or nest of shapes, of the input to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Union[Tuple[Any, Any], Dict[str, Any]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Normalize inputs, preprocess features, run the ensemble backbone and head.

        :param inputs: A ``(x_num, x_cat)`` tuple, a
            ``{'x_num': ..., 'x_cat': ...}`` dict, or a single tensor
            ``x_num`` when there are no categorical features.
        :type inputs: Union[Tuple[Any, Any], Dict[str, Any]]
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Predictions, shape ``(batch_size, 1, n_classes_or_1)`` for
            ``'plain'`` or ``(batch_size, k, n_classes_or_1)`` otherwise.
        :rtype: keras.KerasTensor
        :raises ValueError: If no features are provided, or the input format
            is invalid.
        """
        if isinstance(inputs, dict):
            x_num = inputs.get('x_num')
            x_cat = inputs.get('x_cat')
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            x_num, x_cat = inputs
        else:
            x_num = inputs
            x_cat = None

        features = []

        if x_num is not None and self.n_num_features > 0:
            features.append(x_num)

        if x_cat is not None and self.cat_cardinalities:
            cat_encoded = self.cat_encoder(x_cat)
            features.append(cat_encoded)

        if len(features) == 0:
            raise ValueError("No valid features provided")
        elif len(features) == 1:
            x = features[0]
        else:
            x = ops.concatenate(features, axis=-1)

        if self.k is not None:
            batch_size = ops.shape(x)[0]

            if self.share_training_batches or not training:
                x = ops.expand_dims(x, axis=1)  # (B, 1, D)
                x = ops.tile(x, [1, self.k, 1])  # (B, K, D)
            else:
                # Caller must supply a batch already sized B * k.
                x = ops.reshape(x, (batch_size // self.k, self.k, -1))

            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)

        x = self.backbone(x, training=training)

        x = self.output_layer(x)

        if self.k is None:
            x = ops.expand_dims(x, axis=1)

        return x

    def predict_with_uncertainty(
            self,
            x_data: Union[Tuple[Any, Any], Dict[str, Any], Any],
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate the member axis into a point estimate and a spread.

        `call` returns `(batch, k, d_out)` without aggregating it; this
        returns the mean `ŷ = (1/k) Σ fᵢ(x)` and the across-member standard
        deviation, each shaped `(batch, d_out)`.

        The spread is the ensemble's disagreement over whatever `call` emits
        (logits, for a classifier), not a calibrated predictive variance. For
        `arch_type='plain'` there is one member and the spread is zero.

        :param x_data: Input data in any format ``call`` accepts.
        :type x_data: Union[Tuple[Any, Any], Dict[str, Any], Any]
        :param kwargs: Forwarded to ``keras.Model.predict``.
        :return: ``(mean, std)``, both shaped ``(batch_size, d_out)``.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        predictions = self.predict(x_data, **kwargs)
        return np.mean(predictions, axis=1), np.std(predictions, axis=1)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int] = None,
        **kwargs: Any
    ) -> "TabMModel":
        """Create a TabM model from a predefined variant.

        :param variant: One of ``"micro"``, ``"tiny"``, ``"small"``, ``"base"``,
            ``"large"``, ``"xlarge"``.
        :type variant: str
        :param n_num_features: Number of numerical features.
        :type n_num_features: int
        :param cat_cardinalities: Cardinality of each categorical feature.
        :type cat_cardinalities: List[int]
        :param n_classes: Number of output classes, or ``None`` for regression.
        :type n_classes: Optional[int]
        :param kwargs: Additional arguments passed to the constructor.
        :return: A configured ``TabMModel``.
        :rtype: TabMModel
        :raises ValueError: If ``variant`` is not recognized.

        Example:
            >>> # Small ensemble for medium datasets
            >>> model = TabMModel.from_variant(
            ...     "small",
            ...     n_num_features=15,
            ...     cat_cardinalities=[5, 3, 8],
            ...     n_classes=3
            ... )
            >>>
            >>> # Large ensemble for big tabular data
            >>> model = TabMModel.from_variant(
            ...     "large",
            ...     n_num_features=100,
            ...     cat_cardinalities=[20, 15, 10, 5],
            ...     n_classes=None  # Regression
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        # `description` is metadata, not a constructor argument; __init__ raises
        # on an unrecognized kwarg, so it is popped before the splat below.
        description = config.pop("description", "")
        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset dict
        # before update(kwargs); splatting named preset fields alongside
        # **kwargs raised on every override. See decisions.md.
        config.update(kwargs)

        logger.info(f"Creating TabM-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        return cls(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for serialization.

        :return: All constructor parameters needed to recreate this instance.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "n_num_features": self.n_num_features,
            "cat_cardinalities": self.cat_cardinalities,
            "n_classes": self.n_classes,
            "hidden_dims": self.hidden_dims,
            "arch_type": self.arch_type,
            "k": self.k,
            "activation": serialize_activation(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "share_training_batches": self.share_training_batches,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TabMModel":
        """Create a model from its configuration dictionary, deserializing initializers and regularizers.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model instance.
        :rtype: TabMModel
        """
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "bias_initializer" in config and isinstance(config["bias_initializer"], dict):
            config["bias_initializer"] = keras.initializers.deserialize(
                config["bias_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "bias_regularizer" in config and config["bias_regularizer"]:
            config["bias_regularizer"] = keras.regularizers.deserialize(
                config["bias_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print the standard Keras summary, then TabM-specific configuration details.

        :param kwargs: Additional arguments forwarded to
            ``keras.Model.summary``.
        """
        super().summary(**kwargs)
        logger.info("=" * 60)
        logger.info("TabM: Deep Ensemble Tabular Model")
        logger.info("=" * 60)
        logger.info("Architecture Configuration:")
        logger.info(f"  - Architecture type: {self.arch_type}")
        logger.info(f"  - Ensemble members: {self.k if self.k else 'N/A (plain)'}")
        logger.info(f"  - Hidden layers: {self.hidden_dims}")
        logger.info(f"  - Total parameters: {self.count_params():,}")

        logger.info("\nFeature Configuration:")
        logger.info(f"  - Numerical features: {self.n_num_features}")
        logger.info(f"  - Categorical features: {len(self.cat_cardinalities)}")
        logger.info(f"  - Categorical cardinalities: {self.cat_cardinalities}")
        logger.info(f"  - Total input dimension: {self.d_flat}")

        logger.info("\nTask Configuration:")
        task_type = f"{self.n_classes}-class classification" if self.n_classes else "Regression"
        logger.info(f"  - Task type: {task_type}")
        logger.info(f"  - Activation: {self.activation}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Shared batches: {self.share_training_batches}")

        if self.k:
            logger.info("\nEnsemble Benefits:")
            logger.info("  - Improved robustness through ensemble diversity")
            logger.info("  - Uncertainty estimation via prediction variance")
            logger.info("  - Parameter-efficient batched training")

    def __repr__(self) -> str:
        """Return a one-line representation with the key architecture parameters.

        :rtype: str
        """
        return (
            f"TabMModel(arch_type='{self.arch_type}', k={self.k}, "
            f"hidden_dims={self.hidden_dims}, features=({self.n_num_features}, {len(self.cat_cardinalities)}), "
            f"name='{self.name}')"
        )

# ---------------------------------------------------------------------
# Factory functions for easy model creation
# ---------------------------------------------------------------------

def create_tabm_model(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: Sequence[int] = (256, 256),
        arch_type: Literal[
            'plain', 'tabm', 'tabm-mini', 'tabm-packed',
            'tabm-normal', 'tabm-mini-normal'
        ] = 'tabm',
        k: Optional[int] = 8,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        share_training_batches: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model with sensible defaults for typical tabular tasks.

    :param n_num_features: Number of numerical features.
    :type n_num_features: int
    :param cat_cardinalities: Cardinality of each categorical feature.
    :type cat_cardinalities: List[int]
    :param n_classes: Number of output classes, or ``None`` for regression.
    :type n_classes: Optional[int]
    :param hidden_dims: Width of each backbone hidden layer.
    :type hidden_dims: Sequence[int]
    :param arch_type: See :class:`TabMModel`'s Variants table.
    :type arch_type: str
    :param k: Number of ensemble members.
    :type k: Optional[int]
    :param activation: Activation function for hidden layers.
    :type activation: str
    :param dropout_rate: Dropout probability after each hidden layer.
    :type dropout_rate: float
    :param use_bias: Whether linear layers use a bias term.
    :type use_bias: bool
    :param share_training_batches: See :class:`TabMModel`.
    :type share_training_batches: bool
    :param kernel_initializer: Initializer for linear-layer weights.
    :param bias_initializer: Initializer for bias terms.
    :param kwargs: Additional arguments forwarded to :class:`TabMModel`.
    :return: A configured, uncompiled ``TabMModel``.
    :rtype: TabMModel

    Example:
        >>> # Binary classification with mixed features
        >>> model = create_tabm_model(
        ...     n_num_features=10,
        ...     cat_cardinalities=[5, 3, 8],
        ...     n_classes=2,
        ...     arch_type='tabm',
        ...     k=8
        ... )

        >>> # Regression with only numerical features
        >>> model = create_tabm_model(
        ...     n_num_features=15,
        ...     cat_cardinalities=[],
        ...     n_classes=None,
        ...     arch_type='tabm-mini',
        ...     k=4
        ... )
    """
    return TabMModel(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type=arch_type,
        k=k,
        activation=activation,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        share_training_batches=share_training_batches,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_tabm_plain(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: Sequence[int] = (256, 256),
        **kwargs: Any
) -> TabMModel:
    """Create a plain MLP baseline without ensembling.

    :param n_num_features: Number of numerical features.
    :param cat_cardinalities: Cardinality of each categorical feature.
    :param n_classes: Number of output classes, or ``None`` for regression.
    :param hidden_dims: Width of each backbone hidden layer.
    :param kwargs: Additional arguments forwarded to :func:`create_tabm_model`.
    :return: A plain ``TabMModel`` with no ensembling.
    :rtype: TabMModel
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='plain',
        k=None,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_tabm_ensemble(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: Sequence[int] = (256, 256),
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model with the full per-layer efficient ensemble (``arch_type='tabm'``).

    :param n_num_features: Number of numerical features.
    :param cat_cardinalities: Cardinality of each categorical feature.
    :param n_classes: Number of output classes, or ``None`` for regression.
    :param k: Number of ensemble members.
    :param hidden_dims: Width of each backbone hidden layer.
    :param kwargs: Additional arguments forwarded to :func:`create_tabm_model`.
    :return: A configured ``TabMModel``.
    :rtype: TabMModel
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm',
        k=k,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_tabm_mini(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: Sequence[int] = (256, 256),
        **kwargs: Any
) -> TabMModel:
    """Create a TabM model with only the input-side ScaleEnsemble adapter (``arch_type='tabm-mini'``).

    :param n_num_features: Number of numerical features.
    :param cat_cardinalities: Cardinality of each categorical feature.
    :param n_classes: Number of output classes, or ``None`` for regression.
    :param k: Number of ensemble members.
    :param hidden_dims: Width of each backbone hidden layer.
    :param kwargs: Additional arguments forwarded to :func:`create_tabm_model`.
    :return: A configured ``TabMModel``.
    :rtype: TabMModel
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm-mini',
        k=k,
        **kwargs
    )

# ---------------------------------------------------------------------

def ensemble_predict(
        model: TabMModel,
        x_data: Union[Tuple, Dict, Any],
        method: Literal['mean', 'best', 'greedy'] = 'mean'
) -> np.ndarray:
    """Predict with a TabM model and aggregate the ensemble axis.

    :param model: A trained TabM model.
    :type model: TabMModel
    :param x_data: Input data in any format the model's ``call`` accepts.
    :type x_data: Union[Tuple, Dict, Any]
    :param method: ``'mean'`` averages across members. ``'best'`` and
        ``'greedy'`` are not implemented and fall back to ``'mean'`` with a
        warning; both need external validation data to select members.
    :type method: Literal['mean', 'best', 'greedy']
    :return: Aggregated predictions, shape ``(batch_size, n_outputs)``.
    :rtype: np.ndarray
    :raises ValueError: If ``method`` is not one of the three above.
    """
    predictions = model.predict(x_data)

    if method == 'mean':
        return np.mean(predictions, axis=1)

    elif method == 'best':
        logger.warning("Best member selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    elif method == 'greedy':
        logger.warning("Greedy selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

# ---------------------------------------------------------------------

def create_tabm_for_dataset(
        X_train: np.ndarray,
        y_train: np.ndarray,
        categorical_indices: Optional[List[int]] = None,
        categorical_cardinalities: Optional[List[int]] = None,
        arch_type: str = 'tabm',
        k: int = 8,
        hidden_dims: Sequence[int] = (256, 256),
        **kwargs: Any
) -> TabMModel:
    """Infer the problem type and feature layout from a dataset, then build a TabM model.

    :param X_train: Training features, shape ``(n_samples, n_features)``.
    :type X_train: np.ndarray
    :param y_train: Training targets.
    :type y_train: np.ndarray
    :param categorical_indices: Indices of categorical columns in ``X_train``.
    :type categorical_indices: Optional[List[int]]
    :param categorical_cardinalities: Cardinality of each categorical feature.
    :type categorical_cardinalities: Optional[List[int]]
    :param arch_type: See :class:`TabMModel`'s Variants table.
    :type arch_type: str
    :param k: Number of ensemble members.
    :type k: int
    :param hidden_dims: Width of each backbone hidden layer.
    :type hidden_dims: Sequence[int]
    :param kwargs: Additional arguments forwarded to :func:`create_tabm_model`.
    :return: A ``TabMModel`` sized for the given dataset.
    :rtype: TabMModel

    Example:
        >>> import numpy as np
        >>> from dl_techniques.models.tabular.tabm.model import create_tabm_for_dataset

        >>> # Generate sample tabular data
        >>> X_train = np.random.randn(1000, 15)  # 15 features
        >>> y_train = np.random.randint(0, 3, 1000)  # 3-class classification

        >>> # Specify categorical features (first 3 columns)
        >>> categorical_indices = [0, 1, 2]
        >>> categorical_cardinalities = [5, 3, 8]

        >>> model = create_tabm_for_dataset(
        ...     X_train, y_train,
        ...     categorical_indices=categorical_indices,
        ...     categorical_cardinalities=categorical_cardinalities,
        ...     arch_type='tabm',
        ...     k=8
        ... )
        >>>
        >>> # Model ready for compilation and training
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    """
    if len(y_train.shape) == 1:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            n_classes = 2
            problem_type = "Binary classification"
        elif len(unique_labels) > 2 and np.all(unique_labels == np.arange(len(unique_labels))):
            n_classes = len(unique_labels)
            problem_type = f"{n_classes}-class classification"
        else:
            n_classes = None
            problem_type = "Regression"
    else:
        n_classes = y_train.shape[1]
        problem_type = f"Multi-output ({n_classes} outputs)"

    if categorical_indices is None:
        categorical_indices = []
    if categorical_cardinalities is None:
        categorical_cardinalities = []

    n_total_features = X_train.shape[1]
    n_categorical = len(categorical_indices)
    n_numerical = n_total_features - n_categorical

    logger.info("Dataset Analysis for TabM Configuration:")
    logger.info(f"  - Total samples: {X_train.shape[0]:,}")
    logger.info(f"  - Total features: {n_total_features}")
    logger.info(f"  - Numerical features: {n_numerical}")
    logger.info(f"  - Categorical features: {n_categorical}")
    if categorical_cardinalities:
        logger.info(f"  - Categorical cardinalities: {categorical_cardinalities}")
    logger.info(f"  - Problem type: {problem_type}")
    logger.info(f"  - Architecture: {arch_type} with k={k}")

    return create_tabm_model(
        n_num_features=n_numerical,
        cat_cardinalities=categorical_cardinalities,
        n_classes=n_classes,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **kwargs
    )

# ---------------------------------------------------------------------