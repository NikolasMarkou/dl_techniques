"""
MDNModel and its factory create_mdn_model, a Dense feature stack whose output
head emits the parameters of a conditional Gaussian mixture over the target.

Regression under squared error is maximum likelihood under a fixed-variance
unimodal Gaussian, so a plain network can only ever return E[y|x]. When the
conditional distribution is multi-modal, the conditional mean falls between
the modes and is a value the process never generates. This model replaces the
point head with a density:

p(y|x) = sum_i pi_i(x) * N(y; mu_i(x), diag sigma_i(x)^2)

trained under the mixture negative log-likelihood. The gradient now rewards
placing probability mass where the data is, so components separate onto modes
and sigma_i absorbs local noise; sigma_i is a function of x, so the model can
be confident in one region and diffuse in another (heteroscedasticity).

Compile does not accept a loss argument: it hard-wires the MDN layer's own
negative-log-likelihood loss, since an ordinary regression loss applied to a
concatenated [mu, sigma, pi] vector is meaningless. The head output is a single
tensor of width 2*output_dim*num_mixtures + num_mixtures, laid out
[mu | sigma | pi]; sampling and uncertainty estimation are methods on this
model because that split is the layer's contract.

References:
    - Bishop, 1994. Mixture Density Networks. Aston University Technical Report
      NCRG/94/004.
    - Graves, 2013. Generating Sequences With Recurrent Neural Networks.
      (https://arxiv.org/abs/1308.0850)
    - Ha and Schmidhuber, 2018. World Models. (https://arxiv.org/abs/1803.10122)
    - Kendall and Gal, 2017. What Uncertainties Do We Need in Bayesian Deep Learning
      for Computer Vision? (https://arxiv.org/abs/1703.04977)
"""

import keras
import numpy as np
from keras import ops
from keras import layers
from typing import List, Union, Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer,
    get_uncertainty,
    get_point_estimate,
    get_prediction_intervals
)
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.mdn.model")
class MDNModel(keras.Model):
    """A complete Mixture Density Network model.

    Combines a configurable Dense feature-extraction stack with an
    :class:`MDNLayer` output head, predicting the parameters of a
    Gaussian-mixture distribution over the target instead of a point estimate.
    All sublayers are created in ``__init__``; ``build()`` only threads shapes
    through them, so a ``.keras`` weight restore lands losslessly.

    Architecture:

    .. code-block:: text

        input [B, D]
             |
             v
        ┌──────────────┐
        │ dense         │  (repeated per hidden_layers entry)
        │ batchnorm     │  (optional)
        │ activation    │
        │ dropout       │  (optional)
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ mdn_layer     │  emits [mu | sigma | pi]
        └──────────────┘
             |
             v
        output [B, 2*output_dim*num_mixtures + num_mixtures]

    :param hidden_layers: Sizes of the hidden feature-extraction layers.
    :type hidden_layers: List[int]
    :param output_dimension: Dimensionality of the target being predicted.
    :type output_dimension: int
    :param num_mixtures: Number of Gaussian components in the mixture.
    :type num_mixtures: int
    :param hidden_activation: Activation function for hidden layers.
    :type hidden_activation: str
    :param kernel_initializer: Initializer for the kernel weight matrices.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer applied to the kernel weight matrices.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_batch_norm: Whether to insert batch normalization between hidden
        layers.
    :type use_batch_norm: bool
    :param dropout_rate: Dropout rate applied after each hidden layer, or
        ``None`` for no dropout.
    :type dropout_rate: Optional[float]
    :param kwargs: Additional arguments passed to the parent ``Model`` class.

    Example::

        model = MDNModel(
            hidden_layers=[64, 32],
            output_dimension=2,
            num_mixtures=5,
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.L2(1e-5)
        )
        model.compile(optimizer='adam')
        model.fit(x_train, y_train, epochs=100)
        samples = model.sample(x_test, num_samples=10)

    Note:
        The model uses the MDN negative-log-likelihood loss automatically on
        compile. Sampling and ``predict_with_uncertainty`` give probabilistic
        predictions rather than a single point estimate.
    """

    # Class-level defaults (discoverability).
    DEFAULT_HIDDEN_ACTIVATION: str = "relu"
    DEFAULT_KERNEL_INITIALIZER: str = "glorot_uniform"
    DEFAULT_USE_BATCH_NORM: bool = False

    def __init__(
            self,
            hidden_layers: List[int],
            output_dimension: int,
            num_mixtures: int,
            hidden_activation: str = DEFAULT_HIDDEN_ACTIVATION,
            kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
            dropout_rate: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sublayer.

        Sublayers are created here, not in ``build()``, so that on a
        ``.keras`` weight restore they already exist and the saved weights
        land instead of being silently re-initialized.

        :raises ValueError: If ``hidden_layers`` is empty or holds a
            non-positive value, if ``output_dimension`` or ``num_mixtures``
            is not a positive integer, or if ``dropout_rate`` is outside
            ``[0, 1)``.
        """
        super().__init__(**kwargs)

        # Validate architecture parameters
        if not hidden_layers or any(units <= 0 for units in hidden_layers):
            raise ValueError("hidden_layers must be a non-empty list of positive integers")

        if output_dimension <= 0:
            raise ValueError("output_dimension must be a positive integer")

        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be a positive integer")

        # Validate regularization parameters
        if dropout_rate is not None and (dropout_rate < 0 or dropout_rate >= 1):
            raise ValueError("dropout_rate must be in the range [0, 1) or None")

        # Store configuration parameters for use in build()
        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.hidden_activation = deserialize_activation(hidden_activation)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Convert string initializers/regularizers to objects
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        self._build_input_shape = None  # For serialization

        # Each hidden layer expands to Dense -> [BatchNorm] -> Activation -> [Dropout].
        self.feature_layers = []  # [Dense, BatchNorm?, Activation, Dropout?]*N
        for i, units in enumerate(self.hidden_layers_sizes):
            self.feature_layers.append(layers.Dense(
                units,
                activation=None,  # Activation applied separately after BatchNorm
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}",
            ))
            if self.use_batch_norm:
                self.feature_layers.append(
                    layers.BatchNormalization(name=f"batch_norm_{i}"))
            self.feature_layers.append(layers.Activation(
                self.hidden_activation, name=f"activation_{i}"))
            if self.dropout_rate is not None:
                self.feature_layers.append(
                    layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # The final MDN output layer (μ, σ, π parameters).
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="mdn_layer",
        )

        logger.info(f"Initialized MDNModel with {len(hidden_layers)} hidden layers, "
                   f"{output_dimension}D output, {num_mixtures} mixtures")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Thread the input shape through every sublayer, in order.

        Sublayers are created in ``__init__``; this only calls each
        sublayer's own ``build()`` explicitly, since a deferred build would
        leave sublayers unbuilt at ``.keras`` weight-restore time and the
        restored weights would be silently re-initialized.

        :param input_shape: ``(batch_size, feature_dim)``; ``batch_size`` may
            be ``None``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Store input shape for serialization support
        self._build_input_shape = input_shape

        logger.info(f"Building MDNModel with input shape: {input_shape}")

        # Each layer needs the output shape of the previous one.
        current_shape = input_shape
        for layer in self.feature_layers:
            layer.build(current_shape)
            if hasattr(layer, 'compute_output_shape'):
                current_shape = layer.compute_output_shape(current_shape)

        # Build the final MDN layer with the shape after all feature layers.
        self.mdn_layer.build(current_shape)

        logger.info("MDNModel built successfully")

        # Mark the model as built (must be the LAST statement of build()).
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the feature stack, then the MDN head.

        :param inputs: Input tensor, ``[batch_size, input_dim]``.
        :type inputs: keras.KerasTensor
        :param training: Whether BatchNorm/Dropout run in training mode.
        :type training: Optional[bool]
        :return: Mixture parameters, ``[batch_size, 2*output_dim*num_mixtures + num_mixtures]``,
            laid out ``[mu_1, ..., mu_n, sigma_1, ..., sigma_n, pi_1, ..., pi_m]``
            where ``n = num_mixtures * output_dim`` and ``m = num_mixtures``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        for layer in self.feature_layers:
            x = layer(x, training=training)

        return self.mdn_layer(x, training=training)

    def sample(
            self,
            inputs: keras.KerasTensor,
            num_samples: int = 1,
            temperature: float = 1.0,
            seed: Optional[int] = None
    ) -> keras.KerasTensor:
        """Draw samples from the predicted mixture distribution.

        Runs one forward pass to get ``[mu, sigma, pi]``, then for each
        sample selects a component via a categorical draw over ``pi`` and
        samples from that component's Gaussian.

        :param inputs: Input tensor, ``[batch_size, input_dim]``.
        :type inputs: keras.KerasTensor
        :param num_samples: Number of samples to draw per input.
        :type num_samples: int
        :param temperature: Scales the mixture logits before the categorical
            draw. Below 1 concentrates on dominant components; above 1
            flattens toward uniform selection.
        :type temperature: float
        :param seed: If given, sample ``i`` uses ``seed + i`` for a
            reproducible but uncorrelated draw.
        :type seed: Optional[int]
        :return: Samples, ``[batch_size, num_samples, output_dim]``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``num_samples`` or ``temperature`` is not
            positive.

        Example::

            samples = model.sample(x_test, num_samples=100)
            sample_mean = ops.mean(samples, axis=1)
            sample_std = ops.std(samples, axis=1)
            confidence_intervals = ops.percentile(samples, [5, 95], axis=1)
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        # Inference mode, for predictions consistent across calls.
        predictions = self(inputs, training=False)

        samples = []
        for i in range(num_samples):
            sample_seed = None if seed is None else seed + i

            # DECISION plan_2026-06-09_be55db55/D-004: forward sample_seed into
            # mdn_layer.sample; it was previously computed and discarded, so
            # `seed=` was a no-op. Do not drop this argument. See decisions.md D-004.
            sample = self.mdn_layer.sample(
                predictions, temperature=temperature, seed=sample_seed)
            samples.append(sample)

        return ops.stack(samples, axis=1)

    def predict_with_uncertainty(
            self,
            inputs: keras.KerasTensor,
            confidence_level: float = 0.95
    ) -> Dict[str, keras.KerasTensor]:
        """Decompose the mixture's predictive variance and give an interval.

        Applies the law of total variance: ``Var[y|x] = aleatoric +
        epistemic``, where ``aleatoric = sum_i pi_i * sigma_i^2`` and
        ``epistemic = sum_i pi_i * (mu_i - E[y|x])^2``. The interval is a
        Gaussian approximation, ``point +/- z * sqrt(total_variance)``; it is
        calibrated for a unimodal mixture and only indicative when the
        mixture is genuinely multi-modal.

        :param inputs: Input tensor, ``[batch_size, input_dim]``.
        :type inputs: keras.KerasTensor
        :param confidence_level: Confidence level for the prediction
            interval, in ``(0, 1)``.
        :type confidence_level: float
        :return: Dictionary with keys ``point_estimates``, ``total_variance``,
            ``aleatoric_variance``, ``epistemic_variance``, ``lower_bound`` and
            ``upper_bound``, each ``[batch_size, output_dim]``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``confidence_level`` is not in ``(0, 1)``.

        Example::

            uncertainty = model.predict_with_uncertainty(x_test, confidence_level=0.95)
            predictions = uncertainty['point_estimates']
            data_noise = uncertainty['aleatoric_variance']
            model_unc = uncertainty['epistemic_variance']
            pred_width = uncertainty['upper_bound'] - uncertainty['lower_bound']
        """
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in the range (0, 1)")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-117: no separate
        # `self.predict(inputs)` call here; get_point_estimate/get_uncertainty
        # below each already run a forward pass. Do not re-add it.
        point_estimates = get_point_estimate(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer
        )

        total_variance, aleatoric_variance = get_uncertainty(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer,
            point_estimates=point_estimates
        )

        epistemic_variance = total_variance - aleatoric_variance

        lower_bound, upper_bound = get_prediction_intervals(
            point_estimates=point_estimates,
            total_variance=total_variance,
            confidence_level=confidence_level
        )

        return {
            'point_estimates': ops.convert_to_tensor(point_estimates),
            'total_variance': ops.convert_to_tensor(total_variance),
            'aleatoric_variance': ops.convert_to_tensor(aleatoric_variance),
            'epistemic_variance': ops.convert_to_tensor(epistemic_variance),
            'lower_bound': ops.convert_to_tensor(lower_bound),
            'upper_bound': ops.convert_to_tensor(upper_bound)
        }

    def compile(
            self,
            optimizer: Union[str, keras.optimizers.Optimizer],
            metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
            **kwargs: Any
    ) -> None:
        """Configure the model for training with the MDN negative-log-likelihood loss.

        The loss argument is not exposed here: the loss is always
        ``self.mdn_layer.loss_func``, since an ordinary regression loss
        applied to the concatenated ``[mu, sigma, pi]`` output would not be
        meaningful.

        :param optimizer: Optimizer instance or name (``'adam'``, ``'rmsprop'``,
            ``'sgd'``, ...).
        :type optimizer: Union[str, keras.optimizers.Optimizer]
        :param metrics: Metrics to track during training. Standard regression
            metrics may not apply directly, since the model outputs
            distribution parameters rather than point predictions.
        :type metrics: Optional[List[Union[str, keras.metrics.Metric]]]
        :param kwargs: Additional compile arguments (``loss_weights``,
            ``run_eagerly``, ...).

        Example::

            model.compile(optimizer='adam')
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                metrics=['mae']
            )
        """
        super().compile(
            optimizer=optimizer,
            loss=self.mdn_layer.loss_func,  # Negative log-likelihood loss
            metrics=metrics,
            **kwargs
        )
        logger.info(f"MDNModel compiled with optimizer: {optimizer}")

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Every constructor argument, plus the base model config.
        :rtype: Dict[str, Any]
        """
        config = {
            "hidden_layers": self.hidden_layers_sizes,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "hidden_activation": serialize_activation(self.hidden_activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate
        }
        # Merge with base model configuration
        base_config = super().get_config()
        return {**base_config, **config}

    def get_build_config(self) -> Dict[str, Any]:
        """Return the build configuration, separate from the constructor config.

        :return: Dictionary with the stored ``input_shape``.
        :rtype: Dict[str, Any]
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild the model layers from a stored build configuration.

        Called automatically when loading a saved model.

        :param config: Dictionary from :meth:`get_build_config`.
        :type config: Dict[str, Any]
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNModel":
        """Reconstruct a model from a configuration dictionary.

        :param config: Dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model instance with the same architecture.
        :rtype: MDNModel
        """
        config_copy = config.copy()

        # Deserialize complex objects that were serialized in get_config()
        config_copy["kernel_initializer"] = keras.initializers.deserialize(
            config["kernel_initializer"]
        )

        # Handle optional regularizer (may be None)
        if config["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config_copy)

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Save the model, adding a ``.keras`` extension if missing.

        :param filepath: Path to save to.
        :type filepath: str
        :param kwargs: Additional arguments passed to the parent ``save()``.

        Example::

            model.save('my_mdn_model')  # Becomes 'my_mdn_model.keras'
            loaded_model = keras.models.load_model(
                'my_mdn_model.keras',
                custom_objects={'MDNModel': MDNModel, 'MDNLayer': MDNLayer}
            )
        """
        if not filepath.endswith('.keras'):
            filepath += '.keras'

        logger.info(f"Saving MDNModel to: {filepath}")
        super().save(filepath, **kwargs)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape for a given input shape.

        :param input_shape: ``(batch_size, input_features)``; ``batch_size``
            may be ``None``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, 2*output_dim*num_mixtures + num_mixtures)``.
        :rtype: Tuple[Optional[int], ...]

        Example::

            input_shape = (None, 10)
            output_shape = model.compute_output_shape(input_shape)
            # output_shape == (None, 21) for output_dim=2, num_mixtures=3:
            # (2*2*3) + 3 = 12 + 9 = 21
        """
        input_shape_list = list(input_shape)

        output_features = (2 * self.output_dim * self.num_mix) + self.num_mix

        return tuple(input_shape_list[:-1] + [output_features])


def create_mdn_model(
        hidden_layers: List[int],
        output_dimension: int,
        num_mixtures: int,
        input_dimension: int,
        hidden_activation: str = MDNModel.DEFAULT_HIDDEN_ACTIVATION,
        kernel_initializer: Union[str, keras.initializers.Initializer] = MDNModel.DEFAULT_KERNEL_INITIALIZER,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = MDNModel.DEFAULT_USE_BATCH_NORM,
        dropout_rate: Optional[float] = None,
        **kwargs: Any
) -> MDNModel:
    """Construct and build an :class:`MDNModel`.

    Convenience factory mirroring ``create_nbeats_model``/``create_deepar``: it
    instantiates the model and runs a tiny inference-mode dummy forward pass so
    the returned model is already built (weights materialized), ready for
    ``.summary()``, ``.save()``, or weight transfer without a separate warmup.

    :param hidden_layers: Sizes of the hidden feature-extraction layers.
    :type hidden_layers: List[int]
    :param output_dimension: Dimensionality of the target/output space.
    :type output_dimension: int
    :param num_mixtures: Number of Gaussian mixture components.
    :type num_mixtures: int
    :param input_dimension: Feature dimension of the input, used only for the
        dummy forward pass that builds the model.
    :type input_dimension: int
    :param hidden_activation: Activation for hidden layers.
    :type hidden_activation: str
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional kernel regularizer.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_batch_norm: Whether to insert BatchNormalization.
    :type use_batch_norm: bool
    :param dropout_rate: Dropout rate in ``[0, 1)`` or ``None``.
    :type dropout_rate: Optional[float]
    :param kwargs: Forwarded to :class:`MDNModel` (e.g. ``name``).
    :return: A built :class:`MDNModel` instance.
    :rtype: MDNModel

    Example::

        model = create_mdn_model(
            hidden_layers=[64, 32],
            output_dimension=2,
            num_mixtures=5,
            input_dimension=10,
        )
        model.compile(optimizer="adam")
    """
    model = MDNModel(
        hidden_layers=hidden_layers,
        output_dimension=output_dimension,
        num_mixtures=num_mixtures,
        hidden_activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        **kwargs
    )

    # Build via a tiny inference-mode dummy forward pass.
    dummy = np.zeros((1, input_dimension), dtype="float32")
    model(dummy, training=False)
    return model
