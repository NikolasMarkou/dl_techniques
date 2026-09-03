"""
MothNet: a three-stage model of the insect olfactory network, built by the
`MothNet` class and used as a feature generator via `create_cyborg_features`.

Gradient descent needs many examples per class because it moves a shared
parameter set by small increments. A moth learns a new odor from a handful
of exposures using a different arrangement: an antennal lobe stage does
competitive inhibition (each unit's output is measured against the
population mean, removing the common, non-discriminative part of the
input), then a mushroom body stage expands the signal through a frozen
random sparse projection and keeps only the top-k activations, turning
each input into a small, nearly disjoint set of active units. Because two
classes then barely share active units, a single Hebbian readout weight
per unit is enough to tell them apart, and there is nothing for the
projection to learn — it stays frozen.

The readout trains by a Hebbian rule, `W <- W + alpha * (1/N) *
mb_output^T y`, not by backpropagation, so labels must be one-hot: the
rule reads `y` as the post-synaptic activation, not as a class index.
`train_hebbian` runs a plain Python loop, not a Keras `train_step`, since
there is no gradient to compute. The cross-entropy it reports is for
monitoring only and is not what drives the weight updates.

`create_cyborg_features` concatenates the model's readout activations
(`num_classes` wide, not `mb_units`) onto the original input for use by a
conventional classifier. The wider sparse code is available separately
through `extract_mb_features`.

References:
    - Delahunt & Kutz, 2019. Putting a bug in ML: The moth olfactory network learns
      to read MNIST. Neural Networks 118, 54-64.
      (https://arxiv.org/abs/1802.05405)
    - Delahunt & Kutz, 2018. Insect cyborgs: Bio-mimetic feature generators improve
      machine learning accuracy on limited data.
      (https://arxiv.org/abs/1808.08124)
    - Dasgupta et al., 2017. A neural algorithm for a fundamental computing problem.
      Science 358, 793-796.
    - Olsen et al., 2010. Divisive normalization in olfactory population codes.
      Neuron 66, 287-299.
    - Hebb, 1949. The Organization of Behavior. Wiley.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.mothnet_blocks import (
    AntennalLobeLayer,
    MushroomBodyLayer,
    HebbianReadoutLayer
)
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mothnet.model")
class MothNet(keras.Model):
    """
    Antennal lobe, mushroom body, and Hebbian readout, chained into one model.

    Use it either as a standalone classifier trained by :meth:`train_hebbian`,
    or as a feature generator: pass a trained instance to
    :func:`create_cyborg_features` to concatenate its readout activations onto
    the original input for a conventional classifier.

    Architecture:

    .. code-block:: text

        Input [B, input_dim]
               │
               ▼
        ┌─────────────────────────┐
        │  Antennal Lobe (AL)     │  competitive inhibition
        │  [B, al_units]          │
        └────────────┬────────────┘
                      ▼
        ┌─────────────────────────┐
        │  Mushroom Body (MB)     │  frozen sparse projection, top-k
        │  [B, mb_units]          │
        └────────────┬────────────┘
                      ▼
        ┌─────────────────────────┐
        │  Hebbian Readout        │  trained by train_hebbian, not fit()
        │  [B, num_classes]       │
        └────────────┬────────────┘
                      ▼
        Class logits, or features for create_cyborg_features

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param al_units: Number of antennal-lobe units. Defaults to the input
        dimension (no compression) when ``None``.
    :type al_units: Optional[int]
    :param mb_units: Number of mushroom-body units; 2000-4000 gives a
        20-50x expansion over a typical ``al_units``.
    :type mb_units: int
    :param mb_sparsity: Fraction of mushroom-body units that fire per input.
    :type mb_sparsity: float
    :param connection_sparsity: Fraction of nonzero AL-to-MB connections.
    :type connection_sparsity: float
    :param hebbian_learning_rate: Learning rate for the Hebbian readout update.
    :type hebbian_learning_rate: float
    :param inhibition_strength: Competitive-inhibition strength in the
        antennal lobe, in ``[0, 1]``.
    :type inhibition_strength: float
    :param al_activation: Activation function for the antennal-lobe layer.
    :type al_activation: str
    :param mb_activation: Activation function for the mushroom-body layer.
    :type mb_activation: str
    :param kwargs: Additional keyword arguments for the base ``keras.Model``.

    :ivar antennal_lobe: Competitive-inhibition layer.
    :vartype antennal_lobe: AntennalLobeLayer
    :ivar mushroom_body: Sparse projection layer.
    :vartype mushroom_body: MushroomBodyLayer
    :ivar readout: Hebbian readout layer.
    :vartype readout: HebbianReadoutLayer

    Input shape:
        ``(batch_size, input_dim)``.

    Output shape:
        ``(batch_size, num_classes)`` class logits, before softmax.

    Example:
        >>> model = MothNet(num_classes=10, al_units=85, mb_units=4000)
        >>> model.build((None, 784))
        >>> y_onehot = keras.utils.to_categorical(y_train, 10)
        >>> model.train_hebbian(x_train, y_onehot, epochs=5, batch_size=32)
        >>> features = model.extract_features(x_train)

    Note:
        Labels passed to :meth:`train_hebbian` must be one-hot: the update
        rule reads ``y`` as a post-synaptic activation, not a class index.
    """

    def __init__(
        self,
        num_classes: int,
        al_units: Optional[int] = None,
        mb_units: int = 2000,
        mb_sparsity: float = 0.1,
        connection_sparsity: float = 0.1,
        hebbian_learning_rate: float = 0.01,
        inhibition_strength: float = 0.5,
        al_activation: str = 'relu',
        mb_activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.al_units = al_units
        self.mb_units = mb_units
        self.mb_sparsity = mb_sparsity
        self.connection_sparsity = connection_sparsity
        self.hebbian_learning_rate = hebbian_learning_rate
        self.inhibition_strength = inhibition_strength
        self.al_activation = deserialize_activation(al_activation)
        self.mb_activation = deserialize_activation(mb_activation)

        # Sub-layers will be initialized in build()
        self.antennal_lobe = None
        self.mushroom_body = None
        self.readout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create and explicitly build the three sub-layers, for reliable serialization.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is ``None``.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        al_units = self.al_units if self.al_units is not None else input_dim

        self.antennal_lobe = AntennalLobeLayer(
            units=al_units,
            inhibition_strength=self.inhibition_strength,
            activation=self.al_activation,
            name='antennal_lobe'
        )

        self.mushroom_body = MushroomBodyLayer(
            units=self.mb_units,
            sparsity=self.mb_sparsity,
            connection_sparsity=self.connection_sparsity,
            activation=self.mb_activation,
            trainable_projection=False,
            name='mushroom_body'
        )

        self.readout = HebbianReadoutLayer(
            units=self.num_classes,
            learning_rate=self.hebbian_learning_rate,
            name='hebbian_readout'
        )

        self.antennal_lobe.build(input_shape)

        al_output_shape = self.antennal_lobe.compute_output_shape(input_shape)
        self.mushroom_body.build(al_output_shape)

        mb_output_shape = self.mushroom_body.compute_output_shape(al_output_shape)
        self.readout.build(mb_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the antennal lobe, mushroom body, and readout in sequence.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Class logits of shape ``(batch_size, num_classes)``.
        :rtype: keras.KerasTensor
        """
        al_output = self.antennal_lobe(inputs, training=training)
        mb_output = self.mushroom_body(al_output, training=training)
        output = self.readout(mb_output, training=training)
        return output

    def extract_features(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Return readout activations for concatenation via `create_cyborg_features`.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Feature tensor of shape ``(batch_size, num_classes)``.
        :rtype: keras.KerasTensor
        """
        return self(inputs, training=False)

    def extract_mb_features(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Return the mushroom-body sparse code, before the readout layer.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Sparse feature tensor of shape ``(batch_size, mb_units)``.
        :rtype: keras.KerasTensor
        """
        al_output = self.antennal_lobe(inputs, training=False)
        mb_output = self.mushroom_body(al_output, training=False)
        return mb_output

    def train_hebbian(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, list]:
        """Train the readout weights by the Hebbian update rule, not backpropagation.

        :param x: Training data of shape ``(num_samples, input_dim)``.
        :type x: np.ndarray
        :param y: Training labels of shape ``(num_samples, num_classes)``. Must be
            one-hot; use ``keras.utils.to_categorical(labels, num_classes)``.
        :type y: np.ndarray
        :param epochs: Number of training epochs. 1-5 is typically enough.
        :type epochs: int
        :param batch_size: Batch size for training.
        :type batch_size: int
        :param verbose: 0 silent, 1 one line per epoch, 2 same as 1.
        :type verbose: int
        :return: Training history with a ``'loss'`` key holding one value per epoch.
        :rtype: Dict[str, list]

        Example:
            >>> y_onehot = keras.utils.to_categorical(y_train, num_classes=10)
            >>> history = model.train_hebbian(x_train, y_onehot, epochs=5)

        Note:
            The reported loss is cross-entropy for monitoring only; the Hebbian
            update does not minimize it.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-017: build() creates the
        # sublayers, so an unbuilt model here would call None(...) and crash.
        # Do not remove this guard. See decisions.md.
        if not self.built:
            self.build((None, x.shape[-1]))

        num_samples = x.shape[0]
        history = {'loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                batch_x = x_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                # to_categorical returns float64; hebbian_update requires it to match
                # the float32 mushroom-body output.
                batch_x_tensor = keras.ops.cast(batch_x, self.compute_dtype)
                batch_y_tensor = keras.ops.cast(batch_y, self.compute_dtype)

                al_output = self.antennal_lobe(batch_x_tensor, training=True)
                mb_output = self.mushroom_body(al_output, training=True)

                self.readout.hebbian_update(mb_output, batch_y_tensor)

                logits = self.readout(mb_output, training=True)
                loss = keras.ops.mean(
                    keras.losses.categorical_crossentropy(
                        batch_y_tensor, logits, from_logits=True
                    )
                )
                epoch_loss += keras.ops.convert_to_numpy(loss)
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)

            if verbose > 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

        return history

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary of every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'al_units': self.al_units,
            'mb_units': self.mb_units,
            'mb_sparsity': self.mb_sparsity,
            'connection_sparsity': self.connection_sparsity,
            'hebbian_learning_rate': self.hebbian_learning_rate,
            'inhibition_strength': self.inhibition_strength,
            'al_activation': serialize_activation(self.al_activation),
            'mb_activation': serialize_activation(self.mb_activation),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MothNet':
        """Create a model from a `get_config()` dictionary.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new `MothNet` instance.
        :rtype: MothNet
        """
        return cls(**config)


def create_cyborg_features(
    mothnet: MothNet,
    x_data: np.ndarray
) -> np.ndarray:
    """Concatenate a trained MothNet's readout activations onto the original input.

    :param mothnet: A `MothNet` instance already trained via `train_hebbian`.
    :type mothnet: MothNet
    :param x_data: Original feature data of shape ``(num_samples, input_dim)``.
    :type x_data: np.ndarray
    :return: Augmented features of shape
        ``(num_samples, input_dim + num_classes)``: the first ``input_dim``
        columns are the original features, the rest are readout activations.
    :rtype: np.ndarray

    Example:
        >>> mothnet.train_hebbian(x_train, y_train_onehot, epochs=5)
        >>> x_train_cyborg = create_cyborg_features(mothnet, x_train)
        >>> x_test_cyborg = create_cyborg_features(mothnet, x_test)

    Note:
        The concatenated features tend to be complementary to what a
        norm-based ML method finds on its own, which is why the plain
        concatenation improves SVM, kNN, and neural network accuracy alike.
    """
    mothnet_features = keras.ops.convert_to_numpy(
        mothnet.extract_features(x_data)
    )
    cyborg_features = np.concatenate([x_data, mothnet_features], axis=1)
    return cyborg_features


