"""
Matrix Product State (MPS) layer, built by the ``MPSLayer`` class.

A dense layer stores one weight matrix connecting every input to every
output, which grows quadratically with input size. This layer instead gives
each input feature its own small "core" tensor and contracts them in
sequence along a chain, in the style of a tensor-train decomposition from
quantum many-body physics. Parameter count grows linearly with input
dimension instead of quadratically, at the cost of a fixed-capacity "bond
dimension" limiting how much correlation the chain can carry between
features.

References:
    - Stoudenmire & Schwab, 2016. Supervised Learning with Tensor Networks.
    - Novikov et al., 2015. Tensorizing Neural Networks.
    - Cohen et al., 2016. On the Expressive Power of Deep Learning: A Tensor
      Analysis.
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.structured_linear.mps_layer")
class MPSLayer(keras.layers.Layer):
    """Matrix Product State (MPS) layer for efficient tensor decomposition.

    This layer implements a quantum-inspired tensor network that efficiently
    parameterizes correlations between input features. It decomposes the
    high-dimensional weight tensor into a chain of smaller core tensors,
    reducing parameters from ``O(n^2)`` to ``O(n * bond_dim^2)`` while
    capturing long-range dependencies through sequential tensor contractions:
    ``output = B * (A^1[x_1] * A^2[x_2] * ... * A^n[x_n]) * P``, where
    ``B`` is the boundary vector, ``A^i`` are core tensors activated by
    input features ``x_i``, and ``P`` is the projection matrix to output space.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │    Input [batch, input_dim]          │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Initialize boundary vector B        │
        │  [batch, bond_dim] = ones            │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  For i = 1 to input_dim:             │
        │    M_i = x_i * core[i]               │
        │    B = B @ M_i                       │
        │  (sequential tensor contraction)     │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Project: output = B @ P + bias      │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │    Output [batch, output_dim]        │
        └──────────────────────────────────────┘

    :param output_dim: Dimension of the output tensor. Must be positive.
    :type output_dim: int
    :param bond_dim: Internal bond dimension controlling expressiveness.
        Higher values capture more complex correlations. Defaults to 16.
    :type bond_dim: int
    :param use_bias: Whether to include bias in the final projection.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for core tensors and projection.
        Defaults to ``'glorot_uniform'``. ``build()`` draws ``mps_cores`` and
        ``projection`` from their own clones of this initializer, so a RANDOM
        SEEDLESS instance yields independent draws at both sites. Differing
        shapes are NOT what makes two sites independent: one shared seedless
        instance replays the same underlying sample everywhere. Three
        exemptions, all correct behaviour: a caller-supplied SEEDED instance
        (e.g. ``GlorotUniform(seed=7)``) -- cloning reproduces an explicit seed
        deliberately and by contract
        (``dl_techniques.initializers.clone_initializer``), so the two weights
        stay tied; a DETERMINISTIC initializer (``'zeros'``, ``'ones'``,
        ``Constant``, and ``Identity`` where the weight is 2-D) -- it holds no
        random state, so every site is bit-identical and that is what it is
        meant to do; and a CUSTOM initializer whose
        ``get_config()``/``from_config()`` round trip raises --
        ``clone_initializer`` then falls back to ``copy.deepcopy``, which
        copies the already-resolved seed rather than drawing a new one
        (measured: a deepcopy of a seedless glorot_uniform draws
        bit-identically), so such a site can silently stay tied. For
        reproducibility WITHOUT the tie, call ``keras.utils.set_random_seed()``
        and leave this argument seedless.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for core tensors and projection.
        Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_initializer: Initializer for bias terms.
        Defaults to ``'zeros'``. There is a single bias site, so nothing can
        alias against it and it is not cloned.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Regularizer for bias terms. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Regularizer for layer output. Defaults to None.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If output_dim or bond_dim are not positive.
    """

    def __init__(
        self,
        output_dim: int,
        bond_dim: int = 16,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the MPSLayer."""
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if bond_dim <= 0:
            raise ValueError(f"bond_dim must be positive, got {bond_dim}")

        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Created in build(), once input_dim is known.
        self.cores = None
        self.projection = None
        self.bias_weight = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights based on input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input_shape is invalid.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"MPSLayer requires input with at least 2 dimensions, "
                f"got shape {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "Last dimension of input must be defined (not None). "
                f"Got input_shape: {input_shape}"
            )

        input_dim = int(input_dim)

        # DECISION plan-2026-09-07T161712-985e4d31/D-008: mps_cores and
        # projection each draw from their OWN clone of self.kernel_initializer.
        # Do NOT "simplify" them back to the bare attribute: one seedless Keras
        # 3 initializer instance self-assigns a seed at construction and
        # replays THE SAME UNDERLYING SAMPLE at every later site, so these two
        # weights -- the chain's per-feature core tensors and the single
        # bond-to-output map, architecturally unrelated -- were the same random
        # numbers. Measured before this change at output_dim=8, bond_dim=8:
        # projection was mps_cores' flattened prefix times 2.828427 (ratio std
        # 9.5e-07), Pearson r = 0.9999999999999968.
        #
        # What is NOT the criterion: matching SHAPES. mps_cores is (input_dim,
        # bond_dim, bond_dim) and projection is (bond_dim, output_dim) --
        # different ranks and different sizes -- and they aliased anyway,
        # because the shorter draw came out as the longer draw's prefix up to
        # the fan-based scale -- measured on this backend for every shape pair
        # tried, the durable mechanism being the replayed seed and the prefix
        # relation how the stateless RNG realises it. An earlier revision of
        # this plan CLEARED this layer on the rule that differing shapes cannot
        # alias; measurement refuted that rule. Every add_weight fed by a
        # shared initializer instance gets its own clone here.
        #
        # Scope of the claim, exactly: independence holds for a RANDOM SEEDLESS
        # initializer. Three exemptions, all correct behaviour, none a defect:
        #   (1) a caller-supplied SEEDED instance (e.g. GlorotUniform(seed=7)):
        #       clone_initializer reproduces an explicit seed deliberately and
        #       by contract (initializers/clone.py), so the clones stay tied --
        #       measured r = 0.9999999999999957 across these very shapes;
        #   (2) a DETERMINISTIC initializer ('zeros', 'ones', Constant, and
        #       Identity where the weight is 2-D): it holds no random state, so
        #       every site is bit-identical and that is what it is meant to do;
        #       cloning it is a no-op;
        #   (3) a CUSTOM initializer whose get_config()/from_config() round
        #       trip raises: clone_initializer falls back to copy.deepcopy,
        #       which copies the already-resolved seed rather than drawing a
        #       new one (measured: a deepcopy of a seedless glorot_uniform
        #       draws bit-identically), so such a site can silently stay tied
        #       with no diagnostic.
        #
        # Callers wanting reproducibility WITHOUT the tie should use
        # keras.utils.set_random_seed() and leave the initializer seedless. The
        # default and exemptions (1) and (2) are pinned by
        # TestInitializerAliasing in
        # tests/test_layers/test_structured_linear/test_mps_layer.py. Exemption
        # (3) has NO test here: it is a property of clone_initializer, not of
        # this layer. See decisions.md D-008.
        #
        # Behaviour change: initial weights move, so a training run seeded only
        # by a fixed process seed will not reproduce pre-change results
        # bit-for-bit, and any checkpoint of this layer taken before the change
        # holds weights drawn under the old, tied scheme.
        #
        # The single bias site has nothing to alias against and is not cloned.
        # One core tensor per input feature.
        self.cores = self.add_weight(
            name="mps_cores",
            shape=(input_dim, self.bond_dim, self.bond_dim),
            initializer=clone_initializer(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.projection = self.add_weight(
            name="projection",
            shape=(self.bond_dim, self.output_dim),
            initializer=clone_initializer(self.kernel_initializer),
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        if self.use_bias:
            self.bias_weight = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing MPS tensor contraction.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, output_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(inputs)[0]
        input_dim = keras.ops.shape(inputs)[1]

        # Left boundary vector of the MPS chain.
        boundary = keras.ops.ones((batch_size, self.bond_dim))

        # B x A^1[x_1] x A^2[x_2] x ... x A^n[x_n], one core per step.
        for i in range(input_dim):
            x_i = keras.ops.expand_dims(inputs[:, i], axis=-1)
            core_i = self.cores[i, :, :]
            weighted_core = keras.ops.expand_dims(x_i, axis=-1) * keras.ops.expand_dims(core_i, axis=0)
            boundary = keras.ops.einsum('bi,bij->bj', boundary, weighted_core)

        output = keras.ops.matmul(boundary, self.projection)

        if self.use_bias:
            output = keras.ops.add(output, self.bias_weight)

        return output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'bond_dim': self.bond_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
