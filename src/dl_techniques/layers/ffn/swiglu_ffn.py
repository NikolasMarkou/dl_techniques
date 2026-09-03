"""
SwiGLU feed-forward network, implemented by ``SwiGLUFFN``.

SwiGLU is the FFN used by PaLM, LLaMA and most models since. It is a Gated
Linear Unit: one projection of the input gates another projection of the
same input, so the layer can suppress or amplify a feature per token rather
than per weight. Three Dense layers do the work: ``gate_proj`` and
``up_proj`` read the input in parallel, then ``down_proj`` maps their
product to ``output_dim``. ``SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down``.

The hidden width defaults to a 2/3 rule, not the usual 4x expansion, rounded
up to a multiple of ``ffn_multiple_of``, unless ``hidden_dim`` is given
directly. Bias terms are optional and off by default.

References:
    - Shazeer, 2020. GLU Variants Improve Transformer. (https://arxiv.org/abs/2002.05202)
    - Chowdhery et al., 2022. PaLM: Scaling Language Modeling with
      Pathways. (https://arxiv.org/abs/2204.02311)
    - Touvron et al., 2023. LLaMA: Open and Efficient Foundation Language
      Models. (https://arxiv.org/abs/2302.13971)
"""

import keras
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.swiglu_ffn")
class SwiGLUFFN(keras.layers.Layer):
    """
    SwiGLU feed-forward network.

    Three projections. ``gate_proj`` and ``up_proj`` both read the input and
    produce ``hidden_dim`` features. The gate goes through SiLU and multiplies
    the up branch element-wise. ``down_proj`` maps the product to
    ``output_dim``:
    ``SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down``.

    The input width does not have to equal ``output_dim``. ``gate_proj`` and
    ``up_proj`` are built from whatever width arrives, and ``output_dim`` sets
    only the output.

    ``hidden_dim`` is derived from ``output_dim`` unless you pass it. The
    derivation, and the two knobs that drive it, are drawn below.

    Architecture:

    .. code-block:: text

               Input  [..., input_dim]
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          ┌───────────┐       ┌────────────┐
          │ gate_proj │       │  up_proj   │
          │  Dense(H) │       │  Dense(H)  │
          └─────┬─────┘       └─────┬──────┘
                ▼                   │
          ┌───────────┐             │
          │   SiLU    │             │
          └─────┬─────┘             │
                └─────────┬─────────┘
                          ▼
                    multiply  [..., H]
                          │
                          ▼
                  ┌───────────────┐
                  │   down_proj   │
                  │   Dense(O)    │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │   dropout     │
                  │  (optional)   │
                  └───────┬───────┘
                          ▼
              Output [..., output_dim]

        H = hidden_dim, O = output_dim. `dropout` really is
        conditional here: at dropout_rate=0.0 the attribute is
        None and no Dropout layer exists in the graph.

    Gate/up split and hidden-dim arithmetic:

    .. code-block:: text

        x  [..., input_dim]
        │
        ├──► gate_proj ──► g  [..., H] ──► SiLU(g) = g*sigmoid(g)
        │                                       │
        └──► up_proj   ──► u  [..., H] ─────────┤
                                                ▼
                                  h = SiLU(g) * u   [..., H]

        Only the gate branch is non-linear. gate_proj and
        up_proj get separate initializer instances, so they do
        not start out as the same function.

        H comes from _calculate_hidden_dim() when hidden_dim is
        None. Defaults: ffn_expansion_factor=4, multiple_of=256.

          step 1   raw = int(output_dim * factor * 2/3)
                   (the PaLM 2/3 rule; int() truncates)
          step 2   H   = multiple_of * ceil(raw / multiple_of)

        With the real defaults:

          output_dim   raw      H
          ----------   ------   -----
          64           170      256
          512          1365     1536
          768          2048     2048
          4096         10922    11008

        An explicit hidden_dim is used verbatim. Both knobs are
        then ignored, and get_config() still stores the None or
        the value you passed, never the computed H.

    :param output_dim: Width of the output, and the input to ``down_proj``.
        Must be positive. It does not constrain the input width.
    :type output_dim: int
    :param hidden_dim: Explicit hidden width. When given it is used verbatim
        and ``ffn_expansion_factor`` / ``ffn_multiple_of`` are ignored. When
        ``None`` (the default) the hidden width is derived from ``output_dim``
        as drawn above. This knob exists so SwiGLU can be sized the same way as
        every other FFN in the factory.
    :type hidden_dim: Optional[int]
    :param ffn_expansion_factor: Expansion factor in the 2/3 rule. Must be
        positive. Defaults to 4. Ignored when ``hidden_dim`` is given.
    :type ffn_expansion_factor: int
    :param ffn_multiple_of: The hidden width is rounded up to a multiple of
        this, for hardware efficiency. Must be positive. Defaults to 256.
        Ignored when ``hidden_dim`` is given.
    :type ffn_multiple_of: int
    :param dropout_rate: Dropout rate applied to the output, after
        ``down_proj``, in ``[0.0, 1.0]``. At 0.0 no Dropout layer is created.
        Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the three projections carry a bias. Defaults to
        False, which is what LLaMA-style models use.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels. ``up_proj`` and
        ``down_proj`` receive clones of it, never the same instance.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. All three Dense
        layers receive clones of it, never the same instance -- the same
        rule as the kernels. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar output_dim: Width of the output.
    :vartype output_dim: int
    :ivar hidden_dim: The resolved hidden width, either the explicit argument
        or the result of the 2/3 rule.
    :vartype hidden_dim: int
    :ivar _hidden_dim_arg: The hidden width as requested, possibly ``None``.
        This is what ``get_config()`` stores.
    :vartype _hidden_dim_arg: Optional[int]
    :ivar ffn_expansion_factor: The stored expansion factor.
    :vartype ffn_expansion_factor: int
    :ivar ffn_multiple_of: The stored rounding multiple.
    :vartype ffn_multiple_of: int
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the projections carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer. It is the source
        the per-layer clones are rebuilt from, and is not handed to any
        Dense layer itself.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar gate_proj: ``Dense(hidden_dim)``, the gate branch.
    :vartype gate_proj: keras.layers.Dense
    :ivar up_proj: ``Dense(hidden_dim)``, the value branch.
    :vartype up_proj: keras.layers.Dense
    :ivar down_proj: ``Dense(output_dim)``, the final projection.
    :vartype down_proj: keras.layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``, or ``None`` when the rate is 0.0.
    :vartype dropout: Optional[keras.layers.Dropout]

    :raises ValueError: If ``output_dim``, ``ffn_expansion_factor`` or
        ``ffn_multiple_of`` is not positive, if ``dropout_rate`` is outside
        ``[0.0, 1.0]``, or if ``hidden_dim`` is given and not positive.
    :raises ValueError: If a sub-layer constructor fails. The constructor
        catches any exception from sub-layer creation and re-raises it as
        ``ValueError``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The input width is
        independent of ``output_dim``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            ffn = SwiGLUFFN(output_dim=512)
            ffn.hidden_dim          # 1536
            y = ffn(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 512)

    Note:
        Sub-layers are created in ``__init__`` and built explicitly in
        ``build()``. Keras does not build them on its own here, because
        ``down_proj`` sees ``hidden_dim`` rather than the input width.
    """

    def __init__(
            self,
            output_dim: int,
            hidden_dim: Optional[int] = None,
            ffn_expansion_factor: int = 4,
            ffn_multiple_of: int = 256,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the three projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :raises ValueError: If ``output_dim``, ``ffn_expansion_factor`` or
            ``ffn_multiple_of`` is not positive, if ``dropout_rate`` is outside
            ``[0.0, 1.0]``, if ``hidden_dim`` is given and not positive, or if
            a sub-layer constructor fails.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        self._validate_inputs(output_dim, ffn_expansion_factor, ffn_multiple_of, dropout_rate)

        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        # Store every constructor argument; get_config() returns all of them.
        self.output_dim = output_dim
        # The requested hidden_dim (may be None), kept apart from the
        # resolved self.hidden_dim so get_config() round-trips the caller's
        # intent instead of pinning a computed size on reload.
        self._hidden_dim_arg = hidden_dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # An explicit hidden_dim wins; otherwise size via the 2/3 rule. Every
        # other FFN type in the factory is sized by hidden_dim, so this one
        # takes the same knob rather than a SwiGLU-only sizing path.
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None else self._calculate_hidden_dim()
        )

        # Create every sub-layer here, unbuilt. build() builds them.
        # A failure below is re-raised as ValueError so callers see one
        # exception type from this constructor.
        try:
            # The gate branch. SiLU is applied in call(), not here.
            self.gate_proj = keras.layers.Dense(
                self.hidden_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='gate_proj'
            )

            # DECISION plan-2026-08-19T163559-499b6f0e/D-070: cloned
            # initializers, never shared -- sharing made the gate a no-op. See decisions.md.
            self.up_proj = keras.layers.Dense(
                self.hidden_dim,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='up_proj'
            )

            # Back down to output_dim.
            self.down_proj = keras.layers.Dense(
                self.output_dim,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='down_proj'
            )

            # At rate 0.0 no Dropout layer is created at all.
            if self.dropout_rate > 0.0:
                self.dropout = keras.layers.Dropout(
                    self.dropout_rate,
                    name='dropout'
                )
            else:
                self.dropout = None

        except Exception as e:
            logger.error(f"Failed to create SwiGLUFFN sub-layers: {e}")
            raise ValueError(
                f"Failed to create SwiGLUFFN sub-layers. This might be due to "
                f"incompatible parameters or missing dependencies. Original error: {e}"
            )

        logger.info(f"SwiGLUFFN initialized: output_dim={output_dim}, "
                   f"hidden_dim={self.hidden_dim}, expansion_factor={ffn_expansion_factor}")

    def _validate_inputs(
            self,
            output_dim: int,
            ffn_expansion_factor: int,
            ffn_multiple_of: int,
            dropout_rate: float
    ) -> None:
        """
        Check the four numeric constructor arguments.

        Called from ``__init__`` before anything is stored. ``hidden_dim`` is
        checked separately in ``__init__``, because ``None`` is a valid value
        for it.

        :param output_dim: Output width. Must be positive.
        :type output_dim: int
        :param ffn_expansion_factor: Expansion factor. Must be positive.
        :type ffn_expansion_factor: int
        :param ffn_multiple_of: Rounding multiple. Must be positive.
        :type ffn_multiple_of: int
        :param dropout_rate: Dropout rate. Must be in ``[0.0, 1.0]``.
        :type dropout_rate: float
        :return: ``None``. The function is called for its exceptions.
        :rtype: None
        :raises ValueError: If any of the four is out of range.
        """
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if ffn_expansion_factor <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}")
        if ffn_multiple_of <= 0:
            raise ValueError(f"ffn_multiple_of must be positive, got {ffn_multiple_of}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def _calculate_hidden_dim(self) -> int:
        """
        Return the hidden width derived from ``output_dim``.

        Two steps, in order:

        1. ``int(output_dim * ffn_expansion_factor * 2 / 3)`` -- the 2/3 rule
           from the PaLM paper. ``int()`` truncates.
        2. Round that UP to the next multiple of ``ffn_multiple_of``.

        With the defaults (factor 4, multiple 256), ``output_dim=4096`` gives
        ``int(10922.67) = 10922``, rounded up to ``11008``.

        This runs only when ``hidden_dim`` was not supplied. An explicit
        ``hidden_dim`` is used verbatim and neither knob applies.

        :return: The rounded hidden width.
        :rtype: int
        """
        hidden_dim = int(self.output_dim * self.ffn_expansion_factor * 2 / 3)

        # Round up to a multiple of ffn_multiple_of for hardware efficiency.
        hidden_dim = self.ffn_multiple_of * (
                (hidden_dim + self.ffn_multiple_of - 1) // self.ffn_multiple_of
        )

        return hidden_dim

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the weights of every sub-layer.

        Each sub-layer is built explicitly so that all weight variables exist
        before Keras restores saved weights. A lazily-built sub-layer would be
        skipped on load and would silently keep its fresh initialization.

        :param input_shape: Shape of the input tensor. Its last axis is the
            input width and may differ from ``output_dim``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Both projections read the raw input, so both take input_shape.
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)

        # down_proj and dropout see (..., hidden_dim).
        down_input_shape = list(input_shape)
        down_input_shape[-1] = self.hidden_dim
        self.down_proj.build(tuple(down_input_shape))

        if self.dropout is not None:
            self.dropout.build(tuple(down_input_shape))

        # Keras requires the parent build() call last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the SwiGLU forward pass.

        Computes ``(SiLU(x @ W_gate) * (x @ W_up)) @ W_down``, then dropout.

        :param inputs: Input tensor of any rank, shape ``(..., input_dim)``.
            The input width need not equal ``output_dim``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, passed to the dropout sub-layer
            when one exists.
        :type training: Optional[bool]
        :return: Tensor with the same rank as ``inputs`` and last axis
            ``output_dim``.
        :rtype: keras.KerasTensor
        """
        # Two parallel projections of the same input.
        # Both produce shape (..., hidden_dim).
        gate = self.gate_proj(inputs)
        up = self.up_proj(inputs)

        # SiLU, that is x * sigmoid(x). Only the gate branch gets it.
        gate_activated = keras.ops.silu(gate)

        hidden = gate_activated * up

        # Project (..., hidden_dim) down to (..., output_dim).
        output = self.down_proj(hidden)

        # self.dropout is None when dropout_rate == 0.0, so in that case there
        # is no Dropout layer in the graph at all.
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the input shape with its last axis set to ``output_dim``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the last axis replaced by ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        # DECISION plan-2026-07-30T140922-8af1028f/D-013: return output_dim,
        # not input_shape -- BaseVLMHead derives its width assertion from this. See decisions.md.
        return tuple(input_shape)[:-1] + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        ``hidden_dim`` is returned as requested, not as resolved, so a layer
        built with ``hidden_dim=None`` re-sizes itself on reload instead of
        being pinned to whatever width it happened to compute.

        :return: The base layer config plus every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            # The requested value (may be None), not the resolved one.
            "hidden_dim": self._hidden_dim_arg,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "ffn_multiple_of": self.ffn_multiple_of,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @property
    def num_parameters(self) -> int:
        """
        Return the total number of parameters, or 0 if not built.

        :return: The sum of ``keras.ops.size`` over ``self.weights``,
            trainable and non-trainable. Returns 0 before ``build()`` has run,
            because no weights exist yet.
        :rtype: int
        """
        if not self.built:
            return 0

        total_params = 0
        for weight in self.weights:
            # ops.size keeps this backend-agnostic.
            total_params += keras.ops.size(weight)
        return int(total_params)

# ---------------------------------------------------------------------
