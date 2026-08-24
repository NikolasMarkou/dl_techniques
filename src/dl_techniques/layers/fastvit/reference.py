"""Reference-fidelity constants shared by every ``layers/fastvit/`` block.

Both facts below are properties of the timm / MobileCLIP2 reference implementation,
not of any one block, and BOTH are silent when wrong: a normalization epsilon and a
padding convention are invisible to every shape assertion. They were each shipped
wrong once (an undisclosed 100x epsilon divergence in 86 of 114 BatchNormalizations,
and a one-pixel sampling shift in every strided convolution), so they live here in
ONE place and every call site imports them.

Interface contract for consumers: import the constant, never re-declare its value.
A second literal ``1e-5`` or ``'reference'`` in a block module is a defect — the two
copies drift and nothing in the suite can tell.

:var REFERENCE_NORM_EPSILON: Variance epsilon for every normalization layer in the
    port. PyTorch's ``BatchNorm2d`` and ``LayerNorm`` both default to ``1e-5``,
    while Keras' ``BatchNormalization`` defaults to ``1e-3``, Keras'
    ``LayerNormalization`` to ``1e-3``, and this repo's ``create_normalization_layer``
    ``setdefault``s ``1e-6``. Every one of those differs from the reference, so the
    value must be passed EXPLICITLY at every construction site.
:var REFERENCE_PADDING_MODE: ``padding_mode`` to pass to
    :class:`~dl_techniques.layers.mobile_one_block.MobileOneBlock` (and to use for
    the convolutions authored here). PyTorch pads SYMMETRICALLY by
    ``kernel_size // 2``; Keras' ``padding='same'`` pads asymmetrically, which at
    stride > 1 makes the sampled grid depend on the kernel size. See
    :func:`~dl_techniques.layers.mobile_one_block.resolve_conv_padding`.
"""

#: See the module docstring.
REFERENCE_NORM_EPSILON = 1e-5

#: See the module docstring.
REFERENCE_PADDING_MODE = 'reference'
