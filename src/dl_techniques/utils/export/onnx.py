"""
ONNX Export Utility for Keras 3 Models
======================================

This module provides utilities for exporting Keras 3 models (with TensorFlow backend)
to ONNX format for deployment and interoperability.

Requirements:
    pip install tf2onnx onnx onnxruntime --break-system-packages

Usage:
    from export_to_onnx import export_keras_model_to_onnx

    # Export a saved model
    export_keras_model_to_onnx(
        model_path="path/to/model.keras",
        output_path="path/to/model.onnx",
        opset_version=17
    )

    # Export a model instance
    export_keras_model_to_onnx(
        model=my_model,
        output_path="path/to/model.onnx"
    )
"""

import os
import keras
import numpy as np
import tensorflow as tf
from typing import Optional, List, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..logger import logger

# ---------------------------------------------------------------------


def export_keras_model_to_onnx(
        model: Optional[keras.Model] = None,
        model_path: Optional[str] = None,
        output_path: str = "model.onnx",
        opset_version: int = 17,
        input_signature: Optional[List[Tuple[int, ...]]] = None,
        custom_objects: Optional[Dict[str, Any]] = None,
        validate_output: bool = True,
        optimization_level: str = "basic",
) -> str:
    """
    Export a Keras 3 model to ONNX format.

    This function handles the conversion of Keras models (saved or in-memory)
    to ONNX format using tf2onnx as the conversion backend.

    :param model: A Keras model instance to export. Either this or model_path
        must be provided.
    :type model: Optional[keras.Model]
    :param model_path: Path to a saved Keras model (.keras format). Either this
        or model must be provided.
    :type model_path: Optional[str]
    :param output_path: Path where the ONNX model will be saved.
    :type output_path: str
    :param opset_version: ONNX opset version to use. Higher versions support
        more operators. Recommended: 13-17.
    :type opset_version: int
    :param input_signature: Optional list of input shapes. If not provided,
        will be inferred from the model.
    :type input_signature: Optional[List[Tuple[int, ...]]]
    :param custom_objects: Dictionary of custom objects needed to load the model.
    :type custom_objects: Optional[Dict[str, Any]]
    :param validate_output: Whether to validate the ONNX model after conversion.
    :type validate_output: bool
    :param optimization_level: Optimization level for tf2onnx. One of
        "basic", "extended", or "none".
    :type optimization_level: str
    :returns: Path to the exported ONNX model.
    :rtype: str
    :raises ValueError: If neither model nor model_path is provided.
    :raises ImportError: If required dependencies are not installed.

    Example::

        # From saved model
        onnx_path = export_keras_model_to_onnx(
            model_path="trained_model.keras",
            output_path="deployed_model.onnx",
            opset_version=17
        )

        # From model instance
        model = keras.models.load_model("trained_model.keras")
        onnx_path = export_keras_model_to_onnx(
            model=model,
            output_path="deployed_model.onnx"
        )
    """
    # Validate inputs
    if model is None and model_path is None:
        raise ValueError("Either 'model' or 'model_path' must be provided.")

    # Import dependencies
    try:
        import tf2onnx
        import onnx
    except ImportError as e:
        raise ImportError(
            "Required dependencies not found. Install with:\n"
            "pip install tf2onnx onnx onnxruntime --break-system-packages"
        ) from e

    # Load model if path provided
    if model is None:
        logger.info(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

    logger.info(f"Model summary:")
    model.summary(print_fn=logger.info)

    # Get input specifications
    input_specs = _get_input_specs(model, input_signature)
    logger.info(f"Input specifications: {input_specs}")

    # Create a concrete function for conversion
    logger.info("Creating TensorFlow concrete function...")

    @tf.function(input_signature=input_specs)
    def model_func(*args):
        """Wrapper function for model inference."""
        if len(args) == 1:
            return model(args[0], training=False)
        return model(args, training=False)

    # Get concrete function
    concrete_func = model_func.get_concrete_function()

    # Convert to ONNX
    logger.info(f"Converting to ONNX (opset version {opset_version})...")

    # Set optimization options
    extra_opset = []
    if optimization_level == "extended":
        extra_opset = [tf2onnx.constants.CONTRIB_OPS_DOMAIN]

    model_proto, _ = tf2onnx.convert.from_function(
        concrete_func,
        opset=opset_version,
        output_path=output_path,
        extra_opset=extra_opset,
    )

    logger.info(f"ONNX model saved to: {output_path}")

    # Validate the exported model
    if validate_output:
        _validate_onnx_model(output_path, model, input_specs)

    return output_path


def _get_input_specs(
        model: keras.Model,
        input_signature: Optional[List[Tuple[int, ...]]] = None,
) -> List:
    """
    Get TensorFlow input specifications from a Keras model.

    :param model: The Keras model to extract input specs from.
    :type model: keras.Model
    :param input_signature: Optional explicit input shapes.
    :type input_signature: Optional[List[Tuple[int, ...]]]
    :returns: List of TensorFlow TensorSpecs.
    :rtype: List[tf.TensorSpec]
    """

    if input_signature is not None:
        # Use provided input signature
        return [
            tf.TensorSpec(shape=shape, dtype=tf.float32, name=f"input_{i}")
            for i, shape in enumerate(input_signature)
        ]

    # Infer from model
    input_specs = []

    if hasattr(model, 'input_shape'):
        # Single input
        if isinstance(model.input_shape, tuple):
            # Replace None batch dimension with dynamic size
            shape = list(model.input_shape)
            input_specs.append(
                tf.TensorSpec(shape=shape, dtype=tf.float32, name="input")
            )
        elif isinstance(model.input_shape, list):
            # Multiple inputs
            for i, inp_shape in enumerate(model.input_shape):
                shape = list(inp_shape)
                input_specs.append(
                    tf.TensorSpec(shape=shape, dtype=tf.float32, name=f"input_{i}")
                )
    elif hasattr(model, 'inputs') and model.inputs:
        # Functional API model
        for i, inp in enumerate(model.inputs):
            shape = list(inp.shape)
            dtype = inp.dtype
            # Convert Keras dtype to TF dtype
            tf_dtype = tf.as_dtype(dtype) if dtype else tf.float32
            input_specs.append(
                tf.TensorSpec(shape=shape, dtype=tf_dtype, name=inp.name or f"input_{i}")
            )
    else:
        raise ValueError(
            "Could not infer input shape from model. "
            "Please provide 'input_signature' explicitly."
        )

    return input_specs


def _validate_onnx_model(
        onnx_path: str,
        keras_model: keras.Model,
        input_specs: List,
) -> None:
    """
    Validate the ONNX model by comparing outputs with the original Keras model.

    :param onnx_path: Path to the ONNX model.
    :type onnx_path: str
    :param keras_model: Original Keras model for comparison.
    :type keras_model: keras.Model
    :param input_specs: Input specifications for generating test data.
    :type input_specs: List
    :raises AssertionError: If outputs don't match within tolerance.
    """
    import onnx
    import onnxruntime as ort

    logger.info("Validating ONNX model...")

    # Check ONNX model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure validation passed.")

    # Create ONNX runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options)

    # Generate test inputs
    test_inputs = []
    ort_inputs = {}

    for i, spec in enumerate(input_specs):
        # Create random test input
        shape = [s if s is not None else 2 for s in spec.shape]
        test_input = np.random.randn(*shape).astype(np.float32)
        test_inputs.append(test_input)

        # Get ONNX input name
        ort_input_name = session.get_inputs()[i].name
        ort_inputs[ort_input_name] = test_input

    # Get Keras output
    if len(test_inputs) == 1:
        keras_output = keras_model(test_inputs[0], training=False)
    else:
        keras_output = keras_model(test_inputs, training=False)

    keras_output_np = keras.ops.convert_to_numpy(keras_output)

    # Get ONNX output
    ort_outputs = session.run(None, ort_inputs)
    onnx_output_np = ort_outputs[0]

    # Compare outputs
    np.testing.assert_allclose(
        keras_output_np,
        onnx_output_np,
        rtol=1e-4,
        atol=1e-4,
        err_msg="ONNX output differs from Keras output!"
    )

    logger.info("Output validation passed - ONNX model produces matching results.")

    # Log model info
    logger.info(f"ONNX model inputs: {[inp.name for inp in session.get_inputs()]}")
    logger.info(f"ONNX model outputs: {[out.name for out in session.get_outputs()]}")


def export_with_dynamic_batch(
        model: keras.Model,
        output_path: str = "model.onnx",
        opset_version: int = 17,
        batch_dim_name: str = "batch_size",
) -> str:
    """
    Export a Keras model to ONNX with dynamic batch size.

    This is useful for inference scenarios where batch size varies.

    :param model: The Keras model to export.
    :type model: keras.Model
    :param output_path: Path where the ONNX model will be saved.
    :type output_path: str
    :param opset_version: ONNX opset version.
    :type opset_version: int
    :param batch_dim_name: Name for the dynamic batch dimension.
    :type batch_dim_name: str
    :returns: Path to the exported ONNX model.
    :rtype: str
    """
    import tensorflow as tf
    import tf2onnx
    import onnx
    from onnx import helper

    # First export with fixed batch size
    temp_path = output_path.replace(".onnx", "_temp.onnx")
    export_keras_model_to_onnx(
        model=model,
        output_path=temp_path,
        opset_version=opset_version,
        validate_output=False,
    )

    # Load and modify for dynamic batch
    onnx_model = onnx.load(temp_path)

    # Make batch dimension dynamic
    for input_tensor in onnx_model.graph.input:
        dim = input_tensor.type.tensor_type.shape.dim[0]
        dim.dim_param = batch_dim_name

    for output_tensor in onnx_model.graph.output:
        dim = output_tensor.type.tensor_type.shape.dim[0]
        dim.dim_param = batch_dim_name

    # Save modified model
    onnx.save(onnx_model, output_path)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    logger.info(f"Exported ONNX model with dynamic batch to: {output_path}")
    return output_path


def get_onnx_model_info(onnx_path: str) -> Dict[str, Any]:
    """
    Get information about an ONNX model.

    :param onnx_path: Path to the ONNX model.
    :type onnx_path: str
    :returns: Dictionary containing model information.
    :rtype: Dict[str, Any]
    """
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    session = ort.InferenceSession(onnx_path)

    info = {
        "opset_version": onnx_model.opset_import[0].version,
        "ir_version": onnx_model.ir_version,
        "producer_name": onnx_model.producer_name,
        "producer_version": onnx_model.producer_version,
        "inputs": [
            {
                "name": inp.name,
                "shape": [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim],
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            }
            for inp in onnx_model.graph.input
        ],
        "outputs": [
            {
                "name": out.name,
                "shape": [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim],
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
            }
            for out in onnx_model.graph.output
        ],
        "num_nodes": len(onnx_model.graph.node),
        "file_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
    }

    return info


# ============================================================================
# Example usage and main script
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Keras model to ONNX format"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the Keras model (.keras format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: same name with .onnx extension)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip output validation"
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch size"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        args.output = args.model_path.replace(".keras", ".onnx")

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path)

    # Export
    if args.dynamic_batch:
        output_path = export_with_dynamic_batch(
            model=model,
            output_path=args.output,
            opset_version=args.opset,
        )
    else:
        output_path = export_keras_model_to_onnx(
            model=model,
            output_path=args.output,
            opset_version=args.opset,
            validate_output=not args.no_validate,
        )

    # Print model info
    info = get_onnx_model_info(output_path)
    logger.info(f"\nONNX Model Info:")
    logger.info(f"  Opset version: {info['opset_version']}")
    logger.info(f"  Number of nodes: {info['num_nodes']}")
    logger.info(f"  File size: {info['file_size_mb']:.2f} MB")
    logger.info(f"  Inputs: {info['inputs']}")
    logger.info(f"  Outputs: {info['outputs']}")