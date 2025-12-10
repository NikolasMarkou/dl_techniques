"""
Universal Dataset Loader
========================

A generic utility to stream data from Hugging Face datasets (Text, Images, Audio)
without requiring the 'transformers' library.

This module provides a flexible interface for loading and preprocessing datasets
from the Hugging Face Hub with seamless integration into TensorFlow data pipelines.

Example Usage
-------------
.. code-block:: python

    from universal_dataset_loader import UniversalDatasetLoader
    import tensorflow as tf

    # Create loader for CIFAR-10
    loader = UniversalDatasetLoader(
        path="cifar10",
        split="train",
        streaming=True
    )

    # Define output signature
    signature = {
        "image": tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        "label": tf.TensorSpec(shape=(), dtype=tf.int32)
    }

    # Get TensorFlow dataset
    dataset = loader.to_tf_dataset(
        output_signature=signature,
        batch_size=32,
        transform_fn=my_transform
    )
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import datasets
import tensorflow as tf
from dl_techniques.utils.logger import logger

class UniversalDatasetLoader:
    """
    A generic utility to stream data from Hugging Face datasets.

    This class provides a flexible interface for loading datasets from the
    Hugging Face Hub with support for text, images, and audio data. It supports
    both streaming mode (memory-efficient) and download mode.

    Parameters
    ----------
    path : str
        The dataset identifier on Hugging Face Hub (e.g., 'wikipedia', 'cifar10').
    name : Optional[str], default=None
        The configuration name for datasets with multiple configurations
        (e.g., '20220301.en' for Wikipedia).
    split : str, default="train"
        The dataset split to load. Common values: 'train', 'test', 'validation'.
    streaming : bool, default=True
        If True, streams data without downloading the full dataset.
        If False, downloads the dataset to disk first.
    buffer_size : int, default=1000
        Shuffle buffer size for streaming mode. Larger values provide
        better randomization but use more memory.
    trust_remote_code : bool, default=True
        Whether to trust and execute remote code from the dataset repository.

    Attributes
    ----------
    path : str
        The dataset identifier.
    name : Optional[str]
        The configuration name.
    split : str
        The dataset split.
    streaming : bool
        Whether streaming mode is enabled.
    buffer_size : int
        The shuffle buffer size.
    dataset : datasets.Dataset or datasets.IterableDataset
        The loaded Hugging Face dataset object.

    Examples
    --------
    >>> loader = UniversalDatasetLoader(
    ...     path="imdb",
    ...     split="train",
    ...     streaming=True
    ... )
    >>> for item in loader.get_generator(columns=["text", "label"]):
    ...     print(item.keys())
    ...     break
    dict_keys(['text', 'label'])

    Notes
    -----
    - Streaming mode is recommended for large datasets to avoid memory issues.
    - The shuffle buffer only affects streaming mode; non-streaming datasets
      should be shuffled via the TensorFlow dataset pipeline.
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        streaming: bool = True,
        buffer_size: int = 1000,
        trust_remote_code: bool = True,
    ) -> None:
        """Initialize the UniversalDatasetLoader."""
        self.path = path
        self.name = name
        self.split = split
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.trust_remote_code = trust_remote_code

        self._dataset: Optional[
            Union[datasets.Dataset, datasets.IterableDataset]
        ] = None

        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        Load the dataset from Hugging Face Hub.

        This method initializes the dataset connection and applies
        shuffling for streaming datasets.

        Raises
        ------
        datasets.DatasetNotFoundError
            If the specified dataset is not found on Hugging Face Hub.
        ValueError
            If the specified split does not exist in the dataset.
        """
        config_info = f" ({self.name})" if self.name else " (default config)"
        logger.info(
            f"Connecting to Hugging Face: {self.path}{config_info}, "
            f"split={self.split}, streaming={self.streaming}"
        )

        try:
            self._dataset = datasets.load_dataset(
                self.path,
                self.name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
            )

            if self.streaming and self._dataset is not None:
                self._dataset = self._dataset.shuffle(buffer_size=self.buffer_size)

            logger.info(f"Successfully connected to dataset: {self.path}")

        except Exception as e:
            logger.error(f"Failed to load dataset {self.path}: {e}")
            raise

    @property
    def dataset(
        self,
    ) -> Union[datasets.Dataset, datasets.IterableDataset]:
        """
        Get the underlying Hugging Face dataset object.

        Returns
        -------
        Union[datasets.Dataset, datasets.IterableDataset]
            The loaded dataset object.

        Raises
        ------
        RuntimeError
            If the dataset has not been loaded.
        """
        if self._dataset is None:
            raise RuntimeError(
                "Dataset not loaded. Call _load_dataset() first."
            )
        return self._dataset

    def get_generator(
        self,
        transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        columns: Optional[List[str]] = None,
        skip_errors: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Create a Python generator yielding processed data items.

        This method provides a flexible way to iterate over the dataset with
        optional filtering, column selection, and transformation.

        Parameters
        ----------
        transform_fn : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], default=None
            A function to process each raw item. Receives a dictionary and
            should return a processed dictionary. Common uses include:
            - Resizing images
            - Tokenizing text
            - Normalizing features
        filter_fn : Optional[Callable[[Dict[str, Any]], bool]], default=None
            A function that receives an item and returns True to keep it,
            False to skip it. Applied before transformation.
        columns : Optional[List[str]], default=None
            List of column names to keep (e.g., ['image', 'label']).
            If None, all columns are yielded.
        skip_errors : bool, default=True
            If True, items that fail transformation are skipped with a warning.
            If False, transformation errors are raised.

        Yields
        ------
        Dict[str, Any]
            A dictionary containing the processed data item.

        Raises
        ------
        Exception
            If skip_errors is False and a transformation fails.

        Examples
        --------
        >>> def normalize_image(item):
        ...     item["image"] = item["image"] / 255.0
        ...     return item
        >>> gen = loader.get_generator(
        ...     transform_fn=normalize_image,
        ...     columns=["image", "label"]
        ... )
        >>> batch = [next(gen) for _ in range(10)]

        Notes
        -----
        The processing order is:
        1. Filter (skip unwanted items)
        2. Column selection (reduce data)
        3. Transform (process remaining data)
        """
        iterator = iter(self.dataset)

        for item in iterator:
            # Step 1: Apply filtering
            if filter_fn is not None:
                try:
                    if not filter_fn(item):
                        continue
                except Exception as e:
                    logger.warning(f"Filter function error: {e}")
                    if not skip_errors:
                        raise
                    continue

            # Step 2: Select columns
            if columns is not None:
                item = {
                    key: item[key]
                    for key in columns
                    if key in item
                }

            # Step 3: Apply transformation
            if transform_fn is not None:
                try:
                    item = transform_fn(item)
                except Exception as e:
                    logger.warning(f"Transform function error: {e}")
                    if not skip_errors:
                        raise
                    continue

            yield item

    def to_tf_dataset(
        self,
        output_signature: Dict[str, tf.TensorSpec],
        batch_size: int = 32,
        transform_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        columns: Optional[List[str]] = None,
        skip_errors: bool = True,
        drop_remainder: bool = True,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        enable_auto_sharding: bool = True,
    ) -> tf.data.Dataset:
        """
        Convert the Hugging Face dataset to an optimized TensorFlow Dataset.

        This method creates a TensorFlow data pipeline from the Hugging Face
        dataset with batching, prefetching, and optional auto-sharding for
        distributed training.

        Parameters
        ----------
        output_signature : Dict[str, tf.TensorSpec]
            A dictionary mapping output keys to their TensorSpec definitions.
            This is required for TensorFlow graph mode compatibility.
        batch_size : int, default=32
            The number of samples per batch.
        transform_fn : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], default=None
            A function to process each raw item before batching.
        filter_fn : Optional[Callable[[Dict[str, Any]], bool]], default=None
            A function to filter items before processing.
        columns : Optional[List[str]], default=None
            List of column names to keep from the source dataset.
        skip_errors : bool, default=True
            If True, skip items that fail transformation.
        drop_remainder : bool, default=True
            If True, drop the last incomplete batch.
        prefetch_buffer : int, default=tf.data.AUTOTUNE
            Number of batches to prefetch. Use tf.data.AUTOTUNE for
            automatic tuning.
        enable_auto_sharding : bool, default=True
            If True, enable automatic sharding for distributed training.

        Returns
        -------
        tf.data.Dataset
            An optimized TensorFlow Dataset ready for training.

        Examples
        --------
        >>> import numpy as np
        >>> def preprocess(item):
        ...     return {
        ...         "image": np.array(item["image"], dtype=np.float32) / 255.0,
        ...         "label": np.int32(item["label"])
        ...     }
        >>> signature = {
        ...     "image": tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        ...     "label": tf.TensorSpec(shape=(), dtype=tf.int32)
        ... }
        >>> dataset = loader.to_tf_dataset(
        ...     output_signature=signature,
        ...     batch_size=64,
        ...     transform_fn=preprocess
        ... )
        >>> for batch in dataset.take(1):
        ...     print(batch["image"].shape)
        (64, 32, 32, 3)

        Notes
        -----
        - The output_signature must exactly match the structure and types
          returned by transform_fn.
        - Auto-sharding distributes data across workers in distributed training.
        - Prefetching overlaps data loading with model computation.

        See Also
        --------
        get_generator : Lower-level generator interface.
        """
        # Create generator factory (lambda required for tf.data.Dataset)
        def generator_factory() -> Generator[Dict[str, Any], None, None]:
            return self.get_generator(
                transform_fn=transform_fn,
                filter_fn=filter_fn,
                columns=columns,
                skip_errors=skip_errors,
            )

        # Build TensorFlow Dataset from generator
        tf_dataset = tf.data.Dataset.from_generator(
            generator_factory,
            output_signature=output_signature,
        )

        # Configure data options
        options = tf.data.Options()
        if enable_auto_sharding:
            options.experimental_distribute.auto_shard_policy = (
                tf.data.experimental.AutoShardPolicy.DATA
            )
        tf_dataset = tf_dataset.with_options(options)

        # Apply batching
        tf_dataset = tf_dataset.batch(
            batch_size,
            drop_remainder=drop_remainder,
        )

        # Enable prefetching for performance
        tf_dataset = tf_dataset.prefetch(prefetch_buffer)

        logger.info(
            f"Created TensorFlow dataset: batch_size={batch_size}, "
            f"prefetch={prefetch_buffer}"
        )

        return tf_dataset

    def to_tf_dataset_tuple(
        self,
        output_signature: tuple[tf.TensorSpec, ...],
        batch_size: int = 32,
        transform_fn: Optional[Callable[[Dict[str, Any]], tuple[Any, ...]]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        columns: Optional[List[str]] = None,
        skip_errors: bool = True,
        drop_remainder: bool = True,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        enable_auto_sharding: bool = True,
    ) -> tf.data.Dataset:
        """
        Convert the dataset to a TensorFlow Dataset yielding tuples.

        This variant is useful for Keras model.fit() which expects
        (inputs, targets) or (inputs, targets, sample_weights) tuples.

        Parameters
        ----------
        output_signature : tuple[tf.TensorSpec, ...]
            A tuple of TensorSpec definitions for the output elements.
            Typically (input_spec, target_spec) for supervised learning.
        batch_size : int, default=32
            The number of samples per batch.
        transform_fn : Optional[Callable[[Dict[str, Any]], tuple[Any, ...]]], default=None
            A function that processes each item and returns a tuple.
            Must match the structure of output_signature.
        filter_fn : Optional[Callable[[Dict[str, Any]], bool]], default=None
            A function to filter items before processing.
        columns : Optional[List[str]], default=None
            List of column names to keep from the source dataset.
        skip_errors : bool, default=True
            If True, skip items that fail transformation.
        drop_remainder : bool, default=True
            If True, drop the last incomplete batch.
        prefetch_buffer : int, default=tf.data.AUTOTUNE
            Number of batches to prefetch.
        enable_auto_sharding : bool, default=True
            If True, enable automatic sharding for distributed training.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow Dataset yielding tuples, compatible with model.fit().

        Examples
        --------
        >>> def to_tuple(item):
        ...     image = np.array(item["image"], dtype=np.float32) / 255.0
        ...     label = np.int32(item["label"])
        ...     return (image, label)
        >>> signature = (
        ...     tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
        ...     tf.TensorSpec(shape=(), dtype=tf.int32)
        ... )
        >>> dataset = loader.to_tf_dataset_tuple(
        ...     output_signature=signature,
        ...     batch_size=64,
        ...     transform_fn=to_tuple
        ... )
        >>> model.fit(dataset, epochs=10)
        """

        def tuple_generator() -> Generator[tuple[Any, ...], None, None]:
            """Generate tuples from the underlying generator."""
            iterator = iter(self.dataset)

            for item in iterator:
                # Apply filtering
                if filter_fn is not None:
                    try:
                        if not filter_fn(item):
                            continue
                    except Exception as e:
                        logger.warning(f"Filter function error: {e}")
                        if not skip_errors:
                            raise
                        continue

                # Select columns
                if columns is not None:
                    item = {
                        key: item[key]
                        for key in columns
                        if key in item
                    }

                # Apply transformation
                if transform_fn is not None:
                    try:
                        result = transform_fn(item)
                        yield result
                    except Exception as e:
                        logger.warning(f"Transform function error: {e}")
                        if not skip_errors:
                            raise
                        continue
                else:
                    # Default: yield values as tuple
                    yield tuple(item.values())

        # Build TensorFlow Dataset
        tf_dataset = tf.data.Dataset.from_generator(
            tuple_generator,
            output_signature=output_signature,
        )

        # Configure options
        options = tf.data.Options()
        if enable_auto_sharding:
            options.experimental_distribute.auto_shard_policy = (
                tf.data.experimental.AutoShardPolicy.DATA
            )
        tf_dataset = tf_dataset.with_options(options)

        # Batch and prefetch
        tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_remainder)
        tf_dataset = tf_dataset.prefetch(prefetch_buffer)

        return tf_dataset

    def __repr__(self) -> str:
        """Return a string representation of the loader."""
        return (
            f"{self.__class__.__name__}("
            f"path='{self.path}', "
            f"name={self.name!r}, "
            f"split='{self.split}', "
            f"streaming={self.streaming})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        config = self.name if self.name else "default"
        mode = "streaming" if self.streaming else "downloaded"
        return f"DatasetLoader({self.path}/{config}, {self.split}, {mode})"