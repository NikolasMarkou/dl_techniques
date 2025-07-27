"""
Vision Question Answering (VQA) Data Processor for nanoVLM Training.

This module provides comprehensive data processing capabilities for VQA datasets,
including The Cauldron dataset format, with robust preprocessing pipelines
for both images and text.
"""

from typing import Dict, List, Optional, Tuple, Callable, Union, Any
import keras
from keras import ops
import numpy as np
from pathlib import Path
import json
from abc import ABC, abstractmethod

from dl_techniques.utils.logger import logger


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        pass


class SimpleCharTokenizer(BaseTokenizer):
    """Simple character-level tokenizer for demonstration purposes.

    Args:
        vocab_size: Maximum vocabulary size.
        lowercase: Whether to convert text to lowercase.
    """

    def __init__(self, vocab_size: int = 32000, lowercase: bool = True) -> None:
        self.vocab_size = vocab_size
        self.lowercase = lowercase

    def encode(self, text: str) -> List[int]:
        """Encode text using character-level tokenization.

        Args:
            text: Input text to encode.

        Returns:
            List of character code token IDs.
        """
        if self.lowercase:
            text = text.lower()

        # Convert characters to codes, clamping to vocab size
        tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """
        # Convert codes back to characters
        chars = [chr(min(token_id, 127)) for token_id in token_ids]  # ASCII range
        return ''.join(chars)


class VQADataProcessor:
    """Data processor for Vision Question Answering datasets.

    Handles preprocessing of images and text for nanoVLM training,
    including The Cauldron dataset format with robust error handling
    and configurable preprocessing pipelines.

    Args:
        image_size: Target image size for preprocessing.
        max_text_length: Maximum text sequence length.
        vocab_size: Vocabulary size for tokenization.
        pad_token_id: Padding token ID.
        bos_token_id: Beginning of sequence token ID.
        eos_token_id: End of sequence token ID.
        image_normalization: Normalization method for images.
            Options: 'siglip' ([-1, 1]), 'imagenet' ([0, 1]), 'standard' (mean/std).
        tokenizer: Custom tokenizer instance. If None, uses SimpleCharTokenizer.

    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
            self,
            image_size: int = 224,
            max_text_length: int = 512,
            vocab_size: int = 32000,
            pad_token_id: int = 0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            image_normalization: str = 'siglip',
            tokenizer: Optional[BaseTokenizer] = None
    ) -> None:
        # Validate parameters
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if max_text_length <= 0:
            raise ValueError(f"max_text_length must be positive, got {max_text_length}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if image_normalization not in ['siglip', 'imagenet', 'standard']:
            raise ValueError(f"Invalid image_normalization: {image_normalization}")

        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_normalization = image_normalization

        # Initialize tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else SimpleCharTokenizer(vocab_size)

        # ImageNet normalization constants
        self._imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger.info(
            f"Initialized VQA processor - image_size: {image_size}, "
            f"max_text_length: {max_text_length}, "
            f"vocab_size: {vocab_size}, "
            f"normalization: {image_normalization}"
        )

    def _load_image_array(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image as numpy array using Keras utilities.

        Args:
            image_path: Path to image file.

        Returns:
            Image array with shape (height, width, channels).

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be loaded.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Use Keras image utilities
            img = keras.utils.load_img(
                str(image_path),
                target_size=(self.image_size, self.image_size),
                interpolation='bilinear'
            )
            img_array = keras.utils.img_to_array(img)
            return img_array.astype(np.float32)

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

    def _normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image array based on specified normalization method.

        Args:
            image_array: Input image array with values in [0, 255].

        Returns:
            Normalized image array.
        """
        if self.image_normalization == 'siglip':
            # SigLIP normalization: [0, 255] -> [-1, 1]
            return (image_array / 127.5) - 1.0

        elif self.image_normalization == 'imagenet':
            # ImageNet normalization: [0, 255] -> [0, 1]
            return image_array / 255.0

        elif self.image_normalization == 'standard':
            # Standard ImageNet preprocessing with mean/std normalization
            image_array = image_array / 255.0  # [0, 1]
            # Normalize per channel
            image_array = (image_array - self._imagenet_mean) / self._imagenet_std
            return image_array

        return image_array

    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Preprocess image for nanoVLM input.

        Loads, resizes, and normalizes image according to configuration.

        Args:
            image_path: Path to image file.

        Returns:
            Preprocessed image array with shape (height, width, channels).

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be processed.
        """
        try:
            # Load image
            img_array = self._load_image_array(image_path)

            # Normalize
            img_array = self._normalize_image(img_array)

            logger.debug(f"Preprocessed image {image_path} - shape: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise

    def preprocess_text(
            self,
            question: str,
            answer: Optional[str] = None,
            conversation_format: str = 'simple'
    ) -> Dict[str, np.ndarray]:
        """Preprocess text for training or inference.

        Args:
            question: Question text.
            answer: Answer text (None for inference).
            conversation_format: Format for conversation.
                Options: 'simple', 'chat', 'instruct'.

        Returns:
            Dictionary containing:
                - 'input_ids': Token IDs for input sequence
                - 'attention_mask': Attention mask (1 for real tokens, 0 for padding)
                - 'labels': Target labels for training (same as input_ids with teacher forcing)

        Raises:
            ValueError: If invalid conversation format is provided.
        """
        if conversation_format not in ['simple', 'chat', 'instruct']:
            raise ValueError(f"Invalid conversation_format: {conversation_format}")

        try:
            # Format conversation based on specified format
            formatted_text = self._format_conversation(question, answer, conversation_format)

            # Tokenize
            tokens = self.tokenizer.encode(formatted_text)

            # Add special tokens and truncate
            input_ids = self._prepare_input_sequence(tokens)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (np.array(input_ids) != self.pad_token_id).astype(np.int32)

            # Prepare result dictionary
            result = {
                'input_ids': np.array(input_ids, dtype=np.int32),
                'attention_mask': attention_mask
            }

            # For training, labels are the same as input_ids (teacher forcing)
            if answer is not None:
                result['labels'] = np.array(input_ids, dtype=np.int32)

            logger.debug(f"Preprocessed text - tokens: {len(input_ids)}, format: {conversation_format}")
            return result

        except Exception as e:
            logger.error(f"Failed to preprocess text: {e}")
            raise ValueError(f"Text preprocessing failed: {e}")

    def _format_conversation(
            self,
            question: str,
            answer: Optional[str],
            format_type: str
    ) -> str:
        """Format conversation according to specified format.

        Args:
            question: Question text.
            answer: Answer text (optional).
            format_type: Conversation format type.

        Returns:
            Formatted conversation string.
        """
        if format_type == 'simple':
            if answer is not None:
                return f"Question: {question} Answer: {answer}"
            else:
                return f"Question: {question} Answer:"

        elif format_type == 'chat':
            if answer is not None:
                return f"User: {question}\nAssistant: {answer}"
            else:
                return f"User: {question}\nAssistant:"

        elif format_type == 'instruct':
            if answer is not None:
                return f"### Instruction:\n{question}\n\n### Response:\n{answer}"
            else:
                return f"### Instruction:\n{question}\n\n### Response:\n"

        return question  # Fallback

    def _prepare_input_sequence(self, tokens: List[int]) -> List[int]:
        """Prepare input sequence with special tokens and padding.

        Args:
            tokens: List of token IDs.

        Returns:
            Prepared sequence with special tokens and padding.
        """
        # Add BOS token and truncate to leave room for EOS
        max_content_length = self.max_text_length - 2  # Reserve space for BOS and EOS
        truncated_tokens = tokens[:max_content_length]

        # Add special tokens
        input_ids = [self.bos_token_id] + truncated_tokens + [self.eos_token_id]

        # Pad to max length
        padding_length = self.max_text_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.pad_token_id] * padding_length)

        # Ensure exact length
        return input_ids[:self.max_text_length]


class VQADataSequence(keras.utils.Sequence):
    """Keras Sequence for VQA data loading and batching.

    Args:
        samples: List of data samples.
        processor: VQA data processor instance.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle data between epochs.
        conversation_format: Format for conversation preprocessing.

    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
            self,
            samples: List[Dict[str, Any]],
            processor: VQADataProcessor,
            batch_size: int = 32,
            shuffle: bool = True,
            conversation_format: str = 'simple'
    ) -> None:
        if not samples:
            raise ValueError("samples cannot be empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.samples = samples
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conversation_format = conversation_format

        # Initialize indices
        self.indices = np.arange(len(samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

        logger.info(
            f"Created VQA sequence - samples: {len(samples)}, "
            f"batch_size: {batch_size}, shuffle: {shuffle}"
        )

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get batch at specified index.

        Args:
            idx: Batch index.

        Returns:
            Tuple of (inputs, targets) where:
                - inputs: Dict with 'images' and 'text_tokens'
                - targets: Target labels for text generation
        """
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        # Process batch samples
        batch_images = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for sample_idx in batch_indices:
            try:
                sample = self.samples[sample_idx]

                # Preprocess image
                image = self.processor.preprocess_image(sample['image_path'])

                # Preprocess text
                text_data = self.processor.preprocess_text(
                    sample['question'],
                    sample.get('answer'),
                    self.conversation_format
                )

                batch_images.append(image)
                batch_input_ids.append(text_data['input_ids'])
                batch_attention_masks.append(text_data['attention_mask'])

                # Use labels if available, otherwise use input_ids
                labels = text_data.get('labels', text_data['input_ids'])
                batch_labels.append(labels)

            except Exception as e:
                logger.warning(f"Failed to process sample {sample_idx}: {e}")
                # Skip this sample by duplicating the last valid one
                if batch_images:
                    batch_images.append(batch_images[-1])
                    batch_input_ids.append(batch_input_ids[-1])
                    batch_attention_masks.append(batch_attention_masks[-1])
                    batch_labels.append(batch_labels[-1])

        # Convert to numpy arrays
        try:
            batch_images_array = np.stack(batch_images, axis=0)
            batch_input_ids_array = np.stack(batch_input_ids, axis=0)
            batch_attention_masks_array = np.stack(batch_attention_masks, axis=0)
            batch_labels_array = np.stack(batch_labels, axis=0)

        except Exception as e:
            logger.error(f"Failed to stack batch arrays: {e}")
            raise

        # Prepare inputs and targets
        inputs = {
            'images': batch_images_array,
            'text_tokens': batch_input_ids_array,
            'attention_mask': batch_attention_masks_array
        }

        return inputs, batch_labels_array

    def on_epoch_end(self) -> None:
        """Shuffle indices at the end of each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)
            logger.debug("Shuffled data indices for new epoch")


def create_vqa_dataset(
        data_samples: List[Dict[str, Any]],
        processor: VQADataProcessor,
        batch_size: int = 32,
        shuffle: bool = True,
        conversation_format: str = 'simple'
) -> VQADataSequence:
    """Create VQA dataset from samples.

    Args:
        data_samples: List of data samples with 'image_path', 'question', 'answer'.
        processor: VQA data processor instance.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle data between epochs.
        conversation_format: Format for conversation preprocessing.

    Returns:
        VQA data sequence ready for training.

    Raises:
        ValueError: If invalid parameters are provided.
    """
    return VQADataSequence(
        samples=data_samples,
        processor=processor,
        batch_size=batch_size,
        shuffle=shuffle,
        conversation_format=conversation_format
    )


def load_cauldron_sample() -> List[Dict[str, Any]]:
    """Load sample data in Cauldron format for testing.

    This is a placeholder function that returns mock data.
    In practice, this would load from HuggingFace datasets or local files.

    Returns:
        List of sample data dictionaries with required keys:
            - 'image_path': Path to image file
            - 'question': Question text
            - 'answer': Answer text

    Note:
        Replace this with actual data loading logic for production use.
    """
    sample_data = [
        {
            'image_path': 'path/to/image1.jpg',
            'question': 'What is shown in this image?',
            'answer': 'A cat sitting on a chair.'
        },
        {
            'image_path': 'path/to/image2.jpg',
            'question': 'What color is the car?',
            'answer': 'The car is red.'
        },
        {
            'image_path': 'path/to/image3.jpg',
            'question': 'How many people are in the photo?',
            'answer': 'There are three people in the photo.'
        }
    ]

    logger.info(f"Loaded {len(sample_data)} sample data points")
    return sample_data


def load_cauldron_from_json(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load Cauldron dataset from JSON file.

    Args:
        json_path: Path to JSON file containing Cauldron dataset.

    Returns:
        List of data samples.

    Raises:
        FileNotFoundError: If JSON file doesn't exist.
        ValueError: If JSON format is invalid.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of samples")

        # Validate required keys
        required_keys = {'image_path', 'question', 'answer'}
        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {i} must be a dictionary")
            missing_keys = required_keys - set(sample.keys())
            if missing_keys:
                raise ValueError(f"Sample {i} missing required keys: {missing_keys}")

        logger.info(f"Loaded {len(data)} samples from {json_path}")
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_path}: {e}")
        raise