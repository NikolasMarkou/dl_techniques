"""
A self-supervised loss from the DINO framework.

This loss function is the core objective of the DINO (self-DIstillation
with NO labels) framework. It enables a model to learn rich visual
representations from images alone, without any human-provided labels.
The method is based on a student-teacher knowledge distillation paradigm,
where the student network is trained to match the output of a momentum
teacher network when shown different augmented views of the same image.

Conceptual Overview:
    The fundamental idea is to enforce consistency in the representations of
    different distorted versions ("views") of an image. The student network
    processes a set of views (including small local crops and large global
    crops), while the teacher network only processes the global crops. The
    training objective is to make the student's output distribution for any
    view match the teacher's output distribution for the global views. This
    forces the student to learn features that are invariant to these
    augmentations, capturing high-level semantic content.

Architectural Design & Collapse Prevention:
    A naive implementation would be prone to "collapse," where both networks
    learn a trivial solution, such as outputting a constant vector for all
    inputs. DINO employs two key mechanisms to prevent this:
    1.  Momentum Teacher: The teacher's weights are not updated by back-
        propagation. Instead, they are an exponential moving average (EMA) of
        the student's weights. This provides more stable and slowly evolving
        targets for the student to learn from.
    2.  Centering: The teacher's outputs are centered by subtracting a
        running average of all batch outputs. This normalization prevents any
        single dimension from dominating the output and encourages the model
        to produce features that are uniformly distributed, effectively
        avoiding collapse.

Mathematical Formulation:
    The loss is a cross-entropy calculated between the probability
    distributions produced by the student and teacher networks. Let `z_s` and
    `z_t` be the output logits from the student and teacher, respectively.

    First, the logits are converted to probabilities using a softmax function
    with different temperature parameters (`τ_s` for student, `τ_t` for
    teacher). The teacher's output is also centered using a momentum-updated
    center vector `C`.

        p_s = softmax(z_s / τ_s)
        p_t = softmax((z_t - C) / τ_t)

    A low teacher temperature `τ_t` sharpens its output distribution, creating
    confident targets for the student to match. The loss is then the
    cross-entropy between these two distributions:

        Loss = - Σ p_t * log(p_s)

    The center `C` is updated via an EMA of the teacher's outputs over many
    batches.

References:
    -   Caron, M., et al. (2021). "Emerging Properties in Self-Supervised
        Vision Transformers." https://arxiv.org/abs/2104.14294
"""

import keras
from keras import ops
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOLoss(keras.losses.Loss):
    """
    DINO consistency loss for self-supervised learning with momentum-based center.

    This loss enforces consistency between student and teacher networks' CLS token
    outputs by matching probability distributions. It uses a momentum-updated center
    vector to prevent feature collapse and applies different temperature scaling
    to student and teacher outputs for effective knowledge distillation.

    **Intent**: Implement the core DINO loss mechanism that enables effective
    self-supervised learning by enforcing cross-view consistency while preventing
    trivial solutions through statistical centering.

    **Mathematical Operations**:
    1. **Teacher Processing**: logits_t = teacher_logits - center
    2. **Temperature Scaling**: p_t = softmax(logits_t / τ_teacher)
    3. **Student Processing**: p_s = log_softmax(student_logits / τ_student)
    4. **Cross-Entropy**: L = -Σ p_t * p_s
    5. **Center Update**: center ← α * center + (1-α) * mean(teacher_logits)

    **Architecture**:
    ```
    Teacher Logits → Center → Temp Scale → Softmax → Target Dist
                                                          ↓
    Student Logits → Temp Scale → LogSoftmax → CrossEntropy Loss
    ```

    Args:
        out_dim: Dimensionality of the model's output embeddings/logits.
        student_temp: Temperature for sharpening student's output distribution.
            Lower values create sharper distributions. Defaults to 0.1.
        teacher_temp: Temperature for sharpening teacher's output distribution.
            Should be lower than student_temp. Defaults to 0.04.
        center_momentum: Momentum coefficient for EMA center updates.
            Higher values create more stable centers. Defaults to 0.9.

    Input shapes:
        y_true: Teacher's output logits with shape `(batch_size, out_dim)`.
        y_pred: Student's output logits with shape `(batch_size, out_dim)`.

    Output shape:
        Scalar loss tensor.

    Attributes:
        center: Non-trainable momentum-updated center vector for teacher logits.

    Example:
        ```python
        # Initialize for vision_heads transformer with 65k dimensional output
        dino_loss = DINOLoss(out_dim=65536, student_temp=0.1, teacher_temp=0.04)

        # Compute loss in training loop
        teacher_cls = teacher_model(global_crops)  # Shape: (batch, 65536)
        student_cls = student_model(all_crops)     # Shape: (batch, 65536)

        loss = dino_loss(teacher_cls, student_cls)

        # CRITICAL: Update center after loss computation
        dino_loss.update_center(teacher_cls)
        ```

    Note:
        The `update_center()` method must be called once per training step,
        typically in the model's `train_step()` method, to maintain the
        momentum-updated center that prevents feature collapse.
    """

    def __init__(
            self,
            out_dim: int,
            student_temp: float = 0.1,
            teacher_temp: float = 0.04,
            center_momentum: float = 0.9,
            name: str = 'dino_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize DINO loss with specified parameters.

        Args:
            out_dim: Output dimensionality for center vector initialization.
            student_temp: Temperature for student distribution sharpening.
            teacher_temp: Temperature for teacher distribution sharpening.
            center_momentum: EMA momentum for center updates.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If out_dim <= 0, temperatures <= 0, or center_momentum
                       not in [0, 1).
        """
        super().__init__(name=name, **kwargs)

        # Validate input parameters
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if student_temp <= 0:
            raise ValueError(f"student_temp must be positive, got {student_temp}")
        if teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be positive, got {teacher_temp}")
        if not (0 <= center_momentum < 1):
            raise ValueError(f"center_momentum must be in [0, 1), got {center_momentum}")

        # Store configuration
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Create momentum-updated center as non-trainable weight
        self.center = self.add_weight(
            name='center',
            shape=(1, out_dim),
            initializer='zeros',
            trainable=False,
            dtype=self.dtype
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute DINO loss between teacher and student outputs.

        Args:
            y_true: Teacher's output logits with shape (batch_size, out_dim).
            y_pred: Student's output logits with shape (batch_size, out_dim).

        Returns:
            Scalar tensor representing the computed DINO loss.

        Note:
            This method computes the loss but does NOT update the center.
            Call `update_center()` separately after loss computation.
        """
        # Process teacher output: center and sharpen
        teacher_logits = y_true - self.center
        teacher_probs = ops.softmax(teacher_logits / self.teacher_temp, axis=-1)

        # Process student output: sharpen to log probabilities
        student_log_probs = ops.log_softmax(y_pred / self.student_temp, axis=-1)

        # Compute cross-entropy loss: -sum(p_teacher * log_p_student)
        loss = -ops.sum(teacher_probs * student_log_probs, axis=-1)

        return ops.mean(loss)

    def update_center(self, teacher_logits: keras.KerasTensor) -> None:
        """
        Update center vector using exponential moving average of teacher logits.

        This method must be called once per training step to maintain the
        momentum-updated center that prevents feature collapse. Typically
        called from the model's `train_step()` method.

        Args:
            teacher_logits: Raw output logits from teacher network with shape
                           (batch_size, out_dim).

        Note:
            In distributed training, the batch center is automatically
            averaged across all replicas before the EMA update.
        """
        # Compute mean of current batch
        batch_center = ops.mean(teacher_logits, axis=0, keepdims=True)

        # Handle distributed training
        if keras.distribution.distribution().num_replicas_in_sync > 1:
            # Average across all replicas
            batch_center = keras.distribution.distribution().reduce(
                'mean', batch_center, axis=None
            )

        # EMA update: center ← α * center + (1-α) * batch_center
        self.center.assign(
            self.center * self.center_momentum +
            batch_center * (1.0 - self.center_momentum)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'out_dim': self.out_dim,
            'student_temp': self.student_temp,
            'teacher_temp': self.teacher_temp,
            'center_momentum': self.center_momentum,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class iBOTPatchLoss(keras.losses.Loss):
    """
    iBOT masked patch prediction loss for self-supervised learning.

    This loss extends DINO's consistency objective to patch-level predictions
    using masked image modeling. It matches student predictions for masked
    patches against teacher outputs for the same patches (unmasked), enabling
    learning of local image features through reconstruction tasks.

    **Intent**: Enable patch-level self-supervised learning by predicting
    masked patch representations, combining global consistency (DINO) with
    local reconstruction objectives for comprehensive visual understanding.

    **Mathematical Operations**:
    1. **Mask Selection**: Select only logits for masked patches
    2. **Teacher Processing**: p_t = softmax((logits_t - center) / τ_teacher)
    3. **Student Processing**: p_s = log_softmax(logits_s / τ_student)
    4. **Cross-Entropy**: L = -Σ p_t * p_s (only for masked patches)
    5. **Center Update**: center ← α * center + (1-α) * mean(all_teacher_patches)

    **Architecture**:
    ```
    Teacher Patches → Center → Temp Scale → Softmax → Target Dist
                                                          ↓
    Student Patches → Mask Select → Temp Scale → LogSoftmax → Loss
           ↑
    Boolean Mask (True for masked patches)
    ```

    Args:
        out_dim: Dimensionality of patch token embeddings/logits.
        student_temp: Temperature for student distribution sharpening.
        teacher_temp: Temperature for teacher distribution sharpening.
        center_momentum: EMA momentum for center updates.

    Input shapes:
        y_true: Teacher patch logits with shape `(batch_size, num_patches, out_dim)`.
        y_pred: Student patch logits with shape `(batch_size, num_patches, out_dim)`.
        mask: Boolean mask with shape `(batch_size, num_patches)`. True for masked.

    Output shape:
        Scalar loss tensor.

    Attributes:
        center: Non-trainable center vector for patch token normalization.

    Example:
        ```python
        # Initialize for vision_heads transformer patches
        ibot_loss = iBOTPatchLoss(out_dim=65536, student_temp=0.1)

        # Compute loss with masking
        teacher_patches = teacher_model.patch_tokens(global_crops)  # (B, 196, 65536)
        student_patches = student_model.patch_tokens(masked_crops)  # (B, 196, 65536)
        mask = create_random_mask(batch_size=B, num_patches=196)    # (B, 196)

        loss = ibot_loss(teacher_patches, student_patches, mask)

        # Update center with all teacher patches
        ibot_loss.update_center(teacher_patches)
        ```
    """

    def __init__(
            self,
            out_dim: int,
            student_temp: float = 0.1,
            teacher_temp: float = 0.04,
            center_momentum: float = 0.9,
            name: str = 'ibot_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize iBOT patch loss with specified parameters.

        Args:
            out_dim: Patch token embedding dimensionality.
            student_temp: Temperature for student sharpening.
            teacher_temp: Temperature for teacher sharpening.
            center_momentum: EMA momentum for center updates.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If parameters are outside valid ranges.
        """
        super().__init__(name=name, **kwargs)

        # Validate parameters
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if student_temp <= 0:
            raise ValueError(f"student_temp must be positive, got {student_temp}")
        if teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be positive, got {teacher_temp}")
        if not (0 <= center_momentum < 1):
            raise ValueError(f"center_momentum must be in [0, 1), got {center_momentum}")

        # Store configuration
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Create center for patch tokens (shape for broadcasting over patches)
        self.center = self.add_weight(
            name='center',
            shape=(1, 1, out_dim),
            initializer='zeros',
            trainable=False,
            dtype=self.dtype
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute iBOT loss on masked patches only.

        Args:
            y_true: Teacher patch logits (B, num_patches, out_dim).
            y_pred: Student patch logits (B, num_patches, out_dim).
            mask: Boolean mask (B, num_patches). True for masked patches.

        Returns:
            Scalar loss tensor normalized by number of masked patches.
        """
        # Select only masked patch tokens
        student_masked_logits = ops.boolean_mask(y_pred, mask)
        teacher_masked_logits = ops.boolean_mask(y_true, mask)

        # If no patches are masked, return zero loss
        num_masked = ops.shape(student_masked_logits)[0]
        if num_masked == 0:
            return ops.convert_to_tensor(0.0, dtype=self.dtype)

        # Process teacher output: center and sharpen
        teacher_logits = teacher_masked_logits - ops.squeeze(self.center, axis=[0, 1])
        teacher_probs = ops.softmax(teacher_logits / self.teacher_temp, axis=-1)

        # Process student output: sharpen to log probabilities
        student_log_probs = ops.log_softmax(student_masked_logits / self.student_temp, axis=-1)

        # Compute cross-entropy loss
        loss = -ops.sum(teacher_probs * student_log_probs, axis=-1)

        # Return mean loss over masked patches
        return ops.mean(loss)

    def update_center(self, teacher_patch_logits: keras.KerasTensor) -> None:
        """
        Update center using all teacher patch tokens.

        Args:
            teacher_patch_logits: All patch logits from teacher with shape
                                 (batch_size, num_patches, out_dim).
        """
        # Compute mean over batch and patch dimensions
        batch_center = ops.mean(teacher_patch_logits, axis=[0, 1], keepdims=True)

        # Handle distributed training
        if keras.distribution.distribution().num_replicas_in_sync > 1:
            batch_center = keras.distribution.distribution().reduce(
                'mean', batch_center, axis=None
            )

        # EMA update
        self.center.assign(
            self.center * self.center_momentum +
            batch_center * (1.0 - self.center_momentum)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'out_dim': self.out_dim,
            'student_temp': self.student_temp,
            'teacher_temp': self.teacher_temp,
            'center_momentum': self.center_momentum,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KoLeoLoss(keras.losses.Loss):
    """
    Kozachenko-Leonenko entropic regularizer for uniform distribution on unit sphere.

    This regularization loss prevents feature collapse by encouraging embeddings
    to be uniformly distributed on the unit hypersphere. It maximizes the distance
    to nearest neighbors, promoting diverse feature representations and preventing
    the network from converging to trivial solutions.

    **Intent**: Provide unsupervised regularization that maintains embedding
    diversity without requiring labels or additional supervision, essential
    for self-supervised learning to avoid representational collapse.

    **Mathematical Operations**:
    1. **Normalize**: x̂ = x / ||x||₂ (project to unit sphere)
    2. **Similarity**: S = x̂ᵀx̂ (cosine similarity matrix)
    3. **Nearest Neighbor**: s_nn = max(S - I) (largest off-diagonal)
    4. **Distance**: d = √(2 - 2s_nn) (L2 distance from similarity)
    5. **Loss**: L = -log(d + ε) (maximize log distance)

    **Architecture**:
    ```
    Input Embeddings → L2 Normalize → Unit Sphere
                                         ↓
    Similarity Matrix ← Cosine Similarity Computation
                                         ↓
    Nearest Neighbors ← Max Off-diagonal Selection
                                         ↓
    L2 Distances ← Distance Conversion
                                         ↓
    -Log(Distance) ← Final Loss Computation
    ```

    Args:
        epsilon: Small value for numerical stability in log computation.
                Higher values provide more stability but may affect gradients.

    Input shapes:
        y_true: Ignored (unsupervised loss).
        y_pred: Embeddings with shape `(batch_size, embedding_dim)`.

    Output shape:
        Scalar loss tensor.

    Example:
        ```python
        # Initialize with default stability
        koleo_loss = KoLeoLoss(epsilon=1e-8)

        # Apply to CLS token embeddings
        student_cls = student_model.cls_token(inputs)  # Shape: (batch, 768)

        # y_true is ignored for this unsupervised loss
        reg_loss = koleo_loss(None, student_cls)

        # Combine with main loss
        total_loss = main_loss + 0.1 * reg_loss
        ```

    Note:
        This is an unsupervised regularizer that ignores y_true. It can be
        applied to any embedding layer to encourage diversity. The loss
        magnitude depends on embedding dimensionality and batch size.
    """

    def __init__(
            self,
            epsilon: float = 1e-8,
            name: str = 'koleo_loss',
            **kwargs: Any
    ) -> None:
        """
        Initialize KoLeo loss with numerical stability parameter.

        Args:
            epsilon: Small value added to distances before log for stability.
            name: Name for this loss instance.
            **kwargs: Additional arguments for Loss base class.

        Raises:
            ValueError: If epsilon <= 0.
        """
        super().__init__(name=name, **kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon

    def call(
            self,
            y_true: Optional[keras.KerasTensor],
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute KoLeo regularization loss.

        Args:
            y_true: Ignored (this is an unsupervised loss).
            y_pred: Embeddings to regularize with shape (batch_size, dim).

        Returns:
            Scalar regularization loss encouraging uniform distribution.
        """
        # L2 normalize embeddings to unit sphere
        features = ops.normalize(y_pred, axis=-1)
        batch_size = ops.shape(features)[0]

        # Compute pairwise cosine similarity matrix
        similarity_matrix = ops.matmul(features, ops.transpose(features))

        # Mask diagonal to exclude self-similarity
        # Set diagonal to large negative value to ignore in max reduction
        eye = ops.eye(batch_size, dtype=similarity_matrix.dtype)
        masked_similarity = similarity_matrix - 2.0 * eye

        # Find nearest neighbor similarity for each embedding
        nearest_neighbor_sim = ops.max(masked_similarity, axis=1)

        # Convert cosine similarity to L2 distance on unit sphere
        # For unit vectors: ||a - b||² = 2 - 2(a·b)
        # Clamp similarity to valid range to avoid numerical issues
        clamped_sim = ops.clip(nearest_neighbor_sim, -1.0, 1.0)
        distances_squared = 2.0 - 2.0 * clamped_sim
        distances = ops.sqrt(ops.maximum(distances_squared, 0.0))

        # Compute loss: maximize log distance = minimize -log(distance)
        loss = -ops.log(distances + self.epsilon)

        return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------

