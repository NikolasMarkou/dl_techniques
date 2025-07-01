# AB-UPT: Anchored Branched Universal Physics Transformer

## Keras Implementation for Computational Fluid Dynamics

This repository contains a complete Keras 3.x implementation of the **Anchored Branched Universal Physics Transformer (AB-UPT)** for computational fluid dynamics (CFD) modeling, converted from the original PyTorch implementation using the advanced spatial layers from `dl_techniques`.

---

## 🌟 Key Features

### 🏗️ **Advanced Architecture**
- **Multi-Modal Processing**: Handles geometry, surface, and volume data simultaneously
- **Hierarchical Attention**: Anchor-query attention patterns for computational efficiency
- **Spatial Reasoning**: Continuous coordinate embeddings and RoPE for 3D spatial understanding
- **Cross-Modal Fusion**: Shared-weight attention mechanisms between different data modalities
- **Physics-Aware Design**: Specialized for fluid dynamics with appropriate inductive biases

### 🔧 **Technical Capabilities**
- **Backend Agnostic**: Full Keras 3.x compatibility (TensorFlow, JAX, PyTorch backends)
- **Production Ready**: Complete serialization support for model deployment
- **Scalable Architecture**: Efficient attention mechanisms for large point clouds
- **Comprehensive Pipeline**: End-to-end training, evaluation, and visualization

### 📊 **Data Processing**
- **Multi-Scale Normalization**: Position and quantity normalization for numerical stability
- **Flexible Sampling**: Configurable anchor/query point sampling strategies
- **Physics-Informed Preprocessing**: Log-scale transformations for vorticity and other quantities
- **Robust Data Generators**: Memory-efficient batch processing with on-the-fly sampling

---

## 🏛️ **Architecture Overview**

The AB-UPT model processes three types of CFD data:

```
🔷 GEOMETRY BRANCH
   Point Cloud → Supernode Pooling → Transformer Blocks → Geometry Features

🔶 SURFACE BRANCH  
   Positions → Continuous Embedding → Anchor/Query Attention → Surface Predictions
   ├── Pressure (1D)
   └── Wall Shear Stress (3D)

🔸 VOLUME BRANCH
   Positions → Continuous Embedding → Anchor/Query Attention → Volume Predictions
   ├── Pressure (1D)
   ├── Velocity (3D)
   └── Vorticity (3D)

💫 CROSS-MODAL FUSION
   Shared attention between surface ↔ volume and perceiver attention to geometry
```

### **Key Components**

1. **SupernodePooling**: Graph-based pooling for geometry encoding
2. **ContinuousSincosEmbed**: Spatial coordinate embeddings
3. **ContinuousRoPE**: Rotary position embeddings for 3D coordinates
4. **AnchorAttention**: Hierarchical attention (anchor tokens + query tokens)
5. **SharedWeightsCrossAttention**: Cross-modal attention between surface/volume
6. **PerceiverBlock**: Cross-attention from surface/volume to geometry

---

## 📁 **Project Structure**

```
📦 AB-UPT Implementation
├── 🏗️ abupt_model.py              # Main model architecture
├── 🎯 abupt_training.py           # Training pipeline & data processing
├── 🔮 abupt_inference.py          # Inference & evaluation utilities
├── 🚀 abupt_complete_example.py   # Complete pipeline example
├── 📊 dl_techniques/              # Spatial layer implementations
│   └── layers/geometric/          # 
│       ├── continuous_sin_cos_embed.py
│       ├── continuous_rope.py
│       ├── anchor_attention.py
│       ├── perceiver_block.py
│       ├── shared_weights_cross_attention.py
│       └── supernode_pooling.py
└── 📋 README.md                   # This documentation
```

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Install dependencies
pip install keras>=3.8.0 tensorflow>=2.18.0 numpy matplotlib scikit-learn

# Or install the full dl_techniques package
pip install dl_techniques
```

### **2. Run Complete Pipeline**

```bash
# Run the complete training and evaluation pipeline
python abupt_complete_example.py --samples 200 --epochs 20 --dim 256

# Options:
# --samples: Number of synthetic samples to generate
# --epochs: Training epochs
# --dim: Model hidden dimension
# --exp-dir: Experiment directory
```

### **3. Custom Training**

```python
from abupt_model import create_abupt_model
from abupt_training import CFDTrainer, DataPreprocessor, NormalizationStats

# Create model
model = create_abupt_model(
    dim=192,
    num_heads=3,
    geometry_depth=1,
    blocks="pscscs",
    num_surface_blocks=4,
    num_volume_blocks=4,
    radius=2.5,
    dropout=0.1
)

# Setup training
stats = NormalizationStats()
preprocessor = DataPreprocessor(stats)
trainer = CFDTrainer(model, preprocessor, learning_rate=1e-4)

# Train
history = trainer.train(train_generator, val_generator, epochs=50)
```

### **4. Inference**

```python
from abupt_inference import CFDInference

# Load trained model
inference_engine = CFDInference("path/to/model.keras", preprocessor)

# Run prediction
predictions = inference_engine.predict(inputs)

# Batch inference with evaluation
predictions_list, targets_list = inference_engine.predict_batch(test_generator)
```

---

## 📊 **Data Format**

The model expects CFD data in the following format:

### **Input Data Structure**
```python
inputs = {
    # Geometry (point cloud)
    "geometry_position": (num_points, 3),           # 3D coordinates
    "geometry_supernode_idx": (num_supernodes,),    # Supernode indices
    "geometry_batch_idx": None,                     # Optional batch indices
    
    # Surface data
    "surface_anchor_position": (batch_size, num_anchors, 3),    # Anchor positions
    "surface_query_position": (batch_size, num_queries, 3),    # Query positions (optional)
    
    # Volume data  
    "volume_anchor_position": (batch_size, num_anchors, 3),     # Anchor positions
    "volume_query_position": (batch_size, num_queries, 3),     # Query positions (optional)
}
```

### **Target Data Structure**
```python
targets = {
    # Surface quantities
    "surface_anchor_pressure": (num_anchors, 1),                    # Pressure
    "surface_anchor_wallshearstress": (num_anchors, 3),            # Wall shear stress
    "surface_query_pressure": (num_queries, 1),                    # Query pressure (optional)
    "surface_query_wallshearstress": (num_queries, 3),            # Query wall shear (optional)
    
    # Volume quantities
    "volume_anchor_totalpcoeff": (num_anchors, 1),                 # Total pressure coefficient
    "volume_anchor_velocity": (num_anchors, 3),                    # Velocity
    "volume_anchor_vorticity": (num_anchors, 3),                   # Vorticity
    "volume_query_totalpcoeff": (num_queries, 1),                  # Query pressure (optional)
    "volume_query_velocity": (num_queries, 3),                     # Query velocity (optional)
    "volume_query_vorticity": (num_queries, 3),                    # Query vorticity (optional)
}
```

---

## ⚙️ **Configuration**

### **Model Parameters**
```python
model_config = {
    "ndim": 3,                    # Coordinate dimensions
    "input_dim": 3,               # Input coordinate dimension
    "output_dim_surface": 4,      # Surface outputs (pressure + wall shear stress)
    "output_dim_volume": 7,       # Volume outputs (pressure + velocity + vorticity)
    "dim": 192,                   # Hidden dimension
    "geometry_depth": 1,          # Geometry transformer depth
    "num_heads": 3,               # Attention heads
    "blocks": "pscscs",           # Shared attention pattern
    "num_volume_blocks": 6,       # Volume-specific blocks
    "num_surface_blocks": 6,      # Surface-specific blocks
    "radius": 0.25,               # Supernode pooling radius
    "dropout": 0.1                # Dropout rate
}
```

### **Training Parameters**
```python
training_config = {
    "learning_rate": 1e-4,
    "epochs": 100,
    "patience": 10,
    "batch_size": 1,
    "loss_weights": {
        "surface_anchor_pressure": 1.0,
        "surface_anchor_wallshearstress": 0.5,
        "volume_anchor_velocity": 0.8,
        "volume_anchor_vorticity": 0.3,
        # ... etc
    }
}
```

### **Data Parameters**
```python
data_config = {
    "num_geometry_points": 1000,
    "num_surface_anchor_points": 500,
    "num_volume_anchor_points": 800,
    "num_geometry_supernodes": 200,
    "use_query_positions": True,
    "normalization_scale": 1000.0
}
```

---

## 🔍 **Key Implementation Details**

### **1. Spatial Embeddings**
- **Continuous Coordinates**: Handles arbitrary 3D coordinate systems
- **Multi-Scale Encoding**: Different wavelengths for fine and coarse features
- **RoPE Integration**: Rotary position embeddings for spatial attention

### **2. Attention Mechanisms**
- **Anchor-Query Pattern**: Reduces O(n²) complexity to O(n·k)
- **Cross-Modal Fusion**: Surface and volume data exchange information
- **Geometry Integration**: Perceiver-style attention to geometry features

### **3. Data Processing**
- **Physics-Informed Normalization**: Separate statistics for each quantity
- **Log-Scale Transformations**: For quantities with exponential distributions
- **Flexible Sampling**: Configurable anchor/query splitting strategies

### **4. Training Stability**
- **Multi-Task Loss Weighting**: Balanced training across all quantities
- **Gradient Clipping**: Prevents training instabilities
- **Learning Rate Scheduling**: Adaptive learning rate reduction

---

## 📈 **Performance & Scalability**

### **Computational Complexity**
- **Supernode Pooling**: O(n·k) where k is max neighbors per supernode
- **Anchor Attention**: O(n·a) where a is number of anchor tokens
- **Cross-Modal Attention**: O(n₁·n₂) between modalities

### **Memory Optimization**
- **Hierarchical Processing**: Reduces memory footprint for large point clouds
- **Efficient Batching**: Handles variable-length sequences
- **Graph Operations**: Approximated with dense operations for Keras compatibility

### **Scaling Guidelines**
```python
# For different problem sizes:
small_config = {"dim": 128, "num_heads": 4, "radius": 1.0}    # <1K points
medium_config = {"dim": 192, "num_heads": 6, "radius": 2.0}   # 1K-10K points  
large_config = {"dim": 256, "num_heads": 8, "radius": 3.0}    # >10K points
```

---

## 🎨 **Visualization & Analysis**

The implementation includes comprehensive visualization tools:

### **Training Monitoring**
- Loss curves and learning rate schedules
- Multi-task loss component tracking
- Validation metric monitoring

### **Result Analysis**
- Surface pressure and wall shear stress visualization
- Volume velocity and vorticity field plots
- Error distribution analysis
- Comparative prediction plots

### **CFD-Specific Visualizations**
- Streamline generation and comparison
- Pressure contour plots
- Velocity magnitude visualizations
- Vorticity field analysis

```python
from abupt_inference import CFDVisualizer

visualizer = CFDVisualizer(preprocessor)

# Surface comparison
visualizer.plot_surface_comparison(
    positions, predictions, targets, quantity="pressure"
)

# Volume slice visualization  
visualizer.plot_volume_slice(
    positions, predictions, targets, quantity="velocity", slice_coord=0.0
)

# Error analysis
visualizer.plot_error_distribution(predictions_list, targets_list)
```

---

## 🔄 **Model Conversion Details**

### **PyTorch → Keras Adaptations**

| **PyTorch Component** | **Keras Equivalent** | **Key Changes** |
|----------------------|----------------------|----------------|
| `SupernodePoolingPosonly` | `SupernodePooling` | Dense tensor approximation of graph ops |
| `RopeFrequency` | `ContinuousRoPE` | Integrated frequency generation |
| `ContinuousSincosEmbed` | `ContinuousSincosEmbed` | Direct port with Keras ops |
| `AnchorAttention` | `AnchorAttention` | Keras-native attention implementation |
| `SharedweightsCrossattnAttention` | `SharedWeightsCrossAttention` | Multi-modal attention patterns |
| `PerceiverBlock` | `PerceiverBlock` | Cross-attention with proper normalization |

### **Key Architectural Differences**
1. **Graph Operations**: Approximated with dense computations for broader compatibility
2. **Batch Processing**: Enhanced for Keras-style batching
3. **Serialization**: Full support for model saving/loading
4. **Memory Management**: Optimized for Keras execution patterns

---

## 🧪 **Testing & Validation**

### **Unit Tests**
```bash
# Test individual components
python -m pytest tests/test_spatial_layers.py
python -m pytest tests/test_model_architecture.py
python -m pytest tests/test_data_pipeline.py
```

### **Integration Tests**
```bash
# Test complete pipeline
python abupt_complete_example.py --samples 50 --epochs 5
```

### **Benchmark Performance**
```python
# Performance comparison with PyTorch version
from benchmarks import compare_implementations
results = compare_implementations(model_keras, model_pytorch, test_data)
```

---

## 🔮 **Future Enhancements**

### **Planned Features**
1. **Multi-GPU Training**: Distributed training support
2. **Mixed Precision**: FP16 training for better performance
3. **Dynamic Batching**: Variable-length sequence batching
4. **Advanced Visualization**: Interactive 3D CFD visualizations
5. **Model Optimization**: Quantization and pruning support

### **Research Directions**
1. **Temporal Dynamics**: Extension to time-dependent CFD
2. **Multi-Physics**: Coupling with other physics simulations
3. **Adaptive Meshing**: Dynamic point cloud refinement
4. **Uncertainty Quantification**: Bayesian neural network integration

---

## 📚 **References & Citations**

### **Original Implementation**
- PyTorch AB-UPT: Based on research implementation for automotive CFD
- DrivAerML Dataset: Automotive aerodynamics simulation data

### **Key Papers**
- "Attention Is All You Need" - Transformer architecture
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- "Perceiver: General Perception with Iterative Attention"
- "Point Transformer" - Attention mechanisms for point clouds
- "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"

### **dl_techniques Library**
```python
# Citation for the spatial layers
@software{dl_techniques,
  title={dl_techniques: Advanced Neural Network Components for Keras},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-repo/dl_techniques}
}
```

---

## 🤝 **Contributing**

We welcome contributions! Please see:
- 🐛 **Issues**: Bug reports and feature requests
- 🔧 **Pull Requests**: Code improvements and new features
- 📖 **Documentation**: Examples and tutorials
- 🧪 **Testing**: Additional test cases and benchmarks

### **Development Setup**
```bash
git clone https://github.com/your-repo/abupt-keras
cd abupt-keras
pip install -e .
pip install -r requirements-dev.txt
```

---

## 📄 **License**

This implementation is released under the MIT License. See LICENSE file for details.

---

## 🙏 **Acknowledgments**

- Original PyTorch implementation authors
- Keras/TensorFlow development team
- dl_techniques library contributors
- CFD research community

---

## 📞 **Support**

- 📧 **Email**: [your-email@domain.com]
- 💬 **Discussions**: GitHub Discussions
- 🐛 **Issues**: GitHub Issues
- 📖 **Documentation**: [Link to detailed docs]

---

**Happy CFD Modeling with AB-UPT! 🌊🚀**