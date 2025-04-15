# Deep Learning Techniques

<p align="center">
  <img src="./imgs/minimalist-2d-logo-with-a-left-to-right-_sSVDZkeKR4C_eovommSCFQ_mJekanaZTB2Nbe5dBKOnPQ.png" alt="Deep Learning Techniques" width="512" height="512">
</p>

A versatile, modern library providing **advanced deep learning layers, initializers, constraints, and analysis tools** for Keras/TensorFlow. Use it to experiment with **cutting-edge network components**, build novel architectures, or perform in-depth **model analyses** – all while leveraging a robust test suite and real-world example experiments.

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Installation](#installation)  
3. [Usage Examples](#usage-examples)  
4. [Project Structure](#project-structure)  
5. [Documentation](#documentation)  
6. [Experiments](#experiments)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [References](#references)  

---

## Key Features

- **Wide Range of Layers**  
  - **Normalization**: RMS Norm variations, Global Response Norm, novel bounding norms, etc.  
  - **Attention Mechanisms**: Hopfield Attention, Non-Local Attention, Differential Transformer, Convolutional Transformer, etc.  
  - **Vision**: Convolutional Block Attention Module (CBAM), CLAHE, various ConvNeXt blocks, hierarchical Vision Transformers  
  - **Capsules & Advanced Activations**: Capsule Networks, RBF layers, Mish, GLU, Shearlet transforms, and more

- **Regularization & Optimization**  
  - **Regularizers**: SRIP, Soft Orthogonal, Tri-State Preferences, Orthonormal constraints  
  - **Optimizers & Schedules**: Warmup schedules, selective gradient routing, deep supervision  
  - **Stochastic Techniques**: Stochastic Depth, band-limited RMS Norm for stable training

- **Advanced Analysis & Visualization**  
  - **Activation Distribution Analysis**: Inspect layer-by-layer activation statistics  
  - **WeightWatcher Integration**: Leverage SVD-based metrics to assess generalization  
  - **Logging & Visualization**: Built-in logging infrastructure, multiple plotting/visualization utilities

- **Full Model Implementations**  
  - **Depth Anything**, **CoshNet**, **MobileNet v4**, **Mixture Density Network (MDN) Model**, **CapsNet**, *and more*

- **Extensive Test Suite & Example Experiments**  
  - Over **100+** unit tests covering layers, losses, regularizers, and more  
  - Real-world experiment scripts on MNIST, time series forecasting, and more  
  - Detailed docs explaining each technique, usage scenario, and best practices

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/dl_techniques.git
   cd dl_techniques
   ```

2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: The key dependencies include `tensorflow==2.18.0`, `keras>=3.8.0`, `numpy`, `matplotlib`, `scikit-learn`, `pandas`, and more.

3. **Editable Install** (recommended for development):
   ```bash
   pip install -e .
   ```

---

## Usage Examples

Here’s a simple snippet demonstrating how to use some advanced layers:

```python
import tensorflow as tf
from dl_techniques.layers.global_response_norm import GlobalResponseNorm
from dl_techniques.layers.adaptive_softmax import AdaptiveTemperatureSoftmax

# Example model using advanced layers
inputs = tf.keras.Input(shape=(64,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = GlobalResponseNorm()(x)        # apply GRN
x = AdaptiveTemperatureSoftmax()(x)  # adapt softmax temperature
model = tf.keras.Model(inputs, x)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Example training
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
```

**Other Highlights**  
- **Custom Losses**: `AnyLoss` classes to directly optimize F1, Balanced Accuracy, or other classification metrics.  
- **MDN**: Mixture Density Networks for multi-modal output predictions.  
- **Analysis Tools**: Activation distribution analysis, WeightWatcher-based SVD smoothing, and more.

---

## Project Structure

```
dl_techniques/
├─ README.md                    
├─ LICENSE                     
├─ pyproject.toml
├─ requirements.txt
├─ setup.py
├─ .pre-commit-config.yaml
├─ .pylintrc
├─ docs/                      
│  ├─ anyloss_classification_metrics_loss_functions.md
│  ├─ custom_rms_norm.md
│  ├─ hierarchical_vision_transformers.md
│  ├─ input_normalization.md
│  ├─ mdn.md
│  ├─ orthogonal_regularization.md
│  └─ orthonormal_regularization_autoadjusting.md
├─ imgs/                      
├─ src/
│  └─ dl_techniques/
│     ├─ analysis/
│     ├─ constraints/
│     ├─ initializers/
│     ├─ layers/
│     ├─ losses/
│     ├─ models/
│     ├─ optimization/
│     ├─ regularizers/
│     ├─ utils/
│     └─ visualization/
├─ experiments/
│  ├─ activation_mish/
│  ├─ band_rms/
│  ├─ coupled_rms_norm/
│  ├─ kmeans/
│  ├─ layer_scale_binary_state/
│  ├─ mdn/
│  ├─ rbf/
│  ├─ regularizer_binary/
│  ├─ regularizer_tri_state/
│  └─ rms_norm/
└─ tests/
   ├─ test_analysis/
   ├─ test_initializers/
   ├─ test_layers/
   ├─ test_losses/
   ├─ test_models/
   ├─ test_optimization/
   ├─ test_regularizers/
   └─ test_utils/
```

- **`src/dl_techniques/`** – The core library code (layers, initializers, constraints, analysis, etc.)  
- **`docs/`** – Additional documentation on specialized topics (loss functions, custom RMS norms, hierarchical ViTs, etc.)  
- **`experiments/`** – Real-world usage demos with scripts for classification, segmentation, time series forecasting, etc.  
- **`tests/`** – PyTest-based testing for reliability, covering everything from layers to losses

---

## Documentation

In-depth details are in the `docs/` folder. Some highlights:

1. **Advanced Loss Functions**  
   See [anyloss_classification_metrics_loss_functions.md](./docs/anyloss_classification_metrics_loss_functions.md) for a deep dive into directly optimizing classification metrics like F1.

2. **Custom RMS Norm**  
   See [custom_rms_norm.md](./docs/custom_rms_norm.md) for various RMS norm variants and bounding shells.

3. **Hierarchical Vision Transformers**  
   See [hierarchical_vision_transformers.md](./docs/hierarchical_vision_transformers.md) for the implementation details and insights about the hMLP stem.

4. **Mixture Density Networks**  
   See [mdn.md](./docs/mdn.md) for theory, usage, and best practices around mixture density models.

5. **Orthonormal / Orthogonal Regularization**  
   See [orthogonal_regularization.md](./docs/orthogonal_regularization.md) and [orthonormal_regularization_autoadjusting.md](./docs/orthonormal_regularization_autoadjusting.md) for how to address the BN + weight decay contradiction using orthonormal constraints.

---

## Experiments

Check out the [`experiments/`](./experiments/) directory for real-world training scripts and analyses:
- **MNIST with Band-limited RMS** (`experiments/band_rms`)
- **MDN for Time Series** (`experiments/mdn`)
- **Coupled RMS Norm** (`experiments/coupled_rms_norm`)
- **K-means Layers** (`experiments/kmeans`)
- **Layer Scale Binary** (`experiments/layer_scale_binary_state`)
- And more…

Each experiment folder typically contains:
- One or more `.py` files demonstrating how to build and train a model
- Markdown results or notes documenting outcomes
- Easy-to-modify code for your own experiments

---

## Contributing

1. **Fork** & **Clone** the repository  
2. **Create a Branch** for your feature or bugfix  
3. **Write Tests** covering changes (see `tests/` folder)  
4. **Pre-commit Hooks**: This project uses [pre-commit](https://pre-commit.com/) for checks. Make sure these pass with `pylint`, `pytest`, etc.  
5. **Open a Pull Request** describing your changes

**Coding Standards**:
- Follow [PEP 8](https://peps.python.org/pep-0008/)  
- Prefer **type hints**  
- Use docstrings (Sphinx or NumPy style) for new layers, models, or utilities  
- Write thorough tests in `tests/`

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).  
Refer to the [LICENSE file](LICENSE) for the full text and details.

---

## References

- **Arjovsky, M., Shah, A., & Bengio, Y.** (2016).  
  _Unitary Evolution Recurrent Neural Networks._ Proceedings of the 33rd International Conference on Machine Learning (ICML).

- **Ba, J.L., Kiros, J.R., & Hinton, G.E.** (2016).  
  _Layer Normalization._ [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

- **Bansal, N., Chen, X., & Wang, Z.** (2018).  
  _Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?_ Proceedings of the 32nd Conference on Neural Information Processing Systems (NeurIPS).

- **Bishop, C.M.** (1994).  
  _Mixture Density Networks._ Technical Report NCRG/4288, Aston University.

- **Bishop, C.M.** (2006).  
  _Pattern Recognition and Machine Learning._ Springer.

- **Brock, A., Lim, T., Ritchie, J., & Weston, N.** (2021).  
  _High-Performance Large-Scale Image Recognition Without Normalization._ International Conference on Machine Learning (ICML).

- **Charles Martin.**  
  Original WeightWatcher implementation for neural network analysis (various blog & GitHub resources).

- **Cissé, M., Bojanowski, P., Grave, E., Dauphin, Y., & Usunier, N.** (2017).  
  _Parseval Networks: Improving Robustness to Adversarial Examples._ Proceedings of the 34th International Conference on Machine Learning (ICML).

- **Doheon Han, Nuno Moniz, & Nitesh V. Chawla.** (2023).  
  _AnyLoss: Transforming Classification Metrics into Loss Functions._ (Various preprint/ArXiv references).

- **Dosovitskiy, A., et al.** (2021).  
  _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ International Conference on Learning Representations (ICLR).

- **Glorot, X. & Bengio, Y.** (2010).  
  _Understanding the difficulty of training deep feedforward neural networks._ Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS).

- **Graves, A.** (2013).  
  _Generating Sequences with Recurrent Neural Networks._ [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)

- **He, K., Zhang, X., Ren, S., & Sun, J.** (2015).  
  _Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification._ Proceedings of the IEEE International Conference on Computer Vision (ICCV).

- **Hinton, G.E., et al.**  
  Work on capsule networks and part-whole relationships (various publications and unpublished notes).

- **Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K.Q.** (2016).  
  _Deep Networks with Stochastic Depth._ Proceedings of the European Conference on Computer Vision (ECCV).

- **Jia, X., Song, X., & Sun, M.** (2019).  
  _Orthogonality-based Deep Neural Networks for Ultra-Low Precision Image Classification._ AAAI Conference on Artificial Intelligence.

- **Liu, Z., et al.** (2022).  
  _A ConvNet for the 2020s (ConvNeXt)._ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

- **Misra, D.** (2019).  
  _Mish: A Self Regularized Non-Monotonic Neural Activation Function._ [arXiv:1908.08681](https://arxiv.org/abs/1908.08681)

- **Ramsauer, H., et al.** (2020).  
  _Hopfield Networks is All You Need._ [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)

- **Sabour, S., Frosst, N., & Hinton, G.E.** (2017).  
  _Dynamic Routing Between Capsules._ Proceedings of the 31st Conference on Neural Information Processing Systems (NeurIPS).

- **Saxe, A.M., McClelland, J.L., & Ganguli, S.** (2014).  
  _Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks._ International Conference on Learning Representations (ICLR).

- **Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., & Jégou, H.**  
  _Three Things Everyone Should Know About Vision Transformers._ [arXiv preprint](https://arxiv.org/abs/2208.07339)

- **Trabelsi, C., et al.** (2018).  
  _Deep Complex Networks._ International Conference on Learning Representations (ICLR).

- **Vorontsov, E., Trabelsi, C., Thomas, A.W., & Pal, C.** (2017).  
  _On Orthogonality and Learning Recurrent Networks with Long Term Dependencies._ ICML Workshop on Deep Structured Prediction.

- **Woo, S., Park, J., Lee, J.Y., & Kweon, I.S.** (2018).  
  _CBAM: Convolutional Block Attention Module._ Proceedings of the IEEE International Conference on Computer Vision (ICCV).

- **Wu, Y. & He, K.** (2018).  
  _Group Normalization._ Proceedings of the European Conference on Computer Vision (ECCV).

- **Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K.** (2017).  
  _Aggregated Residual Transformations for Deep Neural Networks._ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

- **Yang, G., et al.** (2020).  
  _Soft Orthogonality Constraints in Neural Networks._ [arXiv:2007.15677](https://arxiv.org/abs/2007.15677)

- **Yu, F. & Koltun, V.** (2016).  
  _Multi-Scale Context Aggregation by Dilated Convolutions._ International Conference on Learning Representations (ICLR).

- **Zhang, H., et al.** (2018).  
  _SRIP: Matrix-Based Spectral Regularization for Fine-Tuning Deep Neural Networks._ Advances in Neural Information Processing Systems (NeurIPS).

---

**Happy experimenting with DL Techniques!** If you have questions, bug reports, or feature requests, feel free to open an issue or start a discussion. Enjoy exploring the cutting-edge features and insights offered by this library!