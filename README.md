# DL Techniques

This project is a playground for experimenting with various deep learning techniques, particularly focusing on neural network layers and transformations. It provides implementations of several advanced techniques and includes experiments to demonstrate their applications.

## Project Structure

- **src/dl_techniques/layers**: Contains implementations of various deep learning layers and techniques, such as convolutional transformers, differentiable KMeans, Gaussian filters, and more.
- **src/dl_techniques/regularizers**: Includes regularization techniques to improve model generalization.
- **src/dl_techniques/utils**: Utility functions for logging, tensor operations, and visualization.
- **src/experiments**: Scripts demonstrating the application of the implemented techniques, such as KMeans clustering and logit normalization experiments.
- **tests**: Unit tests for the implemented layers, regularizers, and utilities.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### KMeans Clustering

To run the KMeans clustering demo, execute:

```bash
python src/experiments/basic.py
```

### Logit Normalization Experiments

To run the logit normalization experiments, execute:

```bash
python src/experiments/coupled_logit_norm.py
```

## Dependencies

- numpy
- pytest
- pytest-cov
- matplotlib
- scikit-learn
- keras~=3.8.0
- tensorflow==2.18.0

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## Contact

For any questions or inquiries, please contact the project maintainer.
