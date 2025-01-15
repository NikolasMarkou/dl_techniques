# Playground for different ml ideas

## Project Structure:

```markdown
dl_playground/
├── .env                        # Environment variables
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
├── setup.py                    # Package installation script
├── pyproject.toml             # Project metadata and build configuration
│
├── src/                       # Source code directory
│   └── dl_playground/         # Main package directory
│       ├── __init__.py
│       ├── layers/            # Custom layers module
│       │   ├── __init__.py
│       │   ├── cfar10_model.py
│       │   ├── convnext_kmeans_block.py
│       │   ├── convolutional_transformer_block.py
│       │   ├── differentiable_kmeans_layer.py
│       │   └── logit_norm.py
│       │
│       └── utils/             # Utility functions
│           ├── __init__.py
│           ├── data_loader.py
│           └── visualization.py
│
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── conftest.py           # pytest configuration
│   ├── test_layers/          # Tests for layers
│   │   ├── __init__.py
│   │   ├── test_cfar10_model.py
│   │   ├── test_convnext_kmeans.py
│   │   └── test_logit_norm.py
│   └── test_utils/           # Tests for utilities
│       ├── __init__.py
│       └── test_data_loader.py
│
├── notebooks/                 # Jupyter notebooks
│   ├── experiments/          # Experimental notebooks
│   │   ├── kmeans_demo.ipynb
│   │   └── logit_norm_analysis.ipynb
│   └── tutorials/            # Tutorial notebooks
│       └── getting_started.ipynb
│
├── docs/                     # Documentation
│   ├── api/                  # API documentation
│   ├── examples/             # Example usage
│   └── tutorials/            # Written tutorials
│
└── scripts/                  # Utility scripts
    ├── train.py             # Training script
    └── evaluate.py          # Evaluation script
```