"""
LSTM Time Series Forecasting Package

This package contains modules for:
- Data preprocessing and normalization
- LSTM model architecture
- Model training
- Model evaluation and visualization
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import data_preprocessing
from . import model
from . import train
from . import evaluate

__all__ = [
    "data_preprocessing",
    "model",
    "train",
    "evaluate"
]

