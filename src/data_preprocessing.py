"""
Data preprocessing module for time series forecasting.

This module handles:
- Loading raw data
- Normalizing values
- Creating sequences for LSTM
- Splitting into train/validation/test sets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import logging
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """Setup logging configuration."""
    Path(config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=config["logging"]["level"],
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(data_path):
    """
    Load time series data from CSV or other format.

    Args:
        data_path (str): Path to the data file

    Returns:
        np.ndarray: Time series data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_path}")

    try:
        df = pd.read_csv(data_path)
        # Assume the last column is the target variable
        data = df.iloc[:, -1].values.reshape(-1, 1)
        logger.info(f"Data shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise


def normalize_data(data, scaler_type="minmax", fit=True, scaler=None):
    """
    Normalize data using specified scaler.

    Args:
        data (np.ndarray): Input data
        scaler_type (str): Type of scaler ("minmax" or "standard")
        fit (bool): Whether to fit the scaler
        scaler: Pre-fitted scaler object

    Returns:
        tuple: (normalized_data, scaler)
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler() if scaler is None else scaler
    elif scaler_type == "standard":
        scaler = StandardScaler() if scaler is None else scaler
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    if fit:
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)

    return normalized_data, scaler


def create_sequences(data, sequence_length):
    """
    Create sequences for LSTM training.

    Args:
        data (np.ndarray): Input time series data
        sequence_length (int): Length of each sequence

    Returns:
        tuple: (X, y) - Features and labels
    """
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)


def split_data(X, y, test_size=0.2, validation_size=0.1):
    """
    Split data into train, validation, and test sets.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        test_size (float): Proportion of test set
        validation_size (float): Proportion of validation set

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = len(X)
    test_split = int(n_samples * (1 - test_size))
    val_split = int(test_split * (1 - validation_size))

    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:test_split], y[val_split:test_split]
    X_test, y_test = X[test_split:], y[test_split:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(config):
    """
    Execute the full preprocessing pipeline.

    Args:
        config (dict): Configuration dictionary
    """
    logger = setup_logging(config)
    logger.info("Starting data preprocessing pipeline")

    # Create directories
    Path(config["data"]["processed_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)

    # Load data
    data_file = Path(config["data"]["raw_path"]) / "data.csv"
    if not data_file.exists():
        logger.warning(f"No data file found at {data_file}")
        logger.info("Please place your data file at data/raw/data.csv")
        return

    raw_data = load_data(str(data_file))

    # Normalize
    if config["preprocessing"]["normalize"]:
        normalized_data, scaler = normalize_data(
            raw_data,
            scaler_type=config["preprocessing"]["scaler_type"],
            fit=True
        )
        # Save scaler
        with open(config["paths"]["scaler"], "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {config['paths']['scaler']}")
    else:
        normalized_data = raw_data

    # Create sequences
    X, y = create_sequences(
        normalized_data,
        sequence_length=config["preprocessing"]["sequence_length"]
    )
    logger.info(f"Sequences created - X shape: {X.shape}, y shape: {y.shape}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=config["data"]["test_size"],
        validation_size=config["data"]["validation_size"]
    )

    # Save processed data
    np.save(Path(config["data"]["processed_path"]) / "X_train.npy", X_train)
    np.save(Path(config["data"]["processed_path"]) / "X_val.npy", X_val)
    np.save(Path(config["data"]["processed_path"]) / "X_test.npy", X_test)
    np.save(Path(config["data"]["processed_path"]) / "y_train.npy", y_train)
    np.save(Path(config["data"]["processed_path"]) / "y_val.npy", y_val)
    np.save(Path(config["data"]["processed_path"]) / "y_test.npy", y_test)

    logger.info("Data preprocessing completed successfully")
    logger.info(f"Train set - X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Validation set - X: {X_val.shape}, y: {y_val.shape}")
    logger.info(f"Test set - X: {X_test.shape}, y: {y_test.shape}")


if __name__ == "__main__":
    config = load_config()
    preprocess_pipeline(config)

