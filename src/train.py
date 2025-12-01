"""
Model training module.

This module handles the training loop, validation, and model checkpointing
for the LSTM time series forecasting model.
"""

import numpy as np
import yaml
import logging
from pathlib import Path
import json
from datetime import datetime

from model import build_lstm_model, get_callbacks
from data_preprocessing import load_config, setup_logging


def load_preprocessed_data(config):
    """
    Load preprocessed data from disk.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger = logging.getLogger(__name__)
    data_path = Path(config["data"]["processed_path"])

    try:
        X_train = np.load(data_path / "X_train.npy")
        X_val = np.load(data_path / "X_val.npy")
        X_test = np.load(data_path / "X_test.npy")
        y_train = np.load(data_path / "y_train.npy")
        y_val = np.load(data_path / "y_val.npy")
        y_test = np.load(data_path / "y_test.npy")

        logger.info("Preprocessed data loaded successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except FileNotFoundError as e:
        logger.error(f"Preprocessed data not found: {e}")
        logger.info("Please run data_preprocessing.py first")
        raise


def train_model(config):
    """
    Train the LSTM model.

    Args:
        config (dict): Configuration dictionary
    """
    logger = setup_logging(config)
    logger.info("Starting model training")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(config)

    # Build model
    model = build_lstm_model(config)

    # Get callbacks
    callbacks = get_callbacks(config)

    # Train model
    logger.info("Training started")
    history = model.fit(
        X_train, y_train,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Training completed")

    # Save training history
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    history_dict = {
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])],
        "mse": [float(x) for x in history.history.get("mse", [])],
        "val_mse": [float(x) for x in history.history.get("val_mse", [])],
        "mae": [float(x) for x in history.history.get("mae", [])],
        "val_mae": [float(x) for x in history.history.get("val_mae", [])],
    }

    history_file = results_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=2)

    logger.info(f"Training history saved to {history_file}")

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_loss, test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "training_timestamp": datetime.now().isoformat()
    }

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test MSE: {test_mse:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Metrics saved to {metrics_file}")

    return model, history


if __name__ == "__main__":
    config = load_config()
    model, history = train_model(config)

