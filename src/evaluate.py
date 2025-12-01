"""
Model evaluation and results visualization module.

This module handles:
- Loading trained models
- Making predictions
- Computing evaluation metrics
- Visualizing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

from data_preprocessing import load_config, setup_logging, normalize_data


def load_model(config):
    """
    Load the trained model.

    Args:
        config (dict): Configuration dictionary

    Returns:
        keras.Model: Loaded model
    """
    logger = logging.getLogger(__name__)
    model_path = config["paths"]["checkpoint"]

    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_scaler(config):
    """
    Load the data scaler.

    Args:
        config (dict): Configuration dictionary

    Returns:
        Scaler object
    """
    logger = logging.getLogger(__name__)
    scaler_path = config["paths"]["scaler"]

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise


def load_test_data(config):
    """
    Load test data.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple: (X_test, y_test)
    """
    logger = logging.getLogger(__name__)
    data_path = Path(config["data"]["processed_path"])

    try:
        X_test = np.load(data_path / "X_test.npy")
        y_test = np.load(data_path / "y_test.npy")
        logger.info(f"Test data loaded - X: {X_test.shape}, y: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Test data not found: {e}")
        raise


def evaluate_model(config):
    """
    Evaluate the trained model on test set.

    Args:
        config (dict): Configuration dictionary
    """
    logger = setup_logging(config)
    logger.info("Starting model evaluation")

    # Load model and data
    model = load_model(config)
    scaler = load_scaler(config)
    X_test, y_test = load_test_data(config)

    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X_test, verbose=0)

    # Inverse transform to original scale
    y_test_scaled = scaler.inverse_transform(y_test)
    y_pred_scaled = scaler.inverse_transform(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2 = r2_score(y_test_scaled, y_pred_scaled)

    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R²:   {r2:.4f}")
    logger.info("=" * 50)

    # Save evaluation metrics
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "test_samples": len(y_test)
    }

    metrics_file = results_dir / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"Evaluation metrics saved to {metrics_file}")

    # Create visualizations
    logger.info("Creating visualizations")
    create_visualizations(y_test_scaled, y_pred_scaled, results_dir)

    return y_test_scaled, y_pred_scaled


def create_visualizations(y_true, y_pred, results_dir):
    """
    Create evaluation visualizations.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        results_dir (Path): Directory to save plots
    """
    logger = logging.getLogger(__name__)

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Predictions vs True Values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series plot
    ax = axes[0, 0]
    ax.plot(y_true, label="True Values", linewidth=2, alpha=0.7)
    ax.plot(y_pred, label="Predictions", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Predictions vs True Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("True vs Predicted Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution
    residuals = y_true - y_pred
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax.grid(True, alpha=0.3)

    # Residuals over time
    ax = axes[1, 1]
    ax.plot(residuals, linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = results_dir / "evaluation_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    logger.info(f"Evaluation plot saved to {plot_file}")
    plt.close()


if __name__ == "__main__":
    config = load_config()
    y_true, y_pred = evaluate_model(config)

