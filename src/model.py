"""
LSTM model definition module.

This module contains the architecture definition for LSTM-based
time series forecasting models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import logging


def build_lstm_model(config):
    """
    Build an LSTM model for time series forecasting.

    Args:
        config (dict): Configuration dictionary containing model parameters

    Returns:
        keras.Model: Compiled LSTM model
    """
    logger = logging.getLogger(__name__)

    model_config = config["model"]
    training_config = config["training"]

    # Input layer
    input_shape = (config["preprocessing"]["sequence_length"], 1)
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # LSTM layers
    lstm_units = model_config["lstm_units"]
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            activation=model_config["activation"],
            dropout=model_config["dropout"]
        ))

    # Dense layers
    model.add(layers.Dense(
        units=model_config["dense_units"],
        activation=model_config["activation"]
    ))
    model.add(layers.Dropout(model_config["dropout"]))

    # Output layer
    model.add(layers.Dense(1, activation=model_config["output_activation"]))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=training_config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=training_config["loss"],
        metrics=training_config["metrics"]
    )

    logger.info("LSTM model built successfully")
    model.summary()

    return model


def get_callbacks(config):
    """
    Get training callbacks.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list: List of keras callbacks
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=config["paths"]["checkpoint"],
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model = build_lstm_model(config)

