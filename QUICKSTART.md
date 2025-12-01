# Quick Start Guide

## Project Setup

This guide will help you get started with the LSTM Time Series Forecasting project.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

1. Place your time series data in `data/raw/` as `data.csv`
2. Expected format (CSV):
   ```
   timestamp,feature1,feature2,...,target
   2023-01-01,1.2,3.4,...,5.6
   2023-01-02,1.3,3.5,...,5.7
   ...
   ```

### Step 3: Configure the Project

Edit `config.yaml` to customize:
- Data paths
- Preprocessing parameters (sequence length, normalization type)
- Model architecture (LSTM units, dense layers)
- Training parameters (epochs, batch size, learning rate)

### Step 4: Run the Pipeline

#### Option A: Run All Steps Manually

**Preprocess Data:**
```bash
python src/data_preprocessing.py
```

**Train Model:**
```bash
python src/train.py
```

**Evaluate Model:**
```bash
python src/evaluate.py
```

#### Option B: Run Everything at Once

```bash
python src/data_preprocessing.py && python src/train.py && python src/evaluate.py
```

### Step 5: View Results

Check the `results/` directory for:
- `training_history.json` - Training history
- `metrics.json` - Training metrics
- `evaluation_metrics.json` - Test set evaluation metrics
- `evaluation_results.png` - Visualization plots

### Project Structure

```
project-name/
├── README.md                    # Project documentation
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── QUICKSTART.md                # This file
│
├── data/
│   ├── raw/                     # Your raw data here
│   └── processed/               # Preprocessed data (auto-generated)
│
├── notebooks/                   # Jupyter notebooks for EDA
│
├── src/                         # Main source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading & preprocessing
│   ├── model.py                 # LSTM model architecture
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation & visualization
│
├── models/                      # Saved trained models
│   ├── best_model.keras         # Best model checkpoint
│   └── scaler.pkl               # Data scaler (for inverse transform)
│
├── results/                     # Results & outputs
│   ├── training_history.json
│   ├── metrics.json
│   ├── evaluation_metrics.json
│   └── evaluation_results.png
│
└── tests/                       # Unit tests
    └── test_preprocessing.py
```

### Key Files Explained

**config.yaml**
- Contains all hyperparameters and configuration
- Modify this file to tune your model

**data_preprocessing.py**
- Loads raw data
- Normalizes values
- Creates sequences for LSTM
- Splits into train/val/test sets
- Saves preprocessed data as numpy arrays

**model.py**
- Defines LSTM architecture
- Returns compiled model ready for training
- Includes training callbacks (early stopping, learning rate reduction)

**train.py**
- Loads preprocessed data
- Trains the LSTM model
- Saves training history and metrics
- Evaluates on test set

**evaluate.py**
- Loads trained model and scaler
- Makes predictions
- Calculates metrics (MSE, RMSE, MAE, R²)
- Creates visualization plots

### Troubleshooting

**ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow
```

**FileNotFoundError: data/raw/data.csv not found**
- Make sure your data file is in `data/raw/` directory
- File should be named `data.csv`

**No preprocessed data files**
- Run `python src/data_preprocessing.py` first
- Check that your raw data file exists

### Running Tests

```bash
pytest tests/
```

Or run individual test files:
```bash
python tests/test_preprocessing.py
```

### Advanced Usage

#### Customize LSTM Architecture

Edit `config.yaml` to change model parameters:
```yaml
model:
  lstm_units: [128, 64, 32]      # Add more LSTM layers
  dropout: 0.3                    # Increase dropout for regularization
  dense_units: 32
```

#### Change Sequence Length

For longer-term dependencies, increase sequence length in `config.yaml`:
```yaml
preprocessing:
  sequence_length: 120            # Use 120 time steps instead of 60
```

#### Use Different Scaler

```yaml
preprocessing:
  scaler_type: "standard"         # Use StandardScaler instead of MinMaxScaler
```

### Next Steps

1. Experiment with different hyperparameters
2. Add more features to your data
3. Create Jupyter notebooks in `notebooks/` for EDA
4. Implement ensemble methods
5. Deploy model to production

For more information, see README.md

