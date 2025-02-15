import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    apply_label,
)

# Fixture for loading data
@pytest.fixture
def data():
    """Load the census dataset for testing."""
    data_path = "data/census.csv"
    return pd.read_csv(data_path)

# Fixture for processed data
@pytest.fixture
def processed_data(data):
    """Process the data for testing."""
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )
    return X_train, y_train, encoder, lb

# Test 1: Check if the model is trained and returns the correct type
def test_train_model(processed_data):
    """
    Test if the train_model function returns a RandomForestClassifier.
    """
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"

# Test 2: Check if compute_model_metrics returns the expected values
def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns the correct precision, recall, and F1 score.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, "Precision is incorrect"
    assert recall == pytest.approx(0.6667, 0.01), "Recall is incorrect"
    assert fbeta == pytest.approx(0.8, 0.01), "F1 score is incorrect"

# Test 3: Check if apply_label returns the correct string labels
def test_apply_label():
    """
    Test if apply_label returns the correct string labels.
    """
    assert apply_label([0]) == "<=50K", "Label for 0 should be <=50K"
    assert apply_label([1]) == ">50K", "Label for 1 should be >50K"
