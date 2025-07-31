import joblib
import os
from sklearn.linear_model import LinearRegression
from src import utils
import pytest

def test_dataset_loading():
    X, y = utils.load_data()
    assert X.shape[0] == len(y)

def test_model_training():
    X, y = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_")

def test_r2_threshold():
    X, y = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2, _ = utils.evaluate_model(model, X_test, y_test)
    assert r2 > 0.5
