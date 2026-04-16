import numpy as np
import pytest
from src.training.evaluator import compute_metrics, ModelMetrics


def test_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    m = compute_metrics(y, y)
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert m.r2 == 1.0


def test_known_mae():
    m = compute_metrics(np.array([100.0, 100.0]), np.array([110.0, 90.0]))
    assert abs(m.mae - 10.0) < 1e-9


def test_known_rmse():
    m = compute_metrics(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert abs(m.rmse - np.sqrt(12.5)) < 1e-9


def test_r2_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(y, np.full(5, np.mean(y)))
    assert abs(m.r2) < 0.01


def test_mape():
    m = compute_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert abs(m.mape - 10.0) < 1e-6


def test_to_dict_keys():
    d = ModelMetrics(1.0, 2.0, 3.0, 0.9).to_dict()
    assert set(d.keys()) == {"mae", "rmse", "mape", "r2"}


def test_length_mismatch_raises():
    with pytest.raises(AssertionError):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))
