import numpy as np
import torch
import pytest
import torch.nn as nn
from src.training.lstm_model import StockLSTM, _make_batches, _evaluate


def test_forward_output_shape():
    model = StockLSTM(input_size=40, hidden_size=32, num_layers=2, dropout=0.1)
    out   = model(torch.randn(16, 30, 40))
    assert out.shape == (16,), f"Expected (16,), got {out.shape}"


def test_forward_single_sample():
    model = StockLSTM(input_size=10, hidden_size=16, num_layers=1, dropout=0.0)
    out   = model(torch.randn(1, 5, 10))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_make_batches_count():
    X       = np.random.randn(100, 30, 10).astype(np.float32)
    y       = np.random.randn(100).astype(np.float32)
    batches = _make_batches(X, y, batch_size=32, shuffle=False)
    assert len(batches) == 4                         # 100 / 32 = 4 batches
    assert batches[0][0].shape == (32, 30, 10)
    assert batches[-1][0].shape == (4, 30, 10)       # last batch is smaller


def test_make_batches_shuffle():
    np.random.seed(0)
    X       = np.arange(64).reshape(64, 1, 1).astype(np.float32)
    y       = np.arange(64).astype(np.float32)
    ordered = _make_batches(X, y, batch_size=32, shuffle=False)
    shuffled = _make_batches(X, y, batch_size=32, shuffle=True)
    assert not torch.equal(ordered[0][0], shuffled[0][0])


def test_evaluate_returns_correct_shapes():
    model   = StockLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    crit    = nn.MSELoss()
    X       = np.random.randn(50, 10, 5).astype(np.float32)
    y       = np.random.randn(50).astype(np.float32)
    loss, preds, targets = _evaluate(model, X, y, batch_size=16, criterion=crit)
    assert np.isfinite(loss)
    assert preds.shape   == (50,)
    assert targets.shape == (50,)


def test_full_training_loop():
    """3 epochs on tiny data — confirms forward + backward + optimizer all work."""
    model     = StockLSTM(input_size=8, hidden_size=16, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    X         = np.random.randn(40, 5, 8).astype(np.float32)
    y         = (np.random.rand(40) * 100 + 50).astype(np.float32)

    for _ in range(3):
        model.train()
        for X_b, y_b in _make_batches(X, y, batch_size=16, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

    loss_val, preds, _ = _evaluate(model, X, y, batch_size=16, criterion=criterion)
    assert np.isfinite(loss_val)
    assert preds.shape == (40,)
