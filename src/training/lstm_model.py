import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import mlflow
from src.utils.logger import get_logger
from src.utils.config import load_config
import sys

logger = get_logger("training.lstm_model")

PROCESSED_DIR = Path("data/processed")
REGISTRY_DIR  = Path("models/registry")

TBPTT_CHUNK = 5   # backprop through 5 timesteps at a time — avoids stack overflow


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM using LSTMCell.
    Trained with truncated BPTT (TBPTT_CHUNK timesteps per backward pass)
    to avoid PyTorch 2.10 autograd stack overflow on macOS ARM + Python 3.14.
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(nn.LSTMCell(in_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full-sequence forward pass — used for evaluation/inference."""
        batch_size, seq_len, _ = x.shape
        h = [torch.zeros(batch_size, self.hidden_size) for _ in self.cells]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in self.cells]

        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = h[i]
                if i < len(self.cells) - 1:
                    inp = self.dropout(inp)

        return self.fc(self.dropout(h[-1])).squeeze(-1)

    def forward_tbptt(
        self,
        x:         torch.Tensor,
        chunk:     int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        y:         torch.Tensor,
    ) -> float:
        """
        Truncated BPTT forward+backward.
        Processes `chunk` timesteps, backprops, then detaches hidden states.
        Returns total loss for the sequence.
        """
        batch_size, seq_len, _ = x.shape
        h = [torch.zeros(batch_size, self.hidden_size) for _ in self.cells]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in self.cells]

        total_loss = 0.0
        n_chunks   = 0

        for start in range(0, seq_len, chunk):
            end    = min(start + chunk, seq_len)
            x_chunk = x[:, start:end, :]

            # Forward through this chunk
            for t in range(end - start):
                inp = x_chunk[:, t, :]
                for i, cell in enumerate(self.cells):
                    h[i], c[i] = cell(inp, (h[i], c[i]))
                    inp = h[i]
                    if i < len(self.cells) - 1:
                        inp = self.dropout(inp)

            # Only compute loss and backprop on the final chunk
            if end == seq_len:
                out  = self.fc(self.dropout(h[-1])).squeeze(-1)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_chunks   += 1

            # Detach hidden states — breaks the graph, prevents stack overflow
            h = [hi.detach() for hi in h]
            c = [ci.detach() for ci in c]

        return total_loss / max(n_chunks, 1)


def _make_batches(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n   = len(X)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    X, y = X[idx], y[idx]
    batches = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        batches.append((
            torch.from_numpy(X[start:end].copy()).float(),
            torch.from_numpy(y[start:end].copy()).float(),
        ))
    return batches


def _evaluate(
    model:      StockLSTM,
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    criterion:  nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for X_b, y_b in _make_batches(X, y, batch_size, shuffle=False):
            out = model(X_b)
            losses.append(criterion(out, y_b).item())
            preds.append(out.numpy())
            targets.append(y_b.numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(targets)


def load_lstm_data(ticker: str) -> dict[str, np.ndarray]:
    lstm_dir = PROCESSED_DIR / ticker / "lstm"
    data = {}
    for split in ["train", "val", "test"]:
        data[f"{split}_X"] = np.load(lstm_dir / f"{split}_X.npy")
        data[f"{split}_y"] = np.load(lstm_dir / f"{split}_y.npy")
    return data


def train_lstm(ticker: str) -> dict:
    config   = load_config()
    lstm_cfg = config["models"]["lstm"]
    env      = config["training"].get("env", "local")
    patience = config["training"].get("early_stopping_patience", 5)

    epochs     = lstm_cfg["local_epochs"] if env == "local" else lstm_cfg["epochs"]
    batch_size = lstm_cfg["batch_size"]
    lr         = lstm_cfg["learning_rate"]

    logger.info(
        f"Training LSTM | ticker={ticker} | env={env} | "
        f"epochs={epochs} | device=cpu | arch=LSTMCell+TBPTT"
    )

    data       = load_lstm_data(ticker)
    input_size = data["train_X"].shape[2]

    model     = StockLSTM(
        input_size,
        lstm_cfg["hidden_size"],
        lstm_cfg["num_layers"],
        lstm_cfg["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    with mlflow.start_run(run_name=f"lstm_{ticker}") as run:
        mlflow.log_params({
            "model":       "LSTM",
            "arch":        "LSTMCell+TBPTT",
            "ticker":      ticker,
            "hidden_size": lstm_cfg["hidden_size"],
            "num_layers":  lstm_cfg["num_layers"],
            "dropout":     lstm_cfg["dropout"],
            "epochs":      epochs,
            "batch_size":  batch_size,
            "lr":          lr,
            "input_size":  input_size,
            "tbptt_chunk": TBPTT_CHUNK,
            "env":         env,
        })

        best_val_loss    = float("inf")
        patience_counter = 0
        best_state       = None
        t0               = time.time()

        for epoch in range(1, epochs + 1):

            # ── Train with TBPTT ──────────────────────────────────────
            model.train()
            epoch_losses = []
            for X_b, y_b in _make_batches(
                data["train_X"], data["train_y"], batch_size, shuffle=True
            ):
                loss = model.forward_tbptt(X_b, TBPTT_CHUNK, optimizer, criterion, y_b)
                epoch_losses.append(loss)

            train_loss = float(np.mean(epoch_losses))

            # ── Validate ──────────────────────────────────────────────
            val_loss, _, _ = _evaluate(
                model, data["val_X"], data["val_y"], batch_size, criterion
            )
            scheduler.step(val_loss)

            mlflow.log_metrics(
                {"epoch_train_loss": train_loss, "epoch_val_loss": val_loss},
                step=epoch,
            )

            log_every = max(1, epochs // 5)
            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    f"LSTM/{ticker} epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )

            # ── Early stopping ─────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"LSTM/{ticker} early stop at epoch {epoch}")
                    break

        elapsed = time.time() - t0
        if best_state:
            model.load_state_dict(best_state)

        # ── Final evaluation ──────────────────────────────────────────
        _, val_preds,  val_true  = _evaluate(
            model, data["val_X"],  data["val_y"],  batch_size, criterion
        )
        _, test_preds, test_true = _evaluate(
            model, data["test_X"], data["test_y"], batch_size, criterion
        )

        from src.training.evaluator import compute_metrics
        val_m  = compute_metrics(val_true,  val_preds,  f"LSTM/{ticker}/val")
        test_m = compute_metrics(test_true, test_preds, f"LSTM/{ticker}/test")

        mlflow.log_metrics({
            "val_mae":       val_m.mae,
            "val_rmse":      val_m.rmse,
            "val_r2":        val_m.r2,
            "test_mae":      test_m.mae,
            "test_rmse":     test_m.rmse,
            "test_r2":       test_m.r2,
            "train_time_s":  elapsed,
            "best_val_loss": best_val_loss,
        })

        # ── Save ──────────────────────────────────────────────────────
        out_dir    = REGISTRY_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "lstm.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size":  input_size,
                "hidden_size": lstm_cfg["hidden_size"],
                "num_layers":  lstm_cfg["num_layers"],
                "dropout":     lstm_cfg["dropout"],
            },
        }, model_path)
        mlflow.log_param("model_path", str(model_path))

        run_id = run.info.run_id
        logger.info(
            f"LSTM/{ticker} done | val_rmse={val_m.rmse:.4f} | "
            f"time={elapsed:.1f}s | run_id={run_id}"
        )

    return {
        "model":        "lstm",
        "ticker":       ticker,
        "run_id":       run_id,
        "model_path":   str(model_path),
        "val_metrics":  val_m.to_dict(),
        "test_metrics": test_m.to_dict(),
    }


def load_lstm_model(model_path: str | Path) -> tuple[StockLSTM, torch.device]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg        = checkpoint["model_config"]
    model      = StockLSTM(
        cfg["input_size"], cfg["hidden_size"],
        cfg["num_layers"], cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, torch.device("cpu")

def train_lstm_subprocess(ticker: str) -> dict:
    """
    Runs train_lstm in a clean subprocess where torch is imported
    before sklearn/mlflow, eliminating the macOS ARM OpenMP conflict.
    The result is written to a temp JSON file and read back.
    """
    import subprocess
    import json
    import tempfile
    import os

    # Write a self-contained training script to a temp file
    script = f"""
import sys, json, os
sys.path.insert(0, '{os.getcwd()}')

# Import torch FIRST — before sklearn or mlflow load
import torch
import numpy as np

# Now safe to import everything else
from src.training.lstm_model import train_lstm

result = train_lstm('{ticker}')

with open(sys.argv[1], 'w') as f:
    json.dump(result, f)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as sf:
        sf.write(script)
        script_path = sf.name

    result_fd, result_path = tempfile.mkstemp(suffix='.json')
    os.close(result_fd)

    try:
        proc = subprocess.run(
            [sys.executable, script_path, result_path],
            capture_output=False,   # let logs stream to terminal
            timeout=3600,           # 1 hour max
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"LSTM subprocess failed for {ticker} "
                f"(returncode={proc.returncode})"
            )

        with open(result_path) as f:
            return json.load(f)

    finally:
        os.unlink(script_path)
        if os.path.exists(result_path):
            os.unlink(result_path)
