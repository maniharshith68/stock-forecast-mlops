import numpy as np
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger("training.evaluator")


@dataclass
class ModelMetrics:
    mae:  float
    rmse: float
    mape: float
    r2:   float

    def to_dict(self) -> dict:
        return {
            "mae":  round(self.mae,  4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "r2":   round(self.r2,   4),
        }

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.4f} | RMSE={self.rmse:.4f} | "
            f"MAPE={self.mape:.4f}% | R²={self.r2:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> ModelMetrics:
    y_true = np.array(y_true, dtype=np.float64).flatten()
    y_pred = np.array(y_pred, dtype=np.float64).flatten()

    assert len(y_true) == len(y_pred), (
        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    ) if nonzero.any() else float("inf")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = ModelMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2)
    if label:
        logger.info(f"Metrics [{label}]: {metrics}")
    return metrics
