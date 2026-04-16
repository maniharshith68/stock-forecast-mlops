from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    version: str
    tickers_available: list[str]


class ModelInfo(BaseModel):
    ticker:     str
    best_model: str
    model_path: str
    val_metrics:  dict[str, float]
    test_metrics: dict[str, float]
    all_results:  list[dict]


class PredictionRequest(BaseModel):
    """Optional — if not provided, predictor fetches live data from yFinance."""
    lookback_days: int = Field(
        default=90,
        ge=60,
        le=365,
        description="Days of OHLCV history to fetch for feature engineering",
    )


class PredictionResponse(BaseModel):
    ticker:            str
    predicted_close:   float
    model_used:        str
    prediction_date:   str
    val_rmse:          float
    val_r2:            float
    features_used:     int
    inference_time_ms: float


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    ticker:  Optional[str] = None
