from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.utils.logger import get_logger
from src.utils.config import get
from src.serving.schemas import (
    HealthResponse, ModelInfo, PredictionRequest,
    PredictionResponse, ErrorResponse,
)
from src.serving.predictor import (
    list_available_tickers, load_best_model_info, run_inference,
)

logger = get_logger("serving.app")

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    tickers = list_available_tickers()
    logger.info(f"API starting | available tickers: {tickers}")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Stock Forecast MLOps API",
    description=(
        "Serves next-day stock price predictions from trained "
        "Random Forest, XGBoost, and LSTM models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Health"],
)
def health():
    """Returns API status and list of available tickers."""
    tickers = list_available_tickers()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        tickers_available=tickers,
    )


@app.get(
    "/models",
    response_model=list[ModelInfo],
    summary="List all available models",
    tags=["Models"],
)
def list_models():
    """Returns best model info for all tickers in the registry."""
    tickers = list_available_tickers()
    if not tickers:
        raise HTTPException(
            status_code=404,
            detail="No trained models found. Run the training pipeline first.",
        )
    results = []
    for ticker in tickers:
        try:
            info = load_best_model_info(ticker)
            results.append(ModelInfo(**info))
        except Exception as e:
            logger.warning(f"Could not load model info for {ticker}: {e}")
    return results


@app.get(
    "/models/{ticker}",
    response_model=ModelInfo,
    summary="Get model info for a specific ticker",
    tags=["Models"],
)
def get_model(ticker: str):
    """Returns best model details for one ticker."""
    ticker = ticker.upper()
    try:
        info = load_best_model_info(ticker)
        return ModelInfo(**info)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for ticker '{ticker}'.",
        )


@app.post(
    "/predict/{ticker}",
    response_model=PredictionResponse,
    summary="Predict next-day closing price",
    tags=["Predictions"],
)
def predict(ticker: str, request: PredictionRequest = PredictionRequest()):
    """
    Fetches recent OHLCV data, engineers features, and returns
    the predicted next-day closing price using the best trained model.
    """
    ticker = ticker.upper()

    # Validate ticker is in registry
    available = list_available_tickers()
    if ticker not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ticker '{ticker}' not found in model registry. "
                f"Available: {available}"
            ),
        )

    try:
        result = run_inference(ticker, lookback_days=request.lookback_days)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Inference failed for {ticker}: {str(e)}",
        )
