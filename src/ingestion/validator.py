from dataclasses import dataclass, field
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("ingestion.validator")

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MIN_ROWS = 10
MAX_NULL_FRACTION = 0.05       # 5% nulls allowed
MIN_PRICE = 0.01
MAX_PRICE = 1_000_000.0
MIN_VOLUME = 0


@dataclass
class ValidationResult:
    ticker: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.ticker} | rows={self.row_count} | "
            f"errors={len(self.errors)} | warnings={len(self.warnings)}"
        )


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """
    Run all validation checks on a raw OHLCV DataFrame.
    Returns a ValidationResult with pass/fail status and details.
    """
    result = ValidationResult(ticker=ticker, passed=True)

    if df is None or df.empty:
        result.passed = False
        result.errors.append("DataFrame is None or empty")
        logger.error(f"Validation FAIL [{ticker}]: empty dataframe")
        return result

    result.row_count = len(df)

    _check_minimum_rows(df, result)
    _check_required_columns(df, result)
    _check_null_fraction(df, result)
    _check_price_ranges(df, result)
    _check_volume(df, result)
    _check_ohlc_consistency(df, result)
    _check_index_type(df, result)

    if result.errors:
        result.passed = False

    logger.info(result.summary())
    for err in result.errors:
        logger.error(f"  Validation error [{ticker}]: {err}")
    for warn in result.warnings:
        logger.warning(f"  Validation warning [{ticker}]: {warn}")

    return result


def _check_minimum_rows(df: pd.DataFrame, result: ValidationResult) -> None:
    if len(df) < MIN_ROWS:
        result.errors.append(
            f"Too few rows: {len(df)} (minimum {MIN_ROWS})"
        )


def _check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.errors.append(f"Missing required columns: {missing}")


def _check_null_fraction(df: pd.DataFrame, result: ValidationResult) -> None:
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    for col in price_cols:
        null_frac = df[col].isnull().mean()
        if null_frac > MAX_NULL_FRACTION:
            result.errors.append(
                f"Column '{col}' has {null_frac:.1%} nulls (max allowed: {MAX_NULL_FRACTION:.0%})"
            )
        elif null_frac > 0:
            result.warnings.append(
                f"Column '{col}' has {null_frac:.1%} nulls"
            )


def _check_price_ranges(df: pd.DataFrame, result: ValidationResult) -> None:
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        if (col_data < MIN_PRICE).any():
            result.errors.append(
                f"Column '{col}' has values below minimum price {MIN_PRICE}"
            )
        if (col_data > MAX_PRICE).any():
            result.warnings.append(
                f"Column '{col}' has unusually high values (>{MAX_PRICE})"
            )


def _check_volume(df: pd.DataFrame, result: ValidationResult) -> None:
    if "Volume" not in df.columns:
        return
    vol = df["Volume"].dropna()
    if (vol < MIN_VOLUME).any():
        result.errors.append("Volume has negative values")
    zero_vol_frac = (vol == 0).mean()
    if zero_vol_frac > 0.1:
        result.warnings.append(
            f"{zero_vol_frac:.1%} of Volume rows are zero"
        )


def _check_ohlc_consistency(df: pd.DataFrame, result: ValidationResult) -> None:
    required = {"High", "Low", "Open", "Close"}
    if not required.issubset(df.columns):
        return
    df_clean = df[list(required)].dropna()
    # High must be >= Open, Close, Low
    bad_high = (df_clean["High"] < df_clean[["Open", "Close", "Low"]].max(axis=1)).sum()
    if bad_high > 0:
        result.errors.append(
            f"{bad_high} rows where High < max(Open, Close, Low)"
        )
    # Low must be <= Open, Close, High
    bad_low = (df_clean["Low"] > df_clean[["Open", "Close", "High"]].min(axis=1)).sum()
    if bad_low > 0:
        result.errors.append(
            f"{bad_low} rows where Low > min(Open, Close, High)"
        )


def _check_index_type(df: pd.DataFrame, result: ValidationResult) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        result.errors.append(
            f"Index must be DatetimeIndex, got {type(df.index).__name__}"
        )
    elif df.index.tz is not None:
        result.warnings.append("DatetimeIndex is timezone-aware; expected tz-naive UTC")
