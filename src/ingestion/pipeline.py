from datetime import date
from src.utils.logger import get_logger
from src.utils.config import get, load_config
from src.ingestion.downloader import download_multiple_tickers
from src.ingestion.validator import validate_ohlcv
from src.ingestion.s3_uploader import store_ohlcv

logger = get_logger("ingestion.pipeline")


def run_ingestion_pipeline(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Full ingestion pipeline:
      1. Download OHLCV from yFinance
      2. Validate each ticker's data
      3. Store locally (and to S3 if credentials present)

    Returns summary dict with per-ticker outcomes.
    """
    config = load_config()

    tickers = tickers or config["data"]["tickers"]
    start_date = start_date or config["data"]["start_date"]
    end_date = end_date or date.today().isoformat()
    interval = config["data"].get("interval", "1d")

    logger.info(
        f"Starting ingestion pipeline | tickers={tickers} | "
        f"{start_date} → {end_date} | interval={interval}"
    )

    # Step 1: Download
    raw_data = download_multiple_tickers(tickers, start_date, end_date, interval)

    summary = {
        "total": len(tickers),
        "downloaded": len(raw_data),
        "validated": 0,
        "stored": 0,
        "failed_validation": [],
        "failed_storage": [],
        "outcomes": {},
    }

    today_str = date.today().isoformat()

    for ticker, df in raw_data.items():
        # Step 2: Validate
        val_result = validate_ohlcv(df, ticker)
        if not val_result.passed:
            logger.error(
                f"Validation failed for {ticker}: {val_result.errors}"
            )
            summary["failed_validation"].append(ticker)
            summary["outcomes"][ticker] = {
                "validation": "FAIL",
                "errors": val_result.errors,
            }
            continue

        summary["validated"] += 1

        # Step 3: Store
        store_outcome = store_ohlcv(df, ticker, today_str)
        summary["outcomes"][ticker] = {
            "validation": "PASS",
            "rows": val_result.row_count,
            "local": store_outcome["local"],
            "s3": store_outcome["s3"],
        }
        summary["stored"] += 1
        logger.info(f"Pipeline complete for {ticker}: {store_outcome}")

    logger.info(
        f"Ingestion pipeline finished | "
        f"downloaded={summary['downloaded']} | "
        f"validated={summary['validated']} | "
        f"stored={summary['stored']} | "
        f"failed_validation={summary['failed_validation']}"
    )
    return summary
