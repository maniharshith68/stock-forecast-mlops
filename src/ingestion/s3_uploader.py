import os
import io
from pathlib import Path
import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger("ingestion.s3_uploader")

LOCAL_FALLBACK_DIR = Path("data/raw")


def save_locally(df: pd.DataFrame, ticker: str, date_str: str) -> Path:
    """
    Save DataFrame as parquet to data/raw/{ticker}/{date_str}.parquet.
    Always succeeds as long as disk has space.
    """
    out_dir = LOCAL_FALLBACK_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.parquet"
    df.to_parquet(out_path, index=True)
    logger.info(f"Saved locally: {out_path} ({len(df)} rows)")
    return out_path


def upload_to_s3(
    df: pd.DataFrame,
    ticker: str,
    date_str: str,
    bucket: str,
    prefix: str,
) -> bool:
    """
    Upload DataFrame as parquet to S3.
    Returns True on success, False on failure.
    Falls back gracefully — never raises.
    """
    s3_key = f"{prefix}/{ticker}/{date_str}.parquet"

    try:
        s3_client = boto3.client("s3")
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.info(f"Uploaded to s3://{bucket}/{s3_key}")
        return True

    except NoCredentialsError:
        logger.warning(
            "AWS credentials not found — skipping S3 upload. "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env to enable."
        )
        return False
    except ClientError as e:
        logger.error(f"S3 ClientError uploading {ticker}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error uploading {ticker} to S3: {e}")
        return False


def store_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    date_str: str,
) -> dict:
    """
    Primary storage function.
    Always saves locally. Attempts S3 upload if credentials are available.
    Returns a dict with storage outcomes.
    """
    outcome = {"ticker": ticker, "date_str": date_str, "local": None, "s3": False}

    # Always save locally
    local_path = save_locally(df, ticker, date_str)
    outcome["local"] = str(local_path)

    # Attempt S3 only if credentials are configured
    bucket = get("aws.s3_bucket")
    prefix = get("data.raw_s3_prefix", "raw/ohlcv")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")

    if aws_key:
        outcome["s3"] = upload_to_s3(df, ticker, date_str, bucket, prefix)
    else:
        logger.info("AWS credentials not set — local storage only (expected for now)")

    return outcome
