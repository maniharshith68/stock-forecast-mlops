#!/usr/bin/env python3
"""
Upload trained models and processed data to S3.
Run this after training to back up artifacts.

Usage:
    python3 scripts/upload_to_s3.py
    python3 scripts/upload_to_s3.py --bucket my-bucket --prefix myproject
"""
import sys
import argparse
import boto3
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils.logger import get_logger

logger = get_logger("scripts.upload_to_s3")

DEFAULT_BUCKET = "stock-forecasting-pipeline"
DEFAULT_PREFIX = "stock-forecast-mlops"


def upload_file(s3_client, local_path: Path, bucket: str, s3_key: str) -> bool:
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Uploaded: {local_path} → s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False


def upload_directory(
    s3_client,
    local_dir:  Path,
    bucket:     str,
    s3_prefix:  str,
) -> dict:
    """Upload all files in a directory recursively."""
    uploaded, failed = 0, 0

    if not local_dir.exists():
        logger.warning(f"Directory not found, skipping: {local_dir}")
        return {"uploaded": 0, "failed": 0}

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip large raw data files — only upload processed + models
        if any(p in str(file_path) for p in ["__pycache__", ".pyc", ".git"]):
            continue

        relative = file_path.relative_to(local_dir.parent)
        s3_key   = f"{s3_prefix}/{relative}"

        if upload_file(s3_client, file_path, bucket, s3_key):
            uploaded += 1
        else:
            failed += 1

    return {"uploaded": uploaded, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Upload artifacts to S3")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip uploading processed data (only upload models)"
    )
    args = parser.parse_args()

    logger.info(f"Connecting to S3 | bucket={args.bucket} | prefix={args.prefix}")

    try:
        s3 = boto3.client("s3", region_name=args.region)
        # Verify bucket exists and is accessible
        s3.head_bucket(Bucket=args.bucket)
        logger.info(f"Bucket accessible: s3://{args.bucket}")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Run: aws configure")
        sys.exit(1)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.info(f"Bucket not found, creating: {args.bucket}")
            s3.create_bucket(Bucket=args.bucket)
        else:
            logger.error(f"Cannot access bucket: {e}")
            sys.exit(1)

    total_uploaded = 0
    total_failed   = 0

    # Upload trained models
    logger.info("Uploading models/registry/ ...")
    result = upload_directory(
        s3, Path("models/registry"), args.bucket, args.prefix
    )
    total_uploaded += result["uploaded"]
    total_failed   += result["failed"]

    # Upload processed data (features, splits, scalers)
    if not args.skip_data:
        logger.info("Uploading data/processed/ ...")
        result = upload_directory(
            s3, Path("data/processed"), args.bucket, args.prefix
        )
        total_uploaded += result["uploaded"]
        total_failed   += result["failed"]

    # Upload MLflow database
    mlflow_db = Path("mlflow.db")
    if mlflow_db.exists():
        logger.info("Uploading mlflow.db ...")
        s3_key = f"{args.prefix}/mlflow.db"
        if upload_file(s3, mlflow_db, args.bucket, s3_key):
            total_uploaded += 1
        else:
            total_failed += 1

    print(f"\n{'='*55}")
    print(f"S3 Upload Complete")
    print(f"{'='*55}")
    print(f"Bucket:   s3://{args.bucket}/{args.prefix}/")
    print(f"Uploaded: {total_uploaded} files")
    print(f"Failed:   {total_failed} files")
    print(f"{'='*55}")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
