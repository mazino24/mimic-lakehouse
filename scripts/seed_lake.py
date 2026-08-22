#!/usr/bin/env python3
"""Upload raw CSVs into the MinIO/S3 raw zone.

In production this is whatever the hospital's extract process is — an SFTP
drop, a DataSync job, a `gcloud storage rsync`. Locally it is this script, so
the pipeline always reads from object storage and never from a laptop path.

    python scripts/seed_lake.py --source-dir data/raw
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

CSV_SUFFIXES = (".csv", ".csv.gz")


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT", "http://localhost:9000")),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"created bucket {bucket}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bucket", default=os.getenv("LAKE_BUCKET", "mimic-lake"))
    parser.add_argument("--prefix", default="raw")
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(
            f"{args.source_dir} does not exist — run "
            "`python scripts/generate_synthetic_mimic.py` first"
        )

    s3 = client()
    ensure_bucket(s3, args.bucket)

    uploaded = 0
    total_bytes = 0
    for path in sorted(args.source_dir.rglob("*")):
        if not path.is_file() or not path.name.lower().endswith(CSV_SUFFIXES):
            continue
        key = f"{args.prefix}/{path.relative_to(args.source_dir).as_posix()}"
        size = path.stat().st_size
        print(f"  {path.name:<28} -> s3://{args.bucket}/{key}  ({size / 1e6:.1f} MB)")
        s3.upload_file(str(path), args.bucket, key)
        uploaded += 1
        total_bytes += size

    print(f"uploaded {uploaded} file(s), {total_bytes / 1e6:.1f} MB total")
    return 0 if uploaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
