import argparse
import subprocess
from pathlib import Path

import json
import gzip
import random
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LISTINGS_DIR = DATA_DIR / "listings"
FURNITURE_TYPES = {
    "SOFA",
    "CHAIR",
    "TABLE",
    "BED",
    "DESK",
    "BOOKCASE",
    "CABINET"
}


def download_metadata():
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("ABO 메타데이터 다운로드 시작...")

    cmd = [
        "aws", "s3", "cp",
        "s3://amazon-berkeley-objects/listings/metadata/",
        str(LISTINGS_DIR),
        "--recursive",
        "--no-sign-request"
    ]

    subprocess.run(cmd, check=True)

    print("다운로드 완료")
    print(f"저장 위치: {LISTINGS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)

    args = parser.parse_args()

    if args.step == "download":
        download_metadata()
    else:
        print("지원하지 않는 step입니다.")
        print("사용 가능: download")


if __name__ == "__main__":
    main()