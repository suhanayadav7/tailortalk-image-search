import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "catalogue.csv"
IMAGE_DIR = ROOT / "data" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="Download only the first N rows; 0 means all.")
parser.add_argument("--delay", type=float, default=0.05)
args = parser.parse_args()

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["image_url"]).drop_duplicates(subset=["SKU"], keep="first")
if args.limit:
    df = df.head(args.limit)

session = requests.Session()
session.headers.update({"User-Agent": "TailorTalk/1.0"})

ok = failed = 0
for n, row in enumerate(df.itertuples(index=False), 1):
    url = str(row.image_url).strip()
    sku = str(row.SKU).strip()
    ext = ".webp"
    out = IMAGE_DIR / f"{sku}{ext}"

    if out.exists() and out.stat().st_size > 0:
        ok += 1
        continue

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        if "jpeg" in content_type or url.lower().endswith((".jpg", ".jpeg")):
            out = IMAGE_DIR / f"{sku}.jpg"
        elif "png" in content_type or url.lower().endswith(".png"):
            out = IMAGE_DIR / f"{sku}.png"
        out.write_bytes(r.content)
        ok += 1
        print(f"[{n}/{len(df)}] downloaded {sku}")
    except Exception as exc:
        failed += 1
        print(f"[{n}/{len(df)}] FAILED {sku}: {exc}")
    time.sleep(args.delay)

print(f"\nDone. Downloaded/available: {ok}; failed: {failed}; rows: {len(df)}")
