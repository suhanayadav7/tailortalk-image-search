import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "images"
CSV_PATH = ROOT / "data" / "catalogue.csv"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/dinov2-small"
device = "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

df = pd.read_csv(CSV_PATH).dropna(subset=["SKU", "image_url"])
rows = {str(r.SKU): r._asdict() for r in df.itertuples(index=False)}

paths = sorted(DATA_DIR.glob("*"))
valid = []
for path in paths:
    sku = path.stem
    if sku in rows:
        valid.append(path)

if not valid:
    raise SystemExit("No catalogue images found. Run: python scripts/download_dataset.py")

def embed(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(vec, dim=-1)
    return vec[0].cpu().numpy().astype("float32")

vectors = []
metadata = []

for i, path in enumerate(valid, 1):
    try:
        vectors.append(embed(path))
        row = rows[path.stem]
        metadata.append({
            "path": str(path.as_posix()),
            "filename": path.name,
            "name": str(row.get("Name", "")),
            "sku": str(row.get("SKU", "")),
            "stock": int(row.get("Stock", 0)) if pd.notna(row.get("Stock")) else 0,
            "retail_price": float(row.get("Retail Price", 0)),
            "discounted_price": float(row.get("Discounted Price", 0)),
            "image_url": str(row.get("image_url", "")),
            "website_link": str(row.get("Website Link", "")),
        })
        print(f"[{i}/{len(valid)}] indexed {path.name}")
    except Exception as exc:
        print(f"Skipping {path}: {exc}")

matrix = np.vstack(vectors).astype("float32")
index = faiss.IndexFlatIP(matrix.shape[1])
index.add(matrix)
faiss.write_index(index, str(OUT_DIR / "index.faiss"))
(OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

print(f"Indexed {len(metadata)} images. Dimension={matrix.shape[1]}")
