import json
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image


class SimilarityEngine:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        self.metadata = json.loads(Path(metadata_path).read_text())
        self.project_root = Path(__file__).resolve().parents[1]

    def color_histogram(self, image, bins=16):
        image = image.convert("HSV").resize((256, 256))
        arr = np.asarray(image)
        h = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256), density=True)[0]
        s = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256), density=True)[0]
        v = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256), density=True)[0]
        x = np.concatenate([h, s, v]).astype("float32")
        return x / (np.linalg.norm(x) + 1e-8)

    def embed_query(self, image_path):
        worker = self.project_root / "scripts" / "query_embed.py"
        result = subprocess.run(
            [sys.executable, str(worker), str(image_path)],
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError("DINOv2 worker failed:\n" + (result.stderr[-3000:] or "No error output."))
        values = np.fromstring(result.stdout.strip(), sep=",", dtype="float32")
        if values.size != self.index.d:
            raise RuntimeError(f"Embedding dimension mismatch: got {values.size}, expected {self.index.d}")
        return values.reshape(1, -1)

    def search(self, query_image, top_k=6, candidate_k=40):
        qvec = self.embed_query(query_image)
        distances, indices = self.index.search(qvec, min(candidate_k, self.index.ntotal))
        qhist = self.color_histogram(Image.open(query_image).convert("RGB"))
        candidates = []
        for d, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            item = self.metadata[int(idx)]
            ref = Image.open(self.project_root / item["path"]).convert("RGB")
            color_score = float(np.dot(qhist, self.color_histogram(ref)))
            visual_score = float(np.clip((d + 1.0) / 2.0, 0.0, 1.0))
            final = 0.82 * visual_score + 0.18 * color_score
            candidates.append({
                **item,
                "score": final,
                "visual_score": visual_score,
                "color_score": color_score,
                "path": str((self.project_root / item["path"]).resolve()),
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
