from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI

try:
    from ray import serve
except Exception:  # pragma: no cover
    serve = None

from src.common import hdfs
from src.common.embeddings import record_embedding
from src.common.image_utils import infer_fallback_category, infer_fallback_labels, make_caption

MODEL_HDFS_PATH = os.getenv("MODEL_HDFS_PATH", "/photos/models/image_classifier/model.pkl")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))

api = FastAPI(title="Big Photos Ray Serve API")


def make_features(image_id: str, image_uri: str, labels: List[str] | None = None, category: str | None = None) -> List[float]:
    labels = labels or infer_fallback_labels(image_id + image_uri)
    category = category or infer_fallback_category(image_id + image_uri)
    caption = make_caption(category, labels, image_id)
    emb = record_embedding(caption=caption, labels=labels, category=category, dim=EMBEDDING_DIM)
    # Ray Serve endpoint is intentionally lightweight for CPU demos. Full image
    # pixels can be read here, but the trained demo model expects embedding +
    # normalized width/height/size. Unknown image dimensions use zeros.
    return emb + [0.0, 0.0, 0.0]


class PredictorImpl:
    def __init__(self):
        self.payload = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                local = tmp.name
            hdfs.download_to_local(MODEL_HDFS_PATH, local)
            with open(local, "rb") as f:
                self.payload = pickle.load(f)
            print(f"Loaded model from {MODEL_HDFS_PATH}")
        except Exception as exc:
            print(f"Model unavailable, using deterministic fallback: {exc}")
            self.payload = None

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_id = str(data.get("image_id") or "unknown")
        image_uri = str(data.get("image_uri") or "")
        fallback_labels = infer_fallback_labels(image_id + image_uri)
        fallback_category = infer_fallback_category(image_id + image_uri)
        if self.payload and self.payload.get("model") is not None:
            try:
                x = np.asarray([make_features(image_id, image_uri, fallback_labels, fallback_category)], dtype=np.float32)
                pred = self.payload["model"].predict(x)[0]
                proba = self.payload["model"].predict_proba(x)[0]
                confidence = float(np.max(proba))
                labels = list(dict.fromkeys([str(pred)] + fallback_labels))[:5]
                return {"image_id": image_id, "predicted_category": str(pred), "confidence": confidence, "labels": labels}
            except Exception as exc:
                print(f"Prediction fallback for {image_id}: {exc}")
        return {"image_id": image_id, "predicted_category": fallback_category, "confidence": 0.5, "labels": fallback_labels}


if serve is not None:
    @serve.deployment(route_prefix="/")
    @serve.ingress(api)
    class PredictorDeployment(PredictorImpl):
        @api.post("/predict")
        async def predict_endpoint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return self.predict(payload)

    app = PredictorDeployment.bind()
else:
    app = None
