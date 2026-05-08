"""
Ray Serve — MobileNet V3 Small Image Classifier

Real Ray Serve deployment using @serve.deployment decorator.
Accepts image data (base64 or file path) and returns multi-class predictions.

  GET  /health    — liveness probe
  POST /predict   — classify an image (multi-label, MobileNet V3 Small)
  GET  /reload-model — hot-reload model from disk
"""

import os
import sys
import io
import json
import base64
import logging
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, "/app")

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/models/image_classifier.pt")
CLASSES_PATH = "/app/models/class_names.json"
HDFS_MODEL  = "/photos/models/image_classifier/model.pt"
IMAGE_SIZE  = 224
THRESHOLD   = 0.5

logger = logging.getLogger("ray-serve")

# ─── Try Ray Serve deployment, fallback to plain FastAPI ─────────────────────

try:
    import ray
    from ray import serve
    RAY_SERVE_AVAILABLE = True
except ImportError:
    RAY_SERVE_AVAILABLE = False
    logger.warning("ray[serve] not installed — running as plain FastAPI")

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Photo Classifier — Ray Serve", version="2.0.0")


# ─── Model loading ───────────────────────────────────────────────────────────

_model = None
_classes = []
_model_loaded_at = None
_transform = None


def _load_model():
    global _model, _classes, _model_loaded_at, _transform

    model_path = Path(MODEL_PATH)

    # Try HDFS download if local model missing
    if not model_path.exists():
        try:
            from src.common.hdfs import HDFSClient
            client = HDFSClient()
            if client.exists(HDFS_MODEL):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                data = client.get(HDFS_MODEL)
                model_path.write_bytes(data)
                print(f"Downloaded model from HDFS → {MODEL_PATH}")
        except Exception as e:
            print(f"Could not download model from HDFS: {e}")

    if model_path.exists():
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            from torchvision import transforms
            import torch.nn as nn

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            classes = checkpoint.get("classes", [])
            num_classes = checkpoint.get("num_classes", len(classes))
            threshold = checkpoint.get("threshold", THRESHOLD)

            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            _model = model
            _classes = classes
            _model_loaded_at = datetime.datetime.utcnow().isoformat()

            _transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            print(f"MobileNet V3 Small loaded ({num_classes} classes) at {_model_loaded_at}")
            return
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")

    # Try legacy sklearn model
    legacy_path = Path("/app/models/image_classifier.pkl")
    if legacy_path.exists():
        try:
            import pickle
            with open(legacy_path, "rb") as f:
                _model = pickle.load(f)
            _classes = []
            _model_loaded_at = datetime.datetime.utcnow().isoformat()
            print(f"Legacy sklearn model loaded at {_model_loaded_at}")
            return
        except Exception as e:
            print(f"Error loading legacy model: {e}")

    print(f"Warning: no model found. Using fallback.")
    _model = None


def _predict_image(image_bytes: bytes) -> dict:
    """Run MobileNet inference on raw image bytes."""
    import torch
    from PIL import Image

    if _model is None or _transform is None or not _classes:
        return {"predicted_labels": [], "confidence": 0.5, "method": "fallback"}

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    # Multi-label: return all labels above threshold
    predicted = []
    for i, (cls, prob) in enumerate(zip(_classes, probs)):
        if prob >= THRESHOLD:
            predicted.append({"label": cls, "confidence": round(float(prob), 4)})

    # Also return top-5 regardless of threshold
    top_indices = np.argsort(probs)[::-1][:5]
    top5 = [{"label": _classes[i], "confidence": round(float(probs[i]), 4)} for i in top_indices]

    # Primary category = highest confidence label
    primary = _classes[top_indices[0]] if len(top_indices) > 0 else "general"
    max_conf = float(probs[top_indices[0]]) if len(top_indices) > 0 else 0.5

    return {
        "predicted_labels": predicted if predicted else top5[:3],
        "top5": top5,
        "predicted_category": primary,
        "confidence": round(max_conf, 4),
        "method": "mobilenet_v3_small",
    }


def _predict_text_fallback(caption: str, labels: list) -> dict:
    """Fallback for legacy sklearn model or no model."""
    import pickle

    if _model is not None and hasattr(_model, "predict"):
        # Legacy sklearn path
        text = " ".join([caption or ""] + (labels or []))
        try:
            pred = _model.predict([text])[0]
            if hasattr(_model, "predict_proba"):
                proba = _model.predict_proba([text])[0]
                conf = float(max(proba))
            else:
                conf = 0.85
            return {
                "predicted_category": str(pred),
                "confidence": round(conf, 4),
                "predicted_labels": [{"label": str(pred), "confidence": round(conf, 4)}],
                "top5": [{"label": str(pred), "confidence": round(conf, 4)}],
                "method": "sklearn_legacy",
            }
        except Exception:
            pass

    return {
        "predicted_category": "general",
        "confidence": 0.5,
        "predicted_labels": [],
        "top5": [],
        "method": "fallback",
    }


# ─── Request/Response models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_id: str
    image_uri: Optional[str] = ""
    image_base64: Optional[str] = ""  # NEW: base64-encoded image data
    caption: Optional[str] = ""
    labels: Optional[list] = []


class PredictResponse(BaseModel):
    image_id: str
    predicted_category: str
    confidence: float
    labels: list
    predicted_labels: Optional[list] = []
    top5: Optional[list] = []
    method: Optional[str] = ""


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ray-serve-photo-classifier",
        "model_type": "MobileNet_V3_Small" if (_model is not None and _transform is not None) else "sklearn_or_fallback",
        "model_loaded": _model is not None,
        "model_loaded_at": _model_loaded_at,
        "classes": _classes[:10],
        "num_classes": len(_classes),
        "model_path": MODEL_PATH,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify an image using MobileNet V3 Small (multi-label).

    Priority:
      1. image_base64 — decode and classify raw pixels
      2. image_uri / file_name — try to load from local disk
      3. caption + labels — text fallback (legacy sklearn)
    """
    result = None

    # Try image-based prediction (MobileNet)
    if _transform is not None:
        image_bytes = None

        # Option 1: base64 encoded image
        if req.image_base64:
            try:
                image_bytes = base64.b64decode(req.image_base64)
            except Exception:
                pass

        # Option 2: load from local image directory
        if not image_bytes:
            for img_dir in ["/app/data/mirflickr/images", "/app/outputs/uploads"]:
                for ext in ["", ".jpg", ".jpeg", ".png"]:
                    candidate = Path(img_dir) / f"{req.image_id}{ext}"
                    if candidate.exists():
                        image_bytes = candidate.read_bytes()
                        break
                    # Also try file_name from URI
                    if req.image_uri:
                        fname = req.image_uri.split("/")[-1]
                        candidate = Path(img_dir) / fname
                        if candidate.exists():
                            image_bytes = candidate.read_bytes()
                            break
                if image_bytes:
                    break

        if image_bytes:
            result = _predict_image(image_bytes)

    # Fallback to text-based
    if result is None:
        result = _predict_text_fallback(req.caption or "", req.labels or [])

    return PredictResponse(
        image_id=req.image_id,
        predicted_category=result.get("predicted_category", "general"),
        confidence=result.get("confidence", 0.5),
        labels=[p["label"] for p in result.get("predicted_labels", [])] or req.labels or ["general"],
        predicted_labels=result.get("predicted_labels", []),
        top5=result.get("top5", []),
        method=result.get("method", "unknown"),
    )


@app.get("/reload-model")
def reload_model():
    _load_model()
    return {"reloaded": _model is not None, "at": _model_loaded_at}


# ─── Ray Serve deployment ────────────────────────────────────────────────────

if RAY_SERVE_AVAILABLE:
    @serve.deployment(
        name="photo-classifier",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
    )
    @serve.ingress(app)
    class PhotoClassifier:
        """Ray Serve deployment wrapping the FastAPI app."""
        pass

    # This is used by `serve run serve.ray_serve_app:deployment`
    deployment = PhotoClassifier.bind()


# ─── Direct entry point (fallback when Ray Serve CLI not available) ──────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
