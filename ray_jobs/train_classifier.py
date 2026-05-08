"""
Part 3: Ray Classifier Training Job — MobileNet V3 Small

Fine-tunes a pretrained MobileNet V3 Small on MIRFLICKR-25K images for
multi-class image classification.  Each image can have multiple labels
(sky, water, people, sunset, etc.) so we use BCE (sigmoid per class).

Reads:
  - Enriched Parquet from outputs/metadata/enriched/ (for label mapping)
  - Real MIRFLICKR images from data/mirflickr/images/

Produces:
  - /app/models/image_classifier.pt          — fine-tuned MobileNet V3 Small
  - /app/models/class_names.json             — ordered list of class names
  - /app/outputs/training_metrics.json       — accuracy, F1, per-class report
  - HDFS: /photos/models/image_classifier/   — model artifact copy
"""

import os
import sys
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/app")

# ─── Config ──────────────────────────────────────────────────────────────────

ENRICHED_LOCAL = "/app/outputs/metadata/enriched"
IMAGES_DIR     = "/app/data/mirflickr/images"
MODEL_PATH     = "/app/models/image_classifier.pt"
CLASSES_PATH   = "/app/models/class_names.json"
METRICS_PATH   = "/app/outputs/training_metrics.json"

BATCH_SIZE     = 16
EPOCHS         = 8
LEARNING_RATE  = 1e-3
IMAGE_SIZE     = 224
THRESHOLD      = 0.5          # sigmoid threshold for multi-label prediction

# MIRFLICKR-25K annotation categories (same order used during training)
ALL_CATEGORIES = [
    "animals", "baby", "bird", "car", "clouds", "dog", "female", "flower",
    "food", "indoor", "lake", "male", "night", "people", "plant_life",
    "portrait", "river", "sea", "sky", "structures", "sunset",
    "transport", "tree", "water",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except Exception:
            return [val] if val.strip() else []
    return []


# ─── Load enriched metadata ─────────────────────────────────────────────────

def load_enriched() -> pd.DataFrame:
    local = Path(ENRICHED_LOCAL)
    if local.exists() and any(local.glob("**/*.parquet")):
        files = list(local.glob("**/*.parquet"))
        print(f"Reading {len(files)} enriched parquet file(s).")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    try:
        from src.common.hdfs import HDFSClient
        import pyarrow as pa, pyarrow.parquet as pq
        client = HDFSClient()

        def _list_rec(path):
            out = []
            for item in client.listdir(path):
                full = f"{path}/{item['pathSuffix']}"
                if item["type"] == "DIRECTORY":
                    out.extend(_list_rec(full))
                elif item["pathSuffix"].endswith(".parquet"):
                    out.append(full)
            return out

        paths = _list_rec("/photos/metadata/enriched")
        if paths:
            dfs = []
            for p in paths:
                buf = pa.BufferReader(pa.py_buffer(client.get(p)))
                dfs.append(pq.read_table(buf).to_pandas())
            return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"HDFS fallback failed: {e}")

    raise FileNotFoundError("No enriched metadata. Run `make part2` first.")


# ─── Build dataset ───────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    """Build image paths and multi-hot label arrays from enriched metadata."""
    import torch
    from torchvision import transforms
    from torch.utils.data import Dataset

    # Determine which categories are actually present
    all_labels = set()
    for _, row in df.iterrows():
        tags = _safe_list(row.get("tags"))
        ann = _safe_list(row.get("annotation_labels"))
        all_labels.update(t.lower().strip() for t in tags + ann if t)

    # Use the standard MIRFLICKR categories that appear in our data
    classes = [c for c in ALL_CATEGORIES if c in all_labels]
    if not classes:
        # Fallback: use whatever labels we have
        classes = sorted(all_labels)[:24]
    if not classes:
        classes = ["general"]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Training with {num_classes} classes: {classes}")

    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    class MirflickrDataset(Dataset):
        def __init__(self, rows, transform):
            self.rows = rows
            self.transform = transform
            self.num_classes = num_classes
            self.class_to_idx = class_to_idx

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            from PIL import Image
            row = self.rows[idx]
            img_path = Path(IMAGES_DIR) / row["file_name"]

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)

            # Multi-hot label vector
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            tags = _safe_list(row.get("tags"))
            ann = _safe_list(row.get("annotation_labels"))
            for t in tags + ann:
                t_lower = t.lower().strip()
                if t_lower in self.class_to_idx:
                    label_vec[self.class_to_idx[t_lower]] = 1.0

            return img, label_vec

    # Filter to rows that have valid image files
    valid_rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        fn = r.get("file_name", "")
        if fn and (Path(IMAGES_DIR) / fn).exists():
            valid_rows.append(r)

    print(f"Valid images with files: {len(valid_rows)}")

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_rows, val_rows = train_test_split(valid_rows, test_size=0.2, random_state=42)

    train_ds = MirflickrDataset(train_rows, train_transform)
    val_ds = MirflickrDataset(val_rows, val_transform)

    return train_ds, val_ds, classes, class_to_idx


# ─── Train MobileNet V3 Small ───────────────────────────────────────────────

def train(df: pd.DataFrame):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    train_ds, val_ds, classes, class_to_idx = build_dataset(df)
    num_classes = len(classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load pretrained MobileNet V3 Small and replace classifier head
    print("Loading pretrained MobileNet V3 Small...")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace final classifier: 576 → num_classes (multi-label, sigmoid output)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Freeze early layers, fine-tune classifier + last few conv layers
    for param in model.features[:8].parameters():
        param.requires_grad = False

    device = torch.device("cpu")  # CPU training for demo
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"  Classes: {num_classes} | Batch size: {BATCH_SIZE}")

    best_f1 = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > THRESHOLD).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Metrics
        from sklearn.metrics import f1_score, accuracy_score
        f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        f1_micro = f1_score(all_targets, all_preds, average="micro", zero_division=0)
        # Subset accuracy (exact match)
        exact_match = accuracy_score(all_targets, all_preds)
        # Per-sample accuracy (fraction of correct labels)
        per_sample = np.mean(np.sum(all_preds == all_targets, axis=1) / num_classes)

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, "
              f"F1_macro={f1_macro:.3f}, F1_micro={f1_micro:.3f}, exact_match={exact_match:.3f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = model.state_dict().copy()

    # Use best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            preds = (torch.sigmoid(outputs) > THRESHOLD).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    from sklearn.metrics import f1_score, classification_report
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(
        all_targets, all_preds,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    print(f"\nFinal evaluation:")
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    metrics = {
        "model": "MobileNet_V3_Small",
        "task": "multi_label_classification",
        "num_classes": num_classes,
        "classes": classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "train_size": len(train_ds),
        "test_size": len(val_ds),
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "best_f1_macro": round(best_f1, 4),
        "classification_report": report,
        "trained_at": datetime.datetime.utcnow().isoformat(),
    }

    return model, classes, metrics


# ─── Save ────────────────────────────────────────────────────────────────────

def save_model(model, classes: list, metrics: dict):
    import torch

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "num_classes": len(classes),
        "threshold": THRESHOLD,
    }, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")

    # Save class names separately (for Ray Serve)
    Path(CLASSES_PATH).write_text(json.dumps(classes, indent=2))
    print(f"Saved class names → {CLASSES_PATH}")

    # Save metrics
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {METRICS_PATH}")

    # Upload to HDFS
    try:
        from src.common.hdfs import HDFSClient
        client = HDFSClient()
        client.mkdirs("/photos/models/image_classifier")
        client.put("/photos/models/image_classifier/model.pt", Path(MODEL_PATH))
        client.put("/photos/models/image_classifier/class_names.json", Path(CLASSES_PATH))
        print("Uploaded model → HDFS /photos/models/image_classifier/")
    except Exception as e:
        print(f"Warning: HDFS upload failed ({e}). Local copy available.")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Ray Classifier Training: MobileNet V3 Small ===")
    df = load_enriched()
    print(f"Loaded {len(df)} enriched rows.")
    model, classes, metrics = train(df)
    save_model(model, classes, metrics)
    print("Done.")
