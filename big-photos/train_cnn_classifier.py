"""
MIRFLICKR Semantic Image Classifier — Transfer Learning with ResNet-50
=======================================================================
Trains a CNN to classify images into semantic buckets derived from
the EDA pipeline (landscape_nature, portrait_people, urban_street,
animals, travel_event, food).  Uses the structured Parquet/CSV
produced by big-photos-eda.ipynb.

Prerequisites
-------------
pip install torch torchvision pandas scikit-learn matplotlib pillow tqdm

Usage
-----
python train_cnn_classifier.py \
    --data-root /path/to/mirflickr \
    --features-csv /path/to/mirflickr_features.csv \
    --epochs 25 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir ./model_output
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────
#  1.  CONFIGURATION & ARGUMENT PARSING
# ──────────────────────────────────────────────
TARGET_CLASSES = [
    "landscape_nature",
    "portrait_people",
    "urban_street",
    "animals",
    "travel_event",
    "food",
]

NUM_CLASSES = len(TARGET_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMG_SIZE = 224  # ResNet-50 default input size


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CNN image classifier on MIRFLICKR semantic buckets"
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing the MIRFLICKR images (im1.jpg … im25000.jpg)",
    )
    p.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Path to mirflickr_features.csv (output of EDA notebook)",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="Freeze ResNet backbone and only train the classifier head",
    )
    p.add_argument(
        "--unfreeze-after",
        type=int,
        default=5,
        help="Unfreeze backbone after N epochs (ignored if --freeze-backbone not set)",
    )
    p.add_argument("--output-dir", type=str, default="./model_output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto'",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
#  2.  DATASET
# ──────────────────────────────────────────────
class MIRFLICKRDataset(Dataset):
    """
    Loads MIRFLICKR images by photo_id and returns (image_tensor, label).
    Images are expected at: <data_root>/im<photo_id>.jpg
    """

    def __init__(self, df, data_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        photo_id = int(row["photo_id"])
        label = CLASS_TO_IDX[row["semantic_bucket"]]

        img_path = self.data_root / f"im{photo_id}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback: return a black image so training doesn't crash
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


# ──────────────────────────────────────────────
#  3.  TRANSFORMS
# ──────────────────────────────────────────────
def get_transforms():
    """ImageNet-normalised transforms with augmentation for training."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────
#  4.  MODEL
# ──────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=False, device="cpu"):
    """
    ResNet-50 with ImageNet pre-trained weights.
    Replaces the final FC layer for our 6-class classification task.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    return model.to(device)


def unfreeze_backbone(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("[INFO] Backbone unfrozen — full fine-tuning enabled")


# ──────────────────────────────────────────────
#  5.  TRAINING & VALIDATION LOOPS
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return running_loss / total, acc, f1, np.array(all_labels), np.array(all_preds)


# ──────────────────────────────────────────────
#  6.  VISUALIZATION HELPERS
# ──────────────────────────────────────────────
def plot_training_curves(history, output_dir):
    """Save loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {output_dir}/training_curves.png")


def plot_confusion_matrix(labels, preds, output_dir):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(TARGET_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(TARGET_CLASSES, fontsize=9)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {output_dir}/confusion_matrix.png")


# ──────────────────────────────────────────────
#  7.  DATA PREPARATION
# ──────────────────────────────────────────────
def prepare_data(features_csv, data_root, seed=42):
    """
    Load features CSV, filter to known semantic buckets,
    verify image files exist, and split 80/10/10.
    """
    print(f"[INFO] Loading features from {features_csv}")
    df = pd.read_csv(features_csv)

    # Filter to target classes only (exclude 'other' and 'unknown')
    df = df[df["semantic_bucket"].isin(TARGET_CLASSES)].copy()
    print(f"[INFO] {len(df)} images with known semantic labels")

    # Print class distribution
    dist = df["semantic_bucket"].value_counts()
    print("[INFO] Class distribution:")
    for cls, cnt in dist.items():
        print(f"       {cls:25s} → {cnt:,}")

    # Verify image files exist
    data_root = Path(data_root)
    valid_mask = df["photo_id"].apply(
        lambda pid: (data_root / f"im{int(pid)}.jpg").exists()
    )
    df = df[valid_mask].copy()
    print(f"[INFO] {len(df)} images verified on disk")

    if len(df) == 0:
        print("[ERROR] No valid images found. Check --data-root path.")
        sys.exit(1)

    # Stratified split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["semantic_bucket"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["semantic_bucket"], random_state=seed
    )

    print(f"[INFO] Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
#  8.  CLASS WEIGHTS (handle imbalanced data)
# ──────────────────────────────────────────────
def compute_class_weights(train_df, device):
    """Inverse frequency weighting for imbalanced semantic buckets."""
    counts = train_df["semantic_bucket"].value_counts()
    total = len(train_df)
    weights = []
    for cls in TARGET_CLASSES:
        cnt = counts.get(cls, 1)
        weights.append(total / (NUM_CLASSES * cnt))
    weights_tensor = torch.FloatTensor(weights).to(device)
    print(f"[INFO] Class weights: {dict(zip(TARGET_CLASSES, [f'{w:.2f}' for w in weights]))}")
    return weights_tensor


# ──────────────────────────────────────────────
#  9.  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    train_df, val_df, test_df = prepare_data(
        args.features_csv, args.data_root, args.seed
    )

    train_tf, val_tf = get_transforms()

    train_ds = MIRFLICKRDataset(train_df, args.data_root, train_tf)
    val_ds = MIRFLICKRDataset(val_df, args.data_root, val_tf)
    test_ds = MIRFLICKRDataset(test_df, args.data_root, val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Model ──
    model = build_model(
        NUM_CLASSES, freeze_backbone=args.freeze_backbone, device=device
    )
    print(f"[INFO] Model: ResNet-50 | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    # ── Loss & Optimizer ──
    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training Loop ──
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_val_f1 = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  Training for {args.epochs} epochs")
    print(f"  Batch size: {args.batch_size} | LR: {args.lr}")
    print(f"  Backbone frozen: {args.freeze_backbone}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after N epochs if requested
        if args.freeze_backbone and epoch == args.unfreeze_after + 1:
            unfreeze_backbone(model)
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )

        print(f"Epoch {epoch}/{args.epochs}  (LR: {scheduler.get_last_lr()[0]:.2e})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "class_to_idx": CLASS_TO_IDX,
                },
                output_dir / "best_model.pth",
            )
            print(f"  ★ New best model saved (F1: {val_f1:.4f})")

        print()

    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")

    # ── Evaluate on Test Set ──
    print("[INFO] Loading best model for test evaluation...")
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"  Accuracy: {test_acc:.4f} | F1 (weighted): {test_f1:.4f}")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(
        classification_report(
            test_labels, test_preds, target_names=TARGET_CLASSES, digits=4
        )
    )

    # ── Save Outputs ──
    plot_training_curves(history, str(output_dir))
    plot_confusion_matrix(test_labels, test_preds, str(output_dir))

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))

    # Save test results
    results = {
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(test_f1),
        "test_loss": float(test_loss),
        "best_epoch": best_epoch,
        "best_val_f1": float(best_val_f1),
        "total_epochs": args.epochs,
        "model": "ResNet-50 (ImageNet V2)",
        "num_classes": NUM_CLASSES,
        "classes": TARGET_CLASSES,
        "training_time_minutes": round(elapsed / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] All outputs saved to {output_dir}/")
    print(f"       ├── best_model.pth")
    print(f"       ├── training_curves.png")
    print(f"       ├── confusion_matrix.png")
    print(f"       ├── training_history.json")
    print(f"       └── test_results.json")


if __name__ == "__main__":
    main()
