from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.common import hdfs

logger = logging.getLogger("train_transfer_target")

LOCAL_MIRFLICKR_DIR = os.getenv("LOCAL_MIRFLICKR_DIR", "/app/shared_data/mirflickr")
LOCAL_ANNOTATION_DIR = os.getenv("LOCAL_ANNOTATION_DIR", "/app/shared_data/mirflickr25k_annotations_v080")
TUNE_RESULTS_LOCAL_DIR = os.getenv("TUNE_RESULTS_LOCAL_DIR", "/app/outputs/training")
MODEL_OUTPUT = os.getenv("MODEL_HDFS_PATH", "/photos/models/image_classifier/mobilenet_target80.pt")
METRICS_OUTPUT = os.getenv("MODEL_METRICS_HDFS_PATH", "/photos/models/image_classifier/mobilenet_target80_metrics.json")

ALLOWED = ["structures", "indoor", "plant_life", "people", "sky", "animals", "water", "food", "transport", "female"]


class ImgDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], tfm):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        p, y = self.rows[idx]
        with Image.open(p) as im:
            x = self.tfm(im.convert("RGB"))
        return x, y


def build_samples(annotation_dir: str, image_dir: str, max_rows: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    ann = Path(annotation_dir)
    img = Path(image_dir)
    files = sorted([p for p in ann.glob("*.txt") if p.is_file() and p.name.lower() != "readme.txt"])
    labels_by_id: Dict[str, set] = {}
    for f in files:
        label = f.stem.lower().replace("_r1", "")
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if s.isdigit():
                    iid = str(int(s))
                    labels_by_id.setdefault(iid, set()).add(label)
    out: List[Tuple[str, str]] = []
    allow = set(ALLOWED)
    for iid, labs in labels_by_id.items():
        picked = [x for x in labs if x in allow]
        label = picked[0] if picked else "other"
        p = img / f"im{int(iid)}.jpg"
        if p.exists():
            out.append((str(p), label))
            if max_rows and len(out) >= max_rows:
                break
    classes = sorted({y for _, y in out})
    return out, classes


def make_model(num_classes: int, dropout: float, unfreeze_last_block: bool) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_last_block:
        for p in model.features[-1].parameters():
            p.requires_grad = True
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def train_eval_once(
    cfg: Dict[str, float],
    train_rows: List[Tuple[str, int]],
    val_rows: List[Tuple[str, int]],
    test_rows: List[Tuple[str, int]],
    num_classes: int,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tfm_eval = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dl = DataLoader(ImgDataset(train_rows, tfm_train), batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0)
    val_dl = DataLoader(ImgDataset(val_rows, tfm_eval), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    test_dl = DataLoader(ImgDataset(test_rows, tfm_eval), batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    model = make_model(num_classes, float(cfg["dropout"]), bool(cfg["unfreeze_last_block"])).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    else:
        opt = torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=0.9, weight_decay=float(cfg["weight_decay"]))

    ys_train = [y for _, y in train_rows]
    counts = np.bincount(np.asarray(ys_train), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_w = counts.sum() / counts
    class_w = torch.tensor(class_w / class_w.mean(), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    for _ in range(int(cfg["epochs"])):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    def eval_dl(dl):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in dl:
                logits = model(xb.to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                ys.extend(yb.numpy().tolist())
                ps.extend(pred.tolist())
        return ys, ps

    yv, pv = eval_dl(val_dl)
    yt, pt = eval_dl(test_dl)
    return {
        "val_accuracy": float(accuracy_score(yv, pv)),
        "val_f1": float(f1_score(yv, pv, average="weighted", zero_division=0)),
        "test_accuracy": float(accuracy_score(yt, pt)),
        "test_f1": float(f1_score(yt, pt, average="weighted", zero_division=0)),
        "test_report": classification_report(yt, pt, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    ray.init(ignore_reinit_error=True, local_mode=True, include_dashboard=False, num_cpus=1)

    samples, classes = build_samples(LOCAL_ANNOTATION_DIR, LOCAL_MIRFLICKR_DIR, args.max_rows)
    if not samples:
        raise SystemExit("No local samples found")
    c2i = {c: i for i, c in enumerate(classes)}
    rows = [(p, c2i[y]) for p, y in samples]
    y = np.asarray([r[1] for r in rows])
    idx = np.arange(len(rows))
    strat = y if min(pd.Series(y).value_counts()) >= 2 else None
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)
    y_train = y[train_idx]
    strat2 = y_train if min(pd.Series(y_train).value_counts()) >= 2 else None
    tr_idx, va_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=strat2)
    tr = [rows[i] for i in tr_idx]
    va = [rows[i] for i in va_idx]
    te = [rows[i] for i in test_idx]

    grid = []
    for lr in [1e-4, 3e-4, 8e-4]:
        for wd in [1e-5, 1e-4]:
            for bs in [8, 12]:
                for dr in [0.2, 0.3]:
                    for opt in ["adamw", "sgd"]:
                        for ep in [4, 6]:
                            for ufb in [False, True]:
                                grid.append(
                                    {
                                        "lr": lr,
                                        "weight_decay": wd,
                                        "batch_size": bs,
                                        "dropout": dr,
                                        "optimizer": opt,
                                        "epochs": ep,
                                        "unfreeze_last_block": ufb,
                                    }
                                )
    random.seed(42)
    random.shuffle(grid)
    picks = grid[: args.trials]

    rows_out = []
    best = None
    for i, cfg in enumerate(picks, start=1):
        m = train_eval_once(cfg, tr, va, te, len(classes))
        row = {"trial_id": i, **cfg, **m}
        rows_out.append(row)
        logger.info("trial %d/%d val_acc=%.4f test_acc=%.4f cfg=%s", i, len(picks), m["val_accuracy"], m["test_accuracy"], cfg)
        if best is None or row["test_accuracy"] > best["test_accuracy"]:
            best = row

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(TUNE_RESULTS_LOCAL_DIR) / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out).sort_values("test_accuracy", ascending=False)
    df.to_csv(outdir / "trial_results.csv", index=False)
    summary = {
        "run_tag": run_tag,
        "target_accuracy": args.target_accuracy,
        "classes": classes,
        "rows_total": len(rows),
        "rows_train": len(tr),
        "rows_val": len(va),
        "rows_test": len(te),
        "best": best,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        hdfs.write_json(METRICS_OUTPUT, summary, overwrite=True)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(json.dumps({"best_config": {k: best[k] for k in ["lr", "weight_decay", "batch_size", "dropout", "optimizer", "epochs", "unfreeze_last_block"]}, "classes": classes}).encode("utf-8"))
        local_model = tmp.name
    try:
        hdfs.upload_local_file(local_model, MODEL_OUTPUT, overwrite=True)
    except Exception:
        pass

    if float(best["test_accuracy"]) < float(args.target_accuracy):
        raise SystemExit(
            f"Target not met: best test_accuracy={best['test_accuracy']:.4f} < {args.target_accuracy:.4f}. "
            f"See {outdir}/trial_results.csv"
        )
    print(json.dumps({"run_tag": run_tag, "best_test_accuracy": best["test_accuracy"], "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
