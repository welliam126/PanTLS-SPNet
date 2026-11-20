#!/usr/bin/env python
"""
Train an MLP classifier on pre-extracted CT-CLIP region features.

This script loads pre-computed .npy features, trains a lightweight MLP
classification head for subtype classification (DE/DIS/EX/ST), and saves
the best model checkpoint.

Dependencies:
    - Pre-extracted features from CT-CLIP (.npy files)
    - Label Excel file with sample annotations
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


# ========== Constants ==========
TYPE_TO_ID = {"DE": 0, "DIS": 1, "EX": 2, "ST": 3}
ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}


# ========== Utility Functions ==========
def _resolve_excel_columns(df):
    """Resolve column names in the Excel file."""
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    id_candidates = ["sample_id", "id", "case_id"]
    type_candidates = ["type", "label", "category"]
    split_candidates = ["split", "dataset", "fold"]

    id_col = next((c for c in id_candidates if c in cols), None)
    type_col = next((c for c in type_candidates if c in cols), None)
    split_col = next((c for c in split_candidates if c in cols), None)

    if id_col is None or type_col is None:
        raise ValueError(
            f"Excel must contain sample_id and type columns. "
            f"Current columns: {cols}"
        )

    return id_col, type_col, split_col


def _stem_precomputed_feat(path, use_region):
    """Extract stem from precomputed feature filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if use_region and stem.endswith("_" + use_region):
        stem = stem[: -(len(use_region) + 1)]
    return stem


def _loose_match_sid_stem(sid, stem):
    """Loose matching between sample ID and feature stem."""
    sid = str(sid).strip()
    if stem == sid or stem.startswith(sid + "_") or stem.startswith(sid + "-"):
        return True
    if sid not in stem:
        return False
    if sid.isdigit():
        import re
        return re.search(r"(^|_|-)" + re.escape(sid) + r"($|_|-)", stem) is not None
    return True


def _pick_path_for_sid(sid, paths, stem_fn):
    """Pick the best matching path for a sample ID."""
    sid = str(sid).strip()
    if not paths:
        return None

    strict, loose = [], []
    for p in paths:
        stem = stem_fn(p)
        if stem == sid or stem.startswith(sid + "_") or stem.startswith(sid + "-"):
            strict.append(p)
        elif _loose_match_sid_stem(sid, stem):
            loose.append(p)

    pool = strict if strict else loose
    if not pool:
        return None

    return min(pool, key=lambda p: len(stem_fn(p))) if len(pool) > 1 else pool[0]


# ========== Dataset ==========
class PrecomputedRegionDataset(Dataset):
    """
    Dataset for loading pre-computed region features.

    Args:
        label_xlsx: Path to the label Excel file
        feature_dir: Directory containing .npy feature files
        use_region: Feature tag (e.g., 'label12')
        max_region_tokens: Maximum number of region tokens (for padding)
    """

    def __init__(self, label_xlsx, feature_dir, use_region="label12", max_region_tokens=512):
        self.samples = []
        self.max_region_tokens = max_region_tokens

        df = pd.read_excel(label_xlsx)
        id_col, type_col, split_col = _resolve_excel_columns(df)

        all_feats = glob.glob(os.path.join(feature_dir, "*.npy"))
        if use_region:
            all_feats = [p for p in all_feats if use_region in os.path.basename(p)]

        for _, row in df.iterrows():
            sid = str(row[id_col]).strip()
            t = str(row[type_col]).strip().upper()

            if t not in TYPE_TO_ID:
                continue

            class_id = TYPE_TO_ID[t]
            split = str(row[split_col]).strip().lower() if split_col in row else "train"

            stem_fn = lambda p: _stem_precomputed_feat(p, use_region)
            candidates = [
                p for p in all_feats
                if stem_fn(p) == sid
                or stem_fn(p).startswith(sid + "_")
                or stem_fn(p).startswith(sid + "-")
            ]

            if not candidates:
                candidates = [p for p in all_feats if _loose_match_sid_stem(sid, stem_fn(p))]

            feat_path = _pick_path_for_sid(sid, candidates, stem_fn)
            if feat_path is None:
                continue

            self.samples.append((feat_path, class_id, split))

        self.feature_dim = None
        if self.samples:
            first_feat = np.load(self.samples[0][0])
            self.feature_dim = int(first_feat.shape[-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, class_id, _ = self.samples[idx]
        feats = np.load(feat_path).astype(np.float32)

        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        return feats, class_id


def collate_precomputed_region(batch, max_region_tokens=512):
    """
    Collate function for PrecomputedRegionDataset.

    Args:
        batch: List of (features, class_id) tuples
        max_region_tokens: Maximum number of region tokens

    Returns:
        region_padded: Padded feature tensor (B, max_N, dim)
        region_mask: Mask tensor (B, max_N)
        class_ids: Class ID tensor (B,)
    """
    feats_list = [b[0] for b in batch]
    class_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)

    max_N = min(max(max(len(f) for f in feats_list), 1), max_region_tokens)
    dim = feats_list[0].shape[1]

    region_padded = torch.zeros(len(batch), max_N, dim, dtype=torch.float32)
    region_mask = torch.zeros(len(batch), max_N, dtype=torch.bool)

    for i, feats in enumerate(feats_list):
        n = min(len(feats), max_N)
        region_padded[i, :n] = torch.from_numpy(feats[:n].astype(np.float32))
        region_mask[i, :n] = True

    return region_padded, region_mask, class_ids


# ========== Model ==========
class MeanPoolMLPClassifier(nn.Module):
    """
    MLP classifier with mean pooling over region features.

    Args:
        dim: Input feature dimension (default: 512)
        hidden_dim: Hidden layer dimension (default: 64)
        num_classes: Number of output classes (default: 4)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, dim=512, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        if hidden_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.mlp = nn.Linear(dim, num_classes)

    def forward(self, region_features, region_mask):
        # Mean pooling over valid tokens
        mask_f = region_mask.float().unsqueeze(-1)
        sum_feats = (region_features * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = sum_feats / count

        # Normalize and classify
        feats = self.norm(pooled)
        logits = self.mlp(feats)

        return logits


# ========== Loss Functions ==========
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weight (optional)
        reduction: Reduction method (default: "mean")
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = float(alpha)

    def forward(self, inputs, targets):
        if inputs.ndim != 2:
            raise ValueError(f"Expected inputs shape (N, C), got {inputs.shape}")

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.view(-1, 1)).view(-1)
        pt = logpt.exp()

        if hasattr(self, "alpha") and isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(inputs.device).gather(0, targets)
        elif self.alpha is not None:
            at = torch.full_like(targets, self.alpha, dtype=inputs.dtype).to(inputs.device)
        else:
            at = torch.ones_like(targets, dtype=inputs.dtype).to(inputs.device)

        loss = -at * (1 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ========== Metrics ==========
def aggregate_metrics(all_labels, all_preds, all_probs):
    """
    Compute aggregated classification metrics.

    Args:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Predicted probabilities

    Returns:
        acc: Accuracy
        auc: AUC-ROC (multi-class OVR)
        f1_w: Weighted F1 score
        f1_macro: Macro F1 score
        bal_acc: Balanced accuracy
        cm: Confusion matrix
    """
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    acc = accuracy_score(all_labels, all_preds)

    # Normalize probabilities
    P = np.asarray(all_probs, dtype=np.float64)
    if P.ndim == 2 and P.shape[1] > 0:
        P = np.clip(P, 0.0, None)
        row_sum = P.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
        P = P / row_sum

    try:
        auc = roc_auc_score(all_labels, P, multi_class="ovr")
    except Exception:
        auc = float("nan")

    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, auc, f1_w, f1_macro, bal_acc, cm


# ========== Training Functions ==========
def train_epoch(classifier, train_loader, criterion, optimizer, scaler, device, args):
    """Train for one epoch."""
    classifier.train()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(train_loader, desc="Train"):
        region_padded, region_mask, class_batch = [x.to(device) for x in batch]

        optimizer.zero_grad()

        with autocast(enabled=args.amp):
            logits = classifier(region_padded, region_mask)
            loss = criterion(logits, class_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        probs = F.softmax(logits, dim=1)
        all_labels.extend(class_batch.cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    acc, auc, f1_w, f1_macro, bal_acc, cm = aggregate_metrics(
        all_labels, all_preds, all_probs
    )

    return avg_loss, acc, auc, f1_w, f1_macro, bal_acc, cm


def validate(classifier, val_loader, criterion, device, args, desc="Eval"):
    """Validate the model."""
    classifier.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            region_padded, region_mask, class_batch = [x.to(device) for x in batch]

            with autocast(enabled=args.amp):
                logits = classifier(region_padded, region_mask)
                loss = criterion(logits, class_batch)

            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            all_labels.extend(class_batch.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    acc, auc, f1_w, f1_macro, bal_acc, cm = aggregate_metrics(
        all_labels, all_preds, all_probs
    )

    return avg_loss, acc, auc, f1_w, f1_macro, bal_acc, cm


# ========== Main ==========
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MLP classifier on pre-extracted CT-CLIP features."
    )

    # Data paths
    parser.add_argument(
        "--label_xlsx",
        type=str,
        required=True,
        help="Path to the label Excel file (.xlsx)."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing pre-extracted .npy feature files."
    )
    parser.add_argument(
        "--precomputed_region_tag",
        type=str,
        default="label12",
        help="Feature filename tag (default: label12)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints (default: ./checkpoints)."
    )

    # Training hyperparameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)."
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)."
    )

    # Model architecture
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="MLP hidden layer dimension (default: 64)."
    )
    parser.add_argument(
        "--max_region_tokens",
        type=int,
        default=128,
        help="Maximum number of region tokens (default: 128)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)."
    )

    # Loss and regularization
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)."
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="focal",
        choices=["ce", "focal"],
        help="Loss type: 'ce' or 'focal' (default: focal)."
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (default: 2.0)."
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="Focal loss alpha parameter (optional)."
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="auto",
        choices=["none", "auto"],
        help="Class weight strategy: 'none' or 'auto' (default: auto)."
    )

    # Optimizer
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)."
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'training_log.txt')

    class Tee:
        """Redirect stdout to both file and console."""
        def __init__(self, fname):
            self.file = open(fname, 'w')
            self.stdout = sys.stdout

        def write(self, msg):
            self.file.write(msg)
            self.stdout.write(msg)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    sys.stdout = Tee(log_file)
    print(f"Training log will be saved to: {log_file}")

    # Device setup
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print(f"Using device: {device}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    print("Loading dataset...")
    dataset = PrecomputedRegionDataset(
        label_xlsx=args.label_xlsx,
        feature_dir=args.feature_dir,
        use_region=args.precomputed_region_tag,
        max_region_tokens=args.max_region_tokens,
    )
    print(f"Total samples: {len(dataset)}")

    # Split dataset
    splits = [s[2] for s in dataset.samples]
    train_idx = [i for i, sp in enumerate(splits) if sp == 'train']
    val_idx = [i for i, sp in enumerate(splits) if sp == 'eval']
    test_idx = [i for i, sp in enumerate(splits) if sp == 'test']

    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    if len(train_idx) == 0:
        raise SystemExit("Train set is empty. Please check the 'split' column.")

    # Compute class weights
    all_labels = [int(s[1]) for s in dataset.samples]
    train_labels = [all_labels[i] for i in train_idx]

    class_weights = None
    if args.class_weight == "auto":
        cw = compute_class_weight("balanced", classes=np.arange(4), y=np.array(train_labels))
        cw = np.sqrt(cw)
        class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
        print(f"[Class weights] { {ID_TO_TYPE[i]: round(float(cw[i]), 4) for i in range(4)} }")

    # Setup loss function
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
    else:
        focal_alpha = class_weights.tolist() if class_weights is not None else args.focal_alpha
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=focal_alpha, reduction="mean")

    # Initialize model
    classifier = MeanPoolMLPClassifier(
        dim=512,
        hidden_dim=args.mlp_hidden_dim,
        num_classes=4,
        dropout=args.dropout,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        classifier = nn.DataParallel(classifier)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=args.amp)

    # DataLoaders
    def _collate(batch):
        return collate_precomputed_region(batch, max_region_tokens=args.max_region_tokens)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    ) if len(test_idx) > 0 else None

    # Training loop
    best_val_monitor = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc, train_auc, train_f1, train_f1_macro, train_bal, _ = train_epoch(
            classifier, train_loader, criterion, optimizer, scaler, device, args
        )

        val_loss, val_acc, val_auc, val_f1, val_f1_macro, val_bal, val_cm = validate(
            classifier, val_loader, criterion, device, args, desc="Val"
        )

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} "
              f"F1: {train_f1:.4f} MacroF1: {train_f1_macro:.4f} BalAcc: {train_bal:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} "
              f"F1: {val_f1:.4f} MacroF1: {val_f1_macro:.4f} BalAcc: {val_bal:.4f}")
        print("Val Confusion Matrix:\n", val_cm)

        val_monitor = float(val_auc) if np.isfinite(val_auc) else float(val_bal)

        if val_monitor > best_val_monitor:
            best_val_monitor = val_monitor
            epochs_no_improve = 0

            ckpt = classifier.module.state_dict() if isinstance(classifier, nn.DataParallel) else classifier.state_dict()
            torch.save(ckpt, os.path.join(args.save_dir, "best_classifier.pth"))
            print(f"Saved best model (val_monitor={best_val_monitor:.4f})")
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping: No improvement for {args.early_stopping_patience} epochs.")
            break

    print(f"\nTraining completed. Best val monitor score: {best_val_monitor:.4f}")

    # Test evaluation
    if test_loader is not None:
        ckpt_path = os.path.join(args.save_dir, "best_classifier.pth")
        if os.path.exists(ckpt_path):
            raw_cls = classifier.module if isinstance(classifier, nn.DataParallel) else classifier
            raw_cls.load_state_dict(torch.load(ckpt_path, map_location=device))

            test_loss, test_acc, test_auc, test_f1, test_f1_macro, test_bal, test_cm = validate(
                classifier, test_loader, criterion, device, args, desc="Test (best)"
            )

            print(f"Test (best) Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f} "
                  f"F1: {test_f1:.4f} MacroF1: {test_f1_macro:.4f} BalAcc: {test_bal:.4f}")
            print("Test Confusion Matrix:\n", test_cm)


if __name__ == "__main__":
    # Add glob import for the dataset
    import glob
    main()