"""
Common utilities for CT-CLIP feature extraction and classification.

This module provides shared components for:
    - Feature extraction from CT images using CT-CLIP
    - Dataset classes for loading pre-computed features
    - MLP classifier model definition
    - Collate functions for DataLoader

Dependencies:
    - torch, numpy, pandas, nibabel, torchio
    - CT-CLIP (for feature extraction)
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchio


# ========== Constants ==========

TYPE_TO_ID = {"DE": 0, "DIS": 1, "EX": 2, "ST": 3}
"""Mapping from subtype names to integer IDs."""

ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}
"""Mapping from integer IDs to subtype names."""

TARGET_SHAPE = (240, 480, 480)  # (D, H, W)
"""Target shape for feature extraction: (depth, height, width)."""


# ========== Excel Column Resolution ==========

def resolve_excel_columns(df, excel_path):
    """
    Resolve column names in the Excel file.

    Args:
        df: pandas DataFrame loaded from Excel
        excel_path: Path to the Excel file (for error messages)

    Returns:
        id_col: Name of the sample ID column
        type_col: Name of the type/label column
        split_col: Name of the split column (or None)
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    id_candidates = ["sample_id", "id", "case_id", "文件名"]
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


# ========== Filename Utilities ==========

def stem_nii_basename(path):
    """
    Extract the stem from a .nii.gz or .nii file path.

    Args:
        path: Path to the file

    Returns:
        Filename without extension
    """
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[: -len(".nii.gz")]
    if base.endswith(".nii"):
        return base[: -len(".nii")]
    return os.path.splitext(base)[0]


def stem_precomputed_feat(path, use_region):
    """
    Extract the stem from a pre-computed feature file path.

    Args:
        path: Path to the .npy file
        use_region: Region tag to strip from the filename

    Returns:
        Stem without region tag
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if use_region and stem.endswith("_" + use_region):
        stem = stem[: -(len(use_region) + 1)]
    return stem


def loose_match_sid_stem(sid, stem):
    """
    Perform loose matching between sample ID and filename stem.

    Args:
        sid: Sample ID (string)
        stem: Filename stem

    Returns:
        True if the stem matches the sample ID loosely
    """
    sid = str(sid).strip()
    if stem == sid:
        return True
    if stem.startswith(sid + "_") or stem.startswith(sid + "-"):
        return True
    if sid not in stem:
        return False
    if sid.isdigit():
        return re.search(r"(^|_|-)" + re.escape(sid) + r"($|_|-)", stem) is not None
    return True


def pick_path_for_sid(sid, paths, stem_fn):
    """
    Pick the best matching path for a sample ID.

    Args:
        sid: Sample ID
        paths: List of file paths
        stem_fn: Function to extract stem from path

    Returns:
        Best matching path, or None if no match found
    """
    sid = str(sid).strip()
    if not paths:
        return None

    strict, loose = [], []
    for p in paths:
        stem = stem_fn(p)
        if stem == sid or stem.startswith(sid + "_") or stem.startswith(sid + "-"):
            strict.append(p)
        elif loose_match_sid_stem(sid, stem):
            loose.append(p)

    pool = strict if strict else loose
    if not pool:
        return None

    if len(pool) == 1:
        return pool[0]

    return min(pool, key=lambda p: len(stem_fn(p)))


# ========== Image Preprocessing ==========

def load_and_resize(img_path, label_path, target_shape=(240, 480, 480)):
    """
    Load and resize CT image and segmentation label using torchio.

    Args:
        img_path: Path to CT image (.nii.gz)
        label_path: Path to segmentation label (.nii.gz)
        target_shape: Target shape (D, H, W)

    Returns:
        img_tensor: Image tensor (D, H, W)
        label_tensor: Label tensor (D, H, W)
    """
    subject = torchio.Subject(
        image=torchio.ScalarImage(img_path),
        label=torchio.LabelMap(label_path)
    )
    resize = torchio.Resize(target_shape)
    subject = resize(subject)

    img_tensor = subject.image.tensor.squeeze(0).float()
    label_tensor = subject.label.tensor.squeeze(0)

    return img_tensor, label_tensor


# ========== Feature Extraction ==========

def extract_region_features_sequence(
    features,
    label_tensor,
    patch_size=(20, 20, 10),
    target_shape=(240, 480, 480),
    use_label12=True
):
    """
    Extract region features based on segmentation labels.

    Args:
        features: Feature tensor of shape (N, D) where N = grid_d * grid_h * grid_w
        label_tensor: Label tensor of shape (D, H, W)
        patch_size: Patch size (D, H, W) for grid
        target_shape: Target shape (D, H, W)
        use_label12: If True, use labels 1 and 2; otherwise use only label 1

    Returns:
        region_feats: Extracted region features (N_region, D)
    """
    device = features.device
    D, H, W = target_shape
    pd, ph, pw = patch_size
    grid_d, grid_h, grid_w = D // pd, H // ph, W // pw

    if features.shape[0] != grid_d * grid_h * grid_w:
        raise ValueError(
            f"Feature length {features.shape[0]} does not match grid size "
            f"{grid_d * grid_h * grid_w}"
        )

    label_tensor = label_tensor.to(device).float()

    if use_label12:
        patch_mask = ((label_tensor == 1) | (label_tensor == 2)).float().unsqueeze(0).unsqueeze(0)
    else:
        patch_mask = (label_tensor == 1).float().unsqueeze(0).unsqueeze(0)

    pooled = F.max_pool3d(patch_mask, kernel_size=patch_size, stride=patch_size)
    pooled = pooled.squeeze().bool()
    patch_mask_flat = pooled.view(-1)

    feat_grid = features.view(grid_d, grid_h, grid_w, -1)
    feat_flat = feat_grid.view(-1, features.size(-1))

    if patch_mask_flat.any():
        region_feats = feat_flat[patch_mask_flat]
    else:
        region_feats = torch.zeros(1, features.size(-1), device=device)

    return region_feats


def extract_region_features_batch(
    features_batch,
    label_batch,
    patch_size=(20, 20, 10),
    target_shape=(240, 480, 480),
    max_region_tokens=512
):
    """
    Extract region features for a batch of samples.

    Args:
        features_batch: Feature tensor of shape (B, N, D)
        label_batch: Label tensor of shape (B, D, H, W)
        patch_size: Patch size (D, H, W)
        target_shape: Target shape (D, H, W)
        max_region_tokens: Maximum number of region tokens

    Returns:
        region_padded: Padded region features (B, max_N, D)
        region_mask: Mask tensor (B, max_N)
    """
    B, seq_len, dim = features_batch.shape
    device = features_batch.device

    D, H, W = target_shape
    pd, ph, pw = patch_size
    grid_d, grid_h, grid_w = D // pd, H // ph, W // pw

    list_feats = []
    list_lens = []

    for b in range(B):
        feats = extract_region_features_sequence(
            features_batch[b],
            label_batch[b],
            patch_size=patch_size,
            target_shape=target_shape
        )
        list_feats.append(feats)
        list_lens.append(feats.size(0))

    max_N = min(max(int(max(list_lens)), 1), max_region_tokens)

    region_padded = torch.zeros(B, max_N, dim, device=device, dtype=features_batch.dtype)
    region_mask = torch.zeros(B, max_N, dtype=torch.bool, device=device)

    for b in range(B):
        feats = list_feats[b]
        n = min(feats.size(0), max_N)
        region_padded[b, :n] = feats[:n]
        region_mask[b, :n] = True

    return region_padded, region_mask


# ========== Dataset Classes ==========

class CTClassificationDataset(Dataset):
    """
    Dataset for CT image classification with segmentation labels.

    Args:
        label_xlsx: Path to label Excel file
        image_dir: Directory containing CT images
        label_dir: Directory containing segmentation labels
        target_shape: Target shape (H, W, D) for resizing
    """

    def __init__(self, label_xlsx, image_dir, label_dir, target_shape=(480, 480, 240)):
        self.target_shape = target_shape
        self.samples = []

        # Handle single or multiple cohorts
        if isinstance(label_xlsx, (list, tuple)):
            label_list = list(label_xlsx)
        else:
            label_list = [label_xlsx]

        if isinstance(image_dir, (list, tuple)):
            image_dir_list = list(image_dir)
        else:
            image_dir_list = [image_dir]

        if isinstance(label_dir, (list, tuple)):
            label_dir_list = list(label_dir)
        else:
            label_dir_list = [label_dir]

        if not (len(label_list) == len(image_dir_list) == len(label_dir_list)):
            raise ValueError("label_xlsx, image_dir, label_dir must have equal lengths.")

        self.n_total_images = 0
        self.n_excel_rows = 0
        self.cohort_ids = []

        for cohort_idx, (lx, img_dir, lab_dir) in enumerate(
            zip(label_list, image_dir_list, label_dir_list)
        ):
            df = pd.read_excel(lx)
            id_col, type_col, split_col = resolve_excel_columns(df, lx)

            image_files = glob.glob(os.path.join(img_dir, "**", "*.nii.gz"), recursive=True)
            self.n_total_images += len(image_files)
            self.n_excel_rows += len(df)

            n_valid_type = 0
            n_match_image = 0
            n_match_label = 0

            for _, row in df.iterrows():
                sid = str(row[id_col]).strip()
                t = str(row[type_col]).strip().upper()

                if t not in TYPE_TO_ID:
                    continue

                n_valid_type += 1
                class_id = TYPE_TO_ID[t]

                if split_col is not None and split_col in row:
                    split = str(row[split_col]).strip().lower()
                    if split not in ['train', 'eval', 'test']:
                        split = 'train'
                else:
                    split = 'train'

                stem_fn = stem_nii_basename
                candidates = [
                    p for p in image_files
                    if stem_fn(p) == sid
                    or stem_fn(p).startswith(sid + "_")
                    or stem_fn(p).startswith(sid + "-")
                ]

                if not candidates:
                    candidates = [
                        p for p in image_files
                        if loose_match_sid_stem(sid, stem_fn(p))
                    ]
                    if candidates:
                        print(
                            f"[CTClassificationDataset] Warning: sample_id={sid} "
                            f"loosely matched to: {[os.path.basename(x) for x in candidates[:3]]}"
                        )

                img_path = pick_path_for_sid(sid, candidates, stem_fn)
                if img_path is None:
                    continue

                n_match_image += 1
                base = os.path.basename(img_path)

                # Try different label filename patterns
                cand0 = os.path.join(lab_dir, base.replace(".nii.gz", "_anno.nii.gz"))
                cand1 = os.path.join(lab_dir, base.replace(".nii.gz", "anno.nii.gz"))
                cand2 = os.path.join(lab_dir, base)

                if os.path.exists(cand0):
                    label_path = cand0
                elif os.path.exists(cand1):
                    label_path = cand1
                elif os.path.exists(cand2):
                    label_path = cand2
                else:
                    continue

                n_match_label += 1
                self.samples.append((img_path, label_path, class_id, split))
                self.cohort_ids.append(cohort_idx)

            print(
                f"[CTClassificationDataset] Cohort {cohort_idx}: "
                f"excel={lx}, valid_types={n_valid_type}, "
                f"matched_images={n_match_image}, matched_labels={n_match_label}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, class_id, _ = self.samples[idx]
        img_tensor, label_tensor = load_and_resize(img_path, label_path)
        return img_tensor, label_tensor, class_id


class PrecomputedRegionDataset(Dataset):
    """
    Dataset for loading pre-computed region features.

    Args:
        label_xlsx: Path to label Excel file
        feature_dir: Directory containing .npy feature files
        use_region: Feature tag (e.g., 'label12')
        max_region_tokens: Maximum number of region tokens
    """

    def __init__(self, label_xlsx, feature_dir, use_region="label12", max_region_tokens=512):
        self.samples = []
        self.cohort_ids = []
        self.max_region_tokens = max_region_tokens

        if isinstance(label_xlsx, (list, tuple)):
            label_list = list(label_xlsx)
        else:
            label_list = [label_xlsx]

        if isinstance(feature_dir, (list, tuple)):
            feat_dir_list = list(feature_dir)
        else:
            feat_dir_list = [feature_dir]

        if len(label_list) != len(feat_dir_list):
            raise ValueError("label_xlsx and feature_dir must have equal lengths.")

        for cohort_idx, (lx, fdir) in enumerate(zip(label_list, feat_dir_list)):
            if not os.path.isdir(fdir):
                raise ValueError(f"feature_dir does not exist: {fdir}")

            df = pd.read_excel(lx)
            id_col, type_col, split_col = resolve_excel_columns(df, lx)

            all_feats = glob.glob(os.path.join(fdir, "*.npy"))
            if use_region:
                all_feats = [p for p in all_feats if use_region in os.path.basename(p)]

            for _, row in df.iterrows():
                sid = str(row[id_col]).strip()
                t = str(row[type_col]).strip().upper()

                if t not in TYPE_TO_ID:
                    continue

                class_id = TYPE_TO_ID[t]

                if split_col is not None and split_col in row:
                    split = str(row[split_col]).strip().lower()
                    if split not in ['train', 'eval', 'test']:
                        split = 'train'
                else:
                    split = 'train'

                def stem_fn(p):
                    return stem_precomputed_feat(p, use_region)

                candidates = [
                    p for p in all_feats
                    if stem_fn(p) == sid
                    or stem_fn(p).startswith(sid + "_")
                    or stem_fn(p).startswith(sid + "-")
                ]

                if not candidates:
                    candidates = [
                        p for p in all_feats
                        if loose_match_sid_stem(sid, stem_fn(p))
                    ]
                    if candidates:
                        print(
                            f"[PrecomputedRegionDataset] Warning: sample_id={sid} "
                            f"loosely matched to: {[os.path.basename(x) for x in candidates[:3]]}"
                        )

                feat_path = pick_path_for_sid(sid, candidates, stem_fn)
                if feat_path is None:
                    continue

                self.samples.append((feat_path, class_id, split))
                self.cohort_ids.append(cohort_idx)

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


# ========== Collate Functions ==========

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
        hidden_dim: Hidden layer dimension (default: 256)
        num_classes: Number of output classes (default: 4)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, dim=512, hidden_dim=256, num_classes=4, dropout=0.3):
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
        pooled = self.norm(pooled)
        return self.mlp(pooled)


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