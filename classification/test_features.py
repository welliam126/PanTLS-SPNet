#!/usr/bin/env python
"""
Extract CT-CLIP region features from CT images and segmentation labels.

This script loads a pretrained CT-CLIP visual encoder, processes CT images
and their corresponding segmentation labels, and extracts region features
saved as .npy files for downstream classification training.

Dependencies:
    - transformer_maskgit (https://github.com/ibrahimethemhamamci/CT-CLIP)
    - CT_CLIP (https://github.com/ibrahimethemhamamci/CT-CLIP)
    - nibabel, torch, numpy, tqdm
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from torch.cuda.amp import autocast

# Import CT-CLIP components
# These need to be installed from the official repository:
# git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
# pip install -e CT-CLIP/
try:
    from transformer_maskgit.transformer_maskgit.ctvit import CTViT
except ImportError:
    raise ImportError(
        "CTViT not found. Please install CT-CLIP first:\n"
        "  git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git\n"
        "  pip install -e CT-CLIP/"
    )


def load_model(device, pretrained_path, use_multi_gpu=True):
    """
    Load the pretrained CT-CLIP visual encoder (vision-only, without text encoder).

    Args:
        device: torch.device
        pretrained_path: Path to the pretrained checkpoint (.pt)
        use_multi_gpu: Whether to use DataParallel for multi-GPU

    Returns:
        image_encoder: The visual transformer model
    """
    # 1. Create the visual encoder with parameters matching original CT-CLIP
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    # 2. Load the full pretrained weights
    state_dict = torch.load(pretrained_path, map_location='cpu')

    # 3. Extract visual encoder parameters (keys starting with 'visual_transformer.')
    visual_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('visual_transformer.'):
            new_key = k.replace('visual_transformer.', '')
            visual_state_dict[new_key] = v

    # 4. Load weights into the encoder (allow missing keys like ema parameters)
    missing, unexpected = image_encoder.load_state_dict(visual_state_dict, strict=False)
    if missing:
        print(f"Note: Missing keys (can be ignored): {missing[:3]}...")
    if unexpected:
        print(f"Note: Unexpected keys (can be ignored): {unexpected[:3]}...")

    # 5. Switch to evaluation mode and move to device
    image_encoder.eval()
    image_encoder = image_encoder.to(device)

    # 6. Multi-GPU support
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        image_encoder = nn.DataParallel(image_encoder)

    return image_encoder


def process_image_label(image_path, label_path, target_shape=(480, 480, 240)):
    """
    Preprocess CT image and segmentation label.

    The cropping region is centered around the label area (values > 0)
    to ensure the lesion region is not cropped out.

    Args:
        image_path: Path to CT image (.nii.gz)
        label_path: Path to segmentation label (.nii.gz)
        target_shape: Target shape (H, W, D), default (480, 480, 240)

    Returns:
        img_padded: Image tensor (1, D, H, W) normalized to [-1, 1]
        label_padded: Label tensor (D, H, W) with values 0, 1, 2
    """
    # Load image
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata().astype(np.float32)
    if img_data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img_data.ndim}D")
    img_data = np.transpose(img_data, (1, 0, 2))  # (H, W, D)

    # Load label
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata().astype(np.int16)
    if label_data.ndim != 3:
        raise ValueError(f"Expected 3D label, got {label_data.ndim}D")
    label_data = np.transpose(label_data, (1, 0, 2))  # (H, W, D)

    # HU value normalization
    img_data = img_data * 1000.0
    hu_min, hu_max = -1000, 200
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data + 400) / 600).astype(np.float32)  # Normalize to [-1, 1]

    # Convert to tensors
    img_tensor = torch.tensor(img_data)      # (H, W, D)
    label_tensor = torch.tensor(label_data)  # (H, W, D)

    H, W, D = img_tensor.shape
    th, tw, td = target_shape

    # Find bounding box of non-zero label regions
    label_np = label_tensor.numpy()
    nonzero = np.argwhere(label_np > 0)

    if len(nonzero) == 0:
        # No label region: use center cropping
        print(f"Warning: No positive region in label {label_path}, using center crop")
        h_start_ideal = max(0, (H - th) // 2)
        w_start_ideal = max(0, (W - tw) // 2)
        d_start_ideal = max(0, (D - td) // 2)
    else:
        h_min, w_min, d_min = nonzero.min(axis=0)
        h_max, w_max, d_max = nonzero.max(axis=0)
        bbox_h = h_max - h_min + 1
        bbox_w = w_max - w_min + 1
        bbox_d = d_max - d_min + 1

        # Check if bbox exceeds target size
        if bbox_h > th or bbox_w > tw or bbox_d > td:
            print(f"Warning: Bbox size ({bbox_h},{bbox_w},{bbox_d}) exceeds target ({th},{tw},{td}), using center crop")
            h_start_ideal = max(0, (H - th) // 2)
            w_start_ideal = max(0, (W - tw) // 2)
            d_start_ideal = max(0, (D - td) // 2)
        else:
            # Calculate ideal start positions to center bbox in crop
            h_low = max(0, h_max - th)
            h_high = min(H - th, h_min)
            if h_low <= h_high:
                h_start_ideal = (h_low + h_high) // 2
            else:
                h_start_ideal = max(0, (H - th) // 2)

            w_low = max(0, w_max - tw)
            w_high = min(W - tw, w_min)
            if w_low <= w_high:
                w_start_ideal = (w_low + w_high) // 2
            else:
                w_start_ideal = max(0, (W - tw) // 2)

            d_low = max(0, d_max - td)
            d_high = min(D - td, d_min)
            if d_low <= d_high:
                d_start_ideal = (d_low + d_high) // 2
            else:
                d_start_ideal = max(0, (D - td) // 2)

    # Actual crop region (clamped to image boundaries)
    h_start_actual = max(0, h_start_ideal)
    h_end_actual = min(H, h_start_ideal + th)
    w_start_actual = max(0, w_start_ideal)
    w_end_actual = min(W, w_start_ideal + tw)
    d_start_actual = max(0, d_start_ideal)
    d_end_actual = min(D, d_start_ideal + td)

    # Perform cropping
    img_cropped = img_tensor[h_start_actual:h_end_actual,
                             w_start_actual:w_end_actual,
                             d_start_actual:d_end_actual]
    label_cropped = label_tensor[h_start_actual:h_end_actual,
                                 w_start_actual:w_end_actual,
                                 d_start_actual:d_end_actual]

    # Calculate padding amounts
    pad_h_before = max(0, -h_start_ideal)
    pad_h_after = max(0, h_start_ideal + th - H)
    pad_w_before = max(0, -w_start_ideal)
    pad_w_after = max(0, w_start_ideal + tw - W)
    pad_d_before = max(0, -d_start_ideal)
    pad_d_after = max(0, d_start_ideal + td - D)

    # Pad to target shape
    img_padded = F.pad(
        img_cropped,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1.0
    )
    label_padded = F.pad(
        label_cropped,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=0
    )

    # Reorder to (D, H, W)
    img_padded = img_padded.permute(2, 0, 1)   # (D, H, W)
    label_padded = label_padded.permute(2, 0, 1)

    # Add channel dimension
    img_padded = img_padded.unsqueeze(0)       # (1, D, H, W)

    return img_padded, label_padded


@torch.no_grad()
def extract_features(model, image_tensor, device, use_amp=True):
    """
    Extract features from an image tensor using the CT-CLIP visual encoder.

    Args:
        model: The visual encoder model
        image_tensor: Tensor of shape (1, D, H, W) or (batch, 1, D, H, W)
        device: torch.device
        use_amp: Whether to use automatic mixed precision

    Returns:
        features: Feature tensor of shape (seq_len, feature_dim)
    """
    model.eval()

    # Add batch dimension if needed
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with autocast(enabled=use_amp):
        if isinstance(model, nn.DataParallel):
            features = model.module(image_tensor, return_encoded_tokens=True)
        else:
            features = model(image_tensor, return_encoded_tokens=True)

    # Reshape to (batch, seq_len, dim)
    batch_size = features.size(0)
    features = features.view(batch_size, -1, features.size(-1))
    features = features.squeeze(0)  # (seq_len, dim)

    return features.cpu().numpy()


def extract_region_features(features, label_tensor, patch_size=(20, 20, 10),
                            target_shape=(240, 480, 480)):
    """
    Extract region features based on segmentation labels.

    Args:
        features: Feature array of shape (seq_len, dim), seq_len = 24*24*24
        label_tensor: Label tensor of shape (D, H, W) with integer labels
        patch_size: Patch size (D, H, W) for the feature grid
        target_shape: Target shape (D, H, W)

    Returns:
        region1: Features from label 1 regions (mean pooled)
        region12: Features from label 1+2 regions (mean pooled)
    """
    device = features.device if isinstance(features, torch.Tensor) else torch.device('cpu')
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).to(device)

    D, H, W = target_shape
    pd, ph, pw = patch_size
    grid_d = D // pd  # 24
    grid_h = H // ph  # 24
    grid_w = W // pw  # 24

    if features.shape[0] != grid_d * grid_h * grid_w:
        raise ValueError(
            f"Feature length {features.shape[0]} does not match grid size "
            f"{grid_d * grid_h * grid_w}"
        )

    # Move label to same device and convert to float
    label_tensor = label_tensor.to(device).float()

    # Create binary masks (1, 1, D, H, W)
    mask1 = (label_tensor == 1).float().unsqueeze(0).unsqueeze(0)
    mask12 = ((label_tensor == 1) | (label_tensor == 2)).float().unsqueeze(0).unsqueeze(0)

    # Max pool to patch grid
    pooled1 = F.max_pool3d(mask1, kernel_size=patch_size, stride=patch_size)
    pooled12 = F.max_pool3d(mask12, kernel_size=patch_size, stride=patch_size)

    # Convert to boolean masks (grid_d, grid_h, grid_w)
    patch_mask1 = pooled1.squeeze().bool()
    patch_mask12 = pooled12.squeeze().bool()

    # Reshape features to grid
    feat_grid = features.view(grid_d, grid_h, grid_w, -1)

    # Flatten
    patch_mask1_flat = patch_mask1.view(-1)
    patch_mask12_flat = patch_mask12.view(-1)
    feat_flat = feat_grid.view(-1, features.size(-1))

    # Extract region features (mean pooling)
    if patch_mask1_flat.any():
        region1 = feat_flat[patch_mask1_flat].mean(dim=0)
    else:
        region1 = torch.zeros(features.size(-1), device=device)

    if patch_mask12_flat.any():
        region12 = feat_flat[patch_mask12_flat].mean(dim=0)
    else:
        region12 = torch.zeros(features.size(-1), device=device)

    return region1.cpu().numpy(), region12.cpu().numpy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CT-CLIP region features from CT images and segmentation labels."
    )

    # Data paths
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing CT images (.nii.gz files)."
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing segmentation labels (.nii.gz files)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted features (.npy files)."
    )

    # Model path
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained CT-CLIP model checkpoint (.pt)."
    )

    # Processing parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)."
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)."
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[480, 480, 240],
        help="Target shape (H W D) for resizing (default: 480 480 240)."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[20, 20, 10],
        help="Patch size (H W D) for feature grid (default: 20 20 10)."
    )
    parser.add_argument(
        "--feature_target_shape",
        type=int,
        nargs=3,
        default=[240, 480, 480],
        help="Target shape (D H W) for feature extraction (default: 240 480 480)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Load model
    print("Loading CT-CLIP model...")
    model = load_model(device, args.pretrained_path, use_multi_gpu=(device.type == "cuda"))
    print("Model loaded successfully.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .nii.gz files
    nii_files = glob.glob(os.path.join(args.input_dir, "**", "*.nii.gz"), recursive=True)
    if not nii_files:
        print(f"No .nii.gz files found in: {args.input_dir}")
        return

    print(f"Found {len(nii_files)} .nii.gz files")

    # Process each file
    for nii_path in tqdm(nii_files, desc="Extracting features"):
        # Build label file path
        rel_path = os.path.relpath(nii_path, args.input_dir)
        label_path = os.path.join(args.label_dir, rel_path)

        if not os.path.exists(label_path):
            # Try alternative naming: {basename}anno.nii.gz
            basename = os.path.basename(nii_path).split('.')[0]
            label_path = os.path.join(args.label_dir, basename + 'anno.nii.gz')
            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {nii_path}, skipping")
                continue

        try:
            # Preprocess image and label
            target_shape = tuple(args.target_shape)
            img_tensor, label_tensor = process_image_label(nii_path, label_path, target_shape)

            # Extract image features
            features = extract_features(model, img_tensor, device, use_amp=args.use_amp)

            # Extract region features
            region1, region12 = extract_region_features(
                features,
                label_tensor,
                patch_size=tuple(args.patch_size),
                target_shape=tuple(args.feature_target_shape)
            )

            # Save features
            base_name = rel_path.replace("/", "_").replace(".nii.gz", "")
            save_path1 = os.path.join(args.output_dir, f"{base_name}_label1.npy")
            save_path12 = os.path.join(args.output_dir, f"{base_name}_label12.npy")
            np.save(save_path1, region1)
            np.save(save_path12, region12)

        except Exception as e:
            print(f"Error processing {nii_path}: {e}")
            continue

    print(f"Feature extraction complete! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()