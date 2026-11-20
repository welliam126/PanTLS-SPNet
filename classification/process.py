#!/usr/bin/env python
"""
Pre-extract CT-CLIP features from CT images and segmentation labels.

This script loads a pretrained CT-CLIP visual encoder, processes CT images
and their corresponding segmentation labels, and extracts region features
saved as .npy files for downstream classification training.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path for module imports (adapt as needed)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import (
    CTClassificationDataset,
    extract_region_features_sequence,
    TARGET_SHAPE,
    load_model,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CT-CLIP region features from CT images."
    )

    # Data paths
    parser.add_argument(
        "--label_xlsx",
        type=str,
        required=True,
        help="Path to the label Excel file (.xlsx) with sample annotations.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing CT images (.nii.gz files).",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing segmentation labels (.nii.gz files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted feature files (.npy).",
    )

    # Model paths
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained CT-CLIP model checkpoint (.pt).",
    )

    # Processing parameters
    parser.add_argument(
        "--use_label12",
        action="store_true",
        default=True,
        help="Use both label 1 and 2 for region extraction. If False, use only label 1.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for feature extraction (default: 1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use: 'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[480, 480, 240],
        help="Target shape (H, W, D) for resizing (default: 480 480 240).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[20, 20, 10],
        help="Patch size (H, W, D) for feature grid (default: 20 20 10).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the dataset
    print("Loading dataset...")
    dataset = CTClassificationDataset(
        label_xlsx=args.label_xlsx,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        target_shape=tuple(args.target_shape),
    )
    print(f"Total samples: {len(dataset)}")

    # 2. Load the CT-CLIP visual encoder
    print("Loading CT-CLIP visual encoder...")
    visual_encoder = load_model(device, args.pretrained_path, use_multi_gpu=False)
    visual_encoder.eval()
    for param in visual_encoder.parameters():
        param.requires_grad = False
    print("Model loaded successfully.")

    # 3. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Determine feature tag
    tag = "label12" if args.use_label12 else "label1"
    total = 0

    print(f"Starting feature extraction (tag: {tag})...")
    print("-" * 60)

    with torch.no_grad():
        for img_batch, label_batch, class_ids in tqdm(loader, desc="Extracting features"):
            # img_batch: (B, D, H, W) -> add channel dimension
            img_batch = img_batch.unsqueeze(1).to(device)  # (B, 1, D, H, W)
            label_batch = label_batch.to(device)

            # CT-CLIP forward pass
            features = visual_encoder(img_batch, return_encoded_tokens=True)  # (B, seq_len, 512)
            features = features.view(features.size(0), -1, features.size(-1))  # (B, N, 512)

            # Extract region features for each sample in the batch
            for i in range(features.size(0)):
                sample_feat = features[i]          # (N, 512)
                sample_label = label_batch[i]      # (D, H, W)

                region_feats = extract_region_features_sequence(
                    sample_feat,
                    sample_label,
                    patch_size=tuple(args.patch_size),
                    target_shape=tuple(args.target_shape),
                    use_label12=args.use_label12,
                )  # (N_region, 512)

                # Get sample ID from the dataset
                img_path = dataset.samples[total + i][0]
                sample_id = os.path.basename(img_path).replace(".nii.gz", "")
                save_path = os.path.join(args.output_dir, f"{sample_id}_{tag}.npy")

                np.save(save_path, region_feats.cpu().numpy())
                total += 1

    print("-" * 60)
    print(f"✅ Feature extraction complete! Saved {total} feature files to: {args.output_dir}")


if __name__ == "__main__":
    main()