#!/usr/bin/env python
"""
Run inference using a trained MLP classifier on pre-extracted features.

This script loads a trained classifier checkpoint and performs inference
on all samples in the dataset, outputting predictions with probabilities.

Dependencies:
    - Pre-extracted features from CT-CLIP (.npy files)
    - Trained classifier checkpoint (.pth)
    - Label Excel file with sample annotations
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import common components
# Ensure common.py is in the same directory or PYTHONPATH
try:
    from common import (
        PrecomputedRegionDataset,
        collate_precomputed_region,
        MeanPoolMLPClassifier,
        ID_TO_TYPE,
    )
except ImportError:
    raise ImportError(
        "Could not import from common.py. "
        "Please ensure common.py is in the same directory or PYTHONPATH."
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained MLP classifier."
    )

    # Required paths
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the trained classifier checkpoint (.pth)."
    )
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

    # Optional paths
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/predictions.txt",
        help="Output file path for predictions (default: ./results/predictions.txt)."
    )
    parser.add_argument(
        "--region_tag",
        type=str,
        default="label12",
        help="Feature filename tag (default: label12)."
    )

    # Model architecture (must match training)
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Input feature dimension (default: 512)."
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="MLP hidden layer dimension (default: 64)."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of output classes (default: 4)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)."
    )

    # Inference parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)."
    )
    parser.add_argument(
        "--max_region_tokens",
        type=int,
        default=128,
        help="Maximum number of region tokens (default: 128)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = PrecomputedRegionDataset(
        label_xlsx=args.label_xlsx,
        feature_dir=args.feature_dir,
        use_region=args.region_tag,
        max_region_tokens=args.max_region_tokens,
    )
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        raise SystemExit("Dataset is empty. Please check the input paths.")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_precomputed_region(
            batch, max_region_tokens=args.max_region_tokens
        ),
        pin_memory=True,
    )

    # Load model with architecture matching training
    model = MeanPoolMLPClassifier(
        dim=args.dim,
        hidden_dim=args.mlp_hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully: {args.ckpt_path}")

    # Run inference
    results = []
    idx_counter = 0

    with torch.no_grad():
        for region_padded, region_mask, class_ids in tqdm(loader, desc="Inference"):
            region_padded = region_padded.to(device)
            region_mask = region_mask.to(device)

            logits = model(region_padded, region_mask)
            probs = torch.softmax(logits, dim=1)
            pred_ids = logits.argmax(dim=1)

            for i in range(region_padded.size(0)):
                # Extract sample ID from filename
                sample_id = os.path.basename(dataset.samples[idx_counter][0])
                sample_id = sample_id.replace(f"_{args.region_tag}.npy", "")

                gt_label = ID_TO_TYPE[dataset.samples[idx_counter][1]]
                pred_label = ID_TO_TYPE[pred_ids[i].item()]
                split = dataset.samples[idx_counter][2]
                prob = probs[i].cpu().numpy()

                results.append([sample_id, gt_label, pred_label, split] + prob.tolist())
                idx_counter += 1

    # Save results
    columns = [
        'sample_id', 'gt_label', 'pred_label'
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(args.output_file, sep='\t', index=False)

    print(f"✅ Results saved to: {args.output_file}")
    print(f"Total records: {len(df)}")
    print(f"Index counter: {idx_counter} (should match total samples: {len(dataset)})")

    # Print summary
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    print(df['pred_label'].value_counts())
    if 'gt_label' in df.columns:
        acc = (df['gt_label'] == df['pred_label']).mean()
        print(f"\nAccuracy: {acc:.4f}")


if __name__ == "__main__":
    main()