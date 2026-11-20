#!/usr/bin/env python3
"""
nnUNet Inference and Evaluation Script

This script runs nnUNet inference on a set of CT images, computes evaluation metrics
(Dice, Sensitivity, Precision, HD95) against ground truth labels, and saves a JSON
summary. 

Usage example:
    python predict_and_evaluate.py \
        --weights /path/to/checkpoint_final.pth \
        --image_dir /path/to/images \
        --label_dir /path/to/labels \
        --output_dir /path/to/output \
        --device cuda
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from medpy.metric import binary
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="nnUNet inference and evaluation")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (e.g., checkpoint_final.pth)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images (nii.gz files, with optional _0000 suffix)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing ground truth label files (nii.gz, same basename as images without _0000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions and metrics summary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--save_prob",
        action="store_true",
        help="Save probability maps (large files)",
    )
    parser.add_argument(
        "--num_preprocess",
        type=int,
        default=2,
        help="Number of preprocessing processes (default: 2)",
    )
    parser.add_argument(
        "--num_export",
        type=int,
        default=2,
        help="Number of segmentation export processes (default: 2)",
    )
    return parser.parse_args()


def compute_metrics(pred_data: np.ndarray, gt_data: np.ndarray) -> Dict[str, float]:
    """
    Compute Dice, Sensitivity, Precision, and 95% Hausdorff Distance.

    Args:
        pred_data: Predicted segmentation array (probabilities or binary).
        gt_data: Ground truth binary array.

    Returns:
        Dictionary with keys: dice, sensitivity, precision, hd95.
    """
    pred_bin = (pred_data > 0.5).astype(np.uint8)
    gt_bin = (gt_data > 0.5).astype(np.uint8)

    # Both empty: perfect match
    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        return {"dice": 1.0, "sensitivity": 1.0, "precision": 1.0, "hd95": 0.0}

    # One empty, the other not: worst case
    if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return {"dice": 0.0, "sensitivity": 0.0, "precision": 0.0, "hd95": float("inf")}

    dice = binary.dc(pred_bin, gt_bin)
    sens = binary.sensitivity(pred_bin, gt_bin)
    prec = binary.precision(pred_bin, gt_bin)
    hd95 = binary.hd95(pred_bin, gt_bin)
    return {"dice": dice, "sensitivity": sens, "precision": prec, "hd95": hd95}


def find_case_name(img_path: Path) -> str:
    """
    Extract a consistent case identifier from an image file path.

    Handles filenames like 'case_0000.nii.gz' or 'case.nii.gz' (and
    possible double .nii extensions).
    """
    stem = img_path.stem  # removes .nii.gz -> e.g., "case_0000.nii" or "case"
    if stem.endswith(".nii"):
        stem = stem[:-4]  # remove extra .nii if present
    if "_0000" in stem:
        return stem.split("_0000")[0]
    return stem


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer model folder and fold from weights path
    model_folder = weights_path.parent.parent
    fold_name = weights_path.parent.name
    fold = int(fold_name.split("_")[1]) if fold_name.startswith("fold_") else 0
    checkpoint_name = weights_path.name

    # Initialize nnUNet predictor
    predictor = nnUNetPredictor(device=torch.device(args.device))
    try:
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )
    except Exception as e:
        print(f"Failed to initialize predictor: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Model loaded from: {model_folder} (fold {fold}, checkpoint {checkpoint_name})")

    # Gather input images
    image_files = sorted(image_dir.glob("*.nii.gz"))
    if not image_files:
        print(f"No .nii.gz files found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    all_metrics = []
    temp_dir = output_dir / "temp_predictions"
    temp_dir.mkdir(exist_ok=True)

    for img_path in image_files:
        case_name = find_case_name(img_path)
        label_path = label_dir / f"{case_name}.nii.gz"
        if not label_path.exists():
            print(f"Warning: label not found for {case_name}, skipping")
            continue

        print(f"Processing: {case_name}")

        # Run prediction
        # Input must be List[List[str]] (one modality per case)
        try:
            predictor.predict_from_files(
                [[str(img_path)]],
                str(temp_dir),
                save_probabilities=args.save_prob,
                num_processes_preprocessing=args.num_preprocess,
                num_processes_segmentation_export=args.num_export,
            )
        except Exception as e:
            print(f"Prediction failed for {case_name}: {e}", file=sys.stderr)
            # Clean up temp files for this case if possible
            continue

        # Locate the predicted file: same name as input (including _0000)
        # If not found, try globbing by case_name
        pred_file = temp_dir / img_path.name
        if not pred_file.exists():
            candidates = list(temp_dir.glob(f"*{case_name}*.nii.gz"))
            if candidates:
                pred_file = candidates[0]
            else:
                print(f"Prediction output not found for {case_name}", file=sys.stderr)
                continue

        # Move prediction to final output
        final_pred = output_dir / f"{case_name}_pred.nii.gz"
        shutil.move(str(pred_file), str(final_pred))

        # Clean up any leftover temp files for this case (if any)
        # But be careful not to delete other files; we'll remove the entire temp dir later.

        # Load prediction and ground truth
        try:
            pred_nii = nib.load(final_pred)
            gt_nii = nib.load(label_path)
            pred_data = pred_nii.get_fdata()
            gt_data = gt_nii.get_fdata()
        except Exception as e:
            print(f"Error loading data for {case_name}: {e}", file=sys.stderr)
            continue

        if pred_data.shape != gt_data.shape:
            print(
                f"Shape mismatch for {case_name}: pred {pred_data.shape} vs gt {gt_data.shape}, skipping eval"
            )
            continue

        metrics = compute_metrics(pred_data, gt_data)
        metrics["case"] = case_name
        all_metrics.append(metrics)
        print(f"  Dice: {metrics['dice']:.4f}, HD95: {metrics['hd95']:.2f}")

    # Remove temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Compute summary statistics
    if not all_metrics:
        print("No cases were successfully processed.", file=sys.stderr)
        sys.exit(1)

    dice_list = [m["dice"] for m in all_metrics]
    hd95_list = [m["hd95"] for m in all_metrics if np.isfinite(m["hd95"])]
    sens_list = [m["sensitivity"] for m in all_metrics]
    prec_list = [m["precision"] for m in all_metrics]

    summary = {
        "mean_dice": float(np.mean(dice_list)),
        "std_dice": float(np.std(dice_list)),
        "mean_hd95": float(np.mean(hd95_list)) if hd95_list else float("inf"),
        "std_hd95": float(np.std(hd95_list)) if hd95_list else float("inf"),
        "mean_sensitivity": float(np.mean(sens_list)),
        "mean_precision": float(np.mean(prec_list)),
        "num_cases": len(all_metrics),
    }

    print("\n=== Overall Metrics ===")
    print(f"Dice:        {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"HD95:        {summary['mean_hd95']:.2f} ± {summary['std_hd95']:.2f}")
    print(f"Sensitivity: {summary['mean_sensitivity']:.4f}")
    print(f"Precision:   {summary['mean_precision']:.4f}")

    # Save results to JSON
    out_json = output_dir / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump({"per_case": all_metrics, "summary": summary}, f, indent=4)
    print(f"Metrics saved to: {out_json}")
    print("All done!")


if __name__ == "__main__":
    main()