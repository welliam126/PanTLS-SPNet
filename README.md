PanTLS-SPNet

---

```markdown
# CT Image-Assisted Diagnosis Toolkit: Lesion Segmentation + Subtype Classification

This repository provides a complete toolkit for CT-based assisted diagnosis with two core functions:

- **Lesion Segmentation** – using nnUNet v2 for binary segmentation (background / lesion).
- **Subtype Classification** – using CT‑CLIP visual encoder + MLP head for four-class classification (DE / DIS / EX / ST).

We recommend reading this guide first, then running the `tutorial.ipynb` notebook for a quick start.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── segmentation/                     # Segmentation module
│   └── infer_seg.py                  # Inference + evaluation script
│
├── classification/                   # Classification module
│   ├── common.py                     # Shared components (dataset, model, loss)
│   ├── process.py                    # CT‑CLIP feature extraction
│   ├── main.py                       # MLP classifier training
│   ├── infer.py                      # Classification inference
│   └── test_features.py              # Feature extraction test (optional)
│
├── models/                           # Pretrained weights (download manually)
│   ├── segmentation_model.pth
│   └── subtype_cls.pth
│
├── example_data/                     # Sample data (see download instructions)
│   ├── images/                       # CT images (.nii.gz)
│   └── labels/                       # Segmentation labels (.nii.gz)
│
└── labels.xlsx                       # Classification label table (columns: sample_id, type, split)
```

---

## Environment Setup

### 1. Clone and Create Virtual Environment

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python -m venv venv
source venv/bin/activate        # Linux / Mac
# .\venv\Scripts\activate       # Windows
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install External Dependency (CT‑CLIP Visual Encoder)

The classification module requires the CT‑CLIP visual encoder. Clone the official repository into the project root:

```bash
git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
```

After cloning, the `CT-CLIP/` folder should be at the same level as `segmentation/` and `classification/`.

---

## Download Sample Data and Pretrained Weights

Sample data and pretrained weights are hosted on **Hugging Face**.

### Direct Download Links

| File               | Size    | Download Link                                                |
| :----------------- | :------ | :----------------------------------------------------------- |
| Sample data        | ~200 MB | [example_data.zip](https://huggingface.co/datasets/Seraphina201/ct-example-data/resolve/main/example_data.zip) |
| Segmentation model | 235 MB  | [segmentation_model.pth](https://huggingface.co/datasets/Seraphina201/ct-example-data/resolve/main/segmentation_model.pth) |
| Classifier         | ~2 MB   | [subtype_cls.pth](https://huggingface.co/datasets/Seraphina201/ct-example-data/resolve/main/subtype_cls.pth) |

### One‑Click Download with huggingface-cli

```bash
pip install huggingface-hub
hf download Seraphina201/ct-example-data --local-dir ./
```

After downloading, place the weight files in the `models/` directory and extract the sample data into the project root.

---

## Running Inference

### 1. Lesion Segmentation (nnUNet v2)

Run `infer_seg.py` to segment CT images and automatically compute Dice, HD95, etc. (if labels are provided).

```bash
python segmentation/infer_seg.py \
    --weights models/segmentation_model.pth \
    --image_dir example_data/images \
    --label_dir example_data/labels \
    --output_dir results_seg \
    --device cuda
```

**Arguments**:

| Argument       | Description                                       |
| :------------- | :------------------------------------------------ |
| `--weights`    | Path to the segmentation model weights            |
| `--image_dir`  | Directory with input CT images (supports `_0000`) |
| `--label_dir`  | Directory with ground truth labels (optional)     |
| `--output_dir` | Output directory                                  |
| `--device`     | Device: `cuda` or `cpu`                           |

**Output**:

- `results_seg/*_pred.nii.gz` – predicted segmentation per case (0=background, 1=lesion)
- `results_seg/metrics_summary.json` – per‑case and overall Dice, HD95, Sensitivity, Precision

---

### 2. Subtype Classification (CT‑CLIP + MLP)

Classification inference consists of two steps: feature extraction and classification.

#### Step 2.1: Extract CT‑CLIP Features

Use `process.py` to extract region features and save them as `.npy` files.

```bash
python classification/process.py \
    --label_xlsx labels.xlsx \
    --image_dir example_data/images \
    --label_dir example_data/labels \
    --output_dir features \
    --pretrained_path /path/to/ct_clip_pretrained.pt \
    --device cuda
```

The `--pretrained_path` must point to the CT‑CLIP visual encoder weights (e.g., `CT_VocabFine_v2.pt`), available from the [CT‑CLIP official Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/tree/main/models).

#### Step 2.2: Run Classification Inference

Use the trained classifier for four‑class prediction.

```bash
python classification/infer.py \
    --ckpt models/subtype_cls.pth \
    --label_xlsx labels.xlsx \
    --feature_dir features \
    --output results_cls/predictions.txt \
    --device cuda
```

**Output**:

- `results_cls/predictions.txt` (tab‑separated) with fields: `sample_id`, `gt_label`, `pred_label`, and class probabilities.

---

## Model Training

### Training the Segmentation Model (nnUNet v2)

Refer to the [official nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet) for full training details. Brief steps:

1. Prepare data in nnUNet format.
2. Preprocess:
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
   ```
3. Train (5‑fold cross‑validation):
   ```bash
   for fold in 0 1 2 3 4; do
       nnUNetv2_train DATASET_ID 3d_fullres $fold --npz
   done
   ```
4. Export the model:
   ```bash
   nnUNetv2_export_model_to_zip -d DATASET_ID -c 3d_fullres -o /path/to/model.zip
   ```

### Training the Classification Model (MLP Head)

Use `main.py` to train the classifier (features must be pre‑extracted).

```bash
python classification/main.py \
    --label_xlsx labels.xlsx \
    --feature_dir features \
    --save_dir models/classifier_trained \
    --epochs 200 \
    --batch_size 64 \
    --device cuda
```

The script uses the `split` column (train/eval/test) in the Excel file to split the dataset automatically. The best model is saved as `best_classifier.pth`.

---

## Jupyter Notebook Tutorial

We provide `tutorial.ipynb` that walks you through the entire pipeline from environment check to final predictions.

```bash
jupyter notebook tutorial.ipynb
```

---

## Result Explanation

- **Segmentation**: Output `.nii.gz` files with labels 0 (background) and 1 (lesion). `metrics_summary.json` provides detailed quantitative metrics.
- **Classification**: `predictions.txt` contains sample IDs, ground truth (if available), predicted labels, and class probabilities.

---

## Third‑Party Dependencies

This project builds upon the following open‑source projects. Please review their licenses before use.

| Project                                                      | Purpose                | License      |
| :----------------------------------------------------------- | :--------------------- | :----------- |
| [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet)              | Segmentation framework | Apache 2.0   |
| [CT‑CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)    | Visual feature encoder | See original |
| [CT‑CLIP pretrained weights](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/tree/main/models) | Model weights          | See original |

---

## Citation

If you use this toolkit in your research, please cite the following key works:

**CT‑CLIP (for subtype classification)**:
```bibtex
@article{hamamci2024ctclip,
  title={CT-CLIP: A Foundation Model for 3D Medical Image Understanding},
  author={Hamamci, Ibrahim Ethem and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```

**nnUNet v2 (for lesion segmentation)**:
```bibtex
@article{isensee2021nnuet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and others},
  journal={Nature Methods},
  volume={18},
  pages={203--211},
  year={2021}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).
```
