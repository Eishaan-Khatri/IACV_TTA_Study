# Colab Guide

Use Colab first. Kaggle is the fallback.

## Why Colab Is Better Here

- Simpler GPU runtime selection.
- Easier Google Drive caching for the 2.9 GB CIFAR-10-C tarball.
- Easier public GitHub workflow after results are clean.
- Less friction for sharing with a professor or lab member.

Kaggle is useful if Colab disconnects repeatedly, but Colab is the better first run.

## Dataset Download Choice

Yes, download CIFAR-10 and CIFAR-10-C.

- CIFAR-10 downloads automatically through torchvision.
- CIFAR-10-C should be downloaded from the official Zenodo record.
- CIFAR-10-C tarball size: about 2.9 GB.
- Put the data root on Google Drive so the download is cached.

Official CIFAR-10-C source:

https://zenodo.org/records/2535967

## Colab Cells

### 1. Runtime

In Colab:

`Runtime > Change runtime type > T4 GPU`

### 2. Clone Your Public Repo

After you create the GitHub repo:

```bash
!git clone https://github.com/Eishaan-Khatri/IACV_TTA_Study.git
%cd IACV_TTA_Study
```

Before the repo is public, upload the folder manually or use Google Drive.

### 3. Install Dependencies

```bash
!pip install -q -r requirements.txt
```

### 4. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Recommended data root:

```python
DATA_ROOT = "/content/drive/MyDrive/iacv_tta_data"
```

### 5. First-Pass Run

Start with 10-15 epochs if time is tight:

```bash
!python scripts/run_study.py \
  --config configs/default.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 15 \
  --results results/first_pass_results.csv
```

If training accuracy is too low, rerun with 30 epochs:

```bash
!python scripts/run_study.py \
  --config configs/default.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 30 \
  --results results/first_pass_results.csv
```

### 6. Summarize

```bash
!python scripts/summarize_results.py --results results/first_pass_results.csv
```

### 7. Final Adaptive Run

After the core run is complete and the source checkpoint exists, run the final suite:

```bash
!python scripts/run_study.py \
  --config configs/default.yaml \
  --data-root /content/drive/MyDrive/iacv_tta_data \
  --epochs 15 \
  --results results/adaptive_results.csv \
  --variant-suite adaptive
```

Analyze variants:

```bash
!python scripts/analyze_variants.py --results results/adaptive_results.csv
```

### 8. Generate Final Plots

```bash
!python scripts/make_final_plots.py \
  --results results/adaptive_results.csv \
  --summary results/variant_summary.csv \
  --by-severity results/variant_by_severity.csv \
  --out-dir results/figures
```

This writes:

- `results/figures/mean_accuracy_by_method.png`
- `results/figures/accuracy_by_severity.png`
- `results/figures/source_mix_delta_vs_tent_by_corruption.png`
- `results/figures/negative_transfer_cases_by_method.png`

### 9. Save Results

```bash
!cp -r results /content/drive/MyDrive/iacv_tta_results
```

## Kaggle Backup

Use Kaggle if Colab disconnects.

Differences:

- Use `/kaggle/working/iacv_tta_data` as data root.
- Internet must be enabled in notebook settings.
- You may need to re-download CIFAR-10-C unless you create a private Kaggle dataset from the extracted files.

Example:

```bash
!python scripts/run_study.py \
  --config configs/default.yaml \
  --data-root /kaggle/working/iacv_tta_data \
  --epochs 30 \
  --results results/first_pass_results.csv
```
