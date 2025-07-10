# 🦷 Dental Segmentation with YOLOv8 + CBAM

## 📁 Important Files

| File/Folder                | Description                                       |
|---------------------------|---------------------------------------------------|
| `dataset.ipynb`           | Notebook for preprocessing and CLAHE + cropping  |
| `dentex-2/`               | Raw original dataset:    https://universe.roboflow.com/dentex/dentex-3xe7e/dataset/2                           |
| `dentex-2-clahe-cropped/` | Preprocessed dataset (CLAHE-enhanced & cropped)   |
| `train-v1-baseline.py`    | Runs YOLOv8 baseline segmentation model           |
| `train-v2-CBAM.py`        | Runs YOLOv8 + CBAM-enhanced segmentation model    |

---

## 🚀 How to Run

### 1. Create Virtual Environment

```bash
python -m venv env
source env/bin/activate.fish  # or use activate if using bash/zsh
```
### 2. Install Ultralytics in Editable Mode
From the root of the cloned ultralytics/ repo:
```bash
pip install -e .
```
This enables local development (e.g., adding CBAM modules).

### 3. Train the Model
Baseline YOLOv8:
```bash
python train-v1-baseline.py
```
YOLOv8 + CBAM:
```bash
python train-v2-CBAM.py
```


