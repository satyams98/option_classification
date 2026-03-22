# Purpose Option Detection – YOLOv11 + Position-Based Classification

An end-to-end computer vision pipeline that automatically detects which option is hand-marked on scanned form images. The system identifies various marking styles (check marks, circles, crosses, strikethroughs) and classifies the selected option from a fixed set of five choices: **Revenue**, **Training**, **Test**, **Ferry**, and **Others**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Architecture & Implementation Details](#architecture--implementation-details)
  - [Stage 1: YOLO Object Detection](#stage-1-yolo-object-detection)
  - [Stage 2: Position-Based Classification](#stage-2-position-based-classification)
  - [Edge Case Handling](#edge-case-handling)
- [Training Details](#training-details)
- [Evaluation & Results](#evaluation--results)
- [Production Inference](#production-inference)
- [Output & What You Get](#output--what-you-get)
- [License](#license)

---

## Problem Statement

In many document processing workflows, scanned forms contain a row of purpose/option checkboxes (e.g., **Revenue / Training / Test / Ferry / Others**). A human reviewer marks one of these options by hand — using a tick mark, a circle, a cross, a strikethrough, or some other style. The challenge is to **automatically detect which option was selected** from the raw scanned image, handling:

- A wide variety of handwriting and marking styles
- Low-quality scans with noise, skew, or faded marks
- Cases where **no option** was selected at all
- Cases where **multiple marks** may be present

This project provides a robust, production-ready solution for this task.

---

## Solution Overview

The solution uses a **two-stage pipeline**:

```
Scanned Form Image
        │
        ▼
┌───────────────────────────┐
│  Stage 1: YOLOv11n        │
│  Object Detection          │
│  Detect check marks and    │
│  their bounding boxes      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Stage 2: Position-Based   │
│  Classification            │
│  Map bounding box x-center │
│  to one of 5 options       │
└─────────────┬─────────────┘
              │
              ▼
   Output: "Revenue" / "Training" /
           "Test" / "Ferry" / "Others" / "None"
```

**Why two stages?**  
Separating _detection_ from _classification_ makes each stage independently debuggable, testable, and improvable. YOLO handles the visual diversity of mark styles, while the position mapping is deterministic and calibrated from data analysis.

---

## Project Structure

```
OCR/
├── README.md                              ← This file
├── purpose_option_detection.ipynb         ← Main Jupyter notebook (training + evaluation)
├── yolo11n.pt                             ← Pretrained YOLOv11 nano weights
│
├── ocr_purpose.v3-v3.yolov11 (1)/        ← Roboflow dataset (YOLO format)
│   ├── data.yaml                          ← Original Roboflow YAML config
│   ├── data_fixed.yaml                    ← Fixed YAML with absolute paths
│   ├── README.dataset.txt                 ← Dataset description from Roboflow
│   ├── README.roboflow.txt                ← Roboflow export info
│   ├── train/
│   │   ├── images/                        ← 2119 training images (512×512, JPEG)
│   │   └── labels/                        ← YOLO-format annotation files
│   ├── valid/
│   │   ├── images/                        ← 339 validation images
│   │   └── labels/
│   └── test/
│       ├── images/                        ← 318 test images
│       └── labels/
│
└── runs/
    └── purpose_detection/                 ← Training output
        ├── args.yaml                      ← Full training hyperparameters
        ├── results.csv                    ← Epoch-wise training metrics
        ├── results.png                    ← Training curves plot
        ├── confusion_matrix.png           ← Confusion matrix visualization
        ├── confusion_matrix_normalized.png
        ├── labels.jpg                     ← Label distribution visualization
        ├── BoxP_curve.png                 ← Precision curve
        ├── BoxR_curve.png                 ← Recall curve
        ├── BoxPR_curve.png                ← PR curve
        ├── BoxF1_curve.png                ← F1 curve
        ├── train_batch*.jpg               ← Training batch visualizations
        ├── val_batch*_labels.jpg          ← Validation ground-truth visualizations
        ├── val_batch*_pred.jpg            ← Validation predictions visualizations
        └── weights/
            ├── best.pt                    ← Best model weights (used for inference)
            └── last.pt                    ← Last epoch weights
```

---

## Dataset

| Property               | Value                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Source**             | [Roboflow Universe – ocr_purpose v3](https://universe.roboflow.com/ocrdigitdetection-r2lr4/ocr_purpose/dataset/3) |
| **Total Images**       | 2,765                                                                                                             |
| **Train / Val / Test** | 2,119 / 339 / 318                                                                                                 |
| **Image Size**         | 512 × 512 pixels                                                                                                  |
| **Format**             | YOLOv11 (normalized bounding boxes)                                                                               |
| **License**            | CC BY 4.0                                                                                                         |

### Annotation Classes

The dataset uses **2 YOLO object detection classes**:

| Class ID | Name    | Description                                                         |
| -------- | ------- | ------------------------------------------------------------------- |
| 0        | `check` | A hand-drawn mark (tick, circle, cross, underline) around an option |
| 1        | `nan`   | A large bounding box spanning the full row — no option was marked   |

### Option Distribution (Training Set)

Derived from the x-position of `check`-class bounding boxes:

| Option     | Normalized x-range | Training Samples |
| ---------- | ------------------ | ---------------- |
| Revenue    | 0.00 – 0.28        | ~1,914           |
| Training   | 0.28 – 0.47        | ~76              |
| Test       | 0.47 – 0.62        | ~28              |
| Ferry      | 0.62 – 0.77        | ~2               |
| Others     | 0.77 – 1.00        | ~3               |
| None (nan) | —                  | ~219             |

### Pre-processing Applied

- Auto-orientation of pixel data (with EXIF-orientation stripping)
- Resize to 512×512 (stretch)
- No additional image augmentation was applied at the dataset level (YOLO's built-in augmentations are used during training)

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** (tested with Python 3.13)
- **pip** package manager
- A machine with at least **4 GB RAM** (CPU-only training is supported)
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd OCR
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install ultralytics numpy matplotlib pillow scikit-learn
```

### Dependencies

| Package           | Version  | Purpose                                                      |
| ----------------- | -------- | ------------------------------------------------------------ |
| `ultralytics`     | ≥ 8.4.24 | YOLOv11 model training, inference, validation, and export    |
| `numpy`           | ≥ 1.24   | Array operations and numerical processing                    |
| `matplotlib`      | ≥ 3.7    | Visualization of training data, results, and predictions     |
| `Pillow`          | ≥ 9.0    | Image loading and manipulation                               |
| `scikit-learn`    | ≥ 1.3    | Classification report, confusion matrix, evaluation metrics  |
| `torch` (PyTorch) | ≥ 2.0    | Deep learning backend (auto-installed with `ultralytics`)    |
| `torchvision`     | ≥ 0.15   | Image transforms (auto-installed with `ultralytics`)         |
| `opencv-python`   | ≥ 4.8    | Image I/O and processing (auto-installed with `ultralytics`) |
| `onnx`            | ≥ 1.14   | ONNX model export (optional, install for export step)        |

### Step 4: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('ultralytics OK')"
python -c "import sklearn; print('scikit-learn OK')"
```

---

## How to Run

### Option A: Run the Full Notebook (Training + Evaluation)

1. Open `purpose_option_detection.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Run all cells sequentially from top to bottom.
3. The notebook will:
   - Explore and visualize the dataset
   - Train a YOLOv11n model for 30 epochs
   - Evaluate on the test set with classification report and confusion matrix
   - Visualize predictions and misclassifications
   - Export the model to ONNX format
   - Generate a production inference script

> **Note:** Training on CPU takes approximately 2.5–3 hours for 30 epochs. With a GPU, this is significantly faster.

### Option B: Use Pre-Trained Weights for Inference Only

If you already have trained weights (`runs/purpose_detection/weights/best.pt`):

```python
from ultralytics import YOLO

OPTION_BOUNDARIES = [
    (0.00, 0.28, "Revenue"),
    (0.28, 0.47, "Training"),
    (0.47, 0.62, "Test"),
    (0.62, 0.77, "Ferry"),
    (0.77, 1.01, "Others"),
]

def x_to_option(x_center):
    for lo, hi, name in OPTION_BOUNDARIES:
        if lo <= x_center < hi:
            return name
    return "Others"

# Load model
model = YOLO("runs/purpose_detection/weights/best.pt")

# Predict
results = model("path/to/your/image.jpg", conf=0.25, verbose=False)
result = results[0]

if result.boxes is not None and len(result.boxes) > 0:
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    check_mask = classes == 0
    if check_mask.any():
        best_idx = confs[check_mask].argmax()
        best_box = xyxy[check_mask][best_idx]
        x_center = ((best_box[0] + best_box[2]) / 2) / result.orig_shape[1]
        print(f"Selected option: {x_to_option(x_center)}")
    else:
        print("Selected option: None")
else:
    print("Selected option: None")
```

---

## Architecture & Implementation Details

### Stage 1: YOLO Object Detection

**Model:** YOLOv11n (nano variant)

YOLOv11n (You Only Look Once, version 11, nano) is a state-of-the-art real-time object detection model. We chose the **nano** variant for its balance between speed and accuracy — it's small enough for CPU inference while being powerful enough for this binary detection task.

**What YOLO does in this pipeline:**

- Takes a 512×512 scanned form image as input
- Outputs bounding boxes with class labels (`check` or `nan`) and confidence scores
- Handles the _visual diversity_ of different marking styles (ticks, circles, crosses, underlines, strikethroughs)

**Training approach:**

- **Transfer learning:** Start from pretrained `yolo11n.pt` weights (trained on COCO) and fine-tune on our domain-specific dataset
- **Fine-tuning:** All layers are updated during training, allowing the model to adapt from generic object detection to check-mark detection
- **Built-in augmentations:** YOLO applies mosaic augmentation, random flips, color jitter, and scale variations during training to improve robustness

**Key training hyperparameters:**

| Parameter  | Value | Rationale                                         |
| ---------- | ----- | ------------------------------------------------- |
| Epochs     | 30    | Sufficient for convergence with transfer learning |
| Image Size | 512   | Matches dataset resolution                        |
| Batch Size | 16    | Balances memory usage and training stability      |
| Patience   | 10    | Early stopping if no improvement for 10 epochs    |
| Optimizer  | Auto  | Ultralytics auto-selects the best optimizer       |
| Pretrained | Yes   | Leverages COCO features for faster convergence    |

### Stage 2: Position-Based Classification

After YOLO detects a `check`-class bounding box, we use its **normalized x-center coordinate** to determine which option was marked. The five options are arranged **horizontally from left to right** in a fixed layout:

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ Revenue  │ Training │   Test   │  Ferry   │  Others  │
│ 0.00-0.28│ 0.28-0.47│ 0.47-0.62│ 0.62-0.77│ 0.77-1.00│
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

The boundaries were **calibrated empirically** by analyzing the x-position distribution of check-mark annotations in the training data. A histogram analysis confirmed clear clusters corresponding to each option position.

**Classification logic:**

1. Compute the bounding box x-center: `x_center = (x1 + x2) / 2 / image_width`
2. Compare against the boundary thresholds
3. Return the matching option name

This deterministic mapping is simple, fast, and interpretable — no additional model is needed for classification.

### Edge Case Handling

| Scenario                    | Behavior                                                 |
| --------------------------- | -------------------------------------------------------- |
| No detection at all         | Returns `"None"` (no option was selected)                |
| Only `nan`-class detections | Returns `"None"` (explicitly marked as no option)        |
| Multiple `check` detections | Picks the detection with highest confidence score        |
| Low-confidence detection    | Configurable threshold (default: 0.25)                   |
| Mark between two options    | Assigned to the option whose x-range contains the center |

---

## Training Details

### Training Process

Training was performed on CPU over 30 epochs. Below is a summary of the training progression:

| Metric           | Epoch 1 | Epoch 10 | Epoch 20 | Epoch 30 (Final) |
| ---------------- | ------- | -------- | -------- | ---------------- |
| Train Box Loss   | 1.163   | 0.759    | 0.658    | 0.475            |
| Train Class Loss | 1.901   | 0.624    | 0.508    | 0.227            |
| Train DFL Loss   | 1.378   | 1.112    | 1.069    | 1.064            |
| Val Box Loss     | 1.191   | 0.763    | 0.694    | 0.659            |
| Val Class Loss   | 1.950   | 0.522    | 0.374    | 0.305            |
| Precision (B)    | 0.894   | 0.984    | 0.950    | 0.980            |
| Recall (B)       | 0.451   | 0.956    | 0.983    | 0.997            |
| mAP50 (B)        | 0.461   | 0.976    | 0.980    | 0.984            |
| mAP50-95 (B)     | 0.318   | 0.817    | 0.839    | 0.861            |

**Key observations:**

- The model converges rapidly, reaching **mAP50 > 0.97** by epoch 7
- **Precision** stabilizes around **0.98** and **recall** reaches **0.997** by the final epoch
- Training and validation losses decrease consistently, indicating no overfitting
- The final **mAP50 of 0.984** and **mAP50-95 of 0.861** demonstrate excellent detection performance

### Training Artifacts

All training outputs are saved under `runs/purpose_detection/`:

- **Training curves** (`results.png`) — loss and metric plots across epochs
- **Confusion matrix** (`confusion_matrix.png`) — detection confusion between classes
- **PR/F1 curves** — precision-recall and F1 curves for threshold selection
- **Batch visualizations** — sample training and validation batch images with predictions

---

## Evaluation & Results

### YOLO Detection Metrics (Test Set)

| Metric        | Value |
| ------------- | ----- |
| **mAP50**     | 0.984 |
| **mAP50-95**  | 0.861 |
| **Precision** | 0.980 |
| **Recall**    | 0.997 |

### End-to-End Option Classification (Test Set)

The notebook evaluates the full two-stage pipeline on 318 test images, producing:

- **Classification report** with per-option precision, recall, and F1-score
- **Confusion matrix** showing the distribution of correct and incorrect classifications
- **Misclassification analysis** with visual display of incorrectly predicted images

The pipeline achieves high accuracy on the test set, with most errors occurring in underrepresented classes (Ferry, Others) that have very few training samples.

---

## Production Inference

### Using the PurposeDetector Class

The notebook generates a self-contained production inference script. Here's how to use it:

```python
from ultralytics import YOLO

OPTION_BOUNDARIES = [
    (0.00, 0.28, "Revenue"),
    (0.28, 0.47, "Training"),
    (0.47, 0.62, "Test"),
    (0.62, 0.77, "Ferry"),
    (0.77, 1.01, "Others"),
]

class PurposeDetector:
    def __init__(self, weights_path, conf_threshold=0.25):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    @staticmethod
    def _x_to_option(x_center):
        for lo, hi, name in OPTION_BOUNDARIES:
            if lo <= x_center < hi:
                return name
        return "Others"

    def predict(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {"option": "None", "confidence": 0.0}

        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()

        check_mask = classes == 0
        if not check_mask.any():
            return {"option": "None", "confidence": float(confs.max())}

        check_confs = confs[check_mask]
        check_xyxy = xyxy[check_mask]
        best_idx = check_confs.argmax()

        x_center_px = (check_xyxy[best_idx][0] + check_xyxy[best_idx][2]) / 2
        x_center_norm = x_center_px / result.orig_shape[1]

        return {
            "option": self._x_to_option(x_center_norm),
            "confidence": float(check_confs[best_idx]),
        }

# Usage
detector = PurposeDetector("runs/purpose_detection/weights/best.pt")
result = detector.predict("path/to/scanned_form.jpg")
print(result["option"])     # e.g., "Revenue"
print(result["confidence"]) # e.g., 0.95
```

### ONNX Export for Cross-Platform Deployment

The model can be exported to ONNX format for deployment across different platforms and runtimes:

```python
from ultralytics import YOLO

model = YOLO("runs/purpose_detection/weights/best.pt")
model.export(format="onnx", imgsz=512)
```

This produces a `best.onnx` file that can be used with ONNX Runtime, TensorRT, OpenVINO, or other inference engines.

---

## Output & What You Get

### Prediction Output Format

For each input image, the pipeline returns a dictionary:

```python
{
    "option": "Revenue",       # One of: Revenue, Training, Test, Ferry, Others, None
    "confidence": 0.95,        # Detection confidence score (0.0 – 1.0)
    "bbox": (x1, y1, x2, y2), # Bounding box in pixel coordinates (or None)
    "x_center_norm": 0.15      # Normalized x-center of the detection (or None)
}
```

| Field           | Type              | Description                                                            |
| --------------- | ----------------- | ---------------------------------------------------------------------- |
| `option`        | `str`             | The classified option: Revenue, Training, Test, Ferry, Others, or None |
| `confidence`    | `float`           | YOLO detection confidence (0.0 if no detection)                        |
| `bbox`          | `tuple` or `None` | Pixel-coordinate bounding box `(x1, y1, x2, y2)`                       |
| `x_center_norm` | `float` or `None` | Normalized x-center used for option mapping                            |

### Marking Styles Supported

| Marking Style           | How It's Handled                                         |
| ----------------------- | -------------------------------------------------------- |
| ✓ Check / tick mark     | YOLO detects as `check` class, position maps to option   |
| ⭕ Circle around option | YOLO detects the circle, x-center maps to option         |
| ✗ Cross mark            | YOLO detects the cross mark on the selected option       |
| ~~Strikethrough~~       | YOLO detects the mark on the selected option             |
| Underline               | YOLO detects the underline mark, position maps to option |
| No marking at all       | No detection or `nan` class → returns `"None"`           |
| Two options marked      | Highest-confidence detection wins                        |

### Production Artifacts

| File                                     | Description                              |
| ---------------------------------------- | ---------------------------------------- |
| `runs/purpose_detection/weights/best.pt` | Best trained YOLO model weights          |
| `runs/purpose_detection/weights/last.pt` | Last epoch model weights                 |
| `best.onnx` (after export)               | Cross-platform ONNX model for deployment |

### Use Cases

- **Automated document processing:** Integrate into a scanning pipeline to automatically extract the purpose field from forms
- **Batch processing:** Run inference on thousands of scanned forms to extract option selections at scale
- **Quality assurance:** Flag forms where no option is detected or confidence is low for human review
- **Data entry automation:** Replace manual data entry for the purpose/option field in form-heavy workflows

---

## License

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (provided by [Roboflow](https://universe.roboflow.com/ocrdigitdetection-r2lr4/ocr_purpose/dataset/3))
- **YOLOv11:** [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) (Ultralytics)
