# Computer Vision Project — Multi-Class Natural Scene Image Classification

An end-to-end image classification pipeline that predicts one of six natural-scene
categories — `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` — from a photo,
using the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

Two models are trained and compared:
- **Baseline CNN** (trained from scratch) — test accuracy **83.70%**, macro F1 **0.8400**
- **Transfer learning** (MobileNetV2, pretrained + fine-tuned) — test accuracy **91.07%**, macro F1 **0.9122**

The transfer learning model is selected as the final deployed model (chosen by
**validation accuracy**, not test accuracy — see Section 17 of the notebook) and saved as
`final_model.keras`.

## Contents

| File | Description |
|---|---|
| `Computer_Vision_Image_Classification.ipynb` | Full notebook: EDA → preprocessing → augmentation → baseline CNN → transfer learning → evaluation → error analysis → Grad-CAM → inference. All interpretation/discussion cells compute their text live from that run's actual results, so they never go stale. |
| `final_model.keras` | Saved final trained model (produced by running the notebook) |
| `app.py` / `requirements.txt` | Streamlit app for uploading an image and getting a live prediction |
| `Project_Report.pdf` | 5-page write-up of the project and results |
| `README.md` | This file |

## Setup & Run Instructions

### 1. Open in Google Colab
Upload `Computer_Vision_Image_Classification.ipynb` to
[Google Colab](https://colab.research.google.com) (`File → Upload notebook`).

### 2. Enable a GPU runtime
`Runtime → Change runtime type → Hardware accelerator → GPU (T4)`.
The notebook will run on CPU too, but training will be significantly slower.

### 3. Kaggle authentication (for dataset download)
The notebook downloads the dataset automatically via `kagglehub`. The first time it runs
it will prompt for Kaggle credentials:
1. Go to your Kaggle account → **Settings** → **API** → **Create New Token**.
2. This downloads a `kaggle.json` file.
3. When prompted by `kagglehub` in the notebook, upload that file (or place it at
   `~/.kaggle/kaggle.json`).

No manual dataset download or unzip is required — Section 4 of the notebook handles it.

### 4. Run all cells, in order
`Runtime → Run all`. The full run (baseline CNN + transfer learning + fine-tuning)
takes roughly 20 minutes on a Colab GPU runtime.

> Sections 1-19 must run in order — later sections (evaluation, error analysis, transfer
> learning, Grad-CAM, inference, and the auto-generated discussion/conclusion cells) reuse
> variables and trained models created earlier.

> **Note on run-to-run variation:** exact numbers (accuracy, F1, which class pairs get
> confused most) can shift slightly between runs, since GPU training is not perfectly
> deterministic. `tf.config.experimental.enable_op_determinism()` is enabled in Section 3
> to minimize this, and every interpretation/discussion cell in the notebook computes its
> text live from that run's real variables — so the notebook is always internally
> consistent with whatever numbers your run actually produces, even if they differ
> slightly from the numbers in this README (captured from one specific run).

### 5. Outputs produced by the notebook
- `baseline_cnn_best.keras` / `transfer_model_best.keras` — best checkpoint of each model
  during training (by validation accuracy).
- `final_model.keras` — the final selected model, saved in Section 19.

## Running the Streamlit app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `final_model.keras` (downloaded from the notebook after it finishes running) is
in the same folder as `app.py`. See `app.py`'s docstring for deployment instructions
(e.g. Streamlit Community Cloud).

## Reloading the saved model later

```python
from tensorflow import keras
model = keras.models.load_model("final_model.keras")
```

## Project Structure (matches the notebook's sections)

1. Project Introduction
2. Business / Real-World Problem
3. Environment Setup
4. Dataset Loading
5. Dataset Understanding
6. Image EDA
7. Data Preprocessing
8. Train / Validation / Test Strategy
9. Data Augmentation
10. Baseline CNN
11. Model Training
12. Training Curves
13. Model Evaluation
14. Error Analysis
15. Transfer Learning Model
16. Model Comparison
17. Prediction on New Images
18. Bonus: Grad-CAM Explainability
19. Final Conclusion & Answers to Required Questions
20. References

## Key Results (from the reference run captured in this README)

| Metric | Baseline CNN | Transfer Learning (MobileNetV2) |
|---|---|---|
| Validation accuracy | 85.13% | 91.40% |
| Test accuracy | 83.70% | 91.07% |
| Macro F1-score | 0.8400 | 0.9122 |
| Total params | 424,006 | 2,265,670 |
| Training time | 812.3s | 406.4s |

**Per-class recall (baseline):** easiest — `forest` (95.8%); hardest — `glacier` (71.1%).

**Most confused classes:** `glacier` → `mountain` (120 test images), `street` →
`buildings` (94) — consistent with the visual similarity identified during EDA (Section 6).

**Model selection:** the final deployed model is chosen using **validation accuracy**
(baseline: 85.13% vs. transfer: 91.40%), which selects the transfer learning model — the
test set is only used afterward to report that model's unbiased final performance.

## References

- Dataset: <https://www.kaggle.com/datasets/puneet6060/intel-image-classification>
- Keras Applications: <https://keras.io/api/applications/>
- TensorFlow image classification guide: <https://www.tensorflow.org/tutorials/images/classification>
- TensorFlow transfer learning guide: <https://www.tensorflow.org/tutorials/images/transfer_learning>
