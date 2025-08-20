# 🩺 Diabetes Risk Screening (Streamlit)

A lightweight Streamlit web app that estimates **diabetes risk probability** from seven survey-style inputs using a trained **XGBoost** (or any sklearn-compatible) model saved as a `.joblib`.

> **Screening, not diagnosis.** Outputs are model-based estimates intended to prompt follow‑up, not to replace clinical testing.

---

## 1) Overview

**Inputs (order matters):**

```
["BMI", "HighBP", "HighChol", "GenHlth", "Age", "PhysActivity", "DiffWalk"]
```

- `BMI` (float, kg/m²)  
- `HighBP`, `HighChol`, `PhysActivity`, `DiffWalk` (binary: 0/1; in the UI these are Yes/No)  
- `GenHlth` (ordinal 1–5; 1 = Excellent … 5 = Poor)  
- `Age` (integer, years)

**Outputs:**  
- Risk probability (0–1, shown as % in the UI)  
- Binary decision (“at risk” / “not at risk”) based on an adjustable threshold (default 0.50; screening often prefers ~0.35 for higher recall).

---

## 2) Project structure

```
diabetes-ui/
  models/
    xgb_7feat.joblib              # trained model artifact (example)
  diabetes_risk_ui_streamlit.py   # Streamlit app (single-model)
  multi_model_diabetes_ui.py      # (optional) multi-model compare/ensemble app
  requirements.txt                # (optional) pinned deps
  README.md
```

> You can rename the model file; keep it inside `models/` or upload it via the app sidebar.

---

## 3) Environment setup

### Option A — Anaconda (recommended, Windows-friendly)

```bash
conda create -n diabetes-ui python=3.10 -y
conda activate diabetes-ui

# Core deps
pip install streamlit scikit-learn joblib pandas numpy
# XGBoost (Windows easiest via conda-forge)
conda install -c conda-forge xgboost -y
```

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter** → choose **Conda (diabetes-ui)**.

### Option B — Python venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install streamlit scikit-learn joblib pandas numpy xgboost
```

---

## 4) Run the app

From the project folder:

```bash
# activate env first (conda activate diabetes-ui OR source .venv/bin/activate)
python -m streamlit run diabetes_risk_ui_streamlit.py
```

Open the **Local URL** (usually `http://localhost:8501`). In the sidebar:
- **Model folder**: set to `models` (or upload a `.joblib`)
- **Select a model file**: choose your model (e.g., `xgb_7feat.joblib`)
- Enter inputs → **Estimate my risk**

---

## 5) Train & save a 7‑feature model (example, XGBoost)

If your data frame is `X` with all columns and target `y`, train on **exactly the 7 features** to match the UI:

```python
FEATURES7 = ["BMI","HighBP","HighChol","GenHlth","Age","PhysActivity","DiffWalk"]

X_train7 = X_train[FEATURES7].copy()
X_test7  = X_test[FEATURES7].copy()

from xgboost import XGBClassifier
from joblib import dump
import os

xgb7 = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    random_state=42, eval_metric="logloss"
)
xgb7.fit(X_train7, y_train)

os.makedirs("models", exist_ok=True)
dump(xgb7, "models/xgb_7feat.joblib")
print("Saved: models/xgb_7feat.joblib")
```

> **Important:** The model must be trained on those 7 columns in that **exact order**. Otherwise you’ll see a “feature_names mismatch” error at inference.

### (Optional) Probability calibration

```python
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

X7 = X[FEATURES7]; Y = y
X_tr, X_cal, y_tr, y_cal = train_test_split(X7, Y, test_size=0.2, stratify=Y, random_state=42)

xgb7.fit(X_tr, y_tr)
cal = CalibratedClassifierCV(xgb7, method="isotonic", cv="prefit")
cal.fit(X_cal, y_cal)

dump(cal, "models/xgb_7feat_cal.joblib")
```

Use `xgb_7feat_cal.joblib` in the app to present better-calibrated probabilities.

---

## 6) Batch predictions (CSV)

Upload a CSV with exactly these headers:

```csv
BMI,HighBP,HighChol,GenHlth,Age,PhysActivity,DiffWalk
31.2,1,1,4,58,0,1
24.5,0,0,2,33,1,0
```

The app will score each row and let you download a results CSV with `risk_probability` and `predicted_high_risk` columns.

---

## 7) Common issues & fixes

- **feature_names mismatch**  
  The model was trained on more/different columns. Retrain on the exact **7-feature** schema above and re‑save.

- **Streamlit warning about ipykernel**  
  You tried to run Streamlit inside Jupyter. Run from terminal:  
  `python -m streamlit run diabetes_risk_ui_streamlit.py`.

- **Dropdown shows no models**  
  Ensure the **Model folder** points to the directory that contains your `.joblib` (e.g., `models`). Press Enter after pasting the path or refresh the browser tab.

- **Wrong interpreter**  
  VS Code bottom-right should show `Python 3.x (diabetes-ui)`; otherwise **Python: Select Interpreter** and open a new terminal.

---

## 8) Optional: Multi‑model / Ensemble app

To compare several saved models or do a soft‑vote ensemble, run:

```bash
python -m streamlit run multi_model_diabetes_ui.py
```

Place multiple `.joblib` files in `models/` (e.g., `lr_pipeline.joblib`, `rf_pipeline.joblib`, `svm_pipeline.joblib`, `xgb_7feat.joblib`, etc.).

---

## 9) Safety & privacy

- No personal identifiers are required; inputs are used only for on‑device inference during the session.  
- This tool is for **screening**; it does **not** diagnose disease. Clinical decisions require confirmatory testing and professional oversight.


