# Fungis — ML pipeline for spectra-based classification (fungi/bacteria)

This repository contains an end-to-end **machine learning pipeline** for classification tasks built around
a structured workflow:

1) **Data loading + preparation**
2) **Feature engineering + train/val/test split**
3) **Model training + evaluation** (multiple model families)

The codebase is script-driven (not a packaged library). File names intentionally encode the pipeline order
(e.g., `1_...`, `2_...`, `3_...`).

---

## What’s in here

### Data folders
The repository includes multiple data folders, including timestamped directories and prepared datasets, e.g.:

- `data/`
- `data_augmented/`
- several timestamped directories such as `Bacteria 20-06-24-...`, `Bacteria 26-06-24-...`, etc. :contentReference[oaicite:1]{index=1}

> Note: If any of these datasets are sensitive/proprietary, consider moving them out of the repo and using
> a download step + `.gitignore` instead.

### Pipeline scripts (high-level)
**Step 1 — ingestion / preparation**
- `1_dataindlaesning_og_forberedelse.py`
- `1_dataindlaesning_og_forberedelse_labels.py`
- variants with “mitgate” and outlier maps:
  - `1_dataindlaesning_og_forberedelse_labels_mitgate.py`
  - `1_dataindlaesning_og_forberedelse_labels_mitgate_outliermaps.py` :contentReference[oaicite:2]{index=2}

**Step 2 — feature engineering / split / visualization**
- `2_Feature engineering og splitting af data til maskinlæring*.py` (multiple iterations)
- `2_Feature engineering ... _outliermaps_7.py`
- `2_Visualisering_af_smoothed_data.py` :contentReference[oaicite:3]{index=3}

**Step 3 — training / evaluation (many model variants)**
A large set of training/evaluation scripts exist, including:
- Neural nets: `3_model_training_and_evaluation_CNN.py`, `..._FNN.py`, many `FNN_...` variants
- Bayesian optimization: `3_model_training_and_evaluation_Bayesian_Optimization.py`
- Gradient boosted trees: `3_model_training_and_evaluation_Gradient_Boosted_Trees*.py`
- Gradient descent experiments: `..._Gradieent_Descent1.py` :contentReference[oaicite:4]{index=4}

---

## Quick start

### 1) Create environment
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
2) Install dependencies
There is currently no pinned requirements.txt in the repo, so install a standard ML stack:

bash
Kopier kode
pip install numpy pandas scipy scikit-learn matplotlib joblib
If you use deep learning scripts (CNN/FNN), install one of these depending on what your scripts expect:

bash
Kopier kode
# TensorFlow (common for Keras-based CNN/FNN)
pip install tensorflow

# or PyTorch (if your CNN/FNN scripts are torch-based)
# pip install torch torchvision torchaudio
Recommendation: add a requirements.txt once you’ve confirmed the exact imports used by your scripts.

How to run (recommended order)
Step 1 — Data ingestion / labeling
Pick the variant you want (plain labels vs. “mitgate” vs. outlier maps):

bash
Kopier kode
python 1_dataindlaesning_og_forberedelse.py
# or
python 1_dataindlaesning_og_forberedelse_labels.py
# or
python 1_dataindlaesning_og_forberedelse_labels_mitgate.py
# or
python 1_dataindlaesning_og_forberedelse_labels_mitgate_outliermaps.py
Step 2 — Feature engineering + split
Choose the latest/most relevant iteration among the 2_... scripts:

bash
Kopier kode
python "2_Feature engineering og splitting af data til maskinlæring.py"
# or one of the numbered iterations:
python "2_Feature engineering og splitting af data til maskinlæring_6.py"
python "2_Feature engineering og splitting af data til maskinlæring_outliermaps_7.py"
Optional visualization:

bash
Kopier kode
python 2_Visualisering_af_smoothed_data.py
Step 3 — Train + evaluate a model family
Examples:

bash
Kopier kode
python 3_model_training_and_evaluation_CNN.py
python 3_model_training_and_evaluation_FNN.py
python 3_model_training_and_evaluation_Bayesian_Optimization.py
python 3_model_training_and_evaluation_Gradient_Boosted_Trees1.py
Outputs / results
The exact output locations depend on the scripts (some projects write to data/, some to new output folders).
If you want reproducible runs, standardize output paths, e.g.:

outputs/figures/

outputs/models/

outputs/metrics/

(Consider adding those folders + writing them consistently across 1_, 2_, 3_ steps.)

Reproducibility checklist (recommended)
To make experiments repeatable:

set random seeds (NumPy + framework)

log:

dataset version / folder used

preprocessing choices (smoothing, normalization, outlier handling)

split parameters

model hyperparameters

persist:

trained model artifacts

a single metrics summary file (CSV/JSON)

Notes on repo hygiene (recommended)
.idea/ suggests the repo was edited in an IDE; consider ignoring it if not needed.

Large raw datasets inside Git can become painful; consider Git LFS or external storage for big files.

Add a short repo description + topics in GitHub settings.

License
No license file is currently visible in the repo root. If you want this to be open source, add a LICENSE
(e.g., MIT/Apache-2.0). If not, state usage restrictions explicitly.

Contact
Maintainer: Tom Stevns
If you use this in a collaboration context, please open an Issue with:

dataset used

which 1_, 2_, 3_ scripts you ran

your environment (Python version + key packages)
