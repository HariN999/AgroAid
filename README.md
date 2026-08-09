# 🌿 AgroAid

AgroAid is an AI-assisted plant disease diagnostic and agricultural decision-support application. The project evolved from a basic plant disease classifier into a complete decision-support system that helps users understand crop conditions, visualizes prediction confidence, maps alternative classifications, and provides vendor-neutral treatment and cultural management guidance.

---

## 🏗️ Architecture & Pipeline Flow

The diagnostic pipeline operates as follows:

```text
       User Uploads Leaf Image (Browser)
                      ↓
          Flask Web Application
                      ↓
       Image Extension & Format Validation
                      ↓
          Pillow Format Verification
                      ↓
          RGB Image Mode Normalization
                      ↓
     Resized to 224 × 224 & Tensor Conversion
                      ↓
       Custom CNN (CNN.py Architecture)
                      ↓
         Inference Logits Computation
                      ↓
           Softmax Score Evaluation
                      ↓
    Index-Safe Pandas Query (disease_info.csv)
                      ↓
   Treatment Option Mapping (treatment_info.csv)
                      ↓
             Interactive Web Result
```

---

## 🤝 Model Attribution & Lineage

AgroAid currently uses a Convolutional Neural Network (CNN) and model training lineage based on the open-source [Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection) repository by `manthan89-py`. 

The original repository provides the architectural foundation for the plant disease classification model. AgroAid's primary contribution is the complete re-engineering of the application layer surrounding the model to support safe, interactive decision-support workloads. We do not claim ownership of the underlying model weights or original dataset.

---

## 🛠️ What AgroAid Adds

AgroAid substantially re-engineers and hardens the application layer around the core classification model to support safe, decision-support operations:

### 1. Robust Diagnostic Ingestion
- **Image Content Validation:** Verifies file structures using Pillow's `verify()` to prevent spoofed uploads.
- **Strict Format Enforcements:** Restricts uploads to trusted extensions (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`) and forces conversion to RGB to prevent crashes on RGBA, Grayscale, or Palette inputs.
- **collison-free Storage:** Generates random UUID4 filenames to prevent directory traversal risks.
- **Immediate Disk Cleanup:** Automatically deletes uploads within `try/finally` blocks immediately post-inference to prevent storage leaks.

### 2. Uncertainty-Aware Presentation
- **Model Confidence Score:** Visualizes the model's output distribution as High ($\ge 85\%$), Moderate ($50–84.9\%$), or Low ($< 50\%$) confidence categories.
- **Alternative Match Listing:** Exposes top-2 alternative matches ("Other possible matches") if their softmax probability exceeds a 5% product heuristic.
- **Low-Confidence Warnings:** Flags prediction uncertainty to prompt the user to capture clearer, better-lit crop images.

### 3. Vendor-Neutral Treatment Guidance
- **Chemical-free Healthy Crops:** Mapped strictly to `None` with zero chemical suggestions.
- **Organic & Active Ingredient Mapping:** Replaces commercial store links and brand-name recommendations with active ingredients and general categories (e.g., *Systemic Fungicide*, *Acaricide*).
- **Verification Badging:** Explicitly highlights verified active ingredients vs. unverified mappings (`Not established`).

### 4. Application Engineering
- **Stateless Request Handling:** The API operates without global mutable prediction variables.
- **Robust Fail-safes:** Implements lazy model-loading and displays controlled error views if the model weights checkpoint is missing.
- **Verification Suite:** Includes integration and ML architecture tests executing via `pytest`.

---

## 💡 Product Philosophy

AgroAid is intentionally designed as an **agricultural decision-support tool**, not an autonomous diagnostic authority. The system operates on the principle of transparency: it communicates what the model predicts, details the model's prediction confidence, maps alternative possibilities, and clearly outlines when treatment data requires external verification. 

> "AI diagnosis is an aid and not a substitute for qualified agricultural advice or local extension guidance."

---

## 🚀 Key Features

- **AI-Assisted Leaf Classification:** Automated plant disease detection across 39 classes.
- **Model Confidence Score:** Clear display of classified model confidence.
- **Other Possible Matches:** Visual list of top alternative classifications.
- **Safe Ingestion Pipeline:** Sanitization, image validation, and auto-cleanup.
- **Vendor-Neutral Treatment Guidance:** Active ingredient recommendations without e-commerce links.
- **Prevention Guidance:** Detailed cultural management suggestions (sanitation, air circulation, crop rotation).
- **Responsive Layout:** Modern, glassmorphic layout suitable for mobile and desktop viewports.
- **Stateless Multilingual Backend:** Translation endpoints supporting English, Hindi, and Telugu.

---

## 🧠 Machine Learning

The classifier is built using PyTorch and trained on the open-source Mendeley PlantVillage distribution:
- **Input Dimension:** $224\times224$ pixels.
- **Architecture:** 4 Convolutional blocks (BatchNorm + MaxPool) followed by 2 fully connected dense layers.
- **Training Hyperparameters:** Adam optimizer, CrossEntropyLoss, batch size 64, trained for 5 epochs.
- **Performance:** A historical accuracy of ~97–98% on the validation/test splits of the PlantVillage distribution was reported in the original training notebook. This has not been independently reproduced on out-of-distribution real-world samples for this release.

### Confidence & Limitations
Softmax outputs are displayed as a *Model Confidence Score*. This score reflects the model's output distribution and is not a calibrated measure of diagnostic certainty. Alternative matches are alternative model outputs, not confirmed diagnoses. Real-world performance may differ due to lighting, crop variety, and crop environment.

---

## 🌿 Treatment Mappings Status

Treatment recommendations inside `data/treatment_info.csv` are audited and verified:
- **Verified Treatments:** **8 classes** (active ingredients verified from source logs).
- **Needs Verification:** **18 classes** (commercial names mapped to `Not established`).
- **Healthy/None:** **13 classes** (mapped strictly to `None`).

---

## 📂 Project Directory Structure

Every path listed below matches the current active state of the repository:

```text
AgroAid/
├── app.py                      # Flask Application Entrypoint & Config
├── CNN.py                      # PyTorch CNN Class Architecture Definition
├── requirements.txt            # Project Dependencies
├── data/
│   ├── disease_info.csv        # Disease description and prevention steps
│   └── treatment_info.csv      # Vendor-neutral treatment options dataset
├── models/
│   ├── Plant Disease...ipynb   # Original training Jupyter Notebook
│   └── plant_disease_model_1_latest.pt  # (Local Only) Excluded 210MB PyTorch Checkpoint
├── static/
│   └── uploads/
│       └── .gitkeep            # Ignored runtime upload folder tracker
├── templates/
│   ├── base.html               # Standardized Glassmorphic Layout
│   ├── home.html               # Supported Crop Catalog
│   ├── index.html              # Diagnostic scanning & symptom checker
│   └── submit.html             # Diagnostic Report page
└── tests/
    ├── __init__.py
    ├── test_app.py             # Route, form, upload, and translation tests
    └── test_model.py           # Model load, preprocessing, and inference tests
```

---

## ⚙️ Setup & Local Runtime

The 210 MB trained weights checkpoint (`plant_disease_model_1_latest.pt`) is excluded from git tracking. You must place it in the `models/` directory prior to running leaf inference.

### 1. Set Up Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Application
By default, the application runs on port `5001`:
```bash
python app.py
```
Open `http://localhost:5001` in your web browser.

### 3. Run Production Server
For WSGI production environments, Gunicorn is supported:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

---

## 🧪 Testing

The repository contains automated verification tests checking static loading, file uploads, translation fallbacks, and model inference limits.

Run the test suite:
```bash
pytest
```
*Current test status:* **`14 passed`**.

---

## ⚠️ Known Limitations

- **Weights Requirement:** Inference fails gracefully if the weights checkpoint is not placed in the `models/` directory.
- **Translation Services:** Translation features depend on Google Translate's free engine via `deep-translator`. Heavy usage may lead to rate-limiting, which falls back to English.
- **Model Calibration:** Softmax probabilities are not calibrated. Low confidence may indicate out-of-distribution input.
- **Treatment verification:** Several classes lack verified active ingredients and require manual extension verification.

---

## 🤝 Acknowledgements

This project extends the plant classification work of the [Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection) project by `manthan89-py`. We credit their repository for providing the foundational CNN architecture and dataset mapping.
