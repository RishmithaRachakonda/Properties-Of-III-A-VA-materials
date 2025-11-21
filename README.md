# Prediction of bandgap, formation energy and bulk modulus of IIIA-VA group materials using ML model (XGBoost)

## Overview
This project develops machine-learning models using XGBoost to predict three key properties of IIIA–VA semiconductor materials:

- **Bandgap (eV)**
- **Bulk Modulus (GPa)**
- **Formation Energy (eV/atom)**

These properties are usually obtained through Density Functional Theory (DFT), which is accurate but computationally expensive. This project replaces DFT with fast ML surrogate models using structural and compositional features extracted from the JARVIS-DFT dataset.

A detailed explanation of the methodology and results is provided in `4_reports/MLMI_Report.pdf`.

---
## Dataset

**Source:** JARVIS-DFT

Only strict IIIA–VA binary compounds were selected.

**Elements used for filtering:**

- **Group IIIA:** B, Al, Ga, In, Tl  
- **Group VA:** N, P, As, Sb, Bi

The dataset includes structural information such as lattice parameters, atomic positions, and DFT-computed material properties.

---

## Method Summary

Each prediction script follows this workflow:

### 1. Load Data
- Load the JARVIS-DFT dataset.
- Apply strict IIIA–VA compositional filtering.

### 2. Extract Structural Features
- Lattice constants: *a, b, c*
- Angles: *alpha, beta, gamma*
- Unit cell volume
- Interatomic distance statistics

### 3. Extract Compositional Features
- Electronegativity difference  
- Atomic number and atomic mass relations  
- Average valence electrons  
- Molar volume  
- Melting point  

### 4. Preprocessing
- Clean the dataset  
- Standardize and scale the features  

### 5. Model Training
- Train an XGBoost regressor  

### 6. Evaluation
- Metrics: R², MAE, RMSE  
- Save trained models, metrics, and parity plots  

---

## How to Run

### Install dependencies
```bash
pip install pandas numpy pymatgen scikit-learn xgboost matplotlib tqdm
```
### Run Scripts
```bash
python 2_ML_train/bandgap_prediction.py
python 2_ML_train/bulk_modulus_prediction.py
python 2_ML_train/formation_energy_prediction.py
```
## Outputs

- **Metrics:** `3_results/metrics/*.txt`
- **Trained Models:** `3_results/models/*.json`
- **Plots:** `3_results/plots/*.png`
- **Final Report:** `4_reports/MLMI_Report.pdf`
- **Presentation:** `presentation/MLMI_PPT.pdf`

---

## Requirements

- Python 3.8+
- pandas
- numpy
- pymatgen
- scikit-learn
- xgboost
- matplotlib
- tqdm

