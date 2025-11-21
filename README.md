# Prediction of Bandgap, Formation Energy, and Bulk Modulus of IIIA–VA Semiconductors using XGBoost

This repository contains a machine-learning based framework for predicting key electronic and mechanical properties of IIIA–VA semiconductor materials. Using XGBoost, the model estimates **bandgap**, **formation energy**, and **bulk modulus** of binary IIIA–VA compounds with high accuracy, offering a fast alternative to Density Functional Theory (DFT) calculations.

---

## ✨ Project Overview
Traditional DFT calculations are accurate but computationally expensive, often requiring hours or days per material.  
This project builds a surrogate ML model using **254 DFT-computed IIIA–VA compounds** (sourced from the Materials Project) to perform high-speed property prediction.

The model uses **145 compositional + structural features**, enabling accurate predictions with significantly reduced computation time.

---

## 🧪 Target Properties
The model predicts the following material properties:

- **Bandgap (eV)** – Governs optical/electronic behavior  
- **Formation Energy (eV/atom)** – Stability indicator  
- **Bulk Modulus (GPa)** – Mechanical stiffness  

---

## 📁 Dataset
- Dataset source: **The Materials Project**  
- Filtering: Only binary combinations of  
  - Group IIIA: B, Al, Ga, In, Tl  
  - Group VA: N, P, As, Sb, Bi  
- Final dataset size: **254 compounds**

Each material entry contains:
- DFT-calculated target properties  
- Crystal structure (lattice parameters, atoms)  
- Elemental descriptors (electronegativity, atomic mass, etc.)

---

## 🧩 Feature Engineering
Two major feature categories were used:

### **1. Compositional Features**
Examples:
- Electronegativity difference (en_diff)  
- Atomic mass sum/diff  
- Average valence electrons  
- Polarizability estimate  
- Average melting point  
- Molar volume  

### **2. Structural Features**
Derived from atomic coordinates and lattice:

- Lattice constants (a, b, c)  
- Lattice angles (α, β, γ)  
- Unit cell volume  
- Structural density  
- Minimum/average interatomic distances  

A total of **145 features** were used after cleaning and normalization.

---

## 🛠️ Methodology
The ML workflow includes:

1. Data acquisition & IIIA–VA filtering  
2. Compositional + structural feature extraction  
3. Feature domain mapping  
4. Data preprocessing (missing value handling, normalization)  
5. Model development using **XGBoost Regressor**  
6. Multi-output regression for 3 target properties  
7. Validation using R², MAE, RMSE  

A complete pipeline diagram is shown in the report (page 8) :contentReference[oaicite:1]{index=1}.

---

## ⚙️ Model Architecture (XGBoost)

Key hyperparameters:
- trees: **2000**  
- learning_rate: **0.02**  
- max_depth: **6**  
- subsample: **0.8**  
- colsample_bytree: **0.8**  
- regularization: α = 1.0, λ = 2.0  
- tree_method: **hist**  

The model was implemented using **MultiOutputRegressor** to predict all three properties simultaneously.

---

## 📊 Results

| Property | Train R² | Test R² | MAE | RMSE |
|---------|----------|---------|------|--------|
| **Bandgap (eV)** | 0.994 | **0.949** | 0.2689 | 0.4362 |
| **Bulk Modulus (GPa)** | 0.999 | **0.946** | 16.971 | 26.446 |
| **Formation Energy (eV/atom)** | 0.999 | **0.736** | 0.153 | 0.334 |

📌 Parity plots (report page 10) show excellent alignment of predictions with DFT data for bandgap and bulk modulus; formation energy shows moderate scatter but remains reliable for screening.

---

## 🔍 Feature Importance Insights
The model captured physically meaningful relationships:

- **Bandgap** → depends on valence electrons & electronic descriptors  
- **Bulk modulus** → influenced by lattice geometry & interatomic spacing  
- **Formation energy** → depends on mass, density, molar volume, melting point  

This confirms that the ML model is not a black box but aligns with materials science principles.

---

## 🚀 Advantages of This ML Model
- **Millions of times faster** than DFT  
- Suitable for **high-throughput screening**  
- Predicts 3 properties simultaneously  
- Requires minimal computational resources  
- Applicable to hypothetical new IIIA–VA materials  

---

## 📦 Repository Structure (recommended)
