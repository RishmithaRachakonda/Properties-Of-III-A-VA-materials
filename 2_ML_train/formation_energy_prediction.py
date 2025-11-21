# pip install jarvis-tools pymatgen xgboost scikit-learn matplotlib

import pandas as pd
import numpy as np
from jarvis.db.figshare import data
from pymatgen.core import Composition, Element
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load dataset
print("Loading JARVIS-DFT data...")
dft_3d = data("dft_3d")
df = pd.DataFrame(dft_3d)
print(f"Total entries: {len(df)}")

# Filter III–V compounds
group_13 = {'B', 'Al', 'Ga', 'In', 'Tl'}
group_15 = {'N', 'P', 'As', 'Sb', 'Bi'}

def is_strict_III_V(formula):
    try:
        comp = Composition(formula)
        elems = list(comp.as_dict().keys())
        if len(elems) != 2:
            return False
        return (elems[0] in group_13 and elems[1] in group_15) or \
               (elems[1] in group_13 and elems[0] in group_15)
    except:
        return False

df = df[df["formula"].apply(is_strict_III_V)].reset_index(drop=True)
print(f"III–V semiconductors found: {len(df)}")

# Target property
target_col = 'formation_energy_peratom'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
print(f"Predicting: {target_col}")

# Structural features
def extract_structural_features(row):
    try:
        atoms_data = row['atoms']
        abc = atoms_data['abc']
        lattice_b = abc[1]
        lattice_mat = np.array(atoms_data['lattice_mat'])
        volume = np.abs(np.linalg.det(lattice_mat))
        num_atoms = len(atoms_data['elements'])
        structural_density = num_atoms / volume if volume > 0 else 0
        return pd.Series({'lattice_b': lattice_b, 'structural_density': structural_density})
    except:
        return pd.Series({'lattice_b': np.nan, 'structural_density': np.nan})

print("Extracting structural features...")
df = pd.concat([df, df.apply(extract_structural_features, axis=1)], axis=1)

# Composition features
def extract_composition_features(row):
    try:
        comp = Composition(row["formula"])
        elems = [Element(e) for e in comp.elements]
        fracs = [comp.get_atomic_fraction(e.symbol) for e in comp.elements]

        if len(elems) < 2:
            return pd.Series([np.nan]*7)

        e1, e2 = elems[0], elems[1]

        Z_sum = e1.Z + e2.Z
        Z_diff = abs(e1.Z - e2.Z)
        mass_sum = e1.atomic_mass + e2.atomic_mass

        mv1 = e1.molar_volume if e1.molar_volume else 0
        mv2 = e2.molar_volume if e2.molar_volume else 0
        avg_molar_volume = mv1 * fracs[0] + mv2 * fracs[1]

        mp1 = e1.melting_point if e1.melting_point else 0
        mp2 = e2.melting_point if e2.melting_point else 0
        avg_melting_point = mp1 * fracs[0] + mp2 * fracs[1]

        r1 = e1.atomic_radius if e1.atomic_radius else 1
        r2 = e2.atomic_radius if e2.atomic_radius else 1
        polarizability_est = (r1**3 + r2**3) / 2

        val1 = 3 if e1.symbol in group_13 else 5
        val2 = 3 if e2.symbol in group_13 else 5
        avg_valence = val1 * fracs[0] + val2 * fracs[1]

        return pd.Series({
            "mass_sum": mass_sum,
            "avg_valence": avg_valence,
            "avg_molar_volume": avg_molar_volume,
            "Z_diff": Z_diff,
            "Z_sum": Z_sum,
            "avg_melting_point": avg_melting_point,
            "polarizability_est": polarizability_est
        })
    except:
        return pd.Series({
            "mass_sum": np.nan, "avg_valence": np.nan, "avg_molar_volume": np.nan,
            "Z_diff": np.nan, "Z_sum": np.nan, "avg_melting_point": np.nan,
            "polarizability_est": np.nan
        })

print("Extracting composition features...")
df = pd.concat([df, df.apply(extract_composition_features, axis=1)], axis=1)

# Feature list
requested_features = [
    "mass_sum", "structural_density", "avg_valence", "avg_molar_volume",
    "lattice_b", "Z_diff", "avg_melting_point", "polarizability_est",
    "spg_number", "Z_sum"
]

available_features = [f for f in requested_features if f in df.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Clean missing values
df = df.dropna(subset=[target_col]).reset_index(drop=True)

numeric_cols = df[available_features].select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

print(f"Final dataset size: {len(df)} samples")

# Preprocessing
X = df[available_features]
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# XGBoost model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Training metrics
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print("\nTraining Performance")
print(f"Train R² = {train_r2:.3f}")

# Test metrics
y_pred = model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nTest Performance")
print(f"Test R²   = {test_r2:.3f}")
print(f"Test MAE  = {test_mae:.3f}")
print(f"Test RMSE = {test_rmse:.3f}")

# Parity plot
plt.figure(figsize=(7,7), dpi=120)
plt.scatter(y_test, y_pred, s=50, alpha=0.75, edgecolor='black')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.title("Parity Plot — Formation Energy", fontsize=20)
plt.xlabel("Actual (eV/atom)", fontsize=18)
plt.ylabel("Predicted (eV/atom)", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:")
for i in range(len(available_features)):
    print(f"{i+1}. {available_features[indices[i]]}: {importances[indices[i]]:.4f}")
