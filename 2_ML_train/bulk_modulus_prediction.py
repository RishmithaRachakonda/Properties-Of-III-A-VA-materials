# pip install jarvis-tools pymatgen

import pandas as pd
import numpy as np
from jarvis.db.figshare import data
from pymatgen.core import Composition, Element
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Load dataset
dft_3d = data("dft_3d")
df = pd.DataFrame(dft_3d)
print(f"Total entries: {len(df)}")

# Strict III–V filter
group_13 = {'B', 'Al', 'Ga', 'In', 'Tl'}
group_15 = {'N', 'P', 'As,', 'Sb', 'Bi'}

def is_strict_III_V(formula):
    try:
        comp = Composition(formula)
        elems = list(comp.as_dict().keys())
        if len(elems) != 2:
            return False
        return ((elems[0] in group_13 and elems[1] in group_15) or
                (elems[1] in group_13 and elems[0] in group_15))
    except:
        return False

df = df[df["formula"].apply(is_strict_III_V)].reset_index(drop=True)
print(f"III–V semiconductors found: {len(df)}")

# Target column
target_col = 'bulk_modulus_kv'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
print(f"Predicting: {target_col}")

# Structural features
def extract_structural_features(row):
    try:
        atoms_data = row['atoms']
        lattice_mat = np.array(atoms_data['lattice_mat'])
        a, b, c = atoms_data['abc']
        alpha, beta, gamma = atoms_data['angles']
        volume = np.abs(np.linalg.det(lattice_mat))
        coords = np.array(atoms_data['coords'])
        elements = atoms_data['elements']

        distances = []
        if len(coords) > 1:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    distances.append(np.linalg.norm(coords[i] - coords[j]))

        min_dist = np.min(distances) if distances else 0
        max_dist = np.max(distances) if distances else 0
        avg_dist = np.mean(distances) if distances else 0

        lattice_ratio_ab = a / b if b != 0 else 1
        lattice_ratio_ac = a / c if c != 0 else 1
        angle_deviation = np.std([alpha, beta, gamma])

        return pd.Series({
            'lattice_a': a, 'lattice_b': b, 'lattice_c': c,
            'lattice_volume': volume,
            'alpha_angle': alpha, 'beta_angle': beta, 'gamma_angle': gamma,
            'lattice_ratio_ab': lattice_ratio_ab, 'lattice_ratio_ac': lattice_ratio_ac,
            'angle_deviation': angle_deviation,
            'min_atomic_dist': min_dist, 'max_atomic_dist': max_dist,
            'avg_atomic_dist': avg_dist,
            'num_atoms': len(elements),
            'is_cubic': 1 if (a == b == c and all(angle == 90 for angle in [alpha, beta, gamma])) else 0,
            'is_tetragonal': 1 if (a == b != c and all(angle == 90 for angle in [alpha, beta, gamma])) else 0,
        })
    except:
        return pd.Series({k: np.nan for k in [
            'lattice_a','lattice_b','lattice_c','lattice_volume',
            'alpha_angle','beta_angle','gamma_angle',
            'lattice_ratio_ab','lattice_ratio_ac','angle_deviation',
            'min_atomic_dist','max_atomic_dist','avg_atomic_dist',
            'num_atoms','is_cubic','is_tetragonal'
        ]})

print("Extracting structural features...")
df = pd.concat([df, df.apply(extract_structural_features, axis=1)], axis=1)

# Composition features
def extract_composition_features(row):
    try:
        comp = Composition(row["formula"])
        elems = [Element(e) for e in comp.elements]
        fracs = [comp.get_atomic_fraction(e.symbol) for e in comp.elements]
        mean_Z = sum(e.Z * w for e, w in zip(elems, fracs))
        mean_en = np.nanmean([e.X for e in elems])
        mean_mass = sum(e.atomic_mass * w for e, w in zip(elems, fracs))
        en_diff = abs(elems[0].X - elems[1].X)
        mean_radius = sum(e.atomic_radius * w for e, w in zip(elems, fracs))

        return pd.Series({
            "mean_Z": mean_Z,
            "mean_en": mean_en,
            "mean_mass": mean_mass,
            "en_diff": en_diff,
            "mean_radius": mean_radius,
        })
    except:
        return pd.Series({
            "mean_Z": np.nan, "mean_en": np.nan, "mean_mass": np.nan,
            "en_diff": np.nan, "mean_radius": np.nan
        })

print("Extracting composition features...")
df = pd.concat([df, df.apply(extract_composition_features, axis=1)], axis=1)

# Top 10 selected features
important_features = [
    'mean_Z', 'mean_mass', 'mean_radius', 'lattice_a',
    'min_atomic_dist', 'density', 'lattice_volume',
    'spg_number', 'en_diff', 'is_cubic'
]

available_features = [f for f in important_features if f in df.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Handle missing values
numeric_cols = df[available_features].select_dtypes(include=[np.number]).columns
categorical_cols = [c for c in available_features if c not in numeric_cols]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna(subset=[target_col]).reset_index(drop=True)
print(f"Final dataset size: {len(df)} samples")

# Prepare inputs
X = df[available_features]
y = df[target_col]

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train XGBoost model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# Training performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print(f"\nTrain R² = {train_r2:.3f}")

# Test performance
y_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nBulk Modulus Prediction Results")
print(f"Test R²   = {test_r2:.3f}")
print(f"Test MAE  = {test_mae:.3f} GPa")
print(f"Test RMSE = {test_rmse:.3f} GPa")

# Parity plot
import matplotlib.pyplot as plt

plt.figure(figsize=(7,7), dpi=120)
plt.scatter(y_test, y_pred, s=50, alpha=0.75, edgecolor='black')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2)

plt.title("Parity Plot — Bulk Modulus", fontsize=20, weight='bold')
plt.xlabel("Actual Bulk Modulus (GPa)", fontsize=18, weight='bold')
plt.ylabel("Predicted Bulk Modulus (GPa)", fontsize=18, weight='bold')

plt.xticks(fontsize=16, weight='bold')
plt.yticks(fontsize=16, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
