import pandas as pd
import numpy as np
from pymatgen.core import Composition, Element
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("Loading JARVIS-DFT data...")

# -------------------------------
# Load dataset (multiple fallback options)
# -------------------------------
try:
    from jarvis.db.figshare import data
    dft_3d = data("dft_3d")
    df = pd.DataFrame(dft_3d)
    print("Loaded from JARVIS package")
except Exception:
    try:
        df = pd.read_json("dft_3d.json")
        print("Loaded from local JSON file")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception:
        try:
            df = pd.read_csv("dft_3d.csv")
            print("Loaded from local CSV file")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            print("\n  JARVIS not installed and no local data file found!")
            print("   Install jarvis-tools or download 'dft_3d.json' / 'dft_3d.csv' into this folder.")
            raise

print(f"Total entries: {len(df)}")

# -------------------------------
# Strict III–V Filter
# -------------------------------
group_13 = {'B','Al','Ga','In','Tl'}
group_15 = {'N','P','As','Sb','Bi'}

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

if "formula" not in df.columns:
    raise ValueError(" 'formula' column not found in dataset. Cannot filter III–V compounds.")

df = df[df["formula"].apply(is_strict_III_V)].reset_index(drop=True)
print(f"Strict III–V semiconductors found: {len(df)}")
if len(df) == 0:
    raise ValueError(" No III–V semiconductors found — check dataset version or 'formula' column.")

# -------------------------------
# Candidate non-DFT features and space-group columns
# -------------------------------
non_dft_features = [
    'spg_number', 'dimensionality', 'nat',
    'frac_Al', 'frac_As', 'frac_Ga', 'frac_In', 'frac_N',
    'frac_B', 'frac_Sb', 'frac_P', 'frac_Bi', 'frac_Tl',
    'X_avg', 'atomic_radius_avg', 'mendeleev_number_avg',
    'electronegativity_avg', 'avg_dist', 'max_dist', 'min_dist'
]
spg_cols = [c for c in df.columns if c.startswith('spg_symbol_')]
available_non_dft = [f for f in non_dft_features if f in df.columns]
print(f"Found {len(available_non_dft)} non-DFT features + {len(spg_cols)} space group features")

# -------------------------------
# Extract structural features from 'atoms' field
# -------------------------------
def extract_structural_features(atoms_dict):
    try:
        if not isinstance(atoms_dict, dict):
            return {k: np.nan for k in ['lattice_a','lattice_b','lattice_c',
                    'alpha','beta','gamma','lattice_anisotropy','unit_cell_volume',
                    'is_cubic','is_orthogonal','avg_lattice_constant','atoms_per_cell',
                    'structural_density']}
        abc = atoms_dict.get('abc', [0, 0, 0])
        a, b, c = abc[0], abc[1], abc[2]
        angles = atoms_dict.get('angles', [90, 90, 90])
        alpha, beta, gamma = angles[0], angles[1], angles[2]
        lattice_anisotropy = max(abc) / min(abc) if min(abc) > 0 else 1
        volume = a * b * c * np.sqrt(
            1 - np.cos(np.radians(alpha))**2
              - np.cos(np.radians(beta))**2
              - np.cos(np.radians(gamma))**2
              + 2*np.cos(np.radians(alpha))*np.cos(np.radians(beta))*np.cos(np.radians(gamma))
        )
        is_cubic = int(abs(a - b) < 0.01 and abs(b - c) < 0.01)
        is_orthogonal = int(all(abs(ang - 90) < 1 for ang in angles))
        avg_lattice = np.mean(abc)
        coords = atoms_dict.get('coords', [])
        num_atoms = len(coords)
        density_estimate = num_atoms / volume if (volume is not None and volume > 0) else 0
        return {
            'lattice_a': a, 'lattice_b': b, 'lattice_c': c,
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'lattice_anisotropy': lattice_anisotropy,
            'unit_cell_volume': volume,
            'is_cubic': is_cubic, 'is_orthogonal': is_orthogonal,
            'avg_lattice_constant': avg_lattice,
            'atoms_per_cell': num_atoms,
            'structural_density': density_estimate
        }
    except:
        return {k: np.nan for k in ['lattice_a','lattice_b','lattice_c',
                'alpha','beta','gamma','lattice_anisotropy','unit_cell_volume',
                'is_cubic','is_orthogonal','avg_lattice_constant','atoms_per_cell',
                'structural_density']}

print("\nExtracting structural features from 'atoms' field...")
structural_features = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Structural"):
    atoms_dict = row.get('atoms', {})
    structural_features.append(extract_structural_features(atoms_dict))
structural_df = pd.DataFrame(structural_features)

# -------------------------------
# Enhanced composition feature extraction
# -------------------------------
def extract_enhanced_features(row):
    try:
        comp = Composition(row.formula)
        elems = [Element(e) for e in comp.elements]
        fracs = [comp.get_atomic_fraction(e.symbol) for e in elems]

        if elems[0].symbol in group_15:
            elems = elems[::-1]
            fracs = fracs[::-1]

        elem_III, elem_V = elems[0], elems[1]
        frac_III, frac_V = fracs[0], fracs[1]

        en_III, en_V = elem_III.X or 0, elem_V.X or 0
        en_diff = abs(en_III - en_V)
        en_ratio = en_V / en_III if en_III else 1.0

        Z_III, Z_V = elem_III.Z, elem_V.Z
        Z_diff, Z_sum = abs(Z_III - Z_V), Z_III + Z_V

        r_III = elem_III.atomic_radius or getattr(elem_III, "atomic_radius_calculated", None) or 1.0
        r_V = elem_V.atomic_radius or getattr(elem_V, "atomic_radius_calculated", None) or 1.0
        r_sum, r_diff = (r_III + r_V), abs(r_III - r_V)
        r_ratio = r_III / r_V if r_V else 1.0
        size_mismatch = r_diff / r_sum if r_sum else 0

        mass_III, mass_V = elem_III.atomic_mass or 0, elem_V.atomic_mass or 0
        mass_sum, mass_diff = mass_III + mass_V, abs(mass_III - mass_V)
        mass_asymmetry = abs(np.log(mass_III / mass_V)) if (mass_III and mass_V) else 0

        ionicity = en_diff / (en_III + en_V) if (en_III + en_V) else 0
        covalency = 1 - ionicity

        molar_vol_III = elem_III.molar_volume or 0
        molar_vol_V = elem_V.molar_volume or 0
        avg_molar_volume = frac_III * molar_vol_III + frac_V * molar_vol_V

        mp_III = elem_III.melting_point or 0
        mp_V = elem_V.melting_point or 0
        avg_melting_point = frac_III * mp_III + frac_V * mp_V
        mp_diff = abs(mp_III - mp_V)

        avg_valence = frac_III * 3 + frac_V * 5
        has_d_electrons = int(Z_III > 30 or Z_V > 30)

        ea_III = elem_III.electron_affinity or 0
        ea_V = elem_V.electron_affinity or 0
        ea_diff = abs(ea_III - ea_V)

        polarizability_est = (r_III**3 + r_V**3) / 2
        period_diff = abs(elem_III.row - elem_V.row)

        return pd.Series({
            "en_diff": en_diff, "en_ratio": en_ratio,
            "Z_diff": Z_diff, "Z_sum": Z_sum,
            "r_sum": r_sum, "r_diff": r_diff, "r_ratio": r_ratio,
            "size_mismatch": size_mismatch,
            "mass_sum": mass_sum, "mass_diff": mass_diff,
            "mass_asymmetry": mass_asymmetry,
            "ionicity": ionicity, "covalency": covalency,
            "avg_molar_volume": avg_molar_volume,
            "avg_melting_point": avg_melting_point, "mp_diff": mp_diff,
            "avg_valence": avg_valence, "has_d_electrons": has_d_electrons,
            "ea_diff": ea_diff, "polarizability_est": polarizability_est,
            "period_diff": period_diff
        })
    except Exception:
        return pd.Series({k: np.nan for k in [
            "en_diff","en_ratio","Z_diff","Z_sum","r_sum","r_diff","r_ratio",
            "size_mismatch","mass_sum","mass_diff","mass_asymmetry","ionicity",
            "covalency","avg_molar_volume","avg_melting_point","mp_diff",
            "avg_valence","has_d_electrons","ea_diff","polarizability_est",
            "period_diff"
        ]})

print("Extracting enhanced composition features...")
enhanced_features = [extract_enhanced_features(r) for r in tqdm(df.itertuples(), total=len(df), desc="Composition")]
enhanced_df = pd.DataFrame(enhanced_features)

df = pd.concat([df.reset_index(drop=True), structural_df, enhanced_df], axis=1)

# -------------------------------
# Use ONLY the user-specified top-10 features
# -------------------------------
selected_features = [
    'mass_sum',
    'structural_density',
    'avg_valence',
    'avg_molar_volume',
    'lattice_b',
    'Z_diff',
    'avg_melting_point',
    'polarizability_est',
    'Z_sum',
    'spg_number'
]

all_candidate_features = (available_non_dft + spg_cols + list(structural_df.columns) + list(enhanced_df.columns))

valid_selected = [f for f in selected_features if f in all_candidate_features]
invalid_selected = [f for f in selected_features if f not in all_candidate_features]

if invalid_selected:
    print(" The following selected features are NOT present in the dataframe:")
    print("   ", invalid_selected)

if len(valid_selected) == 0:
    raise ValueError(" None of the requested selected features exist in the dataframe!")

all_features = valid_selected
print(f"\nUsing {len(all_features)} manually selected features:")
print(all_features)

# -------------------------------
# Select single target property: Bandgap
# -------------------------------
target_col = "optb88vdw_bandgap"
if target_col not in df.columns:
    raise ValueError(f" Target column '{target_col}' not found in dataframe.")
print(f"Predicting target: {target_col}")

# -------------------------------
# Handle missing/infinite/large values -- fill with median
# -------------------------------
for col in all_features + [target_col]:
    if col in df.columns:
        df[col] = df[col].replace('na', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=[target_col]).reset_index(drop=True)

print(f"\nUsing {len(all_features)} features")
print(f"Target: {target_col}")
print(f"Final dataset size: {len(df)}")
if len(df) < 20:
    print("\n  WARNING: Very small dataset! Results may not be reliable.")

# -------------------------------
# Preprocessing
# -------------------------------
X = df[all_features].copy()
y = df[target_col].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------------
# Train/Test Split
# -------------------------------
test_size = 0.2 if len(df) >= 50 else 0.15
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=test_size, random_state=42)

# -------------------------------
# Clean training data for problematic values
# ------------------------------
print("\nCleaning training data before XGBoost...")
for col in X_train.columns:
    if X_train[col].isnull().any() or np.isinf(X_train[col]).any() or (X_train[col].abs() > 1e10).any():
        median_val = X_train[col].median()
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        X_train[col] = X_train[col].fillna(median_val if pd.notna(median_val) else 0)

if y_train.isnull().any() or np.isinf(y_train).any() or (y_train.abs() > 1e10).any():
    median_val = y_train.median()
    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    y_train = y_train.fillna(median_val if pd.notna(median_val) else 0)

# -------------------------------
# Train XGBoost regressor
# -------------------------------
print("\nTraining XGBoost model...")
xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    tree_method="hist",
    verbosity=0
)
xgb.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print("\n" + "="*90)
print(" MODEL PERFORMANCE (Selected Features Only)")
print("="*90)
print(f"{'Property':<35} {'Train R²':<10} {'Test R²':<10} {'MAE':<10} {'RMSE_train':<12} {'RMSE_test':<12}")
print("-"*90)
print(f"{target_col:<35} {r2_train:>10.3f}  {r2_test:>10.3f}  {mae_test:>8.3f}  {rmse_train:>10.3f}  {rmse_test:>10.3f}")
print("="*90)

# -------------------------------
# Feature importance
# -------------------------------
print("\nFeature importances:")
importances = xgb.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': importances
}).sort_values('importance', ascending=False)
print(feature_importance_df)

# -------------------------------
# Visualization: predicted vs true
# -------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_test, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
ax.set_xlabel(f'True {target_col}', fontsize=11, fontweight='bold')
ax.set_ylabel(f'Predicted {target_col}', fontsize=11, fontweight='bold')
ax.set_title(f'{target_col}\nR² = {r2_test:.3f}, MAE = {mae_test:.3f}, RMSE = {rmse_test:.3f}', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('iii_v_bandgap_selected_features_predictions.png', dpi=300, bbox_inches='tight')
print("\nPrediction plot saved to 'iii_v_bandgap_selected_features_predictions.png'")
print("\nAnalysis complete!")
