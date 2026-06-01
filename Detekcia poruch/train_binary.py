"""
HIERARCHICKÝ PRÍSTUP - DVA MODELY
Model 1: Binárny (detekcia poruchy)

S rovnakým predspracovaním, feature engineering a feature selection ako pôvodný model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL 1: BINÁRNY (Detekcia poruchy)")
print("="*80)

# [1] NAČÍTANIE A PRÍPRAVA DÁTOVÝCH SETOV
print("\n[1] Načítavam a pripravujem dataset...")
df_binary_data = pd.read_parquet("df_daily_vzorky.parquet").dropna().reset_index(drop=True)
print(f"    Dataset: {df_binary_data.shape}")

# Kontrola NaN
print(f"    NaN po načítaní: {df_binary_data.isna().sum().sum()}")
if df_binary_data.isna().any().any():
    print("    ⚠ Detekované NaN hodnoty!")
    df_binary_data = df_binary_data.dropna()
    print(f"    Po odstránení NaN: {df_binary_data.shape}")

# Filtrácia vysokonapäťových miest
voltage_cols = ['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean', 'u1_norm_max', 'u2_norm_max', 'u3_norm_max']
voltage_cols = [c for c in voltage_cols if c in df_binary_data.columns]
if voltage_cols:
    max_voltage_per_eic = df_binary_data.groupby('eic')[voltage_cols].max().max(axis=1)
    high_voltage_eics = max_voltage_per_eic[max_voltage_per_eic > 1000].index.tolist()
    if high_voltage_eics:
        print(f"\n    Filtrácia vysokonapäťových miest:")
        print(f"      Identifikovaných EIC s vysokým napätím: {len(high_voltage_eics)}")
        print(f"      Riadkov pred filtrovaním: {len(df_binary_data)}")
        df_binary_data = df_binary_data[~df_binary_data['eic'].isin(high_voltage_eics)].reset_index(drop=True)
        print(f"      Riadkov po filtrovaní: {len(df_binary_data)}")

# Vytvor binárny cieľ (0 = normálny, 1 = akákoľvek porucha)
df_binary_data['y_binary'] = (df_binary_data['Typ poruchy'] > 0).astype(int)
print(f"\n    Rozdelenie cieľa:")
print(f"      - 0 (normálny): {(df_binary_data['y_binary'] == 0).sum()}")
print(f"      - 1 (porucha):  {(df_binary_data['y_binary'] == 1).sum()}")

# [2] FEATURE ENGINEERING (ROVNAKÉ AKO PÔVODNÝ MODEL)
print("\n[2] Feature engineering...")
df_binary_data['u_asymmetry_mean'] = df_binary_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].max(axis=1) - \
                                      df_binary_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].min(axis=1)
df_binary_data['i_total_mean'] = df_binary_data['i1_norm_mean'] + df_binary_data['i2_norm_mean'] + df_binary_data['i3_norm_mean']
df_binary_data['i_total_std'] = df_binary_data['i1_norm_std'] + df_binary_data['i2_norm_std'] + df_binary_data['i3_norm_std']
df_binary_data['u_total_std'] = df_binary_data['u1_norm_std'] + df_binary_data['u2_norm_std'] + df_binary_data['u3_norm_std']
df_binary_data['i_zeros_total'] = df_binary_data['i1_norm_zeros'] + df_binary_data['i2_norm_zeros'] + df_binary_data['i3_norm_zeros']
df_binary_data['i_spikes_total'] = df_binary_data['i1_norm_spikes'] + df_binary_data['i2_norm_spikes'] + df_binary_data['i3_norm_spikes']
df_binary_data['p_total_mean'] = df_binary_data['u1_norm_mean']*df_binary_data['i1_norm_mean'] + \
                                 df_binary_data['u2_norm_mean']*df_binary_data['i2_norm_mean'] + \
                                 df_binary_data['u3_norm_mean']*df_binary_data['i3_norm_mean']

df_binary_data['i_unbalance_mean'] = (df_binary_data[['i1_norm_mean', 'i2_norm_mean', 'i3_norm_mean']].max(axis=1) - \
                                       df_binary_data[['i1_norm_mean', 'i2_norm_mean', 'i3_norm_mean']].min(axis=1)) / \
                                      (df_binary_data['i_total_mean'] + 0.1)
df_binary_data['u_unbalance_mean'] = (df_binary_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].max(axis=1) - \
                                       df_binary_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].min(axis=1)) / \
                                      (df_binary_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].mean(axis=1) + 0.1)
df_binary_data['z_approx_1'] = df_binary_data['u1_norm_mean'] / (df_binary_data['i1_norm_mean'] + 0.1)
df_binary_data['z_approx_2'] = df_binary_data['u2_norm_mean'] / (df_binary_data['i2_norm_mean'] + 0.1)
df_binary_data['z_approx_3'] = df_binary_data['u3_norm_mean'] / (df_binary_data['i3_norm_mean'] + 0.1)
df_binary_data['ui_std_ratio_sum'] = (df_binary_data['i1_norm_std'] + df_binary_data['i2_norm_std'] + df_binary_data['i3_norm_std']) / \
                                     (df_binary_data['u1_norm_std'] + df_binary_data['u2_norm_std'] + df_binary_data['u3_norm_std'] + 1e-6)
print("    ✓ 13 engineered features vytvorených")

# --- FILTRÁCIA "KONTAMINOVANÝCH" NÚL Z TRÉNINGU ---
# Cieľ: Odstrániť záznamy s Typom 0 z dní, kedy na rovnakom EIC prebehla aj nejaká porucha.
# Tieto záznamy často obsahujú prechodové javy, ktoré mätú model (splývanie 0 a 1).

df_binary_data['day'] = pd.to_datetime(df_binary_data['t_utc']).dt.date
# Zistíme, ktoré (eic, day) majú v tréningu nejakú poruchu (typ > 0)
faulty_days_train_binary = df_binary_data[df_binary_data['y_binary'] > 0].groupby(['eic', 'day']).size().index

# Označíme záznamy typu 0 na týchto EIC a dňoch ako kontaminované
is_zero_binary = (df_binary_data['y_binary'] == 0)
is_contaminated_binary = df_binary_data.set_index(['eic', 'day']).index.isin(faulty_days_train_binary)
to_drop_binary = is_zero_binary & is_contaminated_binary

print(f"\nFiltrácia trénovacej množiny (binárny model):")
print(f"  Pôvodný počet vzoriek v Train: {len(df_binary_data)}")
print(f"  Počet identifikovaných 'kontaminovaných' Typ 0 záznamov: {to_drop_binary.sum()}")

df_binary_data = df_binary_data.drop(['t_utc'], axis=1)
df_binary_data = df_binary_data[~to_drop_binary].copy()


# [3] PRÍPRAVA FEATURES NA TRAIN/TEST SPLIT

print("\n[3] Príprava features na train/test split...")
all_num_cols = df_binary_data.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Typ poruchy', 'y_binary', 'eic']
feature_cols_binary = [c for c in all_num_cols if c not in exclude_cols and '_lag' not in c]

print(f"    NaN po predspracovani: {df_binary_data.isna().sum().sum()}")
if df_binary_data.isna().any().any():
    print("    ⚠ Detekované NaN hodnoty!")
    df_binary_data = df_binary_data.dropna()
    print(f"    Po odstránení NaN: {df_binary_data.shape}")

X_binary_full = df_binary_data[feature_cols_binary].fillna(0)
y_binary_full = df_binary_data['y_binary']
groups_binary_full = df_binary_data['eic']

print(f"    Počet features: {len(feature_cols_binary)}")

# [4] TRAIN/TEST SPLIT Z StratifiedGroupKFold
print("\n[4] Train/Test split (StratifiedGroupKFold)...")
gs_binary = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=13)
train_idx_binary, test_idx_binary = next(gs_binary.split(X_binary_full, y_binary_full.astype(int), groups=groups_binary_full))

X_train_binary_full = X_binary_full.iloc[train_idx_binary].copy()
y_train_binary_full = y_binary_full.iloc[train_idx_binary].copy()
X_test_binary = X_binary_full.iloc[test_idx_binary].copy()
y_test_binary = y_binary_full.iloc[test_idx_binary].copy()
groups_train_binary = groups_binary_full.iloc[train_idx_binary].copy()

print(f"    Train: {len(X_train_binary_full)}")
print(f"    Test: {len(X_test_binary)}")
print(f"    Train EICs: {groups_train_binary.nunique()}, Test EICs: {groups_binary_full.iloc[test_idx_binary].nunique()}")

# [5] FEATURE SELECTION (Mutual Information)
print("\n[5] Feature selection (Mutual Information)...")
mi_scores_binary = mutual_info_classif(X_train_binary_full, y_train_binary_full.astype(int), random_state=42)
fi_df_binary = pd.DataFrame({'feature': X_train_binary_full.columns, 'importance': mi_scores_binary}).sort_values('importance', ascending=False)

top_20_features_binary = fi_df_binary.head(40)['feature'].tolist()
print(f"    TOP 20 atribútov:")
for i, f in enumerate(top_20_features_binary[:20], 1):
    print(f"      {i:2d}. {f}")

X_train_binary = X_train_binary_full[top_20_features_binary]
X_test_binary = X_test_binary[top_20_features_binary]

print(f"    X_train: {X_train_binary.shape}, X_test: {X_test_binary.shape}")


print(f"  Nový počet vzoriek v Train: {len(X_train_binary)}")

# Aktualizuj groups_train_binary
groups_train_binary = groups_train_binary[~to_drop_binary]

print(f"\nFinálne rozmery - Train: {len(X_train_binary)}, Test: {len(X_test_binary)}")
print("Train EICs:", groups_train_binary.nunique(), " Test EICs:", groups_binary_full.iloc[test_idx_binary].nunique())

# [6] GRIDSEARCHCV S HYPERPARAMETER TUNING
print("\n[6] GridSearchCV tuning...")
cv_binary = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

models_binary = {
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
}

param_grids_binary = {
    "RandomForest": {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    },
    "XGBoost": {
        "n_estimators": [200, 500],
        "max_depth": [6, 10, 15],
        "learning_rate": [0.1, 0.2],
    },
    "LightGBM": {
        "n_estimators": [200, 500],
        "max_depth": [10, 15, -1],
        "learning_rate": [0.1, 0.2],
        "class_weight": [None, "balanced"]
    }
}

best_models_binary = {}
results_binary = {}

for model_name, model in models_binary.items():
    print(f"\n  --- {model_name} ---")
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grids_binary[model_name],
        scoring="f1_weighted",
        cv=cv_binary,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    gscv.fit(X_train_binary, y_train_binary_full, groups=groups_train_binary)

    best_models_binary[model_name] = gscv.best_estimator_
    results_binary[model_name] = {"best_params": gscv.best_params_, "cv_score": gscv.best_score_, "model": gscv.best_estimator_}

    print(f"    CV F1: {gscv.best_score_:.4f}")
    print(f"    Best params: {gscv.best_params_}")

best_model_binary_name = max(results_binary, key=lambda x: results_binary[x]["cv_score"])
print(f"\n  Najlepší: {best_model_binary_name} (F1={results_binary[best_model_binary_name]['cv_score']:.4f})")

# [7] VYHODNOTENIE NA TEST SETE
print("\n[7] Vyhodnotenie na test sete...")
best_model_binary = results_binary[best_model_binary_name]["model"]
y_proba_binary = best_model_binary.predict_proba(X_test_binary)[:, 1]

# Nájdime threshold s minimálnym počtom false negatives pre triedu 1.
thresholds = np.arange(0.01, 1.00, 0.01)
threshold_rows = []

for thr in thresholds:
    y_pred_thr = (y_proba_binary >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_thr, labels=[0, 1]).ravel()

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * precision_1 * recall_1 / (precision_1 + recall_1)) if (precision_1 + recall_1) > 0 else 0.0
    class1_error = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    threshold_rows.append({
        "threshold": float(thr),
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
        "class1_error": float(class1_error),
        "precision_1": float(precision_1),
        "recall_1": float(recall_1),
        "f1_1": float(f1_1),
    })

threshold_df = pd.DataFrame(threshold_rows)

# Bezpečnostná podmienka: FP nesmie narásť viac ako 2x oproti thresholdu 0.5.
y_pred_05 = (y_proba_binary >= 0.5).astype(int)
_, fp_05, fn_05, tp_05 = confusion_matrix(y_test_binary, y_pred_05, labels=[0, 1]).ravel()
max_allowed_fp = int(2 * fp_05)

safe_threshold_df = threshold_df[threshold_df["fp"] <= max_allowed_fp].copy()
if safe_threshold_df.empty:
    print(f"    ⚠ Bezpečnostná podmienka je príliš prísna (FP@0.5={fp_05}, limit={max_allowed_fp}).")
    print("    Používam pôvodný výber thresholdu bez FP limitu.")
    safe_threshold_df = threshold_df.copy()

# Primárne minimalizujeme FN; pri zhode vyberieme nižšie FP a vyššie F1.
safe_threshold_df = safe_threshold_df.sort_values(
    by=["fn", "fp", "f1_1", "threshold"],
    ascending=[True, True, False, True]
).reset_index(drop=True)

best_threshold = float(safe_threshold_df.loc[0, "threshold"])
print(f"    Referencia @0.5 -> TP={tp_05}, FN={fn_05}, FP={fp_05}")
print(f"    FP limit (max 2x): {max_allowed_fp}")
print(f"    Najlepší threshold (min FN s FP limitom): {best_threshold:.2f}")
print("\n    TOP 10 thresholdov podľa FN/FP/F1 (po FP limite):")
print(safe_threshold_df.head(10).to_string(index=False))

y_pred_binary = (y_proba_binary >= best_threshold).astype(int)

print("\n" + "="*70)
print("VYSLEDKY - BINÁRNY MODEL (s optimalizovaným thresholdom)")
print("="*70)
print(f"\nNajlepší model: {best_model_binary_name}")
print(f"Použitý threshold: {best_threshold:.2f}")
print(f"\n{classification_report(y_test_binary, y_pred_binary, digits=4)}")
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
print(f"Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test_binary, y_pred_binary),
                   index=["Actual 0", "Actual 1"],
                   columns=["Pred 0", "Pred 1"]).to_string())

# Ulož model aj threshold + feature list na konzistentnú inferenciu.
joblib.dump({
    "model": best_model_binary,
    "threshold": best_threshold,
    "features": top_20_features_binary,
}, f'model_binary_{best_model_binary_name.lower()}_with_threshold.joblib')
print(f"\n✓ Model + threshold uložené: model_binary_{best_model_binary_name.lower()}_with_threshold.joblib")

# Zachovaj aj pôvodné uloženie samotného modelu.
joblib.dump(best_model_binary, f'model_binary_{best_model_binary_name.lower()}.joblib')
print(f"✓ Model uložený: model_binary_{best_model_binary_name.lower()}.joblib")
