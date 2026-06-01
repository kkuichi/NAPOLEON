"""
HIERARCHICKÝ PRÍSTUP - DVA MODELY
Model 2: detekcia typu poruchy

S rovnakým predspracovaním, feature engineering a feature selection ako prvy model.
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


print("\n" + "="*80)
print("MODEL 2: MULTICLASS (Klasifikácia typu poruchy)")
print("="*80)

# [1] NAČÍTANIE A PRÍPRAVA DÁTOVÝCH SETOV (len poruchy)
print("\n[1] Načítavam a pripravujem dataset (len poruchy)...")
df_multi_data = pd.read_parquet("df_daily_vzorky.parquet").dropna().reset_index(drop=True)
df_multi_data = df_multi_data[df_multi_data['Typ poruchy'] > 0].copy()
print(f"    Dataset: {df_multi_data.shape}")

# Generalizuj typy prudove a napatove
# poznamka - porucha 13 (vykrateny prud vo faze 1) je interne zaznamenana pocas preprocessingu ako 10, preto aj ta 10tka, ked chceme generalizovat, doplnit typy 14 a 15
mapping_multi = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 13:1}

# Generalizuj typy prudove a napatove a po jednotlivych fazach
#mapping_multi = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 1, 8: 2, 9: 3, 10: 1, 13:1}

df_multi_data['y_multi'] = df_multi_data['Typ poruchy'].map(mapping_multi)

# Preindexuj triedy z 1-4 na 0-3 (XGBoost v CV padá inak)
print(f"    Triedy pred preindexovaním: {sorted(df_multi_data['y_multi'].unique().tolist())}")
df_multi_data['y_multi'] = df_multi_data['y_multi'] - 1  # 1-4 -> 0-3

print(f"    Rozdelenie cieľa (po preindexovaní 0-3):")
print(df_multi_data['y_multi'].value_counts().sort_index().to_string())

# [2] FEATURE ENGINEERING (ROVNAKÉ)
print("\n[2] Feature engineering...")
df_multi_data['u_asymmetry_mean'] = df_multi_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].max(axis=1) - \
                                     df_multi_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].min(axis=1)
df_multi_data['i_total_mean'] = df_multi_data['i1_norm_mean'] + df_multi_data['i2_norm_mean'] + df_multi_data['i3_norm_mean']
df_multi_data['i_total_std'] = df_multi_data['i1_norm_std'] + df_multi_data['i2_norm_std'] + df_multi_data['i3_norm_std']
df_multi_data['u_total_std'] = df_multi_data['u1_norm_std'] + df_multi_data['u2_norm_std'] + df_multi_data['u3_norm_std']
df_multi_data['i_zeros_total'] = df_multi_data['i1_norm_zeros'] + df_multi_data['i2_norm_zeros'] + df_multi_data['i3_norm_zeros']
df_multi_data['i_spikes_total'] = df_multi_data['i1_norm_spikes'] + df_multi_data['i2_norm_spikes'] + df_multi_data['i3_norm_spikes']
df_multi_data['p_total_mean'] = df_multi_data['u1_norm_mean']*df_multi_data['i1_norm_mean'] + \
                                 df_multi_data['u2_norm_mean']*df_multi_data['i2_norm_mean'] + \
                                 df_multi_data['u3_norm_mean']*df_multi_data['i3_norm_mean']
df_multi_data['i_unbalance_mean'] = (df_multi_data[['i1_norm_mean', 'i2_norm_mean', 'i3_norm_mean']].max(axis=1) - \
                                      df_multi_data[['i1_norm_mean', 'i2_norm_mean', 'i3_norm_mean']].min(axis=1)) / \
                                     (df_multi_data['i_total_mean'] + 0.1)
df_multi_data['u_unbalance_mean'] = (df_multi_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].max(axis=1) - \
                                      df_multi_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].min(axis=1)) / \
                                     (df_multi_data[['u1_norm_mean', 'u2_norm_mean', 'u3_norm_mean']].mean(axis=1) + 0.1)
df_multi_data['z_approx_1'] = df_multi_data['u1_norm_mean'] / (df_multi_data['i1_norm_mean'] + 0.1)
df_multi_data['z_approx_2'] = df_multi_data['u2_norm_mean'] / (df_multi_data['i2_norm_mean'] + 0.1)
df_multi_data['z_approx_3'] = df_multi_data['u3_norm_mean'] / (df_multi_data['i3_norm_mean'] + 0.1)
df_multi_data['ui_std_ratio_sum'] = (df_multi_data['i1_norm_std'] + df_multi_data['i2_norm_std'] + df_multi_data['i3_norm_std']) / \
                                     (df_multi_data['u1_norm_std'] + df_multi_data['u2_norm_std'] + df_multi_data['u3_norm_std'] + 1e-6)
print("    ✓ 13 engineered features vytvorených")

# [3] PRÍPRAVA FEATURES
print("\n[3] Príprava features...")
all_num_cols_multi = df_multi_data.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols_multi = ['Typ poruchy', 'y_multi', 'eic']
feature_cols_multi = [c for c in all_num_cols_multi if c not in exclude_cols_multi and '_lag' not in c]

print(f"    NaN po predspracovani: {df_multi_data.isna().sum().sum()}")
if df_multi_data.isna().any().any():
    print("    ⚠ Detekované NaN hodnoty!")
    df_binary_data = df_multi_data.dropna()
    print(f"    Po odstránení NaN: {df_multi_data.shape}")

X_multi_full = df_multi_data[feature_cols_multi].fillna(0)
y_multi_full = df_multi_data['y_multi']
groups_multi_full = df_multi_data['eic']

print(f"    Počet features: {len(feature_cols_multi)}")

# [4] TRAIN/TEST SPLIT
print("\n[4] Train/Test split (StratifiedGroupKFold)...")
gs_multi = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=13)
train_idx_multi, test_idx_multi = next(gs_multi.split(X_multi_full, y_multi_full.astype(int), groups=groups_multi_full))

X_train_multi_full = X_multi_full.iloc[train_idx_multi].copy()
y_train_multi_full = y_multi_full.iloc[train_idx_multi].copy()
X_test_multi = X_multi_full.iloc[test_idx_multi].copy()
y_test_multi = y_multi_full.iloc[test_idx_multi].copy()
groups_train_multi = groups_multi_full.iloc[train_idx_multi].copy()

print(f"    Train: {len(X_train_multi_full)}")
print(f"    Test: {len(X_test_multi)}")
print(f"    Train EICs: {groups_train_multi.nunique()}, Test EICs: {groups_multi_full.iloc[test_idx_multi].nunique()}")

# [5] FEATURE SELECTION
print("\n[5] Feature selection (Mutual Information)...")
mi_scores_multi = mutual_info_classif(X_train_multi_full, y_train_multi_full.astype(int), random_state=42)
fi_df_multi = pd.DataFrame({'feature': X_train_multi_full.columns, 'importance': mi_scores_multi}).sort_values('importance', ascending=False)

top_20_features_multi = fi_df_multi.head(40)['feature'].tolist()
print(f"    TOP 20 atribútov:")
for i, f in enumerate(top_20_features_multi[:20], 1):
    print(f"      {i:2d}. {f}")

X_train_multi = X_train_multi_full[top_20_features_multi]
X_test_multi = X_test_multi[top_20_features_multi]

print(f"    X_train: {X_train_multi.shape}, X_test: {X_test_multi.shape}")

# [6] GRIDsearchcv S HYPERPARAMETER TUNING
print("\n[6] GridSearchCV tuning...")
cv_multi = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

models_multi = {
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
}

param_grids_multi = {
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

best_models_multi = {}
results_multi = {}

for model_name, model in models_multi.items():
    print(f"\n  --- {model_name} ---")
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grids_multi[model_name],
        scoring="f1_weighted",
        cv=cv_multi,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    gscv.fit(X_train_multi, y_train_multi_full, groups=groups_train_multi)

    best_models_multi[model_name] = gscv.best_estimator_
    results_multi[model_name] = {"best_params": gscv.best_params_, "cv_score": gscv.best_score_, "model": gscv.best_estimator_}

    print(f"    CV F1: {gscv.best_score_:.4f}")
    print(f"    Best params: {gscv.best_params_}")

best_model_multi_name = max(results_multi, key=lambda x: results_multi[x]["cv_score"])
print(f"\n  Najlepší: {best_model_multi_name} (F1={results_multi[best_model_multi_name]['cv_score']:.4f})")

# [7] VYHODNOTENIE NA TEST SETE
print("\n[7] Vyhodnotenie na test sete...")
y_pred_multi = results_multi[best_model_multi_name]["model"].predict(X_test_multi)

print("\n" + "="*70)
print("VYSLEDKY - MULTICLASS MODEL")
print("="*70)
print(f"\nNajlepší model: {best_model_multi_name}")
print(f"\n{classification_report(y_test_multi, y_pred_multi, digits=4)}")
print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.4f}")
print(f"Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test_multi, y_pred_multi),
                   index=[f"Actual {i}" for i in range(0, 2)],
                   columns=[f"Pred {i}" for i in range(0, 2)]).to_string())

# Ulož model + metadata pre scoring
multiclass_bundle = {
    "model": results_multi[best_model_multi_name]["model"],
    "features": top_20_features_multi,
    "class_labels": [1, 2],
    "label_offset": 1,
    "best_model_name": best_model_multi_name,
}
joblib.dump(multiclass_bundle, f'model_multiclass_{best_model_multi_name.lower()}_bundle.joblib')
joblib.dump(results_multi[best_model_multi_name]["model"], f'model_multiclass_{best_model_multi_name.lower()}.joblib')
print(f"\n✓ Model uložený: model_multiclass_{best_model_multi_name.lower()}.joblib")
print(f"✓ Bundle uložený: model_multiclass_{best_model_multi_name.lower()}_bundle.joblib")

print("\n" + "="*80)
print("✅ Multiclass natrenovany!")
print("="*80)
