# Hierarchical pipeline (training + scoring)

Tento dokument pokrýva celú hierarchickú pipeline:

1. tréning binárneho modelu (`train_binary.py`)
2. tréning multiclass modelu (`train_multiclass.py`)
3. scoring eval dát (`score_hierarchical_eval.py`)

---

## 1) Trénovanie binárneho modelu (`train_binary.py`)

Skript trénuje model na detekciu poruchy:

- `0 = bez poruchy`
- `1 = porucha`

### Popis

- načíta `df_daily_vzorky.parquet`
- vykoná čistenie (`dropna`) a filtráciu vysokonapäťových EIC
- vytvorí binárny target (`Typ poruchy > 0`)
- doplní engineered features 
- odstráni „kontaminované nuly“ (typ 0 v dňoch, kde je porucha na tom istom EIC)
- vyberie TOP features cez Mutual Information
- použije `StratifiedGroupKFold` (skupiny podľa `eic`)
- cez `GridSearchCV` porovná `RandomForest`, `XGBoost`, `LightGBM`
- optimalizuje threshold s cieľom minimalizovať FN (s FP bezpečnostným limitom)

### Spustenie

```bash
cd "/Users/martinsarnovsky/Library/Mobile Documents/com~apple~CloudDocs/pycharmProjects/VSDProject"
python train_binary.py
```

### Výstupy

- `model_binary_<best_model>.joblib`
- `model_binary_<best_model>_with_threshold.joblib`

Artefakt `*_with_threshold.joblib` obsahuje:

- `model`
- `threshold`
- `features`

---

## 2) Trénovanie multiclass modelu (`train_multiclass.py`)

Skript trénuje model typu poruchy iba na poruchových záznamoch (`Typ poruchy > 0`).

### Čo robí

- načíta `df_daily_vzorky.parquet`
- nechá len poruchové riadky (`Typ poruchy > 0`)
- mapuje pôvodné typy porúch podľa aktuálneho `mapping_multi`
- reindexuje triedy pre tréning (od 0)
- vytvorí rovnaké engineered features ako binárny model
- vyberie TOP features cez Mutual Information
- použije `StratifiedGroupKFold` (skupiny podľa `eic`)
- cez `GridSearchCV` porovná `RandomForest`, `XGBoost`, `LightGBM`

### Spustenie

```bash
cd "/Users/martinsarnovsky/Library/Mobile Documents/com~apple~CloudDocs/pycharmProjects/VSDProject"
python train_multiclass.py
```

### Výstupy

- `model_multiclass_<best_model>.joblib`
- `model_multiclass_<best_model>_bundle.joblib`

Bundle obsahuje:

- `model`
- `features`
- `class_labels`
- `label_offset`
- `best_model_name`

---

## 3) Scoring eval dát (`score_hierarchical_eval.py`)

Skript skóruje `Data/Eval/eval_data.parquet` (a mapovanie z `Data/Eval/eval_data.csv`) pomocou oboch modelov.

### Čo robí

- aplikuje rovnaké predspracovanie a feature engineering ako pri tréningu
- najprv predikuje poruchu binárnym modelom
- pre binárne pozitívne záznamy predikuje typ poruchy multiclass modelom
- zlučuje súvislé poruchové dni do intervalov `OD - DO`
- pravdepodobnosti agreguje priemerom za interval
- výstupné probability formátuje na max 3 desatinné miesta

### Spustenie (default)

```bash
cd "/Users/martinsarnovsky/Library/Mobile Documents/com~apple~CloudDocs/pycharmProjects/VSDProject"
python score_hierarchical_eval.py
```

### Spustenie s vlastným thresholdom

```bash
python score_hierarchical_eval.py --binary-threshold 0.35
```

### Smoke test na malej vzorke

```bash
python score_hierarchical_eval.py --max-eics 2 --output-csv eval_scored_faults_sample.csv
```

### Výstup

Predvolený súbor:

```text
eval_scored_faults.csv
```

Výstupné stĺpce:

- `EIC`
- `elektromer`
- `OD`
- `DO`
- `dni_v_useku`
- `binary_probability`
- `binary_prediction`
- `multiclass_prediction`
- `multiclass_probability`
- `prob_type_1`
- `prob_type_2`
- `prob_type_3`
- `prob_type_4`

---

## Poznámky

- Všetky tri skripty predpokladajú spúšťanie z koreňa projektu.
- Ak meníš feature engineering v tréning skriptoch, drž ho konzistentný aj v `score_hierarchical_eval.py`.
- Pri staršom multiclass modeli bez bundle sa odporúča znovu spustiť `train_multiclass.py`.
