"""
Hierarchický scoring pipeline pre eval dáta.

Postup:
1) Načíta sample `Data/Eval/eval_data_sample_first50_meters_with_eic.parquet`
2) Spraví rovnaké predspracovanie a denné featury ako tréning
3) Binárny model vyhodnotí prítomnosť poruchy
4) Model 2 (2 triedy) predikuje skupinu poruchy: prudová vs napäťová
5) Uloží intervalový CSV výstup
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EPS_ZERO = 1e-6
SPIKE_K = 3.0
MAX_TIME_GAP_MIN: Optional[float] = None
MAX_NAN_RATIO_PER_DAY = 0.35
MIN_COL_COVERAGE_PER_DAY = 0.60

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_PARQUET = BASE_DIR / "Data" / "Eval" / "eval_data_sample_first50_meters_with_eic.parquet"
DEFAULT_EVAL_CSV = BASE_DIR / "Data" / "Eval" / "eval_data.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "eval_scored_faults_first50_meters.csv"
DEFAULT_BINARY_ARTIFACT = BASE_DIR / "model_binary_randomforest_with_threshold.joblib"
DEFAULT_MULTICLASS_ARTIFACT = BASE_DIR / "model_multiclass_randomforest_bundle.joblib"
DEFAULT_BINARY_THRESHOLD_OVERRIDE: Optional[float] = None

# Mapovanie class label -> názov typu poruchy pre 2-triedny model
# 1: prudové poruchy, 2: napäťové poruchy
FAULT_LABEL_NAMES = {1: "prudova", 2: "napatova"}

# =============================================================================
# FEATURE HELPERS
# =============================================================================

def mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if not s.empty else np.nan


def p2p(s: pd.Series):
    s = s.dropna()
    return (s.max() - s.min()) if not s.empty else np.nan


def robust_p2p(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    q95 = np.percentile(s, 95)
    q05 = np.percentile(s, 5)
    return q95 - q05


def count_zeros(series: pd.Series, eps: float = EPS_ZERO) -> int:
    s = series.dropna()
    if s.empty:
        return 0
    return int((np.abs(s) < eps).sum())


def diffs_abs_stats(df_day: pd.DataFrame, col: str, t_col: str) -> tuple:
    s = df_day[[t_col, col]].dropna().sort_values(t_col)
    if len(s) < 2:
        return np.nan, np.nan
    if MAX_TIME_GAP_MIN is not None:
        dt = s[t_col].diff().dt.total_seconds().fillna(0) / 60.0
        diffs = s[col].diff().where(dt <= MAX_TIME_GAP_MIN)
    else:
        diffs = s[col].diff()
    diffs = diffs.dropna().abs()
    if diffs.empty:
        return np.nan, np.nan
    return float(diffs.mean()), float(diffs.max())


def mad(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def spikes_count(series: pd.Series, k: float = SPIKE_K) -> int:
    s = series.dropna().values
    if s.size == 0:
        return 0
    med = np.median(s)
    m = mad(s)
    if not np.isfinite(m) or m == 0:
        return 0
    thr = med + k * m
    return int(np.sum(s > thr))


def load_main_table(parquet_path: Path, filters: Optional[list] = None) -> pd.DataFrame:
    table = pq.read_table(str(parquet_path), filters=filters)
    return table.to_pandas(types_mapper=None)


def load_eval_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def prepare_time_cols(df: pd.DataFrame, t_col: str = "t_utc") -> pd.DataFrame:
    out = df.copy()
    out[t_col] = pd.to_datetime(out[t_col], utc=True, errors="coerce")
    return out


def build_skipped_nan_output_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_skipped_nan_days.csv")


def write_skipped_nan_report(skipped_days: pd.DataFrame, output_path: Path) -> None:
    skipped_days = normalize_skipped_days(skipped_days)
    if skipped_days.empty:
        if output_path.exists():
            output_path.unlink()
            print(f"    ✓ Zmazaný starý NaN report bez nových záznamov: {output_path}")
        else:
            print("    ✓ Žiadne dni neboli vynechané kvôli veľkému množstvu NaN")
        return

    skipped_days.to_csv(output_path, index=False)
    print(f"    ✓ CSV s vynechanými dňami: {output_path}")


def build_empty_scoring_output() -> pd.DataFrame:
    empty_cols = [
        "EIC", "elektromer", "OD", "DO", "dni_v_useku",
        "binary_probability", "binary_prediction",
        "predikovany_typ_poruchy",
        "prob_prudova", "prob_napatova",
    ]
    return pd.DataFrame(columns=empty_cols)


def normalize_skipped_days(skipped_days: pd.DataFrame) -> pd.DataFrame:
    cols = ["EIC", "elektromer", "datum", "dovod"]
    if skipped_days is None or skipped_days.empty:
        return pd.DataFrame(columns=cols)

    out = skipped_days.copy()
    if "dovod" not in out.columns:
        out["dovod"] = "neznamy"

    for col in cols:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out[cols]
        .drop_duplicates()
        .sort_values(["EIC", "datum", "dovod"])
        .reset_index(drop=True)
    )


def combine_skipped_days(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized = [normalize_skipped_days(frame) for frame in frames if frame is not None]
    if not normalized:
        return normalize_skipped_days(pd.DataFrame())
    return normalize_skipped_days(pd.concat(normalized, ignore_index=True))


def filter_rows_with_nan_model_features(
    df: pd.DataFrame,
    features: list[str],
    eic_to_meter: Dict[Any, Any],
    reason: str,
    eic_col: str = "eic",
    t_col: str = "t_utc",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(
            "Strict feature check failed: v scoring dátach chýbajú stĺpce z modelu. "
            f"Missing count={len(missing)}, first_missing={missing[:10]}"
        )

    if df.empty:
        return df.copy(), normalize_skipped_days(pd.DataFrame())

    skip_mask = df.loc[:, features].isna().any(axis=1)
    if not skip_mask.any():
        return df.reset_index(drop=True), normalize_skipped_days(pd.DataFrame())

    skipped = df.loc[skip_mask, [eic_col, t_col]].copy()
    skipped["elektromer"] = skipped[eic_col].map(eic_to_meter)
    skipped["datum"] = pd.to_datetime(skipped[t_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    skipped["dovod"] = reason
    skipped = skipped.rename(columns={eic_col: "EIC"})[["EIC", "elektromer", "datum", "dovod"]]

    kept = df.loc[~skip_mask].copy().reset_index(drop=True)
    return kept, normalize_skipped_days(skipped)


def filter_days_with_excessive_nans(
    df: pd.DataFrame,
    eic_col: str,
    t_col: str,
    meter_col: str,
    value_cols: list[str],
    max_nan_ratio: float = MAX_NAN_RATIO_PER_DAY,
    min_col_coverage: float = MIN_COL_COVERAGE_PER_DAY,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vyradí EIC/deň, kde by veľké množstvo NaN výrazne skreslilo denné agregáty.

    Deň sa vyradí, ak:
    - celkový podiel NaN naprieč sledovanými prúdmi/napätiami je > max_nan_ratio,
    - alebo má aspoň jeden sledovaný stĺpec pokrytie < min_col_coverage.
    """
    d = df.copy()
    d[t_col] = pd.to_datetime(d[t_col], utc=True, errors="coerce")
    d = d[d[eic_col].notna() & d[t_col].notna()].copy()
    d["_day"] = d[t_col].dt.floor("D")

    if not value_cols:
        d = d.drop(columns=["_day"])
        return d, normalize_skipped_days(pd.DataFrame())

    groups = d.groupby([eic_col, "_day"], dropna=False)
    group_size = groups.size().rename("n_rows")
    non_nan_counts = groups[value_cols].count()
    non_nan_total = non_nan_counts.sum(axis=1).rename("non_nan_total")
    total_cells = (group_size * len(value_cols)).rename("total_cells")
    nan_ratio = (1.0 - (non_nan_total / total_cells)).rename("nan_ratio")
    min_coverage_actual = non_nan_counts.div(group_size, axis=0).min(axis=1).rename("min_col_coverage")

    stats = pd.concat([group_size, nan_ratio, min_coverage_actual], axis=1)
    skip_mask = (stats["nan_ratio"] > max_nan_ratio) | (stats["min_col_coverage"] < min_col_coverage)

    bad_keys = stats[skip_mask].reset_index()
    if bad_keys.empty:
        d = d.drop(columns=["_day"])
        return d, normalize_skipped_days(pd.DataFrame())

    meter_lookup = (
        groups[meter_col]
        .agg(mode_or_nan)
        .reset_index(name=meter_col)
        if meter_col in d.columns
        else pd.DataFrame(columns=[eic_col, "_day", meter_col])
    )
    bad_keys = bad_keys.merge(meter_lookup, on=[eic_col, "_day"], how="left")

    skipped_days = (
        bad_keys.assign(datum=pd.to_datetime(bad_keys["_day"], utc=True).dt.strftime("%Y-%m-%d"))
        [[eic_col, meter_col, "datum"]]
        .rename(columns={eic_col: "EIC", meter_col: "elektromer"})
    )
    skipped_days["dovod"] = "prilis_vela_nan_v_10min"
    skipped_days = normalize_skipped_days(skipped_days)

    keep_keys = stats[~skip_mask].reset_index()[[eic_col, "_day"]]
    d = d.merge(keep_keys, on=[eic_col, "_day"], how="inner")
    d = d.drop(columns=["_day"])
    return d, skipped_days


def daily_base_aggregation(
    df: pd.DataFrame,
    eic_col: str,
    t_col: str,
    value_cols: list[str],
    typ_col: Optional[str] = None,
    typ_strategy: str = "none",
) -> pd.DataFrame:
    d = df.copy()
    d[t_col] = pd.to_datetime(d[t_col], utc=True, errors="coerce")
    d = d[d[eic_col].notna() & d[t_col].notna()].copy()
    d["_day"] = d[t_col].dt.floor("D")

    agg_kwargs: Dict[str, tuple] = {}
    metrics = {"mean": "mean", "median": "median", "max": "max", "min": "min", "std": "std"}
    for col in value_cols:
        for suf, func in metrics.items():
            agg_kwargs[f"{col}_{suf}"] = (col, func)
        agg_kwargs[f"{col}_p2p"] = (col, p2p)

    if typ_strategy == "none" and typ_col and typ_col in d.columns:
        out = (
            d.groupby([eic_col, "_day", typ_col], dropna=False)
            .agg(**agg_kwargs)
            .reset_index()
            .rename(columns={"_day": t_col})
        )
        out[t_col] = pd.to_datetime(out[t_col], utc=True)
        return out.sort_values([eic_col, t_col, typ_col]).reset_index(drop=True)

    if typ_col and (typ_col in d.columns) and typ_strategy == "mode":
        agg_kwargs[typ_col] = (typ_col, mode_or_nan)

    out = (
        d.groupby([eic_col, "_day"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
        .rename(columns={"_day": t_col})
    )
    out[t_col] = pd.to_datetime(out[t_col], utc=True)
    return out.sort_values([eic_col, t_col]).reset_index(drop=True)


def daily_extras(
    df: pd.DataFrame,
    eic_col: str,
    t_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    d = df.copy()
    d[t_col] = pd.to_datetime(d[t_col], utc=True, errors="coerce")
    d = d[d[eic_col].notna() & d[t_col].notna()].copy()
    groups = d.groupby([eic_col, d[t_col].dt.floor("D")], dropna=False)

    extra_cols = [eic_col, t_col]
    for col in value_cols:
        extra_cols.extend([
            f"{col}_p2p_robust",
            f"{col}_zeros",
            f"{col}_mean_abs_diff",
            f"{col}_max_abs_diff",
            f"{col}_spikes",
        ])

    rows: list[Dict[str, object]] = []
    for (eic_val, day), g in groups:
        g_sorted = g.sort_values(t_col)
        rec: Dict[str, object] = {eic_col: eic_val, t_col: pd.to_datetime(day, utc=True)}
        for col in value_cols:
            rec[f"{col}_p2p_robust"] = robust_p2p(g_sorted[col])
            rec[f"{col}_zeros"] = count_zeros(g_sorted[col])
            mean_d, max_d = diffs_abs_stats(g_sorted, col, t_col)
            rec[f"{col}_mean_abs_diff"] = mean_d
            rec[f"{col}_max_abs_diff"] = max_d
            rec[f"{col}_spikes"] = spikes_count(g_sorted[col])
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=extra_cols)
    return pd.DataFrame(rows)


def build_daily_features(
    df: pd.DataFrame,
    eic_col: str = "eic",
    t_col: str = "t_utc",
    u_cols=None,
    i_cols=None,
    typ_col: Optional[str] = None,
) -> pd.DataFrame:
    if u_cols is None:
        u_cols = ["u1_norm", "u2_norm", "u3_norm"]
    if i_cols is None:
        i_cols = ["i1_norm", "i2_norm", "i3_norm"]

    value_cols = [c for c in (u_cols + i_cols) if c in df.columns]
    base = daily_base_aggregation(df, eic_col, t_col, value_cols, typ_col, typ_strategy="none")
    extras = daily_extras(df, eic_col, t_col, value_cols)
    return (
        base.merge(extras, on=[eic_col, t_col], how="left")
        .sort_values([eic_col, t_col])
        .reset_index(drop=True)
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    needed = [
        "u1_norm_mean", "u2_norm_mean", "u3_norm_mean",
        "i1_norm_mean", "i2_norm_mean", "i3_norm_mean",
        "u1_norm_std", "u2_norm_std", "u3_norm_std",
        "i1_norm_std", "i2_norm_std", "i3_norm_std",
        "i1_norm_zeros", "i2_norm_zeros", "i3_norm_zeros",
        "i1_norm_spikes", "i2_norm_spikes", "i3_norm_spikes",
    ]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise KeyError(f"Chýbajú očakávané feature stĺpce: {missing}")

    d["u_asymmetry_mean"] = d[["u1_norm_mean", "u2_norm_mean", "u3_norm_mean"]].max(axis=1) - d[["u1_norm_mean", "u2_norm_mean", "u3_norm_mean"]].min(axis=1)
    d["i_total_mean"] = d["i1_norm_mean"] + d["i2_norm_mean"] + d["i3_norm_mean"]
    d["i_total_std"] = d["i1_norm_std"] + d["i2_norm_std"] + d["i3_norm_std"]
    d["u_total_std"] = d["u1_norm_std"] + d["u2_norm_std"] + d["u3_norm_std"]
    d["i_zeros_total"] = d["i1_norm_zeros"] + d["i2_norm_zeros"] + d["i3_norm_zeros"]
    d["i_spikes_total"] = d["i1_norm_spikes"] + d["i2_norm_spikes"] + d["i3_norm_spikes"]
    d["p_total_mean"] = (
        d["u1_norm_mean"] * d["i1_norm_mean"]
        + d["u2_norm_mean"] * d["i2_norm_mean"]
        + d["u3_norm_mean"] * d["i3_norm_mean"]
    )
    d["i_unbalance_mean"] = (
        d[["i1_norm_mean", "i2_norm_mean", "i3_norm_mean"]].max(axis=1)
        - d[["i1_norm_mean", "i2_norm_mean", "i3_norm_mean"]].min(axis=1)
    ) / (d["i_total_mean"] + 0.1)
    d["u_unbalance_mean"] = (
        d[["u1_norm_mean", "u2_norm_mean", "u3_norm_mean"]].max(axis=1)
        - d[["u1_norm_mean", "u2_norm_mean", "u3_norm_mean"]].min(axis=1)
    ) / (d[["u1_norm_mean", "u2_norm_mean", "u3_norm_mean"]].mean(axis=1) + 0.1)
    d["z_approx_1"] = d["u1_norm_mean"] / (d["i1_norm_mean"] + 0.1)
    d["z_approx_2"] = d["u2_norm_mean"] / (d["i2_norm_mean"] + 0.1)
    d["z_approx_3"] = d["u3_norm_mean"] / (d["i3_norm_mean"] + 0.1)
    d["ui_std_ratio_sum"] = (
        d["i1_norm_std"] + d["i2_norm_std"] + d["i3_norm_std"]
    ) / (d["u1_norm_std"] + d["u2_norm_std"] + d["u3_norm_std"] + 1e-6)
    return d


def filter_high_voltage_eics(df: pd.DataFrame, threshold: float = 1000.0) -> pd.DataFrame:
    d = df.copy()
    voltage_cols = [
        "u1_norm_mean", "u2_norm_mean", "u3_norm_mean",
        "u1_norm_max", "u2_norm_max", "u3_norm_max",
    ]
    voltage_cols = [c for c in voltage_cols if c in d.columns]
    if not voltage_cols:
        return d

    max_voltage_per_eic = d.groupby("eic")[voltage_cols].max().max(axis=1)
    high_voltage_eics = max_voltage_per_eic[max_voltage_per_eic > threshold].index.tolist()
    if high_voltage_eics:
        print(f"    Filtrácia vysokonapäťových EIC: {len(high_voltage_eics)}")
        d = d[~d["eic"].isin(high_voltage_eics)].copy()
    return d


def load_latest_artifact(pattern: str) -> Path:
    matches = sorted(BASE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Nenájdený artifact podľa patternu: {pattern}")
    return matches[0]


def load_binary_artifact(path: Optional[Path] = None) -> Tuple[Any, list[str], float, str]:
    if path is None:
        path = load_latest_artifact("model_binary_*_with_threshold.joblib")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model = obj["model"]
        features = list(obj.get("features", []))
        threshold = float(obj.get("threshold", 0.5))
    else:
        model = obj
        features = list(getattr(model, "feature_names_in_", []))
        threshold = 0.5
    if not features:
        raise RuntimeError(
            f"Binárny model z {path.name} neobsahuje feature list. "
            f"Spusť znovu `train_binary.py`, aby sa uložil bundle s featurami."
        )
    return model, features, threshold, path.name


def load_multiclass_artifact(path: Optional[Path] = None) -> Tuple[Any, list[str], list[int], str]:
    if path is None:
        bundle_matches = sorted(BASE_DIR.glob("model_multiclass_*_bundle.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if bundle_matches:
            path = bundle_matches[0]
        else:
            path = load_latest_artifact("model_multiclass_*.joblib")
    obj = joblib.load(path)

    if isinstance(obj, dict):
        model = obj["model"]
        features = list(obj.get("features", []))
        class_labels = list(obj.get("class_labels", [1, 2, 3, 4]))
    else:
        model = obj
        features = list(getattr(model, "feature_names_in_", []))
        class_labels = [int(c) + 1 for c in getattr(model, "classes_", [0, 1, 2, 3])]

    if not features:
        raise RuntimeError(
            f"Multiclass model z {path.name} neobsahuje feature list. "
            f"Spusť znovu `train_multiclass.py`, aby sa uložil bundle s featurami."
        )
    return model, features, class_labels, path.name


def align_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(
            "Strict feature check failed: v scoring dátach chýbajú stĺpce z modelu. "
            f"Missing count={len(missing)}, first_missing={missing[:10]}"
        )

    X = df.loc[:, features].copy()

    return X


def score_eval(
    eval_parquet: Path,
    eval_csv: Path,
    output_csv: Path,
    binary_artifact: Optional[Path] = None,
    multiclass_artifact: Optional[Path] = None,
    max_eics: Optional[int] = None,
    max_rows: Optional[int] = None,
    binary_threshold_override: Optional[float] = None,
) -> pd.DataFrame:
    print("=" * 80)
    print("HIERARCHICKÝ SCORING PRE EVAL DÁTA")
    print("=" * 80)

    print(f"\n[1] Načítavam mapovanie EIC: {eval_csv}")
    eval_meta = load_eval_csv(eval_csv)
    eval_meta["elektromer"] = pd.to_numeric(eval_meta["elektromer"], errors="coerce")
    print(f"    ✓ Meta: {eval_meta.shape}")

    eic_to_meter = (
        eval_meta.dropna(subset=["eic", "elektromer"])
        .groupby("eic")["elektromer"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )

    raw_filters = None
    print(f"\n[2] Načítavam raw dáta: {eval_parquet}")
    df = load_main_table(eval_parquet, filters=raw_filters)
    print(f"    ✓ Raw dáta: {df.shape}")

    # Sample parquet už obsahuje EIC; fallback je mapovanie cez CSV.
    df["elektromer"] = pd.to_numeric(df["elektromer"], errors="coerce")
    if "eic" in df.columns:
        df["eic"] = df["eic"]
    elif "EIC" in df.columns:
        df["eic"] = df["EIC"]
    else:
        eval_meta["elektromer"] = pd.to_numeric(eval_meta["elektromer"], errors="coerce")
        elm_to_eic = dict(zip(eval_meta["elektromer"], eval_meta["eic"]))
        df["eic"] = df["elektromer"].map(elm_to_eic)

    print(f"    Po priradení EIC - df.shape: {df.shape}, eic notna: {df['eic'].notna().sum()}")

    df = df[df["eic"].notna()].copy()
    print(f"    Po filtrovaní NaN EIC - df.shape: {df.shape}")

    df = prepare_time_cols(df, t_col="t_utc")
    print(f"    Po prepare_time_cols - df.shape: {df.shape}")

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
        print(f"    Po obmedzení na prvých {max_rows} riadkov: {df.shape}")

    value_cols = [c for c in ["u1_norm", "u2_norm", "u3_norm", "i1_norm", "i2_norm", "i3_norm"] if c in df.columns]
    skipped_nan_output_csv = build_skipped_nan_output_path(output_csv)

    print(f"\n[3] Filtrujem EIC/deň s priveľa NaN v 10-min meraniach...")
    df, skipped_nan_days = filter_days_with_excessive_nans(
        df=df,
        eic_col="eic",
        t_col="t_utc",
        meter_col="elektromer",
        value_cols=value_cols,
    )
    skipped_report = normalize_skipped_days(skipped_nan_days)
    print(f"    ✓ Po NaN filtri: {df.shape}")
    print(f"    ✓ Vynechané dni: {len(skipped_report)}")
    write_skipped_nan_report(skipped_report, skipped_nan_output_csv)

    if df.empty:
        print("    ⚠ Po NaN filtri nezostali žiadne dáta na scoring.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    print(f"\n[4] Vytváram denné features...")
    df_daily = build_daily_features(df, eic_col="eic", t_col="t_utc", u_cols=["u1_norm", "u2_norm", "u3_norm"], i_cols=["i1_norm", "i2_norm", "i3_norm"], typ_col=None)
    print(f"    ✓ Denné features: {df_daily.shape}")

    if df_daily.empty:
        print("    ⚠ Po dennej agregácii nevznikli žiadne záznamy.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    print(f"\n[5] Feature engineering ako pri tréningu...")
    df_daily = add_engineered_features(df_daily)
    print(f"    ✓ Po engineered features: {df_daily.shape}")

    print(f"\n[6] Filtrácia vysokonapäťových EIC...")
    before_hv = len(df_daily)
    df_daily = filter_high_voltage_eics(df_daily, threshold=1000.0)
    print(f"    ✓ Riadkov pred HV filtrom: {before_hv}")
    print(f"    ✓ Riadkov po HV filtri: {len(df_daily)}")

    if df_daily.empty:
        print("    ⚠ Po HV filtri nezostali žiadne dáta na scoring.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    print(f"\n[7] Načítavam binárny model a threshold...")
    binary_model, binary_features, binary_threshold, binary_name = load_binary_artifact(binary_artifact)
    if binary_threshold_override is not None:
        print(f"    ⚙ Threshold z artefaktu ({binary_threshold:.4f}) prepisuje používateľský override: {binary_threshold_override:.4f}")
        binary_threshold = binary_threshold_override
    print(f"    ✓ Binary artifact: {binary_name}")
    print(f"    ✓ Binary threshold: {binary_threshold:.2f}")
    print(f"    ✓ Binary features: {len(binary_features)}")

    df_daily, skipped_binary_feature_days = filter_rows_with_nan_model_features(
        df=df_daily,
        features=binary_features,
        eic_to_meter=eic_to_meter,
        reason="nan_v_agregovanych_feature_pre_binary",
    )
    skipped_report = combine_skipped_days(skipped_report, skipped_binary_feature_days)
    if not skipped_binary_feature_days.empty:
        print(f"    ✓ Vynechané agregované dni pre binary kvôli NaN feature: {len(skipped_binary_feature_days)}")
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)

    if df_daily.empty:
        print("    ⚠ Po odfiltrovaní agregovaných NaN feature nezostali žiadne dáta pre binary scoring.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    X_binary = align_features(df_daily, binary_features)
    binary_proba = binary_model.predict_proba(X_binary)[:, 1]
    binary_pred = (binary_proba >= binary_threshold).astype(int)
    df_daily["binary_probability"] = binary_proba
    df_daily["binary_prediction"] = binary_pred

    positive_mask = df_daily["binary_prediction"] == 1
    positives = df_daily[positive_mask].copy()
    print(f"    ✓ Pozitívne binárne záznamy: {len(positives)} / {len(df_daily)}")

    if positives.empty:
        print("    ⚠ Binárny model nenašiel žiadne poruchové záznamy.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    print(f"\n[8] Načítavam model typu poruchy...")
    multi_model, multi_features, class_labels, multi_name = load_multiclass_artifact(multiclass_artifact)
    print(f"    ✓ Model artifact: {multi_name}")
    print(f"    ✓ Model features: {len(multi_features)}")
    print(f"    ✓ Class labels: {class_labels}")

    # Strict kontrola: oba modely musia používať identický feature list.
    if binary_features != multi_features:
        only_binary = [f for f in binary_features if f not in set(multi_features)]
        only_multi = [f for f in multi_features if f not in set(binary_features)]
        raise RuntimeError(
            "Strict feature check failed: feature listy modelov sa líšia. "
            f"len(binary)={len(binary_features)}, len(model2)={len(multi_features)}, "
            f"only_binary(first)={only_binary[:10]}, only_model2(first)={only_multi[:10]}"
        )

    positives, skipped_multi_feature_days = filter_rows_with_nan_model_features(
        df=positives,
        features=multi_features,
        eic_to_meter=eic_to_meter,
        reason="nan_v_agregovanych_feature_pre_typ_model",
    )
    skipped_report = combine_skipped_days(skipped_report, skipped_multi_feature_days)
    if not skipped_multi_feature_days.empty:
        print(f"    ✓ Vynechané agregované dni pre model typu kvôli NaN feature: {len(skipped_multi_feature_days)}")
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)

    if positives.empty:
        print("    ⚠ Po odfiltrovaní agregovaných NaN feature nezostali pozitívne dni pre model typu poruchy.")
        out = build_empty_scoring_output()
        write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
        out.to_csv(output_csv, index=False)
        print(f"\n✓ Uložené prázdne CSV: {output_csv}")
        return out

    X_multi = align_features(positives, multi_features)
    multi_proba = multi_model.predict_proba(X_multi)
    multi_pred_idx = np.argmax(multi_proba, axis=1)
    multi_pred_labels = np.array(class_labels)[multi_pred_idx]
    multi_pred_proba = multi_proba[np.arange(len(multi_proba)), multi_pred_idx]

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    idx_prudova = label_to_idx.get(1)
    idx_napatova = label_to_idx.get(2)

    prob_prudova = multi_proba[:, idx_prudova] if idx_prudova is not None else np.zeros(len(multi_proba))
    prob_napatova = multi_proba[:, idx_napatova] if idx_napatova is not None else np.zeros(len(multi_proba))

    result_daily = pd.DataFrame({
        "eic": positives["eic"].values,
        "t_utc": positives["t_utc"].values,
        "binary_probability": positives["binary_probability"].values,
        "binary_prediction": positives["binary_prediction"].values,
        "predikovany_typ_poruchy": [FAULT_LABEL_NAMES.get(int(x), str(x)) for x in multi_pred_labels],
        "prob_prudova": prob_prudova,
        "prob_napatova": prob_napatova,
    })

    result_daily["elektromer"] = result_daily["eic"].map(eic_to_meter)
    result_daily = result_daily.sort_values(["eic", "t_utc"]).reset_index(drop=True)

    # Zlúč súvislé poruchové dni do intervalov (OD-DO) pre každý EIC.
    day_gap = result_daily.groupby("eic")["t_utc"].diff().dt.total_seconds().div(86400.0)
    new_interval = day_gap.isna() | (day_gap > 1.0)
    result_daily["interval_id"] = new_interval.groupby(result_daily["eic"]).cumsum()

    agg_cols = {
        "elektromer": "first",
        "t_utc": ["min", "max", "count"],
        "binary_probability": "mean",
        "binary_prediction": "max",
        "prob_prudova": "mean",
        "prob_napatova": "mean",
    }

    intervals = result_daily.groupby(["eic", "interval_id"], as_index=False).agg(agg_cols)
    intervals.columns = [
        "eic", "interval_id", "elektromer", "OD", "DO", "dni_v_useku",
        "binary_probability", "binary_prediction", "prob_prudova", "prob_napatova",
    ]

    intervals["predikovany_typ_poruchy"] = np.where(
        intervals["prob_prudova"] >= intervals["prob_napatova"],
        "prudova",
        "napatova",
    )

    output_cols = [
        "eic", "elektromer", "OD", "DO", "dni_v_useku",
        "binary_probability", "binary_prediction",
        "predikovany_typ_poruchy",
        "prob_prudova", "prob_napatova",
    ]
    result = intervals[output_cols].rename(columns={"eic": "EIC"})
    result = result.sort_values(["EIC", "OD"]).reset_index(drop=True)

    # Zaokrúhli desatinné stĺpce na max 3 desatinné miesta
    float_cols = ["binary_probability", "prob_prudova", "prob_napatova"]
    result[float_cols] = result[float_cols].round(3)

    write_skipped_nan_report(skipped_report, skipped_nan_output_csv)
    result.to_csv(output_csv, index=False)

    print(f"\n[9] Hotovo")
    print(f"    ✓ Výstupný CSV: {output_csv}")
    print(f"    ✓ Počet predikovaných poruchových intervalov: {len(result)}")
    unique_eics_with_fault = result["EIC"].nunique()
    print(f"    ✓ Počet unikátnych EIC s poruchou: {unique_eics_with_fault}")
    print(f"    ✓ Stĺpce: {result.columns.tolist()}")
    return result


if __name__ == "__main__":
    score_eval(
        eval_parquet=DEFAULT_EVAL_PARQUET,
        eval_csv=DEFAULT_EVAL_CSV,
        output_csv=DEFAULT_OUTPUT_CSV,
        binary_artifact=DEFAULT_BINARY_ARTIFACT,
        multiclass_artifact=DEFAULT_MULTICLASS_ARTIFACT,
        max_eics=None,
        max_rows=None,
        binary_threshold_override=DEFAULT_BINARY_THRESHOLD_OVERRIDE,
    )
