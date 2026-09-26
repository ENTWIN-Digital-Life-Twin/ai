"""Train the multi-label recommendation ranker.

Builds wellness-shaped features from the Sleep Health & Lifestyle CSV,
derives guideline soft labels, augments to ~6k samples, fits
OneVsRest(HistGradientBoostingClassifier), and exports joblib artifacts.

Examples:
  python scripts/train_recommendation_model.py --evaluate-only
  python scripts/train_recommendation_model.py --no-save
  python scripts/train_recommendation_model.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    label_ranking_average_precision_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.recommendation.catalog import REC_TYPES, TargetDefaults  # noqa: E402
from app.ml.recommendation.features import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_matrix,
)
from app.ml.recommendation.labels import (  # noqa: E402
    LABEL_THRESHOLD,
    multi_hot,
    soft_relevance,
    soft_vector,
)

DATA_PATH = PROJECT_ROOT / "data" / "sleep_health_lifestyle.csv"
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "recommendation_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "app" / "models" / "recommendation_features.joblib"
METADATA_PATH = PROJECT_ROOT / "app" / "models" / "recommendation_model.metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
AUGMENT_TARGET = 6000
N_CV_FOLDS = 5
MODEL_VERSION = "1.0.0"


def csv_row_to_signals(row: pd.Series) -> dict[str, float | None]:
    """Map Kaggle sleep/lifestyle columns to wellness-shaped signals."""
    sleep_hours = float(row["Sleep Duration"])
    activity = float(row["Physical Activity Level"])  # 0–100 intensity proxy
    stress = float(row["Stress Level"])
    quality = float(row["Quality of Sleep"])
    steps = float(row["Daily Steps"])
    person_id = float(row.get("Person ID", 0) or 0)

    # Approximate weekly workout minutes from activity intensity (0–100 → 0–210).
    weekly_workout = activity * 2.1
    # Mood proxy from sleep quality (1–10).
    mood = quality
    # Fatigue proxy: inverse of quality with stress contribution (clamped 1–10).
    fatigue = float(np.clip(11.0 - quality + max(0.0, stress - 5.0) * 0.3, 1.0, 10.0))
    # Hydration is absent from the CSV — synthesize a balanced distribution so
    # HYDRATION is not almost always positive.
    hydration = float(
        np.clip(
            1450.0 + activity * 8.0 - stress * 35.0 + ((person_id * 37) % 900),
            700.0,
            3200.0,
        )
    )

    return {
        "sleep_minutes": sleep_hours * 60.0,
        "hydration_ml": hydration,
        "stress_level": stress,
        "fatigue_level": fatigue,
        "weekly_workout_minutes": weekly_workout,
        "mood_level": mood,
        "daily_steps": steps,
    }


def build_base_dataset(path: Path = DATA_PATH) -> tuple[list[dict[str, float | None]], np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    signals: list[dict[str, float | None]] = []
    y_soft: list[np.ndarray] = []
    y_hard: list[np.ndarray] = []
    targets = TargetDefaults()
    for _, row in df.iterrows():
        sig = csv_row_to_signals(row)
        scores = soft_relevance(sig, targets)
        signals.append(sig)
        y_soft.append(soft_vector(scores))
        y_hard.append(multi_hot(scores))

    # Explicit wellness-shaped anchors (healthy + single-signal extremes).
    anchors: list[dict[str, float | None]] = [
        {
            "sleep_minutes": 480.0,
            "hydration_ml": 2100.0,
            "stress_level": 3.0,
            "fatigue_level": 3.0,
            "weekly_workout_minutes": 180.0,
            "mood_level": 8.0,
            "daily_steps": 9000.0,
        },
        {"sleep_minutes": 330.0, "hydration_ml": None, "stress_level": None,
         "fatigue_level": None, "weekly_workout_minutes": None, "mood_level": None, "daily_steps": None},
        {"sleep_minutes": None, "hydration_ml": 900.0, "stress_level": None,
         "fatigue_level": None, "weekly_workout_minutes": None, "mood_level": None, "daily_steps": None},
        {"sleep_minutes": None, "hydration_ml": None, "stress_level": 8.0,
         "fatigue_level": None, "weekly_workout_minutes": None, "mood_level": None, "daily_steps": None},
        {"sleep_minutes": None, "hydration_ml": None, "stress_level": None,
         "fatigue_level": None, "weekly_workout_minutes": 10.0, "mood_level": None, "daily_steps": None},
        {"sleep_minutes": None, "hydration_ml": None, "stress_level": None,
         "fatigue_level": None, "weekly_workout_minutes": None, "mood_level": 3.0, "daily_steps": None},
        {"sleep_minutes": None, "hydration_ml": None, "stress_level": None,
         "fatigue_level": None, "weekly_workout_minutes": None, "mood_level": None, "daily_steps": 2000.0},
        {"sleep_minutes": 360.0, "hydration_ml": None, "stress_level": 8.0,
         "fatigue_level": 7.0, "weekly_workout_minutes": None, "mood_level": None, "daily_steps": None},
    ]
    for sig in anchors:
        scores = soft_relevance(sig, targets)
        signals.append(sig)
        y_soft.append(soft_vector(scores))
        y_hard.append(multi_hot(scores))

    return signals, np.vstack(y_hard), np.vstack(y_soft)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    y_soft: np.ndarray,
    *,
    target_n: int = AUGMENT_TARGET,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian noise + pairwise interpolation + sparse masking to expand samples."""
    rng = np.random.default_rng(random_state)
    n_base = X.shape[0]
    if n_base == 0:
        return X, y, y_soft

    X_parts = [X]
    y_parts = [y]
    soft_parts = [y_soft]

    # Noise copies
    n_noise = max(0, min(target_n // 3, target_n - n_base))
    if n_noise > 0:
        idx = rng.integers(0, n_base, size=n_noise)
        scale = np.nanstd(X, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        noise = rng.normal(0.0, 0.08, size=(n_noise, X.shape[1])) * scale
        X_noise = X[idx] + noise
        y_noise = y[idx].copy()
        soft_noise = np.clip(
            y_soft[idx] * rng.uniform(0.92, 1.05, size=(n_noise, y_soft.shape[1])),
            0.0,
            1.0,
        )
        X_parts.append(X_noise)
        y_parts.append(y_noise)
        soft_parts.append(soft_noise)

    # Interpolation (mixup-style)
    current = sum(part.shape[0] for part in X_parts)
    n_mix = max(0, (target_n - current) // 2)
    if n_mix > 0:
        i = rng.integers(0, n_base, size=n_mix)
        j = rng.integers(0, n_base, size=n_mix)
        lam = rng.uniform(0.25, 0.75, size=(n_mix, 1))
        X_mix = lam * X[i] + (1.0 - lam) * X[j]
        soft_mix = lam * y_soft[i] + (1.0 - lam) * y_soft[j]
        y_mix = (soft_mix >= LABEL_THRESHOLD).astype(np.int32)
        X_parts.append(X_mix)
        y_parts.append(y_mix)
        soft_parts.append(soft_mix)

    # Sparse / single-signal style rows: zero out most deficits (API-like requests).
    # FEATURE_NAMES indices: raw 0-6, deficits 7-13, interactions 14-17, meta 18-19
    current = sum(part.shape[0] for part in X_parts)
    n_sparse = max(0, target_n - current)
    if n_sparse > 0:
        from app.ml.recommendation.catalog import TargetDefaults

        t = TargetDefaults()
        neutrals = np.asarray(
            [
                t.sleep_minutes,
                t.hydration_ml,
                t.stress_comfort,
                t.fatigue_comfort,
                t.workout_weekly_minutes,
                t.mood_comfort,
                t.steps,
            ],
            dtype=np.float64,
        )
        idx = rng.integers(0, n_base, size=n_sparse)
        X_sp = X[idx].copy()
        y_sp = y[idx].copy()
        soft_sp = y_soft[idx].copy()
        # Keep 1–3 signal groups active; neutralize the rest.
        for row_i in range(n_sparse):
            keep = set(rng.choice(5, size=int(rng.integers(1, 4)), replace=False).tolist())
            # Map label index → feature groups to keep
            # 0 HYDRATION→hydration, 1 REST→sleep/fatigue, 2 STRESS→stress,
            # 3 ACTIVITY→workout/steps, 4 MOOD→mood
            for label_i in range(5):
                if label_i in keep:
                    continue
                soft_sp[row_i, label_i] = 0.0
                y_sp[row_i, label_i] = 0
                if label_i == 0:
                    X_sp[row_i, 1] = neutrals[1]
                    X_sp[row_i, 8] = 0.0  # hydration_deficit
                elif label_i == 1:
                    X_sp[row_i, 0] = neutrals[0]
                    X_sp[row_i, 3] = neutrals[3]
                    X_sp[row_i, 7] = 0.0
                    X_sp[row_i, 10] = 0.0
                elif label_i == 2:
                    X_sp[row_i, 2] = neutrals[2]
                    X_sp[row_i, 9] = 0.0
                elif label_i == 3:
                    X_sp[row_i, 4] = neutrals[4]
                    X_sp[row_i, 6] = neutrals[6]
                    X_sp[row_i, 11] = 0.0
                    X_sp[row_i, 13] = 0.0
                elif label_i == 4:
                    X_sp[row_i, 5] = neutrals[5]
                    X_sp[row_i, 12] = 0.0
            # Recompute interactions / meta lightly
            X_sp[row_i, 14] = X_sp[row_i, 9] * (X_sp[row_i, 7] / max(t.sleep_minutes, 1.0))
            X_sp[row_i, 15] = X_sp[row_i, 10] * (X_sp[row_i, 7] / max(t.sleep_minutes, 1.0))
            X_sp[row_i, 16] = X_sp[row_i, 9] * (X_sp[row_i, 12] / 10.0)
            X_sp[row_i, 17] = (X_sp[row_i, 11] / max(t.workout_weekly_minutes, 1.0)) * (
                X_sp[row_i, 13] / max(t.steps, 1.0)
            )
            X_sp[row_i, 18] = float(len(keep) + 1)
            X_sp[row_i, 19] = 1.0 - (len(keep) + 1) / 7.0
        X_parts.append(X_sp)
        y_parts.append(y_sp)
        soft_parts.append(soft_sp)

    return np.vstack(X_parts), np.vstack(y_parts), np.vstack(soft_parts)


def make_estimator() -> OneVsRestClassifier:
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=120,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    return OneVsRestClassifier(base, n_jobs=1)


def _predict_proba_matrix(model: OneVsRestClassifier, X: np.ndarray) -> np.ndarray:
    """Return (n_samples, n_labels) probabilities aligned with REC_TYPES."""
    # OneVsRest predict_proba returns list of arrays or a 2d array depending on version.
    raw = model.predict_proba(X)
    if isinstance(raw, list):
        cols = []
        for arr in raw:
            if arr.ndim == 2 and arr.shape[1] == 2:
                cols.append(arr[:, 1])
            else:
                cols.append(np.asarray(arr).ravel())
        return np.column_stack(cols)
    return np.asarray(raw, dtype=np.float64)


def evaluate_holdout(
    model: OneVsRestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_soft_test: np.ndarray,
) -> dict:
    y_pred = model.predict(X_test)
    proba = _predict_proba_matrix(model, X_test)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    per_label_ap = {
        name: float(average_precision_score(y_test[:, i], proba[:, i]))
        if y_test[:, i].sum() > 0
        else 0.0
        for i, name in enumerate(REC_TYPES)
    }
    try:
        ranking_ap = float(label_ranking_average_precision_score(y_test, proba))
    except ValueError:
        ranking_ap = 0.0

    # Soft-label ranking correlation proxy: mean AP vs continuous soft targets thresholded.
    soft_binary = (y_soft_test >= LABEL_THRESHOLD).astype(np.int32)
    soft_macro_f1 = float(f1_score(soft_binary, y_pred, average="macro", zero_division=0))

    print(f"\nHoldout macro-F1: {macro_f1:.4f}")
    print(f"Holdout ranking AP: {ranking_ap:.4f}")
    print(f"Soft-aligned macro-F1: {soft_macro_f1:.4f}")
    print("Per-label average precision:")
    for name, ap in per_label_ap.items():
        print(f"  {name}: {ap:.4f}")

    return {
        "macro_f1": macro_f1,
        "ranking_average_precision": ranking_ap,
        "soft_aligned_macro_f1": soft_macro_f1,
        "per_label_average_precision": per_label_ap,
        "n_test": int(X_test.shape[0]),
        "positive_rates": {
            name: float(y_test[:, i].mean()) for i, name in enumerate(REC_TYPES)
        },
    }


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    """Multi-label aware fold metrics (KFold; report per-label + macro)."""
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_macro: list[float] = []
    fold_ap: dict[str, list[float]] = {name: [] for name in REC_TYPES}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        model = make_estimator()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        proba = _predict_proba_matrix(model, X[val_idx])
        y_val = y[val_idx]
        mf1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        fold_macro.append(mf1)
        for i, name in enumerate(REC_TYPES):
            if y_val[:, i].sum() == 0:
                continue
            fold_ap[name].append(float(average_precision_score(y_val[:, i], proba[:, i])))
        print(f"  CV fold {fold}: macro-F1={mf1:.4f}")

    return {
        "n_folds": N_CV_FOLDS,
        "macro_f1_mean": float(np.mean(fold_macro)) if fold_macro else 0.0,
        "macro_f1_std": float(np.std(fold_macro)) if fold_macro else 0.0,
        "per_label_average_precision_mean": {
            name: float(np.mean(vals)) if vals else 0.0 for name, vals in fold_ap.items()
        },
    }


def save_artifacts(
    model: OneVsRestClassifier,
    metrics: dict,
    cv_metrics: dict,
    *,
    n_base: int,
    n_train: int,
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(FEATURE_NAMES), FEATURES_PATH)
    metadata = {
        "name": "recommendation-ranker",
        "version": MODEL_VERSION,
        "algorithm": "OneVsRestClassifier(HistGradientBoostingClassifier)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "label_types": list(REC_TYPES),
        "label_threshold": LABEL_THRESHOLD,
        "n_base_rows": n_base,
        "n_augmented_train": n_train,
        "augmentation_target": AUGMENT_TARGET,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "notes": (
            "Multi-label need predictor trained on Sleep Health & Lifestyle CSV "
            "with guideline-derived soft labels and Gaussian/interpolation augmentation. "
            "Informational lifestyle suggestions only; not a medical diagnosis."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved features -> {FEATURES_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train recommendation multi-label ranker")
    parser.add_argument("--evaluate-only", action="store_true", help="CV + holdout only; do not save")
    parser.add_argument("--no-save", action="store_true", help="Train but skip writing artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--augment-target", type=int, default=AUGMENT_TARGET)
    args = parser.parse_args()

    if MODEL_PATH.is_file() and not args.overwrite and not args.evaluate_only and not args.no_save:
        print(f"Artifact exists: {MODEL_PATH}")
        print("Pass --overwrite to replace, or --evaluate-only / --no-save.")
        return 1

    print("Loading CSV and building guideline labels...")
    signals, y_hard, y_soft = build_base_dataset()
    X = build_feature_matrix(signals)
    n_base = X.shape[0]
    print(f"Base rows: {n_base}, features: {X.shape[1]}, labels: {list(REC_TYPES)}")
    print("Positive rates (base):", {name: float(y_hard[:, i].mean()) for i, name in enumerate(REC_TYPES)})

    print(f"\nAugmenting toward {args.augment_target} samples...")
    X_aug, y_aug, soft_aug = augment_dataset(
        X, y_hard, y_soft, target_n=args.augment_target, random_state=RANDOM_STATE
    )
    print(f"Augmented size: {X_aug.shape[0]}")

    X_train, X_test, y_train, y_test, soft_train, soft_test = train_test_split(
        X_aug,
        y_aug,
        soft_aug,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # soft_train unused beyond split symmetry
    _ = soft_train

    print(f"\n{N_CV_FOLDS}-fold CV on train split...")
    cv_metrics = cross_validate(X_train, y_train)
    print(
        f"CV macro-F1 mean={cv_metrics['macro_f1_mean']:.4f} "
        f"std={cv_metrics['macro_f1_std']:.4f}"
    )

    print("\nFitting final model on train split...")
    model = make_estimator()
    model.fit(X_train, y_train)
    metrics = evaluate_holdout(model, X_test, y_test, soft_test)

    if args.evaluate_only or args.no_save:
        print("\nSkipping artifact save.")
        return 0

    # Refit on full augmented data for production artifact.
    print("\nRefitting on full augmented dataset for export...")
    final = make_estimator()
    final.fit(X_aug, y_aug)
    save_artifacts(final, metrics, cv_metrics, n_base=n_base, n_train=int(X_aug.shape[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
