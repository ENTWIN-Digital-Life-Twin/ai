"""Train or evaluate the sleep-risk DecisionTree.

Training and inference are separate. The API never calls this script.

Examples:
  python scripts/train_model.py --evaluate-only
  python scripts/train_model.py --no-save
  python scripts/train_model.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "sleep_health_lifestyle.csv"
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "sleep_disorder_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "app" / "models" / "sleep_disorder_features.joblib"
METADATA_PATH = PROJECT_ROOT / "app" / "models" / "sleep_disorder_model.metadata.json"
STAGING_DIR = PROJECT_ROOT / "app" / "models" / "staging"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 5
FEATURES_NUM = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
]
TARGET = "Sleep Disorder"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].fillna("None")
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    bmi_encoded = pd.get_dummies(df["BMI Category"], prefix="BMI")
    X = pd.concat([df[FEATURES_NUM + ["Gender"]], bmi_encoded], axis=1)
    y = df[TARGET]
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def evaluate(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    labels = list(model.classes_)
    accuracy = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"\nHoldout accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix labels:", labels)
    print(matrix)
    return {
        "testSize": TEST_SIZE,
        "randomState": RANDOM_STATE,
        "stratified": True,
        "accuracy": round(float(accuracy), 4),
        "macroPrecision": round(float(macro_p), 4),
        "macroRecall": round(float(macro_r), 4),
        "macroF1": round(float(macro_f), 4),
        "weightedPrecision": round(float(weighted_p), 4),
        "weightedRecall": round(float(weighted_r), 4),
        "weightedF1": round(float(weighted_f), 4),
        "confusionMatrixLabels": labels,
        "confusionMatrix": matrix.tolist(),
        "note": (
            "Holdout metrics on the 80/20 stratified split (random_state=42). "
            "Decision-tree predict_proba values are uncalibrated."
        ),
    }


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(
        f"\n5-fold stratified CV accuracy: mean={scores.mean():.2%} std={scores.std():.2%} "
        f"folds={scores.round(4).tolist()}"
    )
    print("CV retrains clones with the same hyperparameters; it is not production validation.")
    return {
        "folds": 5,
        "strategy": "StratifiedKFold",
        "shuffle": True,
        "randomState": RANDOM_STATE,
        "accuracyMean": round(float(scores.mean()), 4),
        "accuracyStd": round(float(scores.std()), 4),
        "note": (
            "CV retrains clones with the same hyperparameters; it does not re-evaluate "
            "the frozen production pickle and is not production validation."
        ),
    }


def baseline_naive(stress_level: float) -> str:
    if stress_level <= 4:
        return "None"
    if stress_level <= 6:
        return "Insomnia"
    return "Sleep Apnea"


def compare_baseline(X_test: pd.DataFrame, y_test: pd.Series, model_accuracy: float) -> None:
    baseline_preds = X_test["Stress Level"].apply(baseline_naive)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    print(f"\nNaive stress-only baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Model holdout accuracy: {model_accuracy:.2%}")
    print(f"Gain vs naive baseline: {(model_accuracy - baseline_accuracy) * 100:.2f} points")


def build_metadata(X: pd.DataFrame, y: pd.Series, holdout: dict, cv: dict) -> dict:
    return {
        "name": "sleep-risk",
        "version": "1.0.0",
        "algorithm": "DecisionTreeClassifier",
        "sklearnVersion": "1.9.0",
        "maxDepth": MAX_DEPTH,
        "randomState": RANDOM_STATE,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "features": list(X.columns),
        "datasetName": "Sleep Health and Lifestyle Dataset",
        "datasetRows": int(len(X)),
        "holdout": holdout,
        "crossValidation": cv,
        "limitations": [
            "Small academic dataset (374 rows)",
            "Class imbalance (None majority class)",
            "Prototype quality; not a medical diagnostic tool",
            "predict_proba is a tree vote share, not a calibrated clinical probability",
        ],
    }


def save_artifacts(model, feature_names: list[str], metadata: dict, overwrite: bool) -> None:
    target_model = MODEL_PATH if overwrite else STAGING_DIR / "sleep_disorder_model.joblib"
    target_features = FEATURES_PATH if overwrite else STAGING_DIR / "sleep_disorder_features.joblib"
    target_metadata = METADATA_PATH if overwrite else STAGING_DIR / "sleep_disorder_model.metadata.json"
    target_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_model)
    joblib.dump(list(feature_names), target_features)
    target_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model: {target_model}")
    print(f"Saved features: {target_features}")
    print(f"Saved metadata: {target_metadata}")
    if not overwrite:
        print("Production artifacts were not overwritten. Pass --overwrite to replace them.")


def evaluate_only() -> None:
    X, y = load_dataset()
    if MODEL_PATH.is_file() and FEATURES_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        X = X[feature_names]
        print("Evaluating the existing production pickle (no retraining).")
    else:
        raise SystemExit("Production model artifacts are missing.")
    X_train, X_test, y_train, y_test = split(X, y)
    del X_train, y_train
    holdout = evaluate(model, X_test, y_test)
    compare_baseline(X_test, y_test, holdout["accuracy"])
    cv = cross_validate(X, y)
    METADATA_PATH.write_text(json.dumps(build_metadata(X, y, holdout, cv), indent=2), encoding="utf-8")
    print(f"\nUpdated metadata: {METADATA_PATH}")


def train(no_save: bool, overwrite: bool) -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {list(X.columns)}")
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    holdout = evaluate(model, X_test, y_test)
    compare_baseline(X_test, y_test, holdout["accuracy"])
    cv = cross_validate(X, y)
    metadata = build_metadata(X, y, holdout, cv)
    if no_save:
        print("\n--no-save set; artifacts were not written.")
        return
    save_artifacts(model, list(X.columns), metadata, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the ENTWIN sleep-risk model")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the existing production pickle without training",
    )
    parser.add_argument("--no-save", action="store_true", help="Train without writing artifacts")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace production joblib files (default writes to app/models/staging/)",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_only()
        return
    train(no_save=args.no_save, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
