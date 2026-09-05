from dataclasses import dataclass

import pandas as pd


def encode_sleep_row(
    *,
    age: int,
    gender: str,
    sleep_duration: float,
    quality_of_sleep: int,
    physical_activity_level: int,
    stress_level: int,
    bmi_category: str,
    heart_rate: int,
    daily_steps: int,
) -> dict[str, float | int]:
    """Encode a sleep profile the same way as scripts/train_model.py.

    Gender: Male=0, Female=1.
    BMI: one-hot after mapping 'Normal Weight' -> 'Normal'.
    """
    normalized_bmi = "Normal" if bmi_category in {"Normal", "Normal Weight"} else bmi_category
    return {
        "Age": age,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity_level,
        "Stress Level": stress_level,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Gender": 0 if gender == "Male" else 1,
        "BMI_Normal": 1 if normalized_bmi == "Normal" else 0,
        "BMI_Obese": 1 if normalized_bmi == "Obese" else 0,
        "BMI_Overweight": 1 if normalized_bmi == "Overweight" else 0,
    }


def build_sleep_feature_frame(encoded_row: dict[str, float | int], feature_names: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in encoded_row]
    if missing:
        raise ValueError(f"Encoded row is missing required features: {missing}")
    return pd.DataFrame([encoded_row])[feature_names]


@dataclass(frozen=True)
class SleepFeatureVector:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
