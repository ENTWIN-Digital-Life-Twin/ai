import joblib
import pandas as pd

from app.core.config import get_settings
from app.ml.preprocessing.sleep import build_sleep_feature_frame, encode_sleep_row
from tests.regression_samples import FROZEN_SLEEP_FEATURES


def test_joblib_feature_order_is_frozen():
    settings = get_settings()
    loaded = joblib.load(settings.resolve_path(settings.sleep_features_path))
    assert list(loaded) == FROZEN_SLEEP_FEATURES


def test_preprocess_uses_authoritative_order():
    encoded = encode_sleep_row(
        age=24,
        gender="Female",
        sleep_duration=7.2,
        quality_of_sleep=8,
        physical_activity_level=60,
        stress_level=4,
        bmi_category="Normal",
        heart_rate=72,
        daily_steps=8000,
    )
    frame = build_sleep_feature_frame(encoded, FROZEN_SLEEP_FEATURES)
    assert list(frame.columns) == FROZEN_SLEEP_FEATURES
    assert frame.shape == (1, 11)
    assert int(frame.loc[0, "Gender"]) == 1
    assert int(frame.loc[0, "BMI_Normal"]) == 1
    assert int(frame.loc[0, "BMI_Obese"]) == 0


def test_wrong_order_without_reindex_would_not_match():
    encoded = encode_sleep_row(
        age=40,
        gender="Male",
        sleep_duration=6.0,
        quality_of_sleep=5,
        physical_activity_level=30,
        stress_level=8,
        bmi_category="Obese",
        heart_rate=85,
        daily_steps=3000,
    )
    shuffled = list(reversed(FROZEN_SLEEP_FEATURES))
    naive = pd.DataFrame([[encoded[name] for name in shuffled]], columns=shuffled)
    aligned = build_sleep_feature_frame(encoded, FROZEN_SLEEP_FEATURES)
    assert list(naive.columns) != list(aligned.columns)
    assert list(aligned.columns) == FROZEN_SLEEP_FEATURES
    assert list(aligned.iloc[0]) != list(naive.iloc[0])
