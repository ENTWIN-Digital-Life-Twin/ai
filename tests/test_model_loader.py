import joblib
import pytest

from app.core.config import Settings
from app.core.exceptions import ModelLoadError
from app.ml.model_loader import load_registry
from tests.regression_samples import FROZEN_SLEEP_FEATURES


def test_missing_required_model_fails_cleanly(tmp_path):
    settings = Settings(
        sleep_model_path=str(tmp_path / "missing_model.joblib"),
        sleep_features_path=str(tmp_path / "missing_features.joblib"),
        sleep_model_required=True,
        lifestyle_model_required=False,
    )
    with pytest.raises(ModelLoadError, match="missing"):
        load_registry(settings)


def test_feature_file_mismatch(tmp_path):
    real_settings = Settings()
    model_src = real_settings.resolve_path(real_settings.sleep_model_path)
    estimator = joblib.load(model_src)
    model_copy = tmp_path / "sleep_disorder_model.joblib"
    features_copy = tmp_path / "sleep_disorder_features.joblib"
    joblib.dump(estimator, model_copy)
    joblib.dump(["Age"], features_copy)

    settings = Settings(
        sleep_model_path=str(model_copy),
        sleep_features_path=str(features_copy),
        sleep_model_required=True,
        lifestyle_model_required=False,
        lifestyle_model_path=str(tmp_path / "no-lifestyle.joblib"),
    )
    with pytest.raises(ModelLoadError, match="does not match"):
        load_registry(settings)


def test_loader_does_not_call_joblib_per_prediction(client, monkeypatch):
    calls = {"n": 0}
    real_load = joblib.load

    def counting_load(path):
        calls["n"] += 1
        return real_load(path)

    monkeypatch.setattr("app.ml.model_loader.joblib.load", counting_load)
    before = calls["n"]
    client.post(
        "/api/v1/ai/sleep-risk",
        json={
            "age": 24,
            "gender": "Female",
            "sleepDuration": 7.2,
            "qualityOfSleep": 8,
            "physicalActivityLevel": 60,
            "stressLevel": 4,
            "bmiCategory": "Normal",
            "heartRate": 72,
            "dailySteps": 8000,
        },
    )
    assert calls["n"] == before
    assert FROZEN_SLEEP_FEATURES == client.app.state.models.sleep.feature_names


def test_optional_lifestyle_missing_does_not_block_startup(tmp_path):
    real = Settings()
    settings = Settings(
        sleep_model_path=str(real.resolve_path(real.sleep_model_path)),
        sleep_features_path=str(real.resolve_path(real.sleep_features_path)),
        sleep_metadata_path=str(real.resolve_path(real.sleep_metadata_path)),
        lifestyle_model_path=str(tmp_path / "absent.joblib"),
        lifestyle_model_required=False,
        sleep_model_required=True,
    )
    registry = load_registry(settings)
    assert registry.sleep is not None
    assert registry.lifestyle is None
