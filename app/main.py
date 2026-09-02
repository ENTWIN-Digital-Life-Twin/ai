import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.baseline import evaluate_risk

app = FastAPI(title="Digital life Twin - AI Service")

# Chargement du modele et de la liste des features
sleep_model = joblib.load("app/models/sleep_disorder_model.joblib")
sleep_features = joblib.load("app/models/sleep_disorder_features.joblib")

class DailyScores(BaseModel):
    sleep_score: float = Field(..., ge=0, le=100)
    hydration_score: float = Field(..., ge=0, le=100)
    activity_score: float = Field(..., ge=0, le=100)
    stress_score: float = Field(..., ge=0, le=100)


class SleepProfile(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(Male|Female)$")
    sleep_duration: float = Field(..., ge=0, le=24)
    quality_of_sleep: int = Field(..., ge=1, le=10)
    physical_activity_level: int = Field(..., ge=0, le=200)
    stress_level: int = Field(..., ge=1, le=10)
    bmi_category: str = Field(..., pattern="^(Normal|Overweight|Obese)$")
    heart_rate: int = Field(..., ge=30, le=220)
    daily_steps: int = Field(..., ge=0, le=50000)


@app.get("/")
def root():
    return {"message": "AI service is running"}


@app.post("/predict/lifestyle_risk")
def predict_lifestyle_risk(scores: DailyScores):
    result = evaluate_risk(
        sleep_score=scores.sleep_score,
        hydration_score=scores.hydration_score,
        activity_score=scores.activity_score,
        stress_score=scores.stress_score
    )
    return result


@app.post("/predict/sleep_disorder")
def predict_sleep_disorder(profile: SleepProfile):
    row = {
        "Age": profile.age,
        "Sleep Duration": profile.sleep_duration,
        "Quality of Sleep": profile.quality_of_sleep,
        "Physical Activity Level": profile.physical_activity_level,
        "Stress Level": profile.stress_level,
        "Heart Rate": profile.heart_rate,
        "Daily Steps": profile.daily_steps,
        "Gender": 0 if profile.gender == "Male" else 1,
        "BMI_Normal": 1 if profile.bmi_category == "Normal" else 0,
        "BMI_Obese": 1 if profile.bmi_category == "Obese" else 0,
        "BMI_Overweight": 1 if profile.bmi_category == "Overweight" else 0,
    }

    X_input = pd.DataFrame([row])[sleep_features]

    prediction = sleep_model.predict(X_input)[0]
    probabilities = sleep_model.predict_proba(X_input)[0]
    confidence = round(max(probabilities), 2)

    return {"risk": prediction, "confidence": confidence}


