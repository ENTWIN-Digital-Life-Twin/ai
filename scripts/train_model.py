import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Charger les donnees
df = pd.read_csv("data/sleep_health_lifestyle.csv")

# 2. Nettoyage
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")

# 3. Preparation des features
FEATURES_NUM = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                 "Stress Level", "Heart Rate", "Daily Steps"]
TARGET = "Sleep Disorder"

df_bmi_encoded = pd.get_dummies(df["BMI Category"], prefix="BMI")

X = pd.concat([df[FEATURES_NUM + ["Gender"]], df_bmi_encoded], axis=1)
y = df[TARGET]

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} lignes | Test: {len(X_test)} lignes")

# 5. Entrainement
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy sur le test set : {accuracy:.2%}")
print("\nRapport detaille :")
print(classification_report(y_test, y_pred))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("Ordre des classes :", list(model.classes_))

# 7. Comparaison a la baseline
def baseline_naive(stress_level):
    if stress_level <= 4:
        return "None"
    elif stress_level <= 6:
        return "Insomnia"
    else:
        return "Sleep Apnea"

baseline_preds = X_test["Stress Level"].apply(baseline_naive)
baseline_accuracy = accuracy_score(y_test, baseline_preds)
print(f"\nAccuracy baseline naive (Stress Level seul) : {baseline_accuracy:.2%}")
print(f"Accuracy modele ML : {accuracy:.2%}")
print(f"Gain du modele ML : {(accuracy - baseline_accuracy)*100:.2f} points")

# 8. Sauvegarde du modele entraine
joblib.dump(model, "app/models/sleep_disorder_model.joblib")
print("\nModele sauvegarde dans app/models/sleep_disorder_model.joblib")

# 9. Sauvegarde de la liste des colonnes (pour reconstruire le bon format plus tard dans l'API)
joblib.dump(list(X.columns), "app/models/sleep_disorder_features.joblib")
print("Liste des features sauvegardee")