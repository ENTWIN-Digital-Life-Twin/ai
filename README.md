# AI Service - Digital Life Twin

## Etat actuel
2 endpoints FastAPI :

### POST /predict/lifestyle-risk
Regle simple (pas un modele ML), basee sur 4 scores 0-100
(sommeil, hydratation, activite, stress)

### POST /predict/sleep-disorder
Modele de machine learning entraine (Decision Tree), 96% de precision
sur donnees de test, valide contre une baseline (+64 points).
Entraine sur un dataset public (Kaggle - Sleep Health and Lifestyle
Dataset)

Features utilisees : Age, Gender, Sleep Duration, Quality of Sleep,
Physical Activity Level, Stress Level, BMI Category, Heart Rate, Daily Steps.

## Limites connues
- Dataset d'entrainement petit (374 lignes)
- Pas de validation croisee, pas d'optimisation des hyperparametres
- Pas encore branche sur les vraies donnees de wellness-service


## Lancer le service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Puis http://127.0.0.1:8000/docs