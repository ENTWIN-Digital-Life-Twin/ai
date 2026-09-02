import numpy as np
import pandas as pd
from app.baseline import evaluate_risk

np.random.seed(42)  # pour que le resultat soit reproductible entre nous

N_ROWS = 3000
RISK_TYPES = ["fatigue", "deshydratation", "sedentarite", "surcharge"]
DOMAIN_TO_SCORE = {
    "fatigue": "sleep_score",
    "deshydratation": "hydration_score",
    "sedentarite": "activity_score",
    "surcharge": "stress_score",
}


def normal_score():
    """Score sain : centre autour de 80, reste au-dessus de 60."""
    return float(np.clip(np.random.normal(loc=80, scale=10), 60, 100))


def low_score():
    """Score a risque : centre autour de 35, reste sous 60."""
    return float(np.clip(np.random.normal(loc=35, scale=15), 0, 59))


def generate_row(force_normal):
    scores = {
        "sleep_score": normal_score(),
        "hydration_score": normal_score(),
        "activity_score": normal_score(),
        "stress_score": normal_score(),
    }

    if not force_normal:
        # Combien de scores vont etre degrades en meme temps :
        # majoritairement 1 seul, parfois 2, plus rarement 3 ou 4
        n_low = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
        chosen_fields = np.random.choice(
            list(DOMAIN_TO_SCORE.values()), size=n_low, replace=False
        )
        for field in chosen_fields:
            scores[field] = low_score()

    return scores


def generate_dataset(n_rows=N_ROWS, normal_ratio=0.7, noise_ratio=0.08):
    rows = []
    n_normal = int(n_rows * normal_ratio)

    for i in range(n_rows):
        force_normal = i < n_normal
        row = generate_row(force_normal)
        rows.append(row)

    np.random.shuffle(rows)
    df = pd.DataFrame(rows)

    labels = df.apply(
        lambda r: evaluate_risk(
            r["sleep_score"], r["hydration_score"], r["activity_score"], r["stress_score"]
        )["risk"],
        axis=1,
    )
    df["risk"] = labels

    # Injecter du bruit : quelques etiquettes sont volontairement changees
    # pour simuler l'imperfection des vraies donnees
    all_classes = df["risk"].unique()
    n_noisy = int(len(df) * noise_ratio)
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)

    for idx in noisy_indices:
        current_label = df.loc[idx, "risk"]
        other_classes = [c for c in all_classes if c != current_label]
        df.loc[idx, "risk"] = np.random.choice(other_classes)

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    print(dataset["risk"].value_counts())
    print(dataset.head())

    dataset.to_csv("data/lifestyle_risk_dataset.csv", index=False)
    print("\nSauvegarde dans data/lifestyle_risk_dataset.csv")