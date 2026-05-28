from __future__ import annotations

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "googleplaystore.csv"
MODEL_PATH = "decision_tree_model.pkl"
TYPE_ENCODER_PATH = "label_encoder.pkl"

FEATURE_COLUMNS = [
    "Category",
    "Rating",
    "Reviews",
    "Size",
    "Installs",
    "Price",
    "Content Rating",
    "Genres",
]


def to_number(value: object) -> float:
    text = str(value).strip().replace(",", "").replace("+", "").replace("$", "")
    if text in {"", "nan", "NaN", "Varies with device"}:
        return 0.0
    multiplier = 1.0
    if text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURE_COLUMNS].copy()
    for column in ["Rating", "Reviews", "Size", "Installs", "Price"]:
        features[column] = features[column].apply(to_number)

    for column in ["Category", "Content Rating", "Genres"]:
        encoder = LabelEncoder()
        features[column] = encoder.fit_transform(features[column].astype(str))

    return features


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["Type"])
    data = data[data["Type"].isin(["Free", "Paid"])].copy()

    x = prepare_features(data)
    type_encoder = LabelEncoder()
    y = type_encoder.fit_transform(data["Type"])

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(x_train, y_train)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(TYPE_ENCODER_PATH, "wb") as file:
        pickle.dump(type_encoder, file)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved type encoder to {TYPE_ENCODER_PATH}")


if __name__ == "__main__":
    main()
