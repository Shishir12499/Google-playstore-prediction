from __future__ import annotations

import pickle
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "googleplaystore.csv"
MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"
TYPE_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

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


def _to_number(value: object) -> float:
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


@st.cache_data
def _load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["App"]).copy()
    data["source_url"] = data["App"].apply(
        lambda name: f"https://play.google.com/store/search?q={quote_plus(str(name))}&c=apps"
    )
    return data


@st.cache_resource
def _load_pickle(path: Path):
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def _encode_feature(data: pd.DataFrame, column: str, value: object) -> float:
    if column in {"Rating", "Reviews", "Size", "Installs", "Price"}:
        return _to_number(value)

    known_values = sorted(data[column].dropna().astype(str).unique())
    value_map = {item: index for index, item in enumerate(known_values)}
    return float(value_map.get(str(value), -1))


def predict_type(row: pd.Series, data: pd.DataFrame, model, type_encoder) -> str | None:
    if model is None or type_encoder is None:
        return None

    features = pd.DataFrame(
        [[_encode_feature(data, column, row.get(column, "")) for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(features)
    return str(type_encoder.inverse_transform(prediction)[0])


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return str(value)


def _find_matches(data: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return data.iloc[0:0]
    return data[data["App"].str.contains(query, case=False, na=False, regex=False)]


def _render_result(row: pd.Series, predicted_type: str | None, suggestions: list[str]) -> None:
    st.subheader(_format_value(row.get("App")))
    st.caption(f"{_format_value(row.get('Category'))} | {_format_value(row.get('Genres'))}")
    st.link_button("Open Google Play Search", _format_value(row.get("source_url")))

    top_cols = st.columns(4)
    top_cols[0].metric("Rating", _format_value(row.get("Rating")))
    top_cols[1].metric("Reviews", _format_value(row.get("Reviews")))
    top_cols[2].metric("Installs", _format_value(row.get("Installs")))
    top_cols[3].metric("Price", _format_value(row.get("Price")))

    details = {
        "Dataset Type": _format_value(row.get("Type")),
        "Predicted Type": predicted_type or "Unavailable",
        "Size": _format_value(row.get("Size")),
        "Content Rating": _format_value(row.get("Content Rating")),
        "Last Updated": _format_value(row.get("Last Updated")),
        "Android Version": _format_value(row.get("Android Ver")),
    }

    st.dataframe(
        pd.DataFrame(details.items(), columns=["Field", "Value"]),
        hide_index=True,
        use_container_width=True,
    )

    if len(suggestions) > 1:
        st.caption("Closest dataset matches: " + ", ".join(suggestions))


def main() -> None:
    st.set_page_config(
        page_title="Google Play Store App Finder",
        layout="centered",
    )

    st.title("Google Play Store App Finder")
    st.write("Search an app from the dataset and open its Google Play source search link.")

    data = _load_data()
    model = _load_pickle(MODEL_PATH)
    type_encoder = _load_pickle(TYPE_ENCODER_PATH)

    with st.form("app_search_form"):
        query = st.text_input("App name", placeholder="Example: WhatsApp Messenger")
        submitted = st.form_submit_button("Find App", type="primary")

    if submitted:
        query = query.strip()
        if not query:
            st.error("Please enter an app name.")
            return

        matches = _find_matches(data, query)
        if matches.empty:
            st.error(f"No app found for '{query}'.")
            return

        row = matches.iloc[0]
        predicted_type = predict_type(row, data, model, type_encoder)
        suggestions = matches["App"].head(8).tolist()
        _render_result(row, predicted_type, suggestions)
    else:
        st.info(f"Dataset loaded with {len(data):,} apps.")


if __name__ == "__main__":
    main()
